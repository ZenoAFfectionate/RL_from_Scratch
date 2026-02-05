import time
import numpy as np
import threading
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty

from vllm import LLM, SamplingParams


# ============================================================================
# Continuous Batching: Sequence State Management
# ============================================================================

class SequenceStatus(Enum):
    """Sequence lifecycle status."""
    PENDING = "pending"
    RUNNING = "running"
    FINISHED_EOS = "finished_eos"
    FINISHED_LENGTH = "finished_length"
    CANCELLED = "cancelled"


@dataclass
class SequenceState:
    """
    State management class for a single sequence.
    
    Tracks all state information from creation to completion.
    """
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int] = field(default_factory=list)
    draft_token_ids: List[int] = field(default_factory=list)
    
    max_tokens: int = 256
    status: SequenceStatus = SequenceStatus.PENDING
    
    # Statistics
    total_draft_proposed: int = 0
    total_draft_accepted: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Optional completion callback
    on_complete: Optional[Callable[['SequenceState'], None]] = None
    
    @property
    def current_input_ids(self) -> List[int]:
        """Get complete input_ids (prompt + generated)."""
        return self.prompt_token_ids + self.generated_token_ids
    
    @property
    def num_generated(self) -> int:
        """Number of generated tokens."""
        return len(self.generated_token_ids)
    
    @property
    def is_finished(self) -> bool:
        """Check if sequence is finished."""
        return self.status in (
            SequenceStatus.FINISHED_EOS, 
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.CANCELLED
        )
    
    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate."""
        if self.total_draft_proposed == 0:
            return 0.0
        return self.total_draft_accepted / self.total_draft_proposed
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate generation speed."""
        elapsed = (self.end_time or time.time()) - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.num_generated / elapsed
    
    def update_with_accepted_tokens(self, accepted_tokens: List[int], 
                                    n_accepted: int, n_proposed: int,
                                    eos_token_id: int):
        """
        Update state with verified tokens.
        
        Args:
            accepted_tokens: Accepted tokens (may include bonus token).
            n_accepted: Number of accepted draft tokens.
            n_proposed: Number of proposed draft tokens.
            eos_token_id: EOS token ID.
        """
        self.generated_token_ids.extend(accepted_tokens)
        self.total_draft_proposed += n_proposed
        self.total_draft_accepted += n_accepted
        self.draft_token_ids = []
        
        # Check completion conditions
        if accepted_tokens and accepted_tokens[-1] == eos_token_id:
            self.status = SequenceStatus.FINISHED_EOS
            self.end_time = time.time()
        elif self.num_generated >= self.max_tokens:
            self.status = SequenceStatus.FINISHED_LENGTH
            self.end_time = time.time()
    
    def set_draft_tokens(self, draft_tokens: List[int]):
        """Set draft tokens pending verification."""
        self.draft_token_ids = draft_tokens
    
    def mark_running(self):
        """Mark sequence as running."""
        if self.status == SequenceStatus.PENDING:
            self.status = SequenceStatus.RUNNING
            self.start_time = time.time()
    
    def cancel(self):
        """Cancel sequence."""
        self.status = SequenceStatus.CANCELLED
        self.end_time = time.time()
    
    def get_result(self, tokenizer) -> Dict[str, Any]:
        """Get final result."""
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "generated_text": tokenizer.decode(self.generated_token_ids),
            "full_text": tokenizer.decode(self.current_input_ids),
            "num_generated_tokens": self.num_generated,
            "status": self.status.value,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_second": self.tokens_per_second,
            "elapsed_time": (self.end_time or time.time()) - self.start_time
        }


class ContinuousBatchScheduler:
    """
    Continuous Batching Scheduler.
    
    Supports dynamic addition/removal of sequences with batched draft and verify.
    """
    
    def __init__(self, decoder: 'SpeculativeDecoder', max_batch_size: int = 32,
                 num_speculative_tokens: int = 16):
        """
        Initialize the scheduler.
        
        Args:
            decoder: SpeculativeDecoder instance.
            max_batch_size: Maximum batch size.
            num_speculative_tokens: Number of tokens to speculate per iteration.
        """
        self.decoder = decoder
        self.max_batch_size = max_batch_size
        self.num_speculative_tokens = num_speculative_tokens
        
        # Sequence management
        self.active_sequences: Dict[str, SequenceState] = {}
        self.pending_queue: Queue = Queue()
        self.completed_sequences: Dict[str, SequenceState] = {}
        
        # Thread safety lock
        self._lock = threading.Lock()
        
        # Running state
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_iterations = 0
        self.total_sequences_processed = 0
    
    def add_request(self, prompt: str, max_tokens: int = 256,
                    request_id: Optional[str] = None,
                    on_complete: Optional[Callable[[SequenceState], None]] = None) -> str:
        """
        Add a new generation request.
        
        Args:
            prompt: Input text.
            max_tokens: Maximum generation length.
            request_id: Request ID (optional, auto-generated if not provided).
            on_complete: Callback function when completed.
        
        Returns:
            request_id: Unique identifier for the request.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]
        
        prompt_token_ids = self.decoder.target_tokenizer.encode(prompt)
        
        seq_state = SequenceState(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            on_complete=on_complete
        )
        
        self.pending_queue.put(seq_state)
        return request_id
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request."""
        with self._lock:
            if request_id in self.active_sequences:
                self.active_sequences[request_id].cancel()
                return True
        return False
    
    def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get request result."""
        if request_id in self.completed_sequences:
            seq = self.completed_sequences[request_id]
            return seq.get_result(self.decoder.target_tokenizer)
        return None
    
    def _merge_pending_requests(self):
        """Merge pending requests into the active batch."""
        available_slots = self.max_batch_size - len(self.active_sequences)
        
        while available_slots > 0:
            try:
                seq_state = self.pending_queue.get_nowait()
                seq_state.mark_running()
                with self._lock:
                    self.active_sequences[seq_state.request_id] = seq_state
                available_slots -= 1
            except Empty:
                break
    
    def _remove_finished_sequences(self) -> List[SequenceState]:
        """Remove finished sequences and return them."""
        finished = []
        with self._lock:
            to_remove = [
                req_id for req_id, seq in self.active_sequences.items()
                if seq.is_finished
            ]
            for req_id in to_remove:
                seq = self.active_sequences.pop(req_id)
                self.completed_sequences[req_id] = seq
                finished.append(seq)
                self.total_sequences_processed += 1
                
                # Trigger callback
                if seq.on_complete:
                    try:
                        seq.on_complete(seq)
                    except Exception as e:
                        print(f"Callback error for {req_id}: {e}")
        
        return finished
    
    def _batch_generate_draft_tokens(self, sequences: List[SequenceState]) -> Dict[str, List[int]]:
        """Generate draft tokens in batch."""
        if not sequences: return {}
        
        all_input_ids = [seq.current_input_ids for seq in sequences]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.num_speculative_tokens,
            detokenize=False,
            stop_token_ids=[self.decoder.target_eos_id]
        )
        
        outputs = self.decoder.draft_llm.generate(
            prompt_token_ids=all_input_ids,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        
        results = {}
        for seq, output in zip(sequences, outputs):
            draft_tokens = list(output.outputs[0].token_ids)
            results[seq.request_id] = draft_tokens
            seq.set_draft_tokens(draft_tokens)
        
        return results
    
    def _batch_verify_tokens(self, sequences: List[SequenceState]) -> Dict[str, Tuple[List[int], int, bool]]:
        """Verify draft tokens in batch."""
        if not sequences: return {}
        
        all_candidate_ids = [
            seq.current_input_ids + seq.draft_token_ids
            for seq in sequences
        ]
        
        outputs = self.decoder.target_llm.generate(
            prompt_token_ids=all_candidate_ids,
            sampling_params=self.decoder.verify_sampling_params,
            use_tqdm=False
        )
        
        results = {}
        for seq, output in zip(sequences, outputs):
            input_len = len(seq.current_input_ids)
            draft_tokens = seq.draft_token_ids
            num_drafts = len(draft_tokens)
            
            full_logprobs = output.prompt_logprobs
            logprobs_slice = full_logprobs[input_len:input_len + num_drafts]
            
            target_predictions = [
                next(iter(lp.keys())) if lp else None
                for lp in logprobs_slice
            ]
            
            accepted_tokens = []
            n_accepted = 0
            finished = False
            
            for draft_token, target_pred in zip(draft_tokens, target_predictions):
                if target_pred is None:
                    continue
                
                if target_pred == draft_token:
                    accepted_tokens.append(draft_token)
                    n_accepted += 1
                    if draft_token == self.decoder.target_eos_id:
                        finished = True
                        break
                else:
                    accepted_tokens.append(target_pred)
                    finished = (target_pred == self.decoder.target_eos_id)
                    break
            
            # Bonus token handling
            if n_accepted == num_drafts and not finished:
                bonus_token = output.outputs[0].token_ids[0]
                accepted_tokens.append(bonus_token)
                if bonus_token == self.decoder.target_eos_id:
                    finished = True
            
            results[seq.request_id] = (accepted_tokens, n_accepted, finished)
        
        return results
    
    def step(self) -> Tuple[int, List[SequenceState]]:
        """
        Execute one continuous batching iteration.
        
        Returns:
            Tuple of (active_count, finished_sequences).
        """
        # 1. Merge pending requests into active batch
        self._merge_pending_requests()
        
        if not self.active_sequences:
            return 0, []
        
        # 2. Get all active sequences
        with self._lock:
            active_seqs = list(self.active_sequences.values())
        
        # 3. Batch generate draft tokens
        self._batch_generate_draft_tokens(active_seqs)
        
        # 4. Batch verify
        verify_results = self._batch_verify_tokens(active_seqs)
        
        # 5. Update sequence states
        for seq in active_seqs:
            if seq.request_id in verify_results:
                accepted_tokens, n_accepted, _ = verify_results[seq.request_id]
                seq.update_with_accepted_tokens(
                    accepted_tokens, n_accepted, len(seq.draft_token_ids),
                    self.decoder.target_eos_id
                )
        
        # 6. Remove finished sequences
        finished = self._remove_finished_sequences()
        
        self.total_iterations += 1
        
        return len(self.active_sequences), finished
    
    def run_until_complete(self, prompts: List[str], max_tokens: int = 256,
                           verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts until all are complete.
        
        Args:
            prompts: List of input texts.
            max_tokens: Maximum generation length per sequence.
            verbose: Whether to print progress info.
        
        Returns:
            List of results for all sequences.
        """
        start_time = time.time()
        request_ids = []
        
        for prompt in prompts:
            req_id = self.add_request(prompt, max_tokens=max_tokens)
            request_ids.append(req_id)
        
        if verbose:
            print(f"Added {len(prompts)} requests to batch")
        
        iteration = 0
        while self.active_sequences or not self.pending_queue.empty():
            active_count, finished = self.step()
            iteration += 1
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: {active_count} active, "
                      f"{len(self.completed_sequences)} completed")
        
        elapsed = time.time() - start_time
        
        results = [self.get_result(req_id) for req_id in request_ids]
        
        if verbose:
            total_tokens = sum(r["num_generated_tokens"] for r in results if r)
            avg_acc_rate = np.mean([r["acceptance_rate"] for r in results if r])
            print(f"\n=== Continuous Batching Complete ===")
            print(f"Total sequences: {len(prompts)}")
            print(f"Total iterations: {iteration}")
            print(f"Total tokens generated: {total_tokens}")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Throughput: {total_tokens / elapsed:.2f} tokens/sec")
            print(f"Average acceptance rate: {avg_acc_rate:.2%}")
        
        return results
    
    def start_background_processing(self):
        """Start background processing thread (for online serving)."""
        if self._running: return
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._scheduler_thread.start()
    
    def stop_background_processing(self):
        """Stop background processing."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
    
    def _background_loop(self):
        """Background processing loop."""
        while self._running:
            if self.active_sequences or not self.pending_queue.empty():
                self.step()
            else:
                time.sleep(0.01)  # Avoid busy waiting
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "active_sequences": len(self.active_sequences),
            "pending_requests": self.pending_queue.qsize(),
            "completed_sequences": len(self.completed_sequences),
            "total_iterations": self.total_iterations,
            "total_sequences_processed": self.total_sequences_processed
        }


class SpeculativeDecoder:
    """Speculative Decoding class that implements vectorized verification."""

    def __init__(self, target_model_name: str, draft_model_name: str, max_tokens: int = 4096,
                 adaptive_k: bool = True, min_k: int = 4, max_k: int = 32, initial_k: int = 16):
        """
        Initialize the speculative decoder with target and draft models.

        Args:
            target_model_name: HuggingFace model ID for the larger target model.
            draft_model_name:  HuggingFace model ID for the smaller draft model.
            max_tokens: Maximum model context length.
            adaptive_k: Whether to use adaptive speculation length.
            min_k: Minimum speculation length.
            max_k: Maximum speculation length.
            initial_k: Initial speculation length.
        """
        print(f"Loading Target Engine: {target_model_name}")
        self.target_llm = LLM(
            model=target_model_name,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True,
            tensor_parallel_size=1,
            enforce_eager=True,  # Disable CUDA graphs
            device='cuda:0',
        )
        self.target_tokenizer = self.target_llm.get_tokenizer()

        print(f"Loading Draft Engine: {draft_model_name}")
        self.draft_llm = LLM(
            model=draft_model_name,
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True,
            tensor_parallel_size=1,
            enforce_eager=True,  # Disable CUDA graphs
            device='cuda:1',
        )
        self.draft_tokenizer = self.draft_llm.get_tokenizer()

        # define sampling params for verification
        self.verify_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,       # Only need 1 bonus token, NOT the full generation length!
            prompt_logprobs=1,  # Critical: request top-1 logprob for verification
            detokenize=False
        )

        # Ensure tokenizers are compatible and set EOS token
        assert self.target_tokenizer.vocab == self.draft_tokenizer.vocab, "Tokenizers must be compatible"
        self.target_eos_id = self.target_tokenizer.eos_token_id

        # Adaptive speculation length parameters
        self.adaptive_k = adaptive_k
        self.min_k = min_k
        self.max_k = max_k
        self.current_k = initial_k
        self.acceptance_history = []
        self._acceptance_window_size = 10

        # Async pipeline executor for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=2)

    def create_batch_scheduler(self, max_batch_size: int = 32,
                               num_speculative_tokens: int = 16) -> ContinuousBatchScheduler:
        """
        Create a Continuous Batching scheduler.
        
        Args:
            max_batch_size: Maximum batch size.
            num_speculative_tokens: Number of tokens to speculate per iteration.
        
        Returns:
            ContinuousBatchScheduler instance.
        """
        return ContinuousBatchScheduler(
            decoder=self,
            max_batch_size=max_batch_size,
            num_speculative_tokens=num_speculative_tokens
        )

    def _update_speculation_length(self, n_accepted: int, n_proposed: int):
        """Update adaptive speculation length based on acceptance rate."""
        if not self.adaptive_k or n_proposed == 0: return

        rate = n_accepted / n_proposed
        self.acceptance_history.append(rate)

        window = self.acceptance_history[-self._acceptance_window_size:]
        avg_rate = sum(window) / len(window)

        if avg_rate > 0.8 and self.current_k < self.max_k:
            self.current_k = min(self.current_k + 2, self.max_k)
        elif avg_rate < 0.5 and self.current_k > self.min_k:
            self.current_k = max(self.current_k - 2, self.min_k)

    def generate_draft_tokens(self, input_ids: List[int], num_speculative_tokens: int = 10) -> List[int]:
        """
        Generate speculative tokens in one forward call using the draft model.

        Args:
            input_ids: Input token IDs.
            num_speculative_tokens: Number of tokens to speculate.

        Returns:
            List of draft token IDs.
        """
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=num_speculative_tokens,
            detokenize=False,
            stop_token_ids=[self.target_eos_id]
        )

        # generate draft tokens in one forward pass
        output = self.draft_llm.generate(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params,
            use_tqdm=False
        )

        return list(output[0].outputs[0].token_ids)
    
    def generate_draft_tree(self, input_ids: List[int], depth: int = 4, width: int = 2):
        """Generate tree-structured candidate tokens."""
        from collections import deque

        tree = []
        queue = deque([(input_ids, 0, [])])  # (current_ids, depth, path)

        sampling_params = SamplingParams(
            temperature=0.7,  # need some randomness
            top_k=width,
            max_tokens=1,
            logprobs=width,
            detokenize=False
        )

        # BFS to generate tree paths
        while queue and len(tree) < width ** depth:
            current_ids, d, path = queue.popleft()
            if d >= depth:
                tree.append(path)
                continue

            output = self.draft_llm.generate(
                prompt_token_ids=current_ids,
                sampling_params=sampling_params,
                use_tqdm=False
            )

            # get top-k candidates
            logprobs = output[0].outputs[0].logprobs[0]
            top_tokens = sorted(logprobs.keys(), key=lambda x: logprobs[x].logprob, reverse=True)[:width]

            for token in top_tokens:
                new_path = path + [token]
                queue.append((current_ids + [token], d + 1, new_path))

        return tree

    def verify_tokens(self, input_ids: List[int], draft_tokens: List[int]) -> Tuple[List[int], int, bool]:
        """
        verify all draft tokens in one forward pass using the target model.

        Args:
            input_ids: The current input token IDs.
            draft_tokens: Draft tokens from the draft model.

        Returns:
            accepted_tokens: List of accepted token IDs.
            n_accepted: Number of draft tokens that were accepted.
            finished: Whether generation should stop (EOS encountered).
        """
        candidate_ids = input_ids + draft_tokens

        # generate target predictions in one pass
        output = self.target_llm.generate(
            prompt_token_ids=candidate_ids,
            sampling_params=self.verify_sampling_params,
            use_tqdm=False
        )
        full_logprobs = output[0].prompt_logprobs

        start_check_idx = len(input_ids)
        num_drafts = len(draft_tokens)

        # Pre-extract all target predictions in one pass using slice to avoid repeat
        logprobs_slice = full_logprobs[start_check_idx:start_check_idx + num_drafts]
        target_predictions = [
            next(iter(lp.keys())) if lp else None
            for lp in logprobs_slice
        ]

        accepted_tokens = []
        n_accepted = 0
        finished = False

        # find first mismatch position
        for draft_token, target_pred in zip(draft_tokens, target_predictions):
            # if target_pred is None, the token is not accepted
            if target_pred is None:
                continue

            if target_pred == draft_token:
                accepted_tokens.append(draft_token)
                n_accepted += 1
                if draft_token == self.target_eos_id:
                    finished = True
                    break
            else:
                accepted_tokens.append(target_pred)
                finished = (target_pred == self.target_eos_id)
                break

        # Bonus token handling
        if n_accepted == num_drafts and not finished:
            bonus_token = output[0].outputs[0].token_ids[0]
            accepted_tokens.append(bonus_token)
            if bonus_token == self.target_eos_id:
                finished = True

        return accepted_tokens, n_accepted, finished

    def speculative_decode(self, prompt: str, max_tokens: int = 256,
                           num_speculative_tokens: int = None,
                           use_async_pipeline: bool = True) -> str:
        """
        Main speculative decoding algorithm with vectorized verification.

        Args:
            prompt: Input text.
            max_tokens: Maximum number of tokens to generate (excluding prompt).
            num_speculative_tokens: Number of tokens to speculate per iteration.
                                    If None and adaptive_k is True, uses adaptive length.
            use_async_pipeline: Whether to use async draft-verify pipeline.

        Returns:
            Generated text.
        """
        input_ids = self.target_tokenizer.encode(prompt)
        initial_len = len(input_ids)

        use_adaptive = num_speculative_tokens is None and self.adaptive_k
        if use_adaptive:
            self.current_k = (self.min_k + self.max_k) // 2
            self.acceptance_history = []

        total_draft_proposed = 0
        total_draft_accepted = 0
        start_time = time.time()
        finished = False

        if use_async_pipeline:
            # ============ Async Pipeline Mode ============
            k = self.current_k if use_adaptive else (num_speculative_tokens or 16)
            
            # First round: synchronously generate initial draft tokens
            draft_tokens = self.generate_draft_tokens(input_ids, k)
            
            while len(input_ids) - initial_len < max_tokens and not finished:
                actual_proposed = len(draft_tokens)
                
                # Optimistic assumption: all current drafts will be accepted
                optimistic_input_ids = input_ids + draft_tokens
                
                # Parallel execution: verify current draft + optimistically generate next draft
                verify_future: Future = self._executor.submit(
                    self.verify_tokens, input_ids, draft_tokens
                )
                
                # Only do optimistic draft if max_tokens not reached
                should_optimistic_draft = (
                    len(optimistic_input_ids) - initial_len < max_tokens
                )
                optimistic_draft_future: Optional[Future] = None
                if should_optimistic_draft:
                    next_k = self.current_k if use_adaptive else (num_speculative_tokens or 16)
                    optimistic_draft_future = self._executor.submit(
                        self.generate_draft_tokens, optimistic_input_ids, next_k
                    )
                
                # Wait for verify result
                valid_tokens, n_accepted, finished = verify_future.result()
                
                total_draft_proposed += actual_proposed
                total_draft_accepted += n_accepted
                input_ids = input_ids + valid_tokens
                
                if use_adaptive:
                    self._update_speculation_length(n_accepted, actual_proposed)
                
                if finished:
                    # Cancel optimistic draft if still running
                    if optimistic_draft_future is not None:
                        optimistic_draft_future.cancel()
                    break
                
                # Decide whether to use optimistic draft result
                if optimistic_draft_future is not None:
                    if n_accepted == actual_proposed:
                        # All accepted: optimistic assumption succeeded, use pre-generated draft
                        draft_tokens = optimistic_draft_future.result()
                    else:
                        # Partial acceptance: optimistic assumption failed, cancel and regenerate
                        optimistic_draft_future.cancel()
                        k = self.current_k if use_adaptive else (num_speculative_tokens or 16)
                        draft_tokens = self.generate_draft_tokens(input_ids, k)
                else:
                    # No optimistic draft, generate synchronously
                    k = self.current_k if use_adaptive else (num_speculative_tokens or 16)
                    draft_tokens = self.generate_draft_tokens(input_ids, k)
        else:
            while len(input_ids) - initial_len < max_tokens and not finished:
                k = self.current_k if use_adaptive else (num_speculative_tokens or 16)

                # generate candidate tokens from draft model
                draft_tokens = self.generate_draft_tokens(input_ids, k)
                actual_proposed = len(draft_tokens)

                # verify all draft tokens in one forward pass using target model
                valid_tokens, n_accepted, finished = self.verify_tokens(input_ids, draft_tokens)

                total_draft_proposed += actual_proposed
                total_draft_accepted += n_accepted
                input_ids = input_ids + valid_tokens

                if use_adaptive:
                    self._update_speculation_length(n_accepted, actual_proposed)

        elapsed = time.time() - start_time
        gen_count = len(input_ids) - initial_len
        acc_rate = total_draft_accepted / total_draft_proposed if total_draft_proposed > 0 else 0

        print(f"Generated {gen_count} tokens in {elapsed:.2f}s")
        print(f"Tokens/sec: {gen_count / elapsed:.2f}")
        print(f"Acceptance Rate: {acc_rate:.2%}")
        if use_adaptive:
            print(f"Final adaptive k: {self.current_k}")
        if use_async_pipeline:
            print(f"Mode: Async Pipeline")

        return self.target_tokenizer.decode(input_ids)


    def benchmark(self, prompt: str, max_tokens: int = 100,
                  num_runs: int = 3, compare_baseline: bool = True,
                  num_speculative_tokens: int = None,
                  use_async_pipeline: bool = True) -> Dict:
        """Benchmark speculative decoding against baseline autoregressive decoding."""
        results = {
            "speculative": {"times": [], "tokens_per_second": []},
            "baseline": {"times": [], "tokens_per_second": []} if compare_baseline else None
        }

        # benchmark speculative decoding
        for i in range(num_runs):
            mode_str = "Async" if use_async_pipeline else "Sync"
            print(f"Speculative Run {i+1} ({mode_str})...")
            start_time = time.time()
            output = self.speculative_decode(
                prompt, max_tokens=max_tokens,
                num_speculative_tokens=num_speculative_tokens,
                use_async_pipeline=use_async_pipeline
            )
            elapsed = time.time() - start_time

            # output now contains only generated tokens (prompt excluded)
            output_tokens = len(self.target_tokenizer.encode(output))
            tps = output_tokens / elapsed
            results["speculative"]["times"].append(elapsed)
            results["speculative"]["tokens_per_second"].append(tps)

        # benchmark baseline decoding
        if compare_baseline:
            print("Baseline Run...")
            baseline_sampling = SamplingParams(
                temperature=0.0, max_tokens=max_tokens, detokenize=False
            )
            prompt_token_ids = self.target_tokenizer.encode(prompt)

            for i in range(num_runs):
                start_time = time.time()
                output = self.target_llm.generate(
                    prompt_token_ids=[prompt_token_ids],
                    sampling_params=baseline_sampling,
                    use_tqdm=False
                )
                elapsed = time.time() - start_time

                output_tokens = len(output[0].outputs[0].token_ids)
                tps = output_tokens / elapsed
                results["baseline"]["times"].append(elapsed)
                results["baseline"]["tokens_per_second"].append(tps)

        # compute averages
        for method in results.keys():
            if results[method] is not None:
                avg_time = sum(results[method]["times"]) / num_runs
                avg_tps = sum(results[method]["tokens_per_second"]) / num_runs
                results[method]["avg_time"] = avg_time
                results[method]["avg_tokens_per_second"] = avg_tps

        if compare_baseline:
            speedup = results["baseline"]["avg_time"] / results["speculative"]["avg_time"]
            results["speedup"] = speedup
            results["latency_reduction"] = (
                1 - results["speculative"]["avg_time"] / results["baseline"]["avg_time"]
            ) * 100
            print(f"Speculative decoding speedup: {speedup:.2f}x")
            print(f"Latency reduction: {results['latency_reduction']:.2f}%")

        return results
