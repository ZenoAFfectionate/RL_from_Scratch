import time
import numpy as np
from typing import List, Tuple, Dict

from vllm import LLM, SamplingParams


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

        # Adaptive speculation length parameters (优化3)
        self.adaptive_k = adaptive_k
        self.min_k = min_k
        self.max_k = max_k
        self.current_k = initial_k
        self.acceptance_history = []
        self._acceptance_window_size = 10

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
                           num_speculative_tokens: int = None) -> str:
        """
        Main speculative decoding algorithm with vectorized verification.

        Args:
            prompt: Input text.
            max_tokens: Maximum number of tokens to generate (excluding prompt).
            num_speculative_tokens: Number of tokens to speculate per iteration.
                                    If None and adaptive_k is True, uses adaptive length.

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

        # speculative decoding loop
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

        return self.target_tokenizer.decode(input_ids)


    def benchmark(self, prompt: str, max_tokens: int = 100,
                  num_runs: int = 3, compare_baseline: bool = True,
                  num_speculative_tokens: int = None) -> Dict:
        """Benchmark speculative decoding against baseline autoregressive decoding."""
        results = {
            "speculative": {"times": [], "tokens_per_second": []},
            "baseline": {"times": [], "tokens_per_second": []} if compare_baseline else None
        }

        # benchmark speculative decoding
        for i in range(num_runs):
            print(f"Speculative Run {i+1}...")
            start_time = time.time()
            output = self.speculative_decode(
                prompt, max_tokens=max_tokens,
                num_speculative_tokens=num_speculative_tokens
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
