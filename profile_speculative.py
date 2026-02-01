"""
Profiling script for SpeculativeDecoder.
Provides detailed timing breakdown of all components.
"""
import os
import sys
import json
import time
import cProfile
import pstats
from io import StringIO
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from algorithms.speculative import SpeculativeDecoder

# Monkey-patch to add detailed profiling
class ProfiledSpeculativeDecoder(SpeculativeDecoder):
    """Extended decoder with detailed profiling capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detailed_timing = defaultdict(float)
        self.call_counts = defaultdict(int)
    
    def _reset_detailed_timing(self):
        self.detailed_timing = defaultdict(float)
        self.call_counts = defaultdict(int)
    
    def generate_draft_tokens(self, sequences, num_speculative_tokens=16):
        """Profiled version of generate_draft_tokens."""
        self.call_counts["generate_draft_tokens"] += 1
        
        # Time: get draft params
        t0 = time.perf_counter()
        sampling_params = self._get_draft_params(num_speculative_tokens)
        self.detailed_timing["draft_get_params"] += time.perf_counter() - t0
        
        # Time: LLM generate call
        t0 = time.perf_counter()
        outputs = self.draft_llm.generate(
            prompt_token_ids=sequences,
            sampling_params=sampling_params,
            use_tqdm=False
        )
        self.detailed_timing["draft_llm_generate"] += time.perf_counter() - t0
        
        # Time: post-processing (EOS truncation)
        t0 = time.perf_counter()
        draft_tokens_list = []
        eos_count = 0
        for output in outputs:
            tokens = list(output.outputs[0].token_ids)
            if self.target_eos_id in tokens:
                eos_idx = tokens.index(self.target_eos_id)
                tokens = tokens[:eos_idx + 1]
                eos_count += 1
            draft_tokens_list.append(tokens)
        self.detailed_timing["draft_post_process"] += time.perf_counter() - t0
        
        return draft_tokens_list, eos_count
    
    def verify_tokens_vectorized(self, sequences, batch_draft_tokens):
        """Profiled version of verify_tokens_vectorized."""
        self.call_counts["verify_tokens_vectorized"] += 1
        batch_size = len(sequences)
        
        # Time: prepare candidate sequences
        t0 = time.perf_counter()
        candidate_sequences = []
        for seq, draft in zip(sequences, batch_draft_tokens):
            candidate = seq.copy()
            candidate.extend(draft)
            candidate_sequences.append(candidate)
        self.detailed_timing["verify_prepare_candidates"] += time.perf_counter() - t0
        
        # Time: LLM generate call
        t0 = time.perf_counter()
        outputs = self.target_llm.generate(
            prompt_token_ids=candidate_sequences,
            sampling_params=self.verify_sampling_params,
            use_tqdm=False
        )
        self.detailed_timing["verify_llm_generate"] += time.perf_counter() - t0
        
        # Time: process verification results
        t0 = time.perf_counter()
        results = [None] * batch_size
        for idx, output in enumerate(outputs):
            results[idx] = self._process_single_verification(
                sequences[idx],
                batch_draft_tokens[idx],
                output.prompt_logprobs,
                output.outputs[0].token_ids[0] if output.outputs[0].token_ids else self.target_eos_id
            )
        self.detailed_timing["verify_process_results"] += time.perf_counter() - t0
        
        return results
    
    def speculative_decode_profiled(
        self,
        prompts,
        max_tokens=256,
        num_speculative_tokens=16,
        adaptive_speculation=True,
        log_interval=50
    ):
        """Fully profiled version of speculative_decode."""
        self._reset_detailed_timing()
        
        # Time: initialization
        t0 = time.perf_counter()
        self.timing_stats = {"draft_time": 0.0, "verify_time": 0.0, "overhead_time": 0.0}
        
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]
        
        batch_size = len(prompts)
        self.detailed_timing["init_setup"] += time.perf_counter() - t0
        
        # Time: tokenization
        t0 = time.perf_counter()
        sequences = [self.target_tokenizer.encode(p) for p in prompts]
        initial_lens = [len(seq) for seq in sequences]
        self.detailed_timing["init_tokenize"] += time.perf_counter() - t0
        
        t0 = time.perf_counter()
        finished = [False] * batch_size
        total_draft_proposed = [0] * batch_size
        total_draft_accepted = [0] * batch_size
        total_mismatches = 0
        total_eos_truncations = 0
        recent_acceptance_rates = []
        current_spec_length = num_speculative_tokens
        self.detailed_timing["init_state"] += time.perf_counter() - t0
        
        start_time = time.time()
        iteration = 0
        pending_draft_future = None
        
        while not all(finished):
            iteration += 1
            
            # Time: find active sequences
            t0 = time.perf_counter()
            active_indices = [i for i in range(batch_size) if not finished[i]]
            if not active_indices:
                break
            active_sequences = [sequences[i] for i in active_indices]
            self.detailed_timing["loop_find_active"] += time.perf_counter() - t0
            
            # Time: adaptive speculation
            t0 = time.perf_counter()
            if adaptive_speculation and recent_acceptance_rates:
                current_spec_length = self._adaptive_speculation_length(
                    recent_acceptance_rates,
                    base_length=num_speculative_tokens
                )
            self.detailed_timing["loop_adaptive_spec"] += time.perf_counter() - t0
            
            # Time: generate draft tokens
            t0 = time.perf_counter()
            if pending_draft_future is not None:
                batch_draft_tokens, eos_count = pending_draft_future.result()
                pending_draft_future = None
            else:
                batch_draft_tokens, eos_count = self.generate_draft_tokens(
                    active_sequences, current_spec_length
                )
            total_eos_truncations += eos_count
            draft_elapsed = time.perf_counter() - t0
            self.timing_stats["draft_time"] += draft_elapsed
            
            # Time: verify tokens
            t0 = time.perf_counter()
            verification_results = self.verify_tokens_vectorized(
                active_sequences, batch_draft_tokens
            )
            verify_elapsed = time.perf_counter() - t0
            self.timing_stats["verify_time"] += verify_elapsed
            
            # Time: update state
            t0 = time.perf_counter()
            iteration_accepted = 0
            iteration_proposed = 0
            
            for idx, active_idx in enumerate(active_indices):
                accepted_tokens, n_accepted, seq_finished = verification_results[idx]
                sequences[active_idx].extend(accepted_tokens)
                
                actual_draft_len = len(batch_draft_tokens[idx])
                total_draft_proposed[active_idx] += actual_draft_len
                total_draft_accepted[active_idx] += n_accepted
                
                if n_accepted < actual_draft_len:
                    total_mismatches += 1
                
                iteration_accepted += n_accepted
                iteration_proposed += actual_draft_len
                
                gen_len = len(sequences[active_idx]) - initial_lens[active_idx]
                if seq_finished or gen_len >= max_tokens:
                    finished[active_idx] = True
            self.detailed_timing["loop_update_state"] += time.perf_counter() - t0
            
            # Time: update acceptance rates
            t0 = time.perf_counter()
            if iteration_proposed > 0:
                iter_rate = iteration_accepted / iteration_proposed
                recent_acceptance_rates.append(iter_rate)
                if len(recent_acceptance_rates) > 5:
                    recent_acceptance_rates.pop(0)
            self.detailed_timing["loop_update_rates"] += time.perf_counter() - t0
            
            # Time: async draft setup
            t0 = time.perf_counter()
            remaining_indices = [i for i in range(batch_size) if not finished[i]]
            if remaining_indices and len(remaining_indices) >= 2:
                remaining_sequences = [sequences[i] for i in remaining_indices]
                pending_draft_future = self._generate_draft_async(
                    remaining_sequences, current_spec_length
                )
            self.detailed_timing["loop_async_setup"] += time.perf_counter() - t0
        
        elapsed = time.time() - start_time
        
        # Time: finalization
        t0 = time.perf_counter()
        if pending_draft_future is not None:
            pending_draft_future.cancel()
        
        total_tokens = sum(len(seq) - init_len for seq, init_len in zip(sequences, initial_lens))
        total_proposed = sum(total_draft_proposed)
        total_accepted = sum(total_draft_accepted)
        self.detailed_timing["finalize_stats"] += time.perf_counter() - t0
        
        # Time: decoding
        t0 = time.perf_counter()
        decoded = [self.target_tokenizer.decode(seq) for seq in sequences]
        self.detailed_timing["finalize_decode"] += time.perf_counter() - t0
        
        # Store results for reporting
        self._profile_results = {
            "elapsed": elapsed,
            "total_tokens": total_tokens,
            "total_proposed": total_proposed,
            "total_accepted": total_accepted,
            "iterations": iteration,
            "batch_size": batch_size
        }
        
        return decoded[0] if single_input else decoded
    
    def print_profile_report(self):
        """Print detailed profiling report."""
        res = self._profile_results
        elapsed = res["elapsed"]
        
        print("\n" + "=" * 70)
        print("DETAILED PROFILING REPORT")
        print("=" * 70)
        
        print(f"\n{'Metric':<30} {'Value':>15}")
        print("-" * 50)
        print(f"{'Total elapsed time':<30} {elapsed:>14.3f}s")
        print(f"{'Total tokens generated':<30} {res['total_tokens']:>15}")
        print(f"{'Throughput':<30} {res['total_tokens']/elapsed:>12.2f} tok/s")
        print(f"{'Iterations':<30} {res['iterations']:>15}")
        print(f"{'Batch size':<30} {res['batch_size']:>15}")
        
        # High-level timing
        print(f"\n{'HIGH-LEVEL TIMING':-^70}")
        draft_time = self.timing_stats["draft_time"]
        verify_time = self.timing_stats["verify_time"]
        overhead = elapsed - draft_time - verify_time
        
        print(f"{'Component':<30} {'Time (s)':>12} {'Percentage':>12} {'Calls':>10}")
        print("-" * 70)
        print(f"{'Draft Generation':<30} {draft_time:>12.3f} {draft_time/elapsed*100:>11.1f}% {self.call_counts['generate_draft_tokens']:>10}")
        print(f"{'Verification':<30} {verify_time:>12.3f} {verify_time/elapsed*100:>11.1f}% {self.call_counts['verify_tokens_vectorized']:>10}")
        print(f"{'Other Overhead':<30} {overhead:>12.3f} {overhead/elapsed*100:>11.1f}%")
        
        # Detailed timing breakdown
        print(f"\n{'DETAILED TIMING BREAKDOWN':-^70}")
        print(f"{'Component':<40} {'Time (s)':>12} {'Percentage':>12}")
        print("-" * 70)
        
        # Group by category
        categories = {
            "Initialization": ["init_setup", "init_tokenize", "init_state"],
            "Draft Model": ["draft_get_params", "draft_llm_generate", "draft_post_process"],
            "Verification Model": ["verify_prepare_candidates", "verify_llm_generate", "verify_process_results"],
            "Loop Overhead": ["loop_find_active", "loop_adaptive_spec", "loop_update_state", 
                             "loop_update_rates", "loop_async_setup"],
            "Finalization": ["finalize_stats", "finalize_decode"]
        }
        
        for category, keys in categories.items():
            cat_total = sum(self.detailed_timing.get(k, 0) for k in keys)
            print(f"\n  {category} (subtotal: {cat_total:.3f}s, {cat_total/elapsed*100:.1f}%)")
            for key in keys:
                t = self.detailed_timing.get(key, 0)
                if t > 0.001:  # only show if > 1ms
                    print(f"    {key:<38} {t:>10.3f} {t/elapsed*100:>11.1f}%")
        
        # Analysis and recommendations
        print(f"\n{'BOTTLENECK ANALYSIS':-^70}")
        
        # Find top 5 time consumers
        sorted_timing = sorted(self.detailed_timing.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 time consumers:")
        for i, (key, t) in enumerate(sorted_timing, 1):
            print(f"  {i}. {key}: {t:.3f}s ({t/elapsed*100:.1f}%)")
        
        # Recommendations
        print(f"\n{'OPTIMIZATION RECOMMENDATIONS':-^70}")
        recommendations = []
        
        draft_llm = self.detailed_timing.get("draft_llm_generate", 0)
        verify_llm = self.detailed_timing.get("verify_llm_generate", 0)
        
        if draft_llm > verify_llm:
            recommendations.append(
                f"- Draft LLM ({draft_llm:.1f}s) > Verify LLM ({verify_llm:.1f}s): "
                "Consider using a smaller/faster draft model, or reduce speculation length."
            )
        
        verify_prep = self.detailed_timing.get("verify_prepare_candidates", 0)
        if verify_prep / elapsed > 0.05:
            recommendations.append(
                f"- Candidate preparation takes {verify_prep/elapsed*100:.1f}%: "
                "Consider optimizing list operations or using numpy arrays."
            )
        
        decode_time = self.detailed_timing.get("finalize_decode", 0)
        if decode_time / elapsed > 0.05:
            recommendations.append(
                f"- Decoding takes {decode_time/elapsed*100:.1f}%: "
                "Consider batch decoding or skip_special_tokens optimization."
            )
        
        loop_overhead = sum(self.detailed_timing.get(k, 0) for k in categories["Loop Overhead"])
        if loop_overhead / elapsed > 0.1:
            recommendations.append(
                f"- Loop overhead is {loop_overhead/elapsed*100:.1f}%: "
                "Consider reducing iteration count via larger speculation lengths."
            )
        
        tokenize_time = self.detailed_timing.get("init_tokenize", 0)
        if tokenize_time / elapsed > 0.05:
            recommendations.append(
                f"- Initial tokenization takes {tokenize_time/elapsed*100:.1f}%: "
                "Consider pre-tokenizing prompts if reusing them."
            )
        
        if not recommendations:
            recommendations.append("- No major bottlenecks detected. System is well-balanced.")
        
        for rec in recommendations:
            print(rec)
        
        print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to profile")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per generation")
    parser.add_argument("--spec_tokens", type=int, default=64, help="Speculative tokens")
    parser.add_argument("--cprofile", action="store_true", help="Run with cProfile")
    args = parser.parse_args()
    
    # Load sample data for profiling
    print("Loading sample data for profiling...")
    with open(f"../data/math/valid.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]
    
    dataset = dataset[:args.n_samples]
    
    R1_ZERO_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.

User: {question}
Assistant: <think>"""
    
    prompts = [R1_ZERO_PROMPT.format(question=d["question"]) for d in dataset]
    
    print(f"Loaded {len(prompts)} prompts for profiling")
    print(f"Settings: max_tokens={args.max_tokens}, spec_tokens={args.spec_tokens}")
    
    # Initialize profiled decoder
    print("\nInitializing ProfiledSpeculativeDecoder...")
    decoder = ProfiledSpeculativeDecoder(
        target_model_name="Qwen/Qwen2.5-Math-7B-Instruct",
        draft_model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    )
    
    def run_profiling():
        print(f"\nRunning profiled speculative decode on {len(prompts)} prompts...")
        outputs = decoder.speculative_decode_profiled(
            prompts,
            max_tokens=args.max_tokens,
            num_speculative_tokens=args.spec_tokens,
            adaptive_speculation=True,
            log_interval=20
        )
        decoder.print_profile_report()
        return outputs
    
    if args.cprofile:
        # Run with cProfile for function-level profiling
        print("\nRunning with cProfile...")
        profiler = cProfile.Profile()
        profiler.enable()
        outputs = run_profiling()
        profiler.disable()
        
        # Print cProfile results
        print(f"\n{'CPROFILE FUNCTION-LEVEL ANALYSIS':-^70}")
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        print(s.getvalue())
    else:
        outputs = run_profiling()
    
    print(f"\nProfiling complete. Generated {len(outputs)} responses.")


if __name__ == "__main__":
    main()