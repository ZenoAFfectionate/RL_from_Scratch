import time
import torch
from typing import List, Tuple, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer


class SpeculativeDecoder:
    """Speculative Decoding class that implements vectorized verification."""
    def __init__(self, target_model_name: str, draft_model_name: str, device: str = "cuda"):
        """
        Initialize the speculative decoder with target and draft models.

        Args:
            target_model_name: HuggingFace model ID for the larger target model.
            draft_model_name:  HuggingFace model ID for the smaller draft model.
            device: Device to run models on ("cuda" or "cpu").
        """
        self.device = device
        self.target_model, self.target_tokenizer = self.initialize_target_model(target_model_name)
        self.draft_model,  self.draft_tokenizer  = self.initialize_draft_model(draft_model_name)
        # ensure tokenizers are compatible
        assert self.target_tokenizer.vocab == self.draft_tokenizer.vocab, "Tokenizers must be compatible"

    def initialize_target_model(self, model_name: str):
        """Initialize the larger target model with caching enabled and proper pad token."""
        print(f"Loading target model: {model_name}")
        # initalize tokenizer and set pad token if needed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # load model with appropriate settings for inference
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            use_cache=True
        )
        model.eval()
        return model, tokenizer

    def initialize_draft_model(self, model_name: str):
        """
        Initialize a smaller, faster draft model with proper pad token.
        Uses lower precision and additional optimizations.
        """
        print(f"Loading draft model: {model_name}")
        # initalize tokenizer and set pad token if needed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # load model with appropriate settings for inference
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            use_cache=True
        )
        model.eval()
        return model, tokenizer

    def generate_draft_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                              num_speculative_tokens: int = 10) -> torch.Tensor:
        """
        Generate speculative tokens in one forward call using the draft model.

        Args:
            input_ids: Input token IDs (tensor of shape [1, seq_len]).
            attention_mask: Corresponding attention mask.
            num_speculative_tokens: Number of tokens to speculate.

        Returns:
            Tensor of shape [1, num_speculative_tokens] containing the draft tokens.
        """
        # use the draft model to generate tokens
        with torch.no_grad():
            draft_output = self.draft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_speculative_tokens,
                do_sample=False,  # greddy decoding for draft
                pad_token_id=self.draft_tokenizer.pad_token_id,
                use_cache=True
            )
        # extract new tokens without the input and returm
        return draft_output[:, input_ids.shape[1]:]

    def verify_tokens_vectorized(self, input_ids: torch.Tensor, draft_tokens: torch.Tensor,
                                 attention_mask: torch.Tensor) -> Tuple[List[int], int]:
        """
        Vectorized verification: verify all draft tokens in one forward pass using the target model.

        Args:
            input_ids: The current input token IDs (shape [1, L]).
            draft_tokens: Draft tokens from the draft model (shape [1, k]).
            attention_mask: The current attention mask for input_ids.

        Returns:
            accepted_tokens: List of accepted token IDs.
            accepted_position: Index of the first rejected token (if all accepted, equals draft_tokens.shape[1]).
        """
        input_len = input_ids.shape[1]     # len of input tokens (existing)
        draft_len = draft_tokens.shape[1]  # len of draft tokens (generate)
        eos_id = self.target_tokenizer.eos_token_id

        # concatenate input_ids and draft_tokens to form candidate sequence
        candidate_input_ids = torch.cat([input_ids, draft_tokens], dim=1)
        # extend attention mask accordingly
        candidate_attention_mask = torch.cat([
            attention_mask, 
            torch.ones((1, draft_tokens.shape[1]), device=self.device)
        ], dim=1)

        # run target model once on the entire candidate sequence
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=candidate_input_ids,
                attention_mask=candidate_attention_mask,
                use_cache=True
            )
            logits = outputs.logits
        
        # get the logits of target model and do greedy decoding
        relevant_logits = logits[:, input_len-1:input_len+draft_len, :]
        target_preds = torch.argmax(relevant_logits, dim=-1)

        n_accepted = 0
        accepted_tokens = []
        finished = False
        for i in range(draft_len):
            d_token = draft_tokens[0, i].item()
            t_token = target_preds[0, i].item()
            # accept draft token if match target token
            if not finished and d_token == t_token:
                accepted_tokens.append(d_token)
                n_accepted += 1
                if d_token == eos_id:
                    finished = True
            # reject draft token and use target token
            else:
                accepted_tokens.append(t_token)
                finished = (t_token == eos_id)
                break
        
        # all draft tokens consumed and haven't hit EOS:
        # accept the 'bonus' token from target model
        if n_accepted == draft_len and not finished:
            bonus_token = target_preds[0, draft_len].item()
            accepted_tokens.append(bonus_token)
            if bonus_token == eos_id: finished = True
        
        return torch.tensor([accepted_tokens], device=self.device), n_accepted, finished


    def speculative_decode(self, prompt: str, max_tokens: int = 100,
                           num_speculative_tokens: int = 15) -> str:
        """
        Main speculative decoding algorithm with vectorized verification.

        Args:
            prompt: Input text.
            max_tokens: Maximum number of tokens to generate (excluding prompt).
            num_speculative_tokens: Number of tokens to speculate per iteration.

        Returns:
            Generated text.
        """
        # Tokenize prompt
        inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        initial_len = input_ids.shape[1]
        current_len = initial_len

        # initialize counters for performance tracking
        total_draft_tokens_proposed = 0
        total_draft_tokens_accepted = 0
        start_time = time.time()

        finished = False
        # 
        while (current_len - initial_len) < max_tokens and not finished:
            # generate k draft tokens:
            draft_tokens = self.generate_draft_tokens(
                input_ids, attention_mask, num_speculative_tokens
            )

            # verify all draft tokens:
            valid_tokens, n_accepted, finished = self.verify_tokens_vectorized(
                input_ids, draft_tokens, attention_mask
            )
            total_draft_tokens_proposed += num_speculative_tokens
            total_draft_tokens_accepted += n_accepted
            # append valid tokens to input
            input_ids = torch.cat([input_ids, valid_tokens], dim=1)

            # update attention mask:
            new_mask = torch.ones((1, valid_tokens.shape[1]), device=self.device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

            current_len = input_ids.shape[1]
            if finished: finished = True

        # Calculate performance metrics
        elapsed_time = time.time() - start_time
        generated_count = current_len - initial_len
        acceptance_rate = total_draft_tokens_accepted / total_draft_tokens_proposed if total_draft_tokens_proposed > 0 else 0

        print(f"Generated {generated_count} tokens in {elapsed_time:.2f} seconds")
        print(f"Tokens per second: {generated_count / elapsed_time:.2f}")
        print(f"Draft token acceptance rate: {acceptance_rate:.2%}")

        return self.target_tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def benchmark(self, prompt: str, max_tokens: int = 100,
                  num_runs: int = 3, compare_baseline: bool = True) -> Dict:
        """
        Benchmark the speculative decoder against baseline decoding.

        Args:
            prompt: Input text.
            max_tokens: Maximum number of tokens to generate.
            num_runs: Number of benchmark runs.
            compare_baseline: Whether to compare with baseline (non-speculative) decoding.

        Returns:
            Dictionary with benchmark results.
        """
        results = {
            "speculative": {"times": [], "tokens_per_second": []},
            "baseline": {"times": [], "tokens_per_second": []} if compare_baseline else None
        }

        # Benchmark speculative decoding.
        for _ in range(num_runs):
            start_time = time.time()
            output = self.speculative_decode(prompt, max_tokens=max_tokens)
            elapsed = time.time() - start_time
            prompt_len = len(self.target_tokenizer(prompt)["input_ids"])
            output_tokens = len(self.target_tokenizer.encode(output)) - prompt_len
            tps = output_tokens / elapsed
            results["speculative"]["times"].append(elapsed)
            results["speculative"]["tokens_per_second"].append(tps)

        # Benchmark baseline decoding.
        if compare_baseline:
            for _ in range(num_runs):
                inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                start_time = time.time()
                with torch.no_grad():
                    output_ids = self.target_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + max_tokens,
                        do_sample=False,
                        pad_token_id=self.target_tokenizer.pad_token_id
                    )
                elapsed = time.time() - start_time
                output_tokens = output_ids.shape[1] - input_ids.shape[1]
                tps = output_tokens / elapsed
                results["baseline"]["times"].append(elapsed)
                results["baseline"]["tokens_per_second"].append(tps)

        for method in results.keys():
            if results[method] is not None:
                avg_time = sum(results[method]["times"]) / num_runs
                avg_tps = sum(results[method]["tokens_per_second"]) / num_runs
                results[method]["avg_time"] = avg_time
                results[method]["avg_tokens_per_second"] = avg_tps

        if compare_baseline:
            speedup = results["baseline"]["avg_time"] / results["speculative"]["avg_time"]
            results["speedup"] = speedup
            results["latency_reduction"] = (1 - results["speculative"]["avg_time"] / results["baseline"]["avg_time"]) * 100
            print(f"Speculative decoding speedup: {speedup:.2f}x")
            print(f"Latency reduction: {results['latency_reduction']:.2f}%")

        return results