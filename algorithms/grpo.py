import os
import time
import wandb
import torch
import torch.nn.functional as F
import numpy as np

from datetime import datetime

from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from vllm import SamplingParams
from utils.vllm_helper import load_policy_into_vllm_instance, evaluate_vllm
from algorithms.utils import tokenize_prompt_and_output


class GRPO_Trainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer.

    Implements the full GRPO training loop:
      1. Generate multiple rollout responses per prompt using vLLM
      2. Compute group-normalized rewards and advantages
      3. Train the policy with clipped / baseline policy-gradient loss
      4. Evaluate periodically on a held-out validation set
    """

    def __init__(self, train_model, valid_model, tokenizer, reward_fun,
                 optimizer, scheduler, train_dataset, valid_dataset, args):
        # model and dataset:
        self.model = train_model
        self.valid_model = valid_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reward_fun = reward_fun
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_device = args.train_device
        self.valid_device = args.valid_device
        self.args = args

        # GRPO hyperparameters:
        self.num_iterations = args.num_iterations
        self.rollout_epochs = args.rollout_epochs
        self.prompt_template = args.prompt_template
        self.num_generations = args.num_generations
        self.beta = args.beta
        self.clip_eps = args.clip_eps
        self.advantage_eps = args.advantage_eps
        self.log_interval = args.log_interval
        self.val_interval = args.val_interval
        self.problem_key  = getattr(args, "problem_key",  "problem")
        self.solution_key = getattr(args, "solution_key", "solution")
        self.output_dir = args.output_dir + f"4{args.dataset}"

        wandb_project = getattr(args, "wandb_project", "GRPO-Reasoning")
        self.wandb_config = vars(args) if hasattr(args, "__dict__") else {}

        self.micro_batch = args.micro_batch
        self.batch_size = args.batch_size
        total_responses = args.batch_size * args.num_generations
        assert total_responses % args.micro_batch == 0, \
            "(batch_size * num_generations) must be divisible by micro_batch."
        self.gradient_accumulation_steps = total_responses // self.micro_batch

        self.global_step = 0

        self.rollout_sampling_params = SamplingParams(
            temperature=args.sampling_temperature,
            max_tokens=args.max_generation_length,
            stop=["</answer>", "</solution>", "<|endoftext|>"],
            include_stop_str_in_output=True,
            n=self.num_generations,
        )
        self.eval_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_generation_length,
            stop=["</answer>", "</solution>", "<|endoftext|>"],
            include_stop_str_in_output=True,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.record_path = os.path.join(self.output_dir, "record.txt")
        with open(self.record_path, "w") as f:
            f.write(f"GRPO Training Record — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("=" * 60 + "\n\n")

        wandb.init(project=wandb_project, config=self.wandb_config)

    def _log(self, message: str):
        """Write *message* to both stdout and the record file."""
        print(message)
        with open(self.record_path, "a") as f:
            f.write(message + "\n")

    def _format_prompt(self, problem: str) -> str:
        """Apply the prompt template to a raw question string."""
        if self.prompt_template is not None:
            return self.prompt_template.format(problem=problem)
        return problem

    def masked_mean(self, tensor, mask, dim=None):
        """Compute the mean of tensor along a given dimension"""
        masked_tensor = tensor * mask
        sum_masked = masked_tensor.sum(dim=dim)
        count_mask = mask.sum(dim=dim)  # no need clamp
        return sum_masked / count_mask  # normalization
    
    def compute_group_normalized_rewards(self, rollout_responses, repeated_ground_truths):
        """
        Compute rewards for each group of rollout responses, normalized by the group size.
        """
        # compute raw rewards based on reward function and reshape to group-wise
        rewards = [self.reward_fun(r, g) for r, g in zip(rollout_responses, repeated_ground_truths)]
        raw_rewards = torch.tensor([res["reward"] for res in rewards], dtype=torch.float32)

        # compute mean for each group
        rewards_matrix = raw_rewards.view(-1, self.num_generations)
        group_means = rewards_matrix.mean(dim=1, keepdim=True)

        # normalize rewards within each group to get advantages
        group_stds = rewards_matrix.std(dim=1, keepdim=True) + self.advantage_eps
        normalized_rewards = (rewards_matrix - group_means) / group_stds
        advantages = normalized_rewards.flatten()

        metadata = {  # metadata for logging
            "raw_rewards": raw_rewards,
            "reward_mean": raw_rewards.mean().item(),
            "format_rewards": [res.get("format_reward", 0.0) for res in rewards],
            "answer_rewards": [res.get("answer_reward", 0.0) for res in rewards],
        }
        return advantages, raw_rewards, metadata

    def generate_samples(self, batch_items):
        """
        Generate rollout samples for a batch of training items using vLLM.

        This method:
          1. Synchronises the current policy weights into the vLLM engine.
          2. Formats each item's question with the prompt template.
          3. Generates ``num_generations`` responses per prompt.
          4. Flattens the results so that every (prompt, response, ground_truth)
             triple is represented once.
        """
        # sync current policy weights into the vLLM inference engine
        load_policy_into_vllm_instance(self.model, self.valid_model)

        # build formatted prompts and ground truths
        prompts = [self._format_prompt(item[self.problem_key]) for item in batch_items]
        grounds = [item[self.solution_key] for item in batch_items]

        # generate N responses per prompt
        rollout_outputs = self.valid_model.generate(
            prompts, self.rollout_sampling_params, use_tqdm=False)

        flat_prompts, flat_responses, repeated_grounds = [], [], []
        for i, output in enumerate(rollout_outputs):
            for sample in output.outputs:
                flat_prompts.append(prompts[i])
                flat_responses.append(sample.text)
                repeated_grounds.append(grounds[i])
        
        return flat_prompts, flat_responses, repeated_grounds

    def generate_experiences(self, flat_prompts, flat_responses, ground_truths):
        """
        Build a training experience dict from rollout data.

        This method:
          1. Computes group-normalized rewards and per-sample advantages.
          2. Tokenises prompt + response pairs and builds the response mask.
          3. Pre-computes reference / old log-probs for GRPO-Clip or KL-penalty.
        """
        # compute group-normalised rewards and advantages
        advantages, raw_rewards, reward_meta = \
            self.compute_group_normalized_rewards(
                rollout_responses=flat_responses,
                repeated_ground_truths=ground_truths
            )
        advantages = advantages.to(self.train_device)

        # tokenize prompt and response pairs
        tokenized = tokenize_prompt_and_output(flat_prompts, flat_responses, self.tokenizer)
        input_ids = tokenized["input_ids"].to(self.train_device, non_blocking=True)
        labels = tokenized["labels"].to(self.train_device, non_blocking=True)
        response_mask = tokenized["response_mask"].to(self.train_device, non_blocking=True)

        # pre-compute old log-probs (fused CE — avoids (B,T,V) materialization)
        with torch.no_grad():
            self.model.eval()
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            logits = self.model(input_ids, attention_mask=attention_mask).logits
            token_nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
            ).view(labels.shape)
            old_log_probs = -token_nll

        return {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask,
            "advantages": advantages,
            "raw_rewards": raw_rewards,
            "old_log_probs": old_log_probs,
            "reward_metadata": reward_meta,
        }

    def get_action_log_probs(self, input_ids, labels):
        """
        Forward pass through the policy model to obtain per-token
        log-probabilities and per-token entropy.

        Materializes the full (B, T, V) log-softmax because entropy
        requires it. Used in train_step (not generate_experiences).
        """
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        # full log-softmax for gather and entropy
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        # compute token entropy
        with torch.no_grad():
            token_entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
        return {"log_probs": token_log_probs, "token_entropy": token_entropy}


    def compute_loss(self, policy_log_probs, response_mask,
                     advantages, old_log_probs):
        """Compute the GRPO-Clip loss for a single micro-batch."""
        # Ensure advantages broadcast correctly: (B,) → (B, 1)
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # Per-token GRPO-Clip loss (inlined)
        ratio = torch.exp(policy_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        per_token_loss = -torch.min(surr1, surr2)

        with torch.no_grad():
            clip_fraction = (surr2 < surr1).float().mean()
        meta_info = {
            "loss": per_token_loss.detach().mean(),
            "clip_fraction": clip_fraction.detach(),
            "ratio_mean": ratio.detach().mean(),
        }

        # compute KL-penalty optionally
        if self.beta > 0 and old_log_probs is not None:
            # unbiased KL estimator: exp(log_ref - log_pi) - 1 - (log_ref - log_pi)
            log_ratio = old_log_probs - policy_log_probs
            kl = torch.exp(log_ratio) - 1.0 - log_ratio
            per_token_loss = per_token_loss + self.beta * kl
            with torch.no_grad():
                meta_info["kl_mean"] = self.masked_mean(kl, response_mask).detach()

        # mask and mean over response tokens
        loss = self.masked_mean(per_token_loss, response_mask)

        return loss, meta_info

    def train_step(self, experiences):
        """
        Execute one full training pass over a batch of experiences, including
        gradient accumulation and optimizer stepping.
        """
        input_ids = experiences["input_ids"]
        labels = experiences["labels"]
        response_mask = experiences["response_mask"]
        advantages = experiences["advantages"]
        old_log_probs = experiences["old_log_probs"]

        dataset = TensorDataset(input_ids, labels, response_mask, advantages, old_log_probs)
        loader = DataLoader(dataset, batch_size=self.micro_batch, shuffle=True)

        step_metrics = {"loss": [], "token_entropy": [], "clip_fraction": [], "kl_mean": []}
        accumu_count = 0

        self.model.train()
        for epoch in range(self.rollout_epochs):
            for batch in loader:
                mb_ids, mb_labels, mb_mask, mb_adv, mb_old_lp = batch

                # forward pass:  compute per-token log-probs and entropy
                model_out = self.get_action_log_probs(mb_ids, mb_labels)
                # compute GRPO-Clip loss without backward pass
                loss, meta = self.compute_loss(
                    policy_log_probs=model_out["log_probs"],
                    response_mask=mb_mask,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                )

                # scale loss for gradient accumulation and backward
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()

                step_metrics["loss"].append(loss.item())
                step_metrics["token_entropy"].append(model_out["token_entropy"].mean().item())
                step_metrics["clip_fraction"].append(meta["clip_fraction"].item())
                step_metrics["kl_mean"].append(meta["kl_mean"].item())

                accumu_count += 1
                # optimizer step when accumulation is complete
                if accumu_count % self.gradient_accumulation_steps == 0:
                    clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

        return step_metrics


    def train(self):
        """
        Run the complete GRPO training loop for ``self.num_iterations`` steps.

        Each iteration:
          1. Sample a batch of prompts from the training dataset.
          2. ``generate_samples``  — produce rollout responses via vLLM.
          3. ``generate_experiences`` — compute rewards, advantages, tokenise.
          4. ``train_step`` — update the policy (with gradient accumulation).
          5. Log training metrics to W&B and the record file.
          6. Periodically evaluate on the validation set.
        """
        self._log_config()
        self._log(">>> Starting GRPO Training Loop")

        last_log_time = time.time()
        last_log_step = 0

        for step in range(self.num_iterations):
            # sample a batch of prompts
            indices = np.random.choice(len(self.train_dataset), self.batch_size, replace=False)
            batch_items = [self.train_dataset[i] for i in indices]

            # generate rollout samples
            flat_prompts, flat_responses, ground_truths = self.generate_samples(batch_items)

            # compute rewards, advantages, and training tensors
            experiences = self.generate_experiences(flat_prompts, flat_responses, ground_truths)

            # perform a single training step on the experiences
            step_metrics = self.train_step(experiences)

            # log training metrics
            reward_meta = experiences["reward_metadata"]
            log_dict = self._build_log_dict(step_metrics, reward_meta, experiences)
            wandb.log(log_dict)

            # Log training metrics periodically
            if self.global_step % self.log_interval == 0:
                now = time.time()
                steps_since = self.global_step - last_log_step
                avg_time = (now - last_log_time) / max(steps_since, 1)
                self._log(
                    f"  Step {self.global_step}: "
                    f"loss={np.mean(step_metrics['loss']):.4f}, "
                    f"reward_mean={reward_meta['reward_mean']:.3f}, "
                    f"time={avg_time:.3f}s"
                )
                last_log_time = now
                last_log_step = self.global_step

            # Evaluate periodically
            if self.global_step % self.val_interval == 0:
                self._evaluate()

        self._save_checkpoint("final")
        self._log(">>> GRPO Training Complete.")
        wandb.finish()

        return self.model


    def _evaluate(self):
        """Run greedy evaluation on the full validation set."""
        self._log(f"Step {self.global_step}: Running Validation...")
        load_policy_into_vllm_instance(self.model, self.valid_model)

        prompts = [
            self._format_prompt(item[self.problem_key])
            for item in self.valid_dataset
        ]
        solutions = [item[self.solution_key] for item in self.valid_dataset]

        acc = evaluate_vllm(
            self.valid_model,
            self.reward_fun,
            prompts,
            solutions,
            self.eval_sampling_params,
            f'../results/grpo4{self.args.dataset}.jsonl'
        )

        self._log(f"Step {self.global_step}: Validation Accuracy = {acc:.2f}%")
        wandb.log({"valid/accuracy": acc, "step": self.global_step})


    def _build_log_dict(self, step_metrics, reward_meta, experiences):
        """Construct the W&B logging dictionary for the current step."""
        log_dict = {
            "step":           self.global_step,
            "train/loss":     np.mean(step_metrics["loss"]),
            "train/entropy":  np.mean(step_metrics["token_entropy"]),
            "train/clip_fra": np.mean(step_metrics["clip_fraction"]),
            "train/mean_adv": experiences["advantages"].mean().item(),
            "train/kl_mean":  np.mean(step_metrics["kl_mean"]) if step_metrics["kl_mean"] else 0,
        }
        # add reward metadata to log dictionary
        for key, val in reward_meta.items():
            log_dict[f"train/rewards/{key}"] = val

        return log_dict

    def _log_config(self):
        config_info = (
            f"\n{'='*60}\n"
            f"GRPO Training Configuration\n"
            f"{'='*60}\n"
            f"Model ID:                    {self.wandb_config.get('model_id', 'N/A')}\n"
            f"Num Iterations:              {self.num_iterations}\n"
            f"Batch Size:                  {self.batch_size} "
            f"(micro: {self.micro_batch})\n"
            f"Num Generations (group):     {self.args.num_generations}\n"
            f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}\n"
            f"Learning Rate:               {self.args.lr}\n"
            f"Loss Type:                   grpo_clip\n"
            f"Clip Eps:                    {self.clip_eps}\n"
            f"Beta (KL penalty):           {self.beta}\n"
            f"Max Generation Length:       {self.args.max_generation_length}\n"
            f"Epochs per Rollout Batch:    {self.rollout_epochs}\n"
            f"Eval Interval:               {self.val_interval} steps\n"
            f"Training Samples:            {len(self.train_dataset)}\n"
            f"Validation Samples:          "
            f"{len(self.valid_dataset) if self.valid_dataset else 0}\n"
            f"Output Directory:            {self.output_dir}\n"
            f"{'='*60}\n"
        )
        self._log(config_info)


    def _save_checkpoint(self, suffix: str = "checkpoint"):
        """Save a named checkpoint and return the path."""
        path = os.path.join(self.output_dir, f"grpo_{suffix}.pt")
        torch.save(self.model.state_dict(), path)
        self._log(f"Checkpoint saved to {path}")
        return path
