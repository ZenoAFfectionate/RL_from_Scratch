import os
import copy
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from datetime import datetime

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from vllm import SamplingParams

from algorithms.utils import tokenize_prompt_and_output, pad_collate
from utils.vllm_helper import (
    load_policy_into_vllm_instance,
    evaluate_vllm,
)


class ValueHead(nn.Module):
    """
    Lightweight scalar-output head that maps transformer hidden states to
    per-token value estimates for the PPO critic.

    Architecture:
        hidden_states (B, T, H) -> Dropout -> Linear(H, 1) -> values (B, T)

    The weights are initialized with small values so that the value network
    starts near zero, which stabilizes early PPO training.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        # Small initialization for stability
        nn.init.normal_(self.head.weight, std=1e-2)
        nn.init.zeros_(self.head.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
        Returns:
            values: (batch_size, seq_len)
        """
        return self.head(self.dropout(hidden_states)).squeeze(-1)


class PPO_Trainer:
    """
    PPO (Proximal Policy Optimization) Trainer for Language Models.

    Key differences from GRPO:
      - Uses a learned value function (critic) rather than group-relative
        baselines, so only **one** response per prompt is needed.
      - Computes per-token advantages via GAE instead of group-level
        normalization.
      - Jointly optimizes the actor (policy) loss **and** the critic
        (value function) loss in a single backward pass through the
        shared transformer backbone.
      - Supports a KL penalty in the per-token reward signal (InstructGPT
        style), allowing the value function to internalize the KL cost.
      - Supports early stopping within PPO epochs when approximate KL
        exceeds a configurable threshold (``target_kl``).
    """

    def __init__(self, train_model, valid_model, tokenizer, reward_func, 
                 optimizer, scheduler, train_dataset, valid_dataset, args):
        # model and dataset:
        self.model = train_model
        self.valid_model = valid_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reward_func = reward_func
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_device = args.train_device
        self.valid_device = args.valid_device
        self.args = args

        # PPO hyperparameters:
        self.rollout_epochs = args.rollout_epochs
        self.num_iterations = args.num_iterations
        self.prompt_template = args.prompt_template
        self.beta = args.beta
        self.gamma = args.gamma
        self.lambd = args.lambd
        self.clip_eps = args.clip_eps
        self.clip_value_eps = args.clip_value_eps
        self.valuefn_coeff = args.valuefn_coeff
        self.entropy_coeff = args.entropy_coeff
        self.target_kl = args.target_kl
        self.log_interval = args.log_interval
        self.val_interval = args.val_interval
        self.problem_key = getattr(args, "problem_key", "problem")
        self.solution_key = getattr(args, "solution_key", "solution")

        wandb_project = getattr(args, "wandb_project", "PPO-MATH")
        self.wandb_config = vars(args) if hasattr(args, "__dict__") else {}

        # PPO generates 1 response per prompt, so:
        self.n_prompts_per_batch = args.batch_size

        assert args.batch_size % args.micro_batch == 0, \
            "batch_size must be divisible by micro_batch."
        self.micro_batch = args.micro_batch
        self.gradient_accumulation_steps = args.batch_size // self.micro_batch

        self.global_step = 0

        # initialize Value head for the critic
        hidden_size = self.model.config.hidden_size
        self.value_head = ValueHead(
            hidden_size,
            dropout=args.dropout,
        ).to(self.train_device)

        # add value-head parameters to the optimizer
        self.optimizer.add_param_group({
            "params": self.value_head.parameters(),
            "lr": args.value_head_lr,
        })

        self._all_params = (
            list(self.model.parameters()) +
            list(self.value_head.parameters())
        )

        # Reference model (optional, for KL penalty)
        # If beta > 0, we store the initial policy weights so we can
        # compute KL against a truly frozen reference.
        self.ref_model = None
        if self.beta > 0:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        self.rollout_sampling_params = SamplingParams(
            temperature=args.sampling_temperature,
            max_tokens=args.max_generation_length,
            stop=["</answer>", "</solution>"],
            include_stop_str_in_output=True,
            n=self.num_generations,
        )
        self.eval_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_generation_length,
            stop=["</answer>", "</solution>"],
            include_stop_str_in_output=True,
        )

        os.makedirs(self.args.output_dir, exist_ok=True)
        self.record_path = os.path.join(self.args.output_dir, "record.txt")
        with open(self.record_path, "w") as f:
            f.write(f"PPO Training Record — {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

        wandb.init(project=wandb_project, config=self.wandb_config)


    def _log(self, message: str):
        """Write *message* to both stdout and the record file."""
        print(message)
        with open(self.record_path, "a") as f:
            f.write(message + "\n")

    def _format_prompt(self, question: str) -> str:
        """Apply the prompt template to a raw question string."""
        if self.prompt_template is not None:
            return self.prompt_template.format(question=question)
        return question

    def _build_attention_mask(self, input_ids):
        """Build an attention mask that excludes padding tokens."""
        return (input_ids != self.tokenizer.pad_token_id).long()


    def masked_mean(self, tensor, mask, dim=None):
        """Mean of *tensor* over positions where mask is nonzero."""
        masked_tensor = tensor * mask
        sum_masked = masked_tensor.sum(dim=dim)
        count = mask.sum(dim=dim).clamp(min=1)
        return sum_masked / count


    def generate_samples(self, batch_items):
        """Generate one rollout response per prompt using vLLM."""
        load_policy_into_vllm_instance(self.model, self.valid_model)

        prompts = [
            self._format_prompt(item[self.problem_key])
            for item in batch_items
        ]
        grounds = [item[self.solution_key] for item in batch_items]

        rollout_outputs = self.valid_model.generate(
            prompts, self.rollout_sampling_params, use_tqdm=True)

        flat_prompts, flat_responses, repeated_grounds = [], [], []
        for i, output in enumerate(rollout_outputs):
            for sample in output.outputs:
                flat_prompts.append(prompts[i])
                flat_responses.append(sample.text)
                repeated_grounds.append(grounds[i])

        return flat_prompts, flat_responses, repeated_grounds


    def get_action_log_probs_and_values(self, input_ids, labels):
        """Joint forward pass to get per-token log-probs, entropy, and values."""
        attention_mask = self._build_attention_mask(input_ids)
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        # ==========================================
        # compute per-token log-probs [Inefficient!]
        # ==========================================
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        with torch.no_grad():  # compute per-token entropy
            token_entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)

        return {
            "log_probs": token_log_probs,
            "token_entropy": token_entropy,
            "values": self.value_head(hidden_states),
        }


    def compute_rewards(self, flat_responses, ground_truths):
        """Score each generated response using the reward function."""
        rewards = [
            self.reward_func(r, g)
            for r, g in zip(flat_responses, ground_truths)
        ]
        raw_rewards = torch.tensor(
            [res["reward"] for res in rewards], dtype=torch.float32
        )

        metadata = {
            "raw_rewards": raw_rewards,
            "reward_mean": raw_rewards.mean().item(),
            "reward_std": raw_rewards.std().item(),
            "format_rewards": [res.get("format_reward", 0.0) for res in rewards],
            "answer_rewards": [res.get("answer_reward", 0.0) for res in rewards],
        }
        return raw_rewards, metadata


    def compute_per_token_rewards(
        self, raw_rewards, old_log_probs, ref_log_probs, response_mask
    ):
        """
        Build the per-token reward signal used by GAE.

        For each response token position *t*:
            r_t = -β · KL_t           (non-terminal)
            r_T = R_env - β · KL_T    (terminal)

        where  KL_t = log π_old(a_t|s_t) − log π_ref(a_t|s_t).

        If β = 0  (default), per-token rewards are simply R_env placed at
        the last response token.
        """
        B, T = old_log_probs.shape
        device = old_log_probs.device
        mask_f = response_mask.float()

        per_token_rewards = torch.zeros(B, T, device=device)

        # KL penalty (only when β > 0)
        if self.beta > 0:
            kl_per_token = old_log_probs - ref_log_probs  # (B, T)
            per_token_rewards = -self.beta * kl_per_token * mask_f

        # Vectorized: place env reward at the last response token
        # For each row, find the rightmost True in response_mask.
        seq_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        # Masked index: -1 where mask is False so argmax picks a True pos
        last_pos = (seq_idx * response_mask.long()).argmax(dim=1)  # (B,)
        # Only scatter for rows that actually have response tokens
        has_response = response_mask.any(dim=1)                    # (B,)
        per_token_rewards[has_response, last_pos[has_response]] += (
            raw_rewards[has_response]
        )

        return per_token_rewards


    def compute_gae_advantages(self, per_token_rewards, values, response_mask):
        """
        Generalized Advantage Estimation (GAE-λ), vectorized across
        the batch dimension.

        For each response token (iterating backwards):
            δ_t = r_t + γ · V(s_{t+1}) − V(s_t)
            A_t = δ_t + (γ λ) · A_{t+1}

        The episode terminates after the last response token, so
        V(s_{T+1}) = 0.
        """
        B, T = per_token_rewards.shape
        device = per_token_rewards.device

        advantages = torch.zeros(B, T, device=device)
        last_gae = torch.zeros(B, device=device)

        for t in reversed(range(T)):
            # mask for the current token
            mask_t = response_mask[:, t].float()

            if t < T - 1:
                next_mask  = response_mask[:, t + 1].float()
                next_value = values[:, t + 1] * next_mask
            else:
                next_mask  = torch.zeros(B, device=device)
                next_value = torch.zeros(B, device=device)

            delta = (
                per_token_rewards[:, t]
                + self.gamma * next_value
                - values[:, t]
            )
            # reset accumulation when leaving the response region
            last_gae = (
                delta + self.gamma * self.lambd * last_gae * next_mask
            ) * mask_t
            advantages[:, t] = last_gae

        returns = advantages + values.detach()
        return advantages, returns


    def generate_experiences(self, flat_prompts, flat_responses, ground_truths):
        """
        Build a complete experience dict from rollout data.

        Steps:
          1. Compute per-response rewards.
          2. Tokenise (prompt, response) pairs → input_ids, labels, mask.
          3. **Single** forward pass → old log-probs **and** old values.
          4. Compute reference log-probs (for KL in per-token rewards).
          5. Build per-token rewards (env reward + KL penalty).
          6. Run GAE to get advantages and value targets.
          7. Normalise advantages across the batch.
        """
        device = self.train_device

        # compute per-response rewards
        raw_rewards, reward_meta = self.compute_rewards(
            flat_responses, ground_truths
        )
        raw_rewards = raw_rewards.to(device)

        # tokenize prompts and responses
        samples = tokenize_prompt_and_output(
            flat_prompts, flat_responses, self.tokenizer
        )
        tokenized = pad_collate(samples, self.tokenizer.pad_token_id)
        input_ids = tokenized["input_ids"].to(device, non_blocking=True)
        labels = tokenized["labels"].to(device, non_blocking=True)
        response_mask = tokenized["response_mask"].to(device, non_blocking=True)

        # single forward pass to get old log-probs and values
        with torch.no_grad():
            self.model.eval()
            self.value_head.eval()

            attention_mask = self._build_attention_mask(input_ids)
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]

            # Old log-probs (fused CE — avoids (B,T,V) materialization)
            token_nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
            ).view(labels.shape)
            old_log_probs = -token_nll

            # Old values
            old_values = self.value_head(hidden_states)

        # compute reference log-probs
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(
                    input_ids, attention_mask=attention_mask
                ).logits
                ref_log_probs = -F.cross_entropy(
                    ref_logits.view(-1, ref_logits.size(-1)),
                    labels.view(-1),
                    reduction="none",
                ).view(labels.shape)
        else:
            ref_log_probs = old_log_probs

        # compute per-token rewards and GAE advantages
        per_token_rewards = self.compute_per_token_rewards(
            raw_rewards, old_log_probs, ref_log_probs, response_mask
        )
        advantages, returns = self.compute_gae_advantages(
            per_token_rewards, old_values, response_mask
        )

        # normalize advantages to have zero mean and unit variance
        mask_f = response_mask.float()
        adv_mean = self.masked_mean(advantages, mask_f)
        adv_vari = self.masked_mean((advantages - adv_mean) ** 2, mask_f)
        advantages = (advantages - adv_mean) / (adv_vari.sqrt() + 1e-8)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask,
            "advantages": advantages.detach(),
            "old_log_probs": old_log_probs,
            "old_values": old_values.detach(),
            "returns": returns.detach(),
            "raw_rewards": raw_rewards,
            "reward_metadata": reward_meta,
        }


    def compute_loss(
        self,
        policy_log_probs,
        response_mask,
        advantages,
        old_log_probs,
        values,
        old_values,
        returns,
        token_entropy,
    ):
        """
        Compute the combined PPO loss for a single micro-batch.

        total_loss = policy_loss + vf_coeff · value_loss
                     − entropy_coeff · entropy_bonus

        **Policy loss** (clipped surrogate):
            L_clip = -min(r_t · A_t,  clip(r_t, 1±ε) · A_t)

        **Value loss** (clipped):
            L_vf = max( (V - V_target)²,  (V_clip - V_target)² )
            where  V_clip = V_old + clip(V - V_old, ±ε_v)

        **Entropy bonus**:
            H = -Σ p log p   (per-token, already computed in forward pass)
        """
        # Broadcast advantages: (B,) → (B, 1) if needed
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        mask_f = response_mask.float()

        # compute policy loss
        ratio = torch.exp(policy_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(
                ratio,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps,
            )
            * advantages
        )
        policy_loss = -torch.min(surr1, surr2)

        # compute clipped value loss
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_value_eps,
            +self.clip_value_eps,
        )
        vf_loss_unclipped = (values - returns) ** 2
        vf_loss_clipped = (values_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped)

        # get per-token loss and mask mean over response tokens
        per_token_loss = (
            policy_loss
            + self.valuefn_coeff * value_loss
            - self.entropy_coeff * token_entropy
        )
        loss = self.masked_mean(per_token_loss, mask_f)

        with torch.no_grad():
            total_resp = mask_f.sum().clamp(min=1)
            clip_frac = ((surr2 < surr1).float() * mask_f).sum() / total_resp
            approx_kl = self.masked_mean(
                0.5 * (old_log_probs - policy_log_probs) ** 2, mask_f
            )

        meta_info = {
            "loss": loss.detach().item(),
            "policy_loss": self.masked_mean(policy_loss, mask_f).detach().item(),
            "value_loss": self.masked_mean(value_loss, mask_f).detach().item(),
            "entropy": self.masked_mean(token_entropy, mask_f).detach().item(),
            "clip_fraction": clip_frac.item(),
            "approx_kl": approx_kl.item(),
            "ratio_mean": ratio.detach().mean().item(),
        }

        return loss, meta_info

    def train_step(self, experiences):
        """
        Execute one full training pass (multiple epochs) over a batch of
        experiences, with gradient accumulation and optimizer stepping.

        Supports early stopping when approximate KL exceeds
        ``1.5 × target_kl`` to prevent catastrophic policy drift.
        """
        input_ids = experiences["input_ids"]
        labels = experiences["labels"]
        response_mask = experiences["response_mask"]
        advantages = experiences["advantages"]
        old_log_probs = experiences["old_log_probs"]
        old_values = experiences["old_values"]
        returns = experiences["returns"]

        dataset = TensorDataset(
            input_ids, labels, response_mask,
            advantages, old_log_probs, old_values, returns,
        )
        loader = DataLoader(dataset, batch_size=self.micro_batch, shuffle=True)


        step_metrics = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "approx_kl": [],
        }
        accumu_count = 0
        early_stopped = False

        self.model.train()
        self.value_head.train()

        for epoch in range(self.rollout_epochs):
            if early_stopped: break

            for batch in loader:
                (mb_ids, mb_labels, mb_mask,
                 mb_adv, mb_old_lp, mb_old_v, mb_ret) = batch

                # forward pass to get per-token log-probs, entropy, and values
                model_out = self.get_action_log_probs_and_values(mb_ids, mb_labels)

                # compute PPO loss
                loss, meta = self.compute_loss(
                    policy_log_probs=model_out["log_probs"],
                    response_mask=mb_mask,
                    advantages=mb_adv,
                    old_log_probs=mb_old_lp,
                    values=model_out["values"],
                    old_values=mb_old_v,
                    returns=mb_ret,
                    token_entropy=model_out["token_entropy"],
                )

                # scale loss for gradient accumulation
                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()

                # Collect metrics
                for key in step_metrics:
                    if key in meta:
                        step_metrics[key].append(meta[key])

                accumu_count += 1

                # Optimizer step
                if accumu_count % self.gradient_accumulation_steps == 0:
                    clip_grad_norm_(self._all_params, self.args.clip_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Early-stop PPO epochs on KL overshoot
                if (
                    self.target_kl is not None
                    and meta["approx_kl"] > 1.5 * self.target_kl
                ):
                    self._log(
                        f"    Early stop at epoch {epoch}: "
                        f"approx_kl={meta['approx_kl']:.4f} "
                        f"> 1.5 × target_kl={self.target_kl}"
                    )
                    early_stopped = True
                    break

        return step_metrics


    def train(self):
        """
        Run the complete PPO training loop for ``num_iterations`` steps.

        Each iteration:
          1. Sample a batch of prompts from the training dataset.
          2. ``generate_samples``      — produce rollout responses via vLLM.
          3. ``generate_experiences``   — rewards, values, GAE advantages.
          4. ``train_step``             — update actor & critic.
          5. Log training metrics to W&B and the record file.
          6. Periodically evaluate on the validation set.
        """
        self._log_config()
        self._log(">>> Starting PPO Training Loop")

        last_log_time = time.time()
        last_log_step = 0

        for step in range(self.num_iterations):
            # sample prompts from training dataset
            indices = np.random.choice(
                len(self.train_dataset),
                self.n_prompts_per_batch,
                replace=len(self.train_dataset) < self.n_prompts_per_batch,
            )
            batch_items = [self.train_dataset[i] for i in indices]

            # rollout responses and build experiences
            flat_prompts, flat_responses, ground_truths = (
                self.generate_samples(batch_items)
            )
            experiences = self.generate_experiences(
                flat_prompts, flat_responses, ground_truths
            )

            # perform one PPO training step
            step_metrics = self.train_step(experiences)

            # log training metrics
            reward_meta = experiences["reward_metadata"]
            log_dict = self._build_log_dict(
                step_metrics, reward_meta, experiences
            )
            wandb.log(log_dict)

            if self.global_step % max(self.log_interval, 1) == 0:
                now = time.time()
                steps_since = self.global_step - last_log_step
                avg_time = (now - last_log_time) / max(steps_since, 1)
                self._log(
                    f"  Step {self.global_step}: "
                    f"loss={np.mean(step_metrics['loss']):.4f}, "
                    f"pi_loss={np.mean(step_metrics['policy_loss']):.4f}, "
                    f"vf_loss={np.mean(step_metrics['value_loss']):.4f}, "
                    f"reward={reward_meta['reward_mean']:.3f}, "
                    f"time={avg_time:.3f}s"
                )
                last_log_time = now
                last_log_step = self.global_step

            # evaluate model periodically
            if (
                self.valid_dataset is not None
                and self.global_step > 0
                and self.global_step % self.val_interval == 0
            ):
                self._evaluate()

        # save final checkpoint
        self._save_checkpoint("final")
        self._log(">>> PPO Training Complete.")
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
        solutions = [
            item[self.solution_key] for item in self.valid_dataset
        ]

        acc = evaluate_vllm(
            self.valid_model,
            self.reward_func,
            prompts,
            solutions,
            self.eval_sampling_params,
        )

        self._log(
            f"Step {self.global_step}: Validation Accuracy = {acc:.2f}%"
        )
        wandb.log({"valid/accuracy": acc, "step": self.global_step})


    def _build_log_dict(self, step_metrics, reward_meta, experiences):
        """Construct the W&B logging dictionary for the current step."""
        log_dict = {
            "step": self.global_step,
            "train/loss": (
                np.mean(step_metrics["loss"])
                if step_metrics["loss"] else 0
            ),
            "train/policy_loss": (
                np.mean(step_metrics["policy_loss"])
                if step_metrics["policy_loss"] else 0
            ),
            "train/value_loss": (
                np.mean(step_metrics["value_loss"])
                if step_metrics["value_loss"] else 0
            ),
            "train/entropy": (
                np.mean(step_metrics["entropy"])
                if step_metrics["entropy"] else 0
            ),
            "train/mean_advantage": experiences["advantages"].mean().item(),
            "train/reward_mean": reward_meta["reward_mean"],
            "train/reward_std": reward_meta["reward_std"],
        }

        if step_metrics["clip_fraction"]:
            log_dict["train/clip_fraction"] = np.mean(
                step_metrics["clip_fraction"]
            )
        if step_metrics["approx_kl"]:
            log_dict["train/approx_kl"] = np.mean(
                step_metrics["approx_kl"]
            )

        return log_dict

    def _log_config(self):
        """Print / record the full training configuration."""
        config_info = (
            f"\n{'='*60}\n"
            f"PPO Training Configuration\n"
            f"{'='*60}\n"
            f"Model ID:                    "
            f"{self.wandb_config.get('model_id', 'N/A')}\n"
            f"Num Iterations:              {self.num_iterations}\n"
            f"Batch Size:                  {self.n_prompts_per_batch} "
            f"(micro: {self.micro_batch})\n"
            f"Gradient Accumulation Steps: "
            f"{self.gradient_accumulation_steps}\n"
            f"Learning Rate:               {self.args.lr}\n"
            f"Clip Eps (policy):           {self.clip_eps}\n"
            f"Clip Eps (value):            {self.clip_value_eps}\n"
            f"Beta (KL penalty):           {self.beta}\n"
            f"Target KL (early stop):      {self.target_kl}\n"
            f"Gamma (discount):            {self.gamma}\n"
            f"Lambda (GAE):                {self.lambd}\n"
            f"VF Coefficient:              {self.valuefn_coeff}\n"
            f"Entropy Coefficient:         {self.entropy_coeff}\n"
            f"Max Generation Length:       "
            f"{self.args.max_generation_length}\n"
            f"Rollout Epochs:              {self.rollout_epochs}\n"
            f"Eval Interval:               {self.val_interval} steps\n"
            f"Training Samples:            {len(self.train_dataset)}\n"
            f"Validation Samples:          "
            f"{len(self.valid_dataset) if self.valid_dataset else 0}\n"
            f"Output Directory:            {self.args.output_dir}\n"
            f"{'='*60}\n"
        )
        self._log(config_info)

    def _save_checkpoint(self, suffix: str = "checkpoint"):
        """
        Save a named checkpoint.

        Writes two files:
          - ``ppo_policy_<suffix>.pt``  — policy weights only (for inference)
          - ``ppo_<suffix>_full.pt``    — full state (policy + value head +
            optimizer + scheduler) for training resumption.
        """
        # Cache state_dict to avoid calling it twice
        state_dict = self.model.state_dict()

        # Lightweight policy-only checkpoint (for inference / deployment)
        policy_path = os.path.join(
            self.args.output_dir, f"ppo_policy_{suffix}.pt"
        )
        torch.save(state_dict, policy_path)

        # Full training state (includes value head; no redundant copy)
        full_path = os.path.join(
            self.args.output_dir, f"ppo_{suffix}_full.pt"
        )
        save_dict = {
            "global_step": self.global_step,
            "model_state_dict": state_dict,
            "value_head_state_dict": self.value_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(save_dict, full_path)

        self._log(f"Checkpoint saved → {policy_path}")
        return policy_path
