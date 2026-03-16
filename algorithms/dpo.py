import os
import time
import wandb
import torch
import torch.nn.functional as F

from typing import List
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import PreTrainedModel



class DPO_Trainer:
    """
    DPO (Direct Preference Optimization) Trainer.

    Implements the full DPO training loop:
      1. Load preference data (instruction, chosen, rejected) via DataLoader
      2. Compute DPO loss using policy & frozen reference model log-probs
      3. Train the policy with gradient accumulation
      4. Evaluate periodically on a held-out validation set
      5. Save best & periodic checkpoints
    """
    def __init__(
        self,
        policy_model,
        reference_model,
        tokenizer,
        optimizer,
        scheduler,
        train_dataset,
        valid_dataset,
        template: str,
        args,
    ):
        # model and dataset:
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.template = template
        self.args = args
        self.policy_device = next(policy_model.parameters()).device
        self.ref_device = next(reference_model.parameters()).device

        # DPO hyperparameters:
        self.epochs = args.epochs
        self.beta = args.beta
        self.batch_size = args.batch_size
        self.micro_batch = args.micro_batch
        self.max_length = args.max_length
        self.log_interval = args.log_interval
        self.val_interval = args.val_interval
        self.save_interval = args.save_interval
        self.clip_grad_norm = args.clip_grad_norm
        self.output_dir = args.output_dir

        # Pre-compute prompt prefix template for response masking
        assert "{response}" in self.template, \
            "Template must contain {response} placeholder for response masking."
        self._prompt_template = self.template.split("{response}")[0]

        # Gradient accumulation
        assert self.batch_size % self.micro_batch == 0, \
            "batch_size must be divisible by micro_batch size."
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch

        # Build DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch,
            collate_fn=self._collate_fn,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.micro_batch,
            collate_fn=self._collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.global_step = 0
        self.best_val_accuracy = 0.0

        os.makedirs(self.output_dir, exist_ok=True)
        self.record_path = os.path.join(self.output_dir, "record.txt")
        with open(self.record_path, "w") as f:
            f.write(
                f"DPO Training Record — {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            )
            f.write("=" * 60 + "\n\n")

        wandb_project = getattr(args, "wandb_project", "DPO-RLHF")
        self.wandb_config = vars(args) if hasattr(args, "__dict__") else {}
        wandb.init(project=wandb_project, config=self.wandb_config)


    @staticmethod
    def _collate_fn(batch):
        """Collate a list of dicts into a batch of string lists."""
        return {
            "instructions": [item["instruction"] for item in batch],
            "chosen": [item["chosen"] for item in batch],
            "rejected": [item["rejected"] for item in batch],
        }

    def _log(self, message: str):
        """Write *message* to both stdout and the record file."""
        print(message)
        with open(self.record_path, "a") as f:
            f.write(message + "\n")

    def _format_text(self, instruction: str, response: str) -> str:
        """Apply the template to an instruction-response pair and add EOS."""
        formatted = self.template.format(instruction=instruction, response=response)
        formatted += self.tokenizer.eos_token
        return formatted


    def _build_response_masks(
        self,
        instructions: List[str],
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build binary response masks for both chosen and rejected sequences.

        The prompt prefix is identical for chosen and rejected (only the
        response differs), so we tokenize each prefix once and reuse the
        length for both masks.

        Returns:
            (chosen_mask, rejected_mask) — each (batch_size, seq_len) float32.
        """
        # tokenize all prefixes once
        prefix_texts = [
            self._prompt_template.format(instruction=inst)
            for inst in instructions
        ]
        prefix_ids = self.tokenizer(
            prefix_texts, add_special_tokens=True
        )["input_ids"]
        prefix_lengths = [len(ids) for ids in prefix_ids]

        # build chosen response mask
        B, T_c = chosen_ids.shape
        chosen_mask = torch.zeros(B, T_c, dtype=torch.float32, device=chosen_ids.device)
        for i, pl in enumerate(prefix_lengths):
            chosen_mask[i, pl:] = 1.0

        # build rejected response mask
        B, T_r = rejected_ids.shape
        rejected_mask = torch.zeros(B, T_r, dtype=torch.float32, device=rejected_ids.device)
        for i, pl in enumerate(prefix_lengths):
            rejected_mask[i, pl:] = 1.0

        return chosen_mask, rejected_mask


    def compute_sequence_log_prob(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the summed log-probability of **response tokens only**
        under a given model (no length normalisation).

        Uses fused ``F.cross_entropy`` internally so the full (B, T, V)
        log-softmax tensor is **never materialised**, saving significant
        GPU memory.
        """
        # forward pass: respect the model's training / eval mode
        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        # only count response tokens (exclude prompt and padding)
        shift_mask = attention_mask[:, 1:].contiguous().float()
        shift_mask = shift_mask * response_mask[:, 1:].contiguous().float()

        # fused cross-entropy to avoids full (B, T, V) log-softmax tensor
        per_token_nll = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)

        # sum token log-probabilities (no length normalisation)
        token_log_probs = -per_token_nll * shift_mask
        return token_log_probs.sum(dim=-1)

    def _dpo_loss_from_log_probs(
        self,
        policy_chosen_lp: torch.Tensor,
        policy_rejected_lp: torch.Tensor,
        ref_chosen_lp: torch.Tensor,
        ref_rejected_lp: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        DPO loss computation from pre-computed sequence log-prob.

        L = -log σ( β * ( (log π_θ(y_w|x) - log π_ref(y_w|x))
                         - (log π_θ(y_l|x) - log π_ref(y_l|x)) ) )
        """
        log_ratio_chosen   = policy_chosen_lp - ref_chosen_lp
        log_ratio_rejected = policy_rejected_lp - ref_rejected_lp
        reward_margin = self.beta * (log_ratio_chosen - log_ratio_rejected)

        losses = -F.logsigmoid(reward_margin)
        loss = losses.mean(dim=-1)

        with torch.no_grad():
            accuracy = (reward_margin > 0).float().mean()

        metadata = {
            "loss": loss.item(),  "accuracy": accuracy.item(),
            "reward_margin_mean": reward_margin.mean().item(),
            "reward_margin_std":  reward_margin.std().item(),
            "chosen_log_ratio_mean": log_ratio_chosen.mean().item(),
            "rejected_log_ratio_mean": log_ratio_rejected.mean().item(),
            "implicit_reward_chosen": (self.beta * log_ratio_chosen).mean().item(),
            "implicit_reward_rejected": (self.beta * log_ratio_rejected).mean().item(),
        }

        return loss, metadata

    def compute_loss(self, batch: dict) -> tuple:
        """
        Compute the DPO loss for a single micro-batch of preference pairs.

        The DPO loss is:
            L = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                           - log π_θ(y_l|x) + log π_ref(y_l|x)))

        This method tokenizes each text exactly **once** and reuses the
        tensors for both the policy and reference models.

        Args:
            batch: dict with keys 'instructions', 'chosen', 'rejected'.

        Returns:
            (loss, metadata) where loss is a scalar tensor and
            metadata is a dict of floats for logging.
        """
        instructions = batch["instructions"]
        # format chosen and rejected texts using template
        chosen_texts = [
            self._format_text(inst, resp)
            for inst, resp in zip(instructions, batch["chosen"])
        ]
        rejected_texts = [
            self._format_text(inst, resp)
            for inst, resp in zip(instructions, batch["rejected"])
        ]

        # tokenize once and reuse for both models
        chosen_enc = self.tokenizer(
            chosen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        rejected_enc = self.tokenizer(
            rejected_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # get token ids and attention masks
        chosen_ids = chosen_enc["input_ids"]
        chosen_mask = chosen_enc["attention_mask"]
        rejected_ids = rejected_enc["input_ids"]
        rejected_mask = rejected_enc["attention_mask"]

        # Build response masks (only response tokens contribute to log-prob)
        chosen_resp_mask, rejected_resp_mask = self._build_response_masks(
            instructions, chosen_ids, rejected_ids
        )

        # compute policy model log-probs for chosen and rejected
        policy_lp_chosen = self.compute_sequence_log_prob(
            self.policy_model,
            chosen_ids.to(self.policy_device),
            chosen_mask.to(self.policy_device),
            chosen_resp_mask.to(self.policy_device),
        )
        policy_lp_rejected = self.compute_sequence_log_prob(
            self.policy_model,
            rejected_ids.to(self.policy_device),
            rejected_mask.to(self.policy_device),
            rejected_resp_mask.to(self.policy_device),
        )

        # compute reference model log-probs for chosen and rejected
        with torch.no_grad():
            ref_lp_chosen = self.compute_sequence_log_prob(
                self.reference_model,
                chosen_ids.to(self.ref_device),
                chosen_mask.to(self.ref_device),
                chosen_resp_mask.to(self.ref_device),
            ).to(self.policy_device)
            ref_lp_rejected = self.compute_sequence_log_prob(
                self.reference_model,
                rejected_ids.to(self.ref_device),
                rejected_mask.to(self.ref_device),
                rejected_resp_mask.to(self.ref_device),
            ).to(self.policy_device)

        return self._dpo_loss_from_log_probs(
            policy_lp_chosen,
            policy_lp_rejected,
            ref_lp_chosen,
            ref_lp_rejected,
        )


    def evaluate(self) -> dict:
        """
        Run evaluation on the full validation set.

        Returns:
            dict with 'loss', 'accuracy', and 'reward_margin_mean'
            averaged over all batches.
        """
        self._log(f"Step {self.global_step}: Running Validation...")
        self.policy_model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        total_reward_margin = 0.0
        num_batches = 0

        with torch.inference_mode():
            for batch in tqdm(self.valid_loader, desc="Validating", ncols=100):
                _, metadata = self.compute_loss(batch)
                total_loss += metadata["loss"]
                total_accuracy += metadata["accuracy"]
                total_reward_margin += metadata["reward_margin_mean"]
                num_batches += 1

        val_metrics = {
            "loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "accuracy": (
                total_accuracy / num_batches if num_batches > 0 else 0.0
            ),
            "reward_margin_mean": (
                total_reward_margin / num_batches if num_batches > 0 else 0.0
            ),
        }

        self._log(
            f"  Validation Loss: {val_metrics['loss']:.4f}, "
            f"Accuracy: {val_metrics['accuracy']:.2%}, "
            f"Reward Margin: {val_metrics['reward_margin_mean']:.4f}"
        )
        wandb.log(
            {
                "valid/loss": val_metrics["loss"],
                "valid/accuracy": val_metrics["accuracy"],
                "valid/reward_margin": val_metrics["reward_margin_mean"],
                "step": self.global_step,
            }
        )

        # Track best model
        if val_metrics["accuracy"] > self.best_val_accuracy:
            self.best_val_accuracy = val_metrics["accuracy"]
            self._save_checkpoint("best")
            self._log(
                f"  New best model! (accuracy: {self.best_val_accuracy:.2%})"
            )

        self.policy_model.train()
        return val_metrics


    def train(self):
        """
        Run the complete DPO training loop for ``self.epochs`` epochs.

        Each epoch:
          1. Iterate over micro-batches from the training DataLoader.
          2. Compute DPO loss and backward for each micro-batch.
          3. Perform optimizer step after ``gradient_accumulation_steps``.
          4. Log training metrics to W&B and the record file.
          5. Periodically evaluate on the validation set.
          6. Periodically save checkpoints.
        """
        self._log_config()
        self._log(">>> Starting DPO Training Loop")

        for epoch in range(self.epochs):
            self._log(f"\n>>> Epoch {epoch + 1}/{self.epochs}")
            self.policy_model.train()
            self.optimizer.zero_grad()

            # Per-epoch accumulators
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            accum_count = 0

            # Accumulators for the current gradient-accumulation window
            accum_loss = 0.0
            accum_accuracy = 0.0
            accum_reward_margin = 0.0
            accum_chosen_lr = 0.0
            accum_rejected_lr = 0.0
            last_log_time = time.time()
            last_log_step = self.global_step

            for _, batch in enumerate(self.train_loader):
                # Forward + backward (scaled for gradient accumulation)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, metadata = self.compute_loss(batch)
                (loss / self.gradient_accumulation_steps).backward()

                # record metrics fir current gradient-accumulation window
                accum_loss += metadata["loss"]
                accum_accuracy += metadata["accuracy"]
                accum_reward_margin += metadata["reward_margin_mean"]
                accum_chosen_lr += metadata["chosen_log_ratio_mean"]
                accum_rejected_lr += metadata["rejected_log_ratio_mean"]
                # record metrics for current epoch
                epoch_loss += metadata["loss"]
                epoch_accuracy += metadata["accuracy"]
                num_batches += 1
                accum_count += 1

                # update optimizer and scheduler
                if accum_count % self.gradient_accumulation_steps == 0:
                    clip_grad_norm_(self.policy_model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # average metrics over the accumulation window
                    n = self.gradient_accumulation_steps
                    log_dict = {
                        "step": self.global_step,
                        "train/loss": accum_loss / n,
                        "train/accuracy": accum_accuracy / n,
                        "train/reward_margin_mean": accum_reward_margin / n,
                        "train/chosen_log_ratio": accum_chosen_lr / n,
                        "train/rejected_log_ratio": accum_rejected_lr / n,
                    }
                    log_dict["train/lr"] = self.scheduler.get_last_lr()[0]
                    wandb.log(log_dict)

                    # print a summary line periodically
                    if self.global_step % self.log_interval == 0:
                        now = time.time()
                        steps_since = self.global_step - last_log_step
                        avg_time = (now - last_log_time) / max(steps_since, 1)
                        self._log(
                            f"  Step {self.global_step}: "
                            f"loss={accum_loss / n:.4f}, "
                            f"acc={accum_accuracy / n:.2%}, "
                            f"margin={accum_reward_margin / n:.4f}, "
                            f"time={avg_time:.3f}s"
                        )
                        last_log_time = now
                        last_log_step = self.global_step

                    # reset accumulation window
                    accum_loss = 0.0
                    accum_accuracy = 0.0
                    accum_reward_margin = 0.0
                    accum_chosen_lr = 0.0
                    accum_rejected_lr = 0.0

                    if self.global_step % self.val_interval == 0:
                        self.evaluate()

                    if self.global_step % self.save_interval == 0:
                        self._save_checkpoint(f"step{self.global_step}")

            self._log(
                f"\nEpoch {epoch + 1} Summary: "
                f"avg_loss={epoch_loss / num_batches:.4f}, "
                f"avg_accuracy={epoch_accuracy / num_batches:.2%}"
            )

        self._log("\n>>> Final Validation...")
        self.evaluate()
        self._save_checkpoint("final")

        self._log(
            f"\n>>> DPO Training Complete. "
            f"Best Validation Accuracy: {self.best_val_accuracy:.2%}"
        )
        wandb.finish()

        return self.policy_model


    def _log_config(self):
        """Print / record the full training configuration."""
        total_steps = (
            len(self.train_loader)
            * self.epochs
            // self.gradient_accumulation_steps
        )
        config_info = (
            f"\n{'='*60}\n"
            f"DPO Training Configuration\n"
            f"{'='*60}\n"
            f"Model Device (policy):       {self.policy_device}\n"
            f"Model Device (reference):    {self.ref_device}\n"
            f"Epochs:                      {self.epochs}\n"
            f"Beta (DPO temperature):      {self.beta}\n"
            f"Batch Size:                  {self.batch_size} "
            f"(micro: {self.micro_batch})\n"
            f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}\n"
            f"Learning Rate:               {self.args.lr}\n"
            f"Gradient Clip Norm:          {self.clip_grad_norm}\n"
            f"Max Sequence Length:         {self.max_length}\n"
            f"Total Training Steps:        {total_steps}\n"
            f"Training Examples:           {len(self.train_loader.dataset)}\n"
            f"Validation Examples:         {len(self.valid_loader.dataset)}\n"
            f"Log Interval:                {self.log_interval} steps\n"
            f"Val Interval:                {self.val_interval} steps\n"
            f"Save Interval:               {self.save_interval} steps\n"
            f"Output Directory:            {self.output_dir}\n"
            f"{'='*60}\n"
        )
        self._log(config_info)


    def _save_checkpoint(self, suffix: str = "checkpoint"):
        """
        Save a named checkpoint and return the path.

        Saves both a lightweight state-dict file and a full training-state
        file (including optimizer) for resumable training.
        """
        # Cache state_dict to avoid calling it twice
        state_dict = self.policy_model.state_dict()

        # Lightweight model-only checkpoint
        model_path = os.path.join(self.output_dir, f"dpo_{suffix}.pt")
        torch.save(state_dict, model_path)
        self._log(f"Checkpoint saved to {model_path}")

        # Full training state (for resumption) — skip for "best" which
        # only needs the model weights
        if suffix not in ("best",):
            full_path = os.path.join(
                self.output_dir, f"dpo_{suffix}_full.pt"
            )
            save_dict = {
                "step": self.global_step,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_accuracy": self.best_val_accuracy,
            }
            if self.scheduler is not None:
                save_dict["scheduler_state_dict"] = (
                    self.scheduler.state_dict()
                )
            torch.save(save_dict, full_path)

        return model_path
