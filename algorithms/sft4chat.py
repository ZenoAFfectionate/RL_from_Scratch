import os
import math
import time
import wandb
import torch
import torch.nn.functional as F

from datetime import datetime

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.vllm_helper import *


class SFT4ChatTrainer:
    """
    Trainer for instruction fine-tuning using packed language modeling datasets.
    Expects InstructionTuningDataset instances (from data.lm_dataset) that return
    fixed-length chunks with 'input_ids' and 'labels' keys.
    """

    def __init__(self,
        model,
        tokenizer,
        train_dataset,
        valid_dataset,
        optimizer,
        scheduler,
        args,
        output_dir: str,
        wandb_project: str,
        run_name: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.run_name = run_name
        self.args = args
        #
        self.epochs = args.epochs
        self.device = args.device
        self.log_interval = args.log_interval
        self.val_interval = args.val_interval
        self.val_batches = args.val_batches
        self.clip_grad_norm = args.clip_grad_norm

        assert args.batch_size % args.micro_batch == 0, \
            "batch_size must be divisible by micro_batch."
        self.gradient_accumulation_steps = args.batch_size // args.micro_batch

        self.train_step_count = 0

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.micro_batch,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=args.micro_batch,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self._valid_iter = iter(self.valid_loader)

        os.makedirs(self.output_dir, exist_ok=True)
        self.record_path = os.path.join(self.output_dir, "record.txt")
        with open(self.record_path, "w") as f:
            f.write(f"SFT Training Record - {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

        wandb.init(project=wandb_project, name=run_name, config=vars(args))
        wandb.define_metric("train_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("valid/*", step_metric="train_step")

    def _log(self, message: str):
        """Write *message* to both stdout and the record file."""
        print(message)
        with open(self.record_path, "a") as f:
            f.write(message + "\n")

    def _log_config(self):
        args = self.args
        config_info = (
            f"\n{'='*60}\n"
            f"Training Configuration\n"
            f"{'='*60}\n"
            f"Model:             {getattr(args, 'model_id', 'N/A')}\n"
            f"Epochs:            {args.epochs}\n"
            f"Learning Rate:     {args.lr}\n"
            f"Batch Size:        {args.batch_size} (micro: {args.micro_batch})\n"
            f"Gradient Accum:    {self.gradient_accumulation_steps} steps\n"
            f"Grad Clip Norm:    {args.clip_grad_norm}\n"
            f"Training Chunks:   {len(self.train_dataset)}\n"
            f"Validate Chunks:   {len(self.valid_dataset)}\n"
            f"Log Interval:      {args.log_interval} steps\n"
            f"Val Interval:      {args.val_interval} steps\n"
            f"Val Batches:       {args.val_batches} batches per validation\n"
            f"Output Directory:  {self.output_dir}\n"
            f"{'='*60}\n"
        )
        self._log(config_info)

    def train_step(self, batch) -> tuple[torch.Tensor, dict]:
        """Execute one micro-batch forward + backward pass."""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        # Forward pass
        logits = self.model(input_ids).logits  # (B, seq_len, V)

        # Fused cross-entropy: avoids materializing the full (B, seq_len, V) log-prob tensor
        microbatch_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='mean',
        )

        # Scale for gradient accumulation and backward
        scaled_loss = microbatch_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        # Compute per-token entropy AFTER backward to reduce peak GPU memory:
        # the autograd graph is freed, so this large (B, seq_len, V) allocation
        # does not overlap with backward activations.
        with torch.no_grad():
            log_probs = F.log_softmax(logits, dim=-1)
            token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            avg_entropy = token_entropy.mean()

        metadata = {"entropy": avg_entropy.item()}
        return scaled_loss, metadata

    def validate(self):
        """Validate the model on a subset of the validation set and log metrics.

        Only iterates over at most `self.val_batches` micro-batches from the
        validation DataLoader to keep validation fast during training.
        """
        self._log(f"    [Validate] Running at step {self.train_step_count} "
                  f"(max {self.val_batches} batches)...")
        self.model.eval()

        total_nll = 0.0
        total_tokens = 0

        with torch.inference_mode():
            for _ in range(self.val_batches):
                try:
                    batch = next(self._valid_iter)
                except StopIteration:
                    self._valid_iter = iter(self.valid_loader)
                    batch = next(self._valid_iter)
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Fused cross-entropy with sum reduction for accumulation
                logits = self.model(input_ids).logits
                nll = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='sum',
                )
                total_nll += nll.item()
                total_tokens += input_ids.numel()

        val_loss = total_nll / total_tokens if total_tokens > 0 else 0.0
        val_ppl = math.exp(min(val_loss, 20.0))  # clamp to avoid overflow

        self._log(
            f"    [Validate] Step {self.train_step_count}: "
            f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}"
        )
        wandb.log({
            "val/loss": val_loss,
            "val/ppl": val_ppl,
            "train_step": self.train_step_count,
        })

        self.model.train()

    def train(self):
        """Run the full SFT training loop with periodic validation."""
        self._log_config()
        self._log("Starting training...")
        self.model.train()

        for epoch in range(self.epochs):
            self._log(f"\n>>> Epoch {epoch + 1}/{self.epochs}")

            # zero gradients at epoch boundary to prevent stale gradient leak
            self.optimizer.zero_grad(set_to_none=True)

            # accumulators for metrics across micro-batches in one optimizer step.
            accumulated_loss = 0.0
            accumulated_entropy = 0.0
            step_start_time = time.time()

            for i, batch in enumerate(self.train_loader):
                loss, metadata = self.train_step(batch)

                accumulated_loss += loss.item()
                accumulated_entropy += metadata.get("entropy", 0.0)

                # Optimizer step after accumulating enough micro-batches
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.train_step_count += 1

                    step_elapsed = time.time() - step_start_time

                    # averaged metrics over the accumulation window
                    avg_loss = accumulated_loss
                    avg_entropy = accumulated_entropy / self.gradient_accumulation_steps
                    ppl = math.exp(min(avg_loss, 100.0))  # clamp to avoid overflow
                    current_lr = self.scheduler.get_last_lr()[0]

                    wandb.log({
                        "train/loss": avg_loss,
                        "train/ppl": ppl,
                        "train/entropy": avg_entropy,
                        "train/lr": current_lr,
                        "train/step_time": step_elapsed,
                        "train_step": self.train_step_count,
                    })

                    if self.train_step_count % self.log_interval == 0:
                        self._log(
                            f"    [Train] Step {self.train_step_count}: "
                            f"loss={avg_loss:.4f}, ppl={ppl:.2f}, "
                            f"entropy={avg_entropy:.4f}, lr={current_lr:.2e}, "
                            f"step_time={step_elapsed:.2f}s"
                        )

                    if self.train_step_count % self.val_interval == 0:
                        self.validate()

                    accumulated_loss = 0.0
                    accumulated_entropy = 0.0
                    step_start_time = time.time()

        self._log("\nRunning final validation...")
        self.validate()

        final_path = os.path.join(self.output_dir, f"{self.run_name}_final.pt")
        torch.save(self.model.state_dict(), final_path)
        self._log(f"\nFinal model saved to {final_path}")

        summary = (
            f"\n{'='*60}\n"
            f"Training Complete!\n"
            f"{'='*60}\n"
            f"Total Training Steps: {self.train_step_count}\n"
            f"Final Model:          {final_path}\n"
            f"Record File:          {self.record_path}\n"
            f"{'='*60}\n"
        )
        self._log(summary)
        wandb.finish()

        return self.model

    def save_checkpoint(self, suffix: str = "checkpoint"):
        """Save a named checkpoint and return the path."""
        path = os.path.join(self.output_dir, f"{self.run_name}_{suffix}.pt")
        torch.save(self.model.state_dict(), path)
        self._log(f"Checkpoint saved to {path}")
        return path
