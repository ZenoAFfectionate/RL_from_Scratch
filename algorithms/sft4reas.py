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
from algorithms.utils import tokenize_prompt_and_output, pad_collate


class SFT4ReasTrainer:
    """
    Supervised Fine-Tuning (SFT) trainer for math and code task.

    This trainer fine-tunes a language model on math problem/solution pairs using
    a standard cross-entropy loss computed only on response tokens (via response_mask).
    It supports gradient accumulation, gradient clipping, learning rate scheduling,
    and periodic validation using vLLM-based generation and reward evaluation.

    Key features:
        - Response-masked loss: only solution tokens contribute to the training loss.
        - Per-token entropy tracking for monitoring model confidence.
        - vLLM-accelerated validation: loads the policy into a vLLM instance for
          fast batched generation and accuracy evaluation.
        - Wandb integration for experiment tracking and metric logging.
        - Checkpoint saving with configurable intervals.
    """
    def __init__(self,
        model,
        tokenizer,
        train_data,
        valid_data,
        reward_fun,
        optimizer,
        scheduler,
        valid_model,
        valid_sampling_params,
        args,
        output_dir: str,
        wandb_project: str,
        run_name: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.reward_fun = reward_fun
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.valid_model = valid_model
        self.valid_sampling_params = valid_sampling_params
        self.output_dir = output_dir
        self.run_name = run_name
        self.args = args
        # 
        self.epochs = args.epochs
        self.train_device = args.train_device
        self.valid_device = args.valid_device
        self.log_interval = args.log_interval
        self.val_interval = args.val_interval
        self.clip_grad_norm = args.clip_grad_norm
        self.prompt_template = args.prompt_template

        assert args.batch_size % args.micro_batch == 0, \
            "batch_size must be divisible by micro_batch size."
        self.gradient_accumulation_steps = args.batch_size // args.micro_batch

        self.train_step_count = 0

        os.makedirs(self.output_dir, exist_ok=True)
        self.record_path = os.path.join(self.output_dir, "record.txt")
        with open(self.record_path, "w") as f:
            f.write(f"SFT Training Record - {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

        def collate_fn(batch):
            """Tokenize math problem/solution pairs with response_mask."""
            prompts   = [self.prompt_template.format(problem=b['problem']) for b in batch]
            responses = [b['solution'] for b in batch]
            samples = tokenize_prompt_and_output(prompts, responses, self.tokenizer)
            return pad_collate(samples, self.tokenizer.pad_token_id)
        
        self._log("Initializing data loaders...")
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=args.micro_batch,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.valid_problems  = [self.prompt_template.format(problem=v['problem']) for v in self.valid_data]
        if self.args.dataset == "code":
            self.valid_solutions = [v["test"] for v in self.valid_data]
        else:
            self.valid_solutions = [v["solution"] for v in self.valid_data]

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
            f"Training Samples:  {len(self.train_data)}\n"
            f"Validate Samples:  {len(self.valid_data)}\n"
            f"Log Interval:      {args.log_interval} steps\n"
            f"Val Interval:      {args.val_interval} steps\n"
            f"Output Directory:  {self.output_dir}\n"
            f"{'='*60}\n"
        )
        self._log(config_info)


    def train_step(self, batch) -> tuple[torch.Tensor, dict]:
        """Execute one micro-batch forward + backward pass."""
        input_ids     = batch["input_ids"].to(self.train_device, non_blocking=True)
        labels        = batch["labels"].to(self.train_device, non_blocking=True)
        response_mask = batch["response_mask"].to(self.train_device, non_blocking=True)

        # build attention mask so the model does not attend to padding tokens
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        logits = self.model(input_ids, attention_mask=attention_mask).logits

        # fused cross-entropy: avoids materializing the full (B, T, V) softmax
        per_token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (batch_size*seq_len, vocab_size)
            labels.view(-1),                   # (batch_size*seq_len,)
            reduction='none',
        ).view(labels.shape)                   # (batch_size, seq_len)

        del logits  # free logits before backward to save memory

        # per-token NLL averaged over response tokens only
        total_response_tokens = response_mask.sum().clamp(min=1)
        microbatch_loss = (per_token_loss * response_mask).sum() / total_response_tokens

        scaled_loss = microbatch_loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        return scaled_loss, {}


    def validate(self):
        """Validate the model on a subset of the validation set and log metrics.

        Only iterates over at most `self.val_batches` micro-batches from the
        validation DataLoader to keep validation fast during training.
        """
        self._log(f"  [Validate] Running at step {self.train_step_count}")
        self.model.eval()

        load_policy_into_vllm_instance(self.model, self.valid_model)
        output_path = f"../results/sft4{self.args.dataset}.jsonl"
        acc = evaluate_vllm(self.valid_model, self.reward_fun, self.valid_problems,
                            self.valid_solutions, self.valid_sampling_params, output_path)

        self.model.train()
        self._log(f"  [Validation] Accuracy: {acc:.2f}%")


    def train(self):
        """Run the full SFT training loop with periodic validation."""
        self._log_config()
        self._log("Starting training...")
        self.model.train()

        for epoch in range(self.epochs):
            self._log(f"\n>>> Epoch {epoch + 1}/{self.epochs}")

            # accumulators for metrics across micro-batches in one optimizer step.
            accumulated_loss    = 0.0
            step_start_time     = time.time()

            for i, batch in enumerate(self.train_loader):
                loss, _ = self.train_step(batch)

                accumulated_loss    += loss.item()

                # Optimizer step after accumulating enough micro-batches
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.train_step_count += 1

                    step_elapsed = time.time() - step_start_time

                    # correctly averaged metrics over the accumulation window
                    ppl = math.exp(min(accumulated_loss, 100))
                    current_lr = self.scheduler.get_last_lr()[0]

                    wandb.log({
                        "train/loss": accumulated_loss,
                        "train/ppl": ppl,
                        "train/lr": current_lr,
                        "train/step_time": step_elapsed,
                        "train_step": self.train_step_count,
                    })

                    if self.train_step_count % self.log_interval == 0:
                        self._log(
                            f"  [Train] Step {self.train_step_count}: "
                            f"loss={accumulated_loss:.4f}, ppl={ppl:.2f}, "
                            f"lr={current_lr:.2e}, "
                            f"step_time={step_elapsed:.2f}s"
                        )

                    # validate model periodically:
                    if self.train_step_count % self.val_interval == 0:
                        self.validate()

                    # Reset accumulators for next optimizer step
                    accumulated_loss    = 0.0
                    step_start_time     = time.time()

        # Final validation before saving
        self._log("\nRunning final validation...")
        self.validate()

        # Save final checkpoint
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