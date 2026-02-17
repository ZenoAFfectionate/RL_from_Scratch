# LLM Alignment: From Supervised Fine-Tuning to Reinforcement Learning

This project implements a comprehensive framework for aligning Large Language Models (LLMs) with human preferences and task-specific objectives. Built upon the Qwen open-source model family, the framework encompasses supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF) algorithms, including Direct Preference Optimization (DPO), Group Relative Policy Optimization (GRPO), and Proximal Policy Optimization (PPO). The implementation is evaluated on multiple tasks: conversational dialogue, mathematical problem-solving, code generation, and safety alignment.

## Overview

- **Foundation Model**: Built on Qwen2.5 series models (Qwen2.5-Math for mathematical reasoning, Qwen3 for dialogue tasks), leveraging state-of-the-art open-source LLM capabilities
- **Chain-of-Thought Data Construction**: Novel approach to construct reasoning datasets using one-shot prompting and N-sampling with Qwen-Math models, achieving 99.2% accuracy
- **Supervised Fine-Tuning**: Implements efficient SFT with gradient accumulation, sequence packing, and response masking for both chat and math domains
- **Reinforcement Learning Algorithms**: Features DPO for preference-based alignment on dialogue tasks, GRPO for reward-based optimization on mathematical reasoning, and PPO for general RL fine-tuning
- **Dual Training Framework**: Provides two parallel implementations — a custom training loop (`trainer/`) and a HuggingFace TRL-based implementation (`trl_trainer/`) — allowing easy comparison and flexibility
- **Speculative Decoding**: High-performance inference optimization with vectorized verification, adaptive speculation length, async pipeline, and continuous batching support
- **Comprehensive Evaluation**: Baseline evaluation scripts for MATH, GSM8K, Code, MMLU, AlpacaEval, and SimpleSafetyTests benchmarks

## Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Running Baseline Evaluations](#running-baseline-evaluations)
  - [Custom Trainers (trainer/)](#custom-trainers-trainer)
  - [TRL-based Trainers (trl_trainer/)](#trl-based-trainers-trl_trainer)
- [Changelog](#changelog)
- [Math Dataset Construction](#math-dataset-construction)
- [Supervised Fine-Tuning](#supervised-fine-tuning)
  - [Data Processing](#data-processing-for-sft)
  - [Algorithm Principles](#sft-algorithm-principles)
  - [Training Process](#sft-training-process)
- [Reinforcement Learning](#reinforcement-learning)
  - [Data Processing](#data-processing-for-rl)
  - [DPO Algorithm](#dpo-algorithm)
  - [GRPO Algorithm](#grpo-algorithm)
- [Speculative Decoding](#speculative-decoding)

## Project Structure

```
Assignment_5/
├── algorithms/             # Core algorithm implementations
│   ├── sft4chat.py         # SFT algorithm for chat
│   ├── sft4reas.py         # SFT algorithm for reasoning
│   ├── dpo.py              # DPO algorithm
│   ├── grpo.py             # GRPO algorithm
│   ├── ppo.py              # PPO algorithm
│   ├── speculative.py      # Speculative decoding
│   └── utils.py            # Shared algorithm utilities
│
├── trainer/                # Custom training loop scripts
│   ├── utils.py            # Shared: prompt loading, data loading, reward functions
│   ├── sft_trainer.py      # Unified SFT (chat + reasoning modes)
│   ├── dpo_trainer.py      # DPO trainer
│   ├── grpo_trainer.py     # GRPO trainer
│   └── ppo_trainer.py      # PPO trainer
│
├── trl_trainer/            # HuggingFace TRL-based training scripts
│   ├── utils.py            # Shared: prompt loading, reward functions, TRL wrappers
│   ├── sft_trainer.py      # TRL SFTTrainer (chat + reasoning modes)
│   ├── dpo_trainer.py      # TRL DPOTrainer
│   ├── grpo_trainer.py     # TRL GRPOTrainer
│   └── ppo_trainer.py      # TRL PPOTrainer
│
├── baseline/               # Baseline evaluation scripts
│   ├── baseline_math.py    # MATH benchmark
│   ├── baseline_gsmk.py    # GSM8K benchmark
│   ├── baseline_code.py    # Code benchmark
│   ├── baseline_mmlu.py    # MMLU benchmark
│   ├── baseline_alpa.py    # AlpacaEval benchmark
│   └── baseline_ssts.py    # SimpleSafetyTests benchmark
│
├── prompts/                # Prompt templates (single source of truth)
│   ├── math.prompt         # Math reasoning template
│   ├── code.prompt         # Code generation template
│   ├── gsmk.prompt         # GSM8K template
│   ├── rlhf.prompt         # RLHF / chat template
│   ├── mmlu.prompt         # MMLU template
│   └── ssts.prompt         # Safety tests template
│
├── data/                   # Datasets
│   ├── math/               # MATH dataset (train.jsonl, valid.jsonl)
│   ├── gsmk/               # GSM8K dataset
│   ├── code/               # Code dataset
│   ├── rlhf/               # RLHF preference data
│   ├── ultrachat/          # UltraChat dataset
│   ├── mmlu/               # MMLU dataset
│   ├── alpa/               # AlpacaEval dataset
│   ├── ssts/               # SimpleSafetyTests dataset
│   └── lm_dataset.py       # Dataset utilities
│
├── utils/                  # Project-level utilities
│   ├── rewards.py          # Reward functions (dsr1, gsmk, code)
│   ├── vllm_helper.py      # vLLM initialization helpers
│   ├── math_utils.py       # Math evaluation utilities
│   └── code_utils.py       # Code evaluation utilities
│
├── checkpoints/            # Checkpoints from trainer/
├── [TRL]checkpoints/       # Checkpoints from trl_trainer/
├── results/                # Evaluation results from trainer/
├── [TRL]results/           # Evaluation results from trl_trainer/
└── tests/                  # Unit tests
```

**Output directory convention:**

| Source | Checkpoints | Results |
|---|---|---|
| `trainer/` | `checkpoints/` | `results/` |
| `trl_trainer/` | `[TRL]checkpoints/` | `[TRL]results/` |

## Quick Start

### Running Baseline Evaluations

The `baseline/` folder contains evaluation scripts for various benchmarks using vLLM for efficient inference. Each script supports greedy decoding and can be run independently. All results are saved to `results/`.

```bash
# Evaluate on MATH dataset
python baseline/baseline_math.py --model_id Qwen/Qwen2.5-Math-7B-Instruct

# Evaluate on GSM8K dataset
python baseline/baseline_gsmk.py --model_id Qwen/Qwen2.5-Math-7B-Instruct

# Evaluate on Code dataset
python baseline/baseline_code.py --model_id Qwen/Qwen2.5-Math-7B-Instruct

# Evaluate on MMLU benchmark
python baseline/baseline_mmlu.py --model_id Qwen/Qwen3-1.7B

# Generate responses for AlpacaEval
python baseline/baseline_alpa.py --model_id Qwen/Qwen3-1.7B

# Run SimpleSafetyTests evaluation
python baseline/baseline_ssts.py --model_id Qwen/Qwen3-1.7B
```

---

### Custom Trainers (`trainer/`)

These trainers use custom training loops with direct PyTorch optimization. They require vLLM for validation-time generation (reasoning/RL tasks use two GPUs: one for training, one for vLLM inference). Checkpoints are saved to `checkpoints/`.

> **Note**: All `trainer/` scripts should be run from the project root directory.

#### SFT — Chat Mode

Fine-tune on the UltraChat dataset for conversational tasks:

```bash
python trainer/sft_trainer.py \
    --mode chat \
    --model_id Qwen/Qwen3-1.7B \
    --lr 1e-5 \
    --epochs 2 \
    --batch_size 32 \
    --micro_batch 2 \
    --max_length 2048
```

Checkpoints: `checkpoints/SFT4Chat/`

#### SFT — Reasoning Mode

Fine-tune on math/code/gsmk datasets with chain-of-thought:

```bash
# Math
python trainer/sft_trainer.py \
    --mode reasoning \
    --dataset math \
    --model_id Qwen/Qwen2.5-Math-1.5B \
    --lr 5e-6 \
    --epochs 2 \
    --batch_size 32 \
    --micro_batch 2

# Code
python trainer/sft_trainer.py \
    --mode reasoning \
    --dataset code \
    --model_id Qwen/Qwen2.5-Math-1.5B

# GSM8K
python trainer/sft_trainer.py \
    --mode reasoning \
    --dataset gsmk \
    --model_id Qwen/Qwen2.5-Math-1.5B
```

Checkpoints: `checkpoints/SFT4{dataset}/` (e.g., `checkpoints/SFT4math/`)

#### DPO

Preference-based alignment on RLHF data (requires 2 GPUs: policy + reference model):

```bash
python trainer/dpo_trainer.py \
    --model_id Qwen/Qwen3-1.7B \
    --dataset rlhf \
    --beta 0.2 \
    --lr 5e-6 \
    --epochs 2 \
    --batch_size 32 \
    --micro_batch 4 \
    --policy_device cuda:1 \
    --ref_device cuda:2
```

Checkpoints: `checkpoints/DPO/`

#### GRPO

Reward-based RL for math/code reasoning (requires 2 GPUs: training + vLLM):

```bash
python trainer/grpo_trainer.py \
    --dataset math \
    --model_id Qwen/Qwen2.5-Math-1.5B \
    --lr 5e-6 \
    --num_iterations 250 \
    --num_generations 8 \
    --batch_size 32 \
    --micro_batch 2 \
    --beta 0.01 \
    --train_device cuda:0 \
    --valid_device cuda:1

# With SFT warm-start
python trainer/grpo_trainer.py \
    --dataset math \
    --init_checkpoint checkpoints/SFT4math/sft_math_lr5e-06_final.pt
```

Checkpoints: `checkpoints/GRPO/`

#### PPO

Proximal Policy Optimization for math/code reasoning (requires 2 GPUs):

```bash
python trainer/ppo_trainer.py \
    --dataset math \
    --model_id Qwen/Qwen2.5-Math-1.5B \
    --lr 1e-5 \
    --num_iterations 200 \
    --batch_size 32 \
    --micro_batch 2 \
    --beta 0.1 \
    --train_device cuda:1 \
    --valid_device cuda:2
```

Checkpoints: `checkpoints/PPO/`

---

### TRL-based Trainers (`trl_trainer/`)

These trainers use HuggingFace's [TRL](https://github.com/huggingface/trl) library, providing a higher-level API with built-in W&B logging, gradient checkpointing, and distributed training support. Checkpoints are saved to `[TRL]checkpoints/`.

> **Note**: All `trl_trainer/` scripts should be run from the project root directory.

#### SFT — Chat Mode

```bash
python trl_trainer/sft_trainer.py \
    --mode chat \
    --model_id Qwen/Qwen3-1.7B \
    --lr 2e-5 \
    --epochs 2 \
    --batch_size 32 \
    --micro_batch 2 \
    --max_length 2048 \
    --packing
```

Checkpoints: `[TRL]checkpoints/SFT4Chat/`

#### SFT — Reasoning Mode

```bash
# Math
python trl_trainer/sft_trainer.py \
    --mode reasoning \
    --dataset math \
    --model_id Qwen/Qwen2.5-Math-1.5B \
    --lr 5e-6 \
    --epochs 2 \
    --batch_size 32 \
    --micro_batch 2

# Code
python trl_trainer/sft_trainer.py \
    --mode reasoning \
    --dataset code \
    --model_id Qwen/Qwen2.5-Math-1.5B
    --lr 5e-6 \
    --epochs 2 \
    --batch_size 32 \
    --micro_batch 2

# GSM8K
python trl_trainer/sft_trainer.py \
    --mode reasoning \
    --dataset gsmk \
    --model_id Qwen/Qwen2.5-Math-1.5B
    --lr 5e-6 \
    --epochs 2 \
    --batch_size 32 \
    --micro_batch 2
```

Checkpoints: `[TRL]checkpoints/SFT4{dataset}/`

#### DPO

```bash
python trl_trainer/dpo_trainer.py \
    --model_id Qwen/Qwen3-1.7B \
    --dataset rlhf \
    --beta 0.2 \
    --lr 5e-6 \
    --epochs 2 \
    --batch_size 32 \
    --micro_batch 4

# With SFT warm-start
python trl_trainer/dpo_trainer.py \
    --init_checkpoint checkpoints/SFT4Chat/sft_final.pt
```

Checkpoints: `[TRL]checkpoints/DPO/`

#### GRPO

```bash
python trl_trainer/grpo_trainer.py \
    --dataset math \
    --model_id Qwen/Qwen2.5-Math-1.5B \
    --lr 5e-6 \
    --max_steps 250 \
    --num_generations 8 \
    --batch_size 32 \
    --micro_batch 2 \
    --beta 0.01

# With SFT warm-start
python trl_trainer/grpo_trainer.py \
    --dataset math \
    --init_checkpoint checkpoints/SFT4math/sft_final.pt
```

Checkpoints: `[TRL]checkpoints/GRPO4{dataset}/`

#### PPO

```bash
python trl_trainer/ppo_trainer.py \
    --dataset math \
    --model_id Qwen/Qwen2.5-Math-1.5B \
    --lr 1e-5 \
    --total_episodes 6400 \
    --batch_size 32 \
    --micro_batch 2 \
    --kl_coef 0.1

# With SFT warm-start
python trl_trainer/ppo_trainer.py \
    --dataset math \
    --init_checkpoint checkpoints/SFT4math/sft_final.pt
```

Checkpoints: `[TRL]checkpoints/PPO4{dataset}/`

---

## Changelog

### Unified Prompt Template System

Prompt templates were previously duplicated as inline Python strings across multiple trainer files. They have been consolidated into a single source of truth under `prompts/`:

```
prompts/
├── math.prompt    # Math reasoning (think/answer format)
├── code.prompt    # Competitive programming
├── gsmk.prompt    # GSM8K step-by-step
├── rlhf.prompt    # Alpaca-style instruction/response
├── mmlu.prompt    # Multiple-choice QA
└── ssts.prompt    # Safety alignment
```

All trainers now load prompts at runtime via `load_prompt_template(dataset_name)`, which reads from `prompts/<dataset_name>.prompt`. No inline prompt strings remain in the codebase.

### Shared Utilities (`utils.py`)

Both `trainer/` and `trl_trainer/` now have a `utils.py` module that centralizes shared logic:

- **`load_prompt_template(dataset_name)`** — Reads the corresponding `.prompt` file from `prompts/`
- **`get_reward_fn(dataset_name)`** — Returns the appropriate reward function (`dsr1_reward_fn`, `gsmk_reward_fn`, or `code_reward_fn`)
- **`load_jsonl(path)`** — Convenience function for loading JSONL data files (in `trainer/utils.py`)
- **`make_trl_reward_fn()`** — Wraps reward functions to match TRL's expected interface (in `trl_trainer/utils.py`)

This eliminates repeated `from utils.rewards import ...` and inline `open()` calls scattered across trainer files.

### Unified SFT Trainer

The previously separate `sft4chat_trainer.py` and `sft4reas_trainer.py` have been merged into a single `sft_trainer.py` with a `--mode` flag:

```bash
# Before (removed):
python trainer/sft4chat_trainer.py ...
python trainer/sft4reas_trainer.py ...

# After:
python trainer/sft_trainer.py --mode chat ...
python trainer/sft_trainer.py --mode reasoning --dataset math ...
```

This applies to both `trainer/sft_trainer.py` and `trl_trainer/sft_trainer.py`.

### TRL-based Training Implementation

A complete parallel implementation using HuggingFace TRL was added under `trl_trainer/`, covering all four training algorithms (SFT, DPO, GRPO, PPO). The TRL trainers provide:

- Built-in W&B integration and logging
- Automatic gradient checkpointing
- HuggingFace-compatible checkpoint saving (model + tokenizer)
- Additionally saves `.pt` state dicts for compatibility with the custom evaluation pipeline

### Separated Output Directories

To avoid checkpoint/result collisions between the two trainer implementations:

| | Custom (`trainer/`) | TRL (`trl_trainer/`) |
|---|---|---|
| Checkpoints | `checkpoints/` | `[TRL]checkpoints/` |
| Results | `results/` | `[TRL]results/` |
| W&B projects | `SFT4Chat`, `DPO-RLHF`, `GRPO-MATH`, ... | `[TRL]SFT4Chat`, `[TRL]DPO-RLHF`, `[TRL]GRPO-MATH`, ... |

### PPO Trainer

A full PPO training pipeline was added to both `trainer/` and `trl_trainer/`, supporting math and code datasets with verifiable reward functions. The TRL version includes a `VerifiableRewardModel` wrapper that bridges the project's reward functions with TRL's expected reward model interface.

---

## Math Dataset Construction

A significant challenge in training models for mathematical reasoning is the lack of datasets with explicit chain-of-thought reasoning in the R1-zero format (with `<think>` and `<answer>` tags). The original MATH dataset provides problems and solutions but not in the structured thinking format required for training reasoning models.

The initial approach was to use the DeepSeek API to generate thinking processes for each problem. While this produces high-quality outputs, it is both expensive and slow due to API rate limits. The `distill_thinking.py` script implements this approach with async processing and concurrency control:

```python
async def process_single_problem(sem, line, index, outfile, wrong_file):
    """Async function to process a single problem using DeepSeek API."""
    async with sem:  # Semaphore to control concurrency
        prompts = [init_prompt, modi_prompt]  # Two-round approach

        for attempt, current_prompt in enumerate(prompts):
            chat_completion = await client.chat.completions.create(
                model='deepseek-reasoner',
                messages=[{"role": "user", "content": current_prompt}],
                max_tokens=4096,
                temperature=0.2,
            )
            response = chat_completion.choices[0].message.content.strip()
            result = r1_zero_reward_fn(response, solution, fast=True)

            if result.get('reward', 0.0) == 1.0:
                is_correct = True
                break
```

To address the cost and speed limitations, a novel local distillation approach was developed using Qwen-2.5-Math-7B. This method employs one-shot prompting with a carefully crafted example to guide the model in generating properly formatted responses, combined with N-sampling (best-of-N) to increase the probability of obtaining correct answers:

```python
# One-shot template with detailed example
init_template = """A conversation between User and Assistant...
User: How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
Assistant: <think>
To find the vertical asymptotes of the rational function...
</think>
<answer>\\boxed{2}</answer>

User: {question}
Assistant:"""

# Multi-round generation with best-of-N sampling
for round_num in range(1, args.max_rounds + 1):
    if round_num == 1:
        prompts = [init_template.format(question=item["problem"]) for _, item in remaining_items]
    else:
        # For retry rounds, provide the ground truth solution as guidance
        prompts = [
            modi_template.format(question=item["problem"], solution=item["solution"])
            for _, item in remaining_items
        ]

    outputs = llm.generate(prompts, sampling_params)  # Generate N samples per prompt

    for idx, output in enumerate(outputs):
        for sample in output.outputs:
            reward_dict = r1_zero_reward_fn(sample.text, ground_truth)
            if bool(reward_dict.get("reward", 0)):
                successful_data.append({"problem": item["problem"], "solution": sample.text})
                found_correct = True
                break
```

The key innovation is the two-round approach: in the first round, the model attempts to solve problems independently using one-shot prompting. For problems that remain unsolved, the second round provides the ground truth solution and asks the model to construct a thinking process that leads to the correct answer. This approach achieves a 99.2% success rate in constructing properly formatted chain-of-thought data, making it a practical and cost-effective alternative to API-based distillation.


## Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is the foundational step in aligning language models with desired behaviors. The goal is to train the model to generate responses that match high-quality demonstrations, effectively teaching the model the format, style, and content expected in its outputs.

### Data Processing for SFT

The data processing pipeline transforms raw instruction-response pairs into training-ready tensors. A critical aspect is the construction of a response mask that ensures the loss is computed only on the response tokens, not the prompt tokens. This prevents the model from being penalized for the prompt content it cannot control.

```python
def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    """
    batch_ids, prompt_lens, output_lens = [], [], []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        output_ids = tokenizer(output, add_special_tokens=False).input_ids
        batch_ids.append(prompt_ids + output_ids + [tokenizer.eos_token_id])
        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(output_ids))

    # Create response mask - only compute loss on response tokens
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, (p_len, o_len, seq) in enumerate(zip(prompt_lens, output_lens, batch_ids)):
        start = max(p_len - 1, 0)
        end = start + o_len
        if end > start: response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
```

For chat tasks, the `InstructionTuningDataset` class implements sequence packing, which concatenates multiple documents into fixed-length chunks to maximize GPU utilization. This approach significantly improves training efficiency by reducing padding overhead.

### SFT Algorithm Principles

The SFT objective is to maximize the log-likelihood of the target response given the prompt. Mathematically, for a prompt $x$ and response $y = (y_1, y_2, ..., y_T)$, the loss function is the negative log-likelihood:

$$\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{T} \log p_\theta(y_t | x, y_{<t})$$

The implementation computes per-token log probabilities and applies the response mask to ensure only response tokens contribute to the gradient:

```python
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute conditional log-probabilities log p_theta(x_t | x_<t)."""
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy
    return result
```

The training step applies masked normalization to compute the loss only over response tokens, then scales the loss for gradient accumulation:

```python
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch."""
    microbatch_loss = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1
    )
    scaled_loss = microbatch_loss.mean() / gradient_accumulation_steps
    scaled_loss.backward()
    return scaled_loss, {}
```

### SFT Training Process

The training loop follows a standard pattern with gradient accumulation to handle large effective batch sizes on limited GPU memory. For mathematical reasoning tasks, the trainer supports experiments with varying dataset sizes (128, 256, 512, 1024, and full dataset) to study the data efficiency of SFT. After each epoch, the model is evaluated using vLLM for efficient inference, and the best checkpoint is saved based on validation accuracy.

```python
for epoch in range(args.epochs):
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        input_ids = batch['input_ids'].to(args.policy_device)
        labels = batch['labels'].to(args.policy_device)
        response_mask = batch['response_mask'].to(args.policy_device)

        results = get_response_log_probs(policy, input_ids, labels, True)
        loss, metadata = sft_microbatch_train_step(
            results['log_probs'], response_mask, args.gradient_accumulation_steps,
            normalize_constant=response_mask.sum()
        )

        if (i + 1) % args.gradient_accumulation_steps == 0:
            clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
```

## Reinforcement Learning

While SFT teaches models to imitate demonstrations, reinforcement learning enables models to optimize for specific objectives beyond simple imitation. This project implements three complementary RL approaches: DPO for learning from human preferences, GRPO for optimizing verifiable rewards, and PPO for general policy optimization.

### Data Processing for RL

For DPO training, the data consists of preference pairs where each example contains an instruction, a chosen (preferred) response, and a rejected response. The `RLHFDataset` class loads these triplets and the collate function prepares them for batch processing:

```python
def collate_fn(batch):
    """Collate function that returns lists of strings."""
    instructions = [item['instruction'] for item in batch]
    chosen = [item['chosen'] for item in batch]
    rejected = [item['rejected'] for item in batch]
    return {'instructions': instructions, 'chosen': chosen, 'rejected': rejected}
```

For GRPO training on mathematical reasoning, the data processing involves generating multiple rollout responses per prompt and computing rewards based on answer correctness. The reward function validates both the format (presence of `<think>` and `<answer>` tags) and the mathematical correctness of the final answer.

### DPO Algorithm

Direct Preference Optimization (DPO) provides a stable and efficient approach to preference-based alignment by reformulating the RLHF objective as a classification problem. Instead of training a separate reward model and using policy gradient methods, DPO directly optimizes the policy to prefer chosen responses over rejected ones.

The DPO loss is derived from the Bradley-Terry preference model and can be expressed as:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)$$

where $y_w$ is the chosen response, $y_l$ is the rejected response, $\pi_\theta$ is the policy being trained, $\pi_{\text{ref}}$ is the frozen reference model, and $\beta$ is a temperature parameter controlling the strength of the KL constraint.

```python
def dpo_loss(
    policy_model: PreTrainedModel,
    reference_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    chosen_response: str,
    rejected_response: str,
    template: str,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the per-instance DPO loss."""
    # Compute log-probabilities under policy model
    policy_log_prob_chosen = compute_sequence_log_prob(policy_model, chosen_ids, chosen_mask)
    policy_log_prob_rejected = compute_sequence_log_prob(policy_model, rejected_ids, rejected_mask)

    # Compute log-probabilities under reference model (no gradient)
    with torch.no_grad():
        ref_log_prob_chosen = compute_sequence_log_prob(reference_model, chosen_ids, chosen_mask)
        ref_log_prob_rejected = compute_sequence_log_prob(reference_model, rejected_ids, rejected_mask)

    # Compute log ratios and reward margin
    log_ratio_chosen = policy_log_prob_chosen - ref_log_prob_chosen
    log_ratio_rejected = policy_log_prob_rejected - ref_log_prob_rejected
    reward_margin = beta * (log_ratio_chosen - log_ratio_rejected)

    # DPO loss using logsigmoid for numerical stability
    loss = -F.logsigmoid(reward_margin)
    return loss.squeeze()
```

The DPO training loop maintains both a policy model (being optimized) and a frozen reference model. The reference model provides a baseline that prevents the policy from deviating too far from the initial distribution.

### GRPO Algorithm

Group Relative Policy Optimization (GRPO) is designed for tasks with verifiable rewards, such as mathematical problem-solving where answer correctness can be automatically checked. Unlike DPO which requires pre-collected preference data, GRPO generates multiple responses per prompt and uses the reward signal to compute advantages.

The key insight of GRPO is to normalize rewards within each group of responses to the same prompt, which reduces variance and provides a natural baseline. For a group of $G$ responses to the same prompt, the advantage for response $i$ is computed as:

$$A_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}$$

where $\mu_g$ and $\sigma_g$ are the mean and standard deviation of rewards within the group.

```python
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """Compute rewards for each group of rollout responses, normalized by the group size."""
    rewards = [reward_fn(r, g) for r, g in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([res["reward"] for res in rewards], dtype=torch.float32)

    # Reshape to group-wise and compute statistics
    rewards_matrix = raw_rewards.view(-1, group_size)
    group_means = rewards_matrix.mean(dim=1, keepdim=True)

    if normalize_by_std:
        group_stds = rewards_matrix.std(dim=1, keepdim=True) + advantage_eps
        normalized_rewards = (rewards_matrix - group_means) / group_stds
    else:
        normalized_rewards = rewards_matrix - group_means

    advantages = normalized_rewards.flatten()
    return advantages, raw_rewards, metadata
```

The GRPO loss supports multiple variants including vanilla REINFORCE, REINFORCE with baseline, and a PPO-style clipped objective:

```python
def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Computes the per-token GRPO-Clip loss."""
    ratio = torch.exp(policy_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
    loss = -torch.min(surr1, surr2)

    with torch.no_grad():
        clip_fraction = (surr2 < surr1).float().mean()
    return loss, {"clip_fraction": clip_fraction, "ratio_mean": ratio.detach().mean()}
```

## Speculative Decoding

Speculative decoding is an inference optimization technique that accelerates autoregressive generation by using a smaller, faster draft model to propose multiple tokens that are then verified in parallel by the larger target model. This project implements a comprehensive speculative decoding system with several advanced features.

The core principle of speculative decoding is based on the observation that a smaller model can often predict the same tokens as a larger model, especially for common patterns. By generating $K$ draft tokens and verifying them in a single forward pass of the target model, we can potentially accept multiple tokens per target model call, significantly improving throughput.

The verification process compares draft tokens against the target model's predictions. If a draft token matches, it is accepted; otherwise, the target model's prediction is used and subsequent draft tokens are discarded:

```python
def verify_tokens(self, input_ids: List[int], draft_tokens: List[int]) -> Tuple[List[int], int, bool]:
    """Verify all draft tokens in one forward pass using the target model."""
    candidate_ids = input_ids + draft_tokens

    output = self.target_llm.generate(
        prompt_token_ids=candidate_ids,
        sampling_params=self.verify_sampling_params,
        use_tqdm=False
    )
    full_logprobs = output[0].prompt_logprobs

    # Extract target predictions for draft positions
    logprobs_slice = full_logprobs[len(input_ids):len(input_ids) + len(draft_tokens)]
    target_predictions = [next(iter(lp.keys())) if lp else None for lp in logprobs_slice]

    accepted_tokens = []
    n_accepted = 0

    for draft_token, target_pred in zip(draft_tokens, target_predictions):
        if target_pred == draft_token:
            accepted_tokens.append(draft_token)
            n_accepted += 1
        else:
            accepted_tokens.append(target_pred)  # Use target's prediction
            break  # Reject remaining drafts

    # Bonus token: if all drafts accepted, add one more from target
    if n_accepted == len(draft_tokens) and not finished:
        bonus_token = output[0].outputs[0].token_ids[0]
        accepted_tokens.append(bonus_token)

    return accepted_tokens, n_accepted, finished
```

The implementation includes adaptive speculation length, which dynamically adjusts the number of draft tokens based on the recent acceptance rate. When acceptance is high, more tokens are speculated; when low, fewer tokens are proposed to avoid wasted computation:

```python
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
```

For high-throughput scenarios, the `ContinuousBatchScheduler` enables processing multiple sequences simultaneously with dynamic batching. Sequences can be added and removed from the batch as they complete, maximizing GPU utilization:

```python
def step(self) -> Tuple[int, List[SequenceState]]:
    """Execute one continuous batching iteration."""
    # 1. Merge pending requests into active batch
    self._merge_pending_requests()

    # 2. Batch generate draft tokens for all active sequences
    self._batch_generate_draft_tokens(active_seqs)

    # 3. Batch verify all drafts in parallel
    verify_results = self._batch_verify_tokens(active_seqs)

    # 4. Update sequence states and remove finished sequences
    for seq in active_seqs:
        accepted_tokens, n_accepted, _ = verify_results[seq.request_id]
        seq.update_with_accepted_tokens(accepted_tokens, n_accepted, ...)

    return len(self.active_sequences), self._remove_finished_sequences()
```

The async pipeline mode further optimizes single-sequence latency by overlapping draft generation with verification. While the target model verifies current drafts, the draft model optimistically generates the next batch of tokens assuming all current drafts will be accepted:

```python
# Parallel execution: verify current draft + optimistically generate next draft
verify_future = self._executor.submit(self.verify_tokens, input_ids, draft_tokens)

optimistic_input_ids = input_ids + draft_tokens
optimistic_draft_future = self._executor.submit(
    self.generate_draft_tokens, optimistic_input_ids, next_k
)

# Wait for verify result
valid_tokens, n_accepted, finished = verify_future.result()

# Use optimistic draft if all accepted, otherwise regenerate
if n_accepted == len(draft_tokens):
    draft_tokens = optimistic_draft_future.result()
else:
    optimistic_draft_future.cancel()
    draft_tokens = self.generate_draft_tokens(input_ids + valid_tokens, k)
```

To run speculative decoding inference:

```bash
# Single mode with async pipeline
python inference.py --mode single --use_async_pipeline \
    --draft_model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --target_model Qwen/Qwen2.5-Math-7B-Instruct

# Batch mode for high throughput
python inference.py --mode batch --batch_size 32 \
    --num_speculative_tokens 16 --adaptive_k
```

The speculative decoding implementation achieves significant speedups over standard autoregressive decoding, with the exact improvement depending on the acceptance rate between the draft and target models. For the Qwen-Math model family, typical acceptance rates range from 60-80%, resulting in 1.5-2.5x throughput improvements.


## Experiment Result



## Reference



## License

