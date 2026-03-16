"""
Tests for TRL-based trainers.

These tests verify that:
  1. Datasets can be correctly built and formatted
  2. Reward function wrappers return the expected format
  3. TRL configs can be instantiated with the right parameters
  4. Trainers can be created (import/instantiation test)

Note: Full training tests require GPU and are skipped in CPU-only environments.
Run with: conda activate llm && cd /home/kemove/Courses/STF_LLM/Assignment_5 && python -m pytest tests/test_trl_trainers.py -v
"""

import os
import sys
import json
import pytest

# Add project root to path so that both `utils` and `trl_trainer` are importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from trl_trainer.utils import (
    ALPACA_TEMPLATE, MATH_TEMPLATE, CODE_TEMPLATE, GSMK_TEMPLATE, RLHF_TEMPLATE,
    make_trl_reward_fn,
)


# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_ultrachat_data(tmp_path):
    """Create a small temporary ultrachat-style dataset."""
    data = [
        {"prompt": "What is 2+2?", "response": "2+2 equals 4."},
        {"prompt": "Explain gravity.", "response": "Gravity is a force of attraction."},
        {"prompt": "Write hello world.", "response": "print('hello world')"},
    ]
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    for path in [train_path, valid_path]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    return str(train_path), str(valid_path)


@pytest.fixture
def tmp_rlhf_data(tmp_path):
    """Create a small temporary RLHF preference dataset."""
    data = [
        {"instruction": "What is AI?", "chosen": "AI is artificial intelligence.", "rejected": "I don't know."},
        {"instruction": "Explain ML.", "chosen": "ML is machine learning.", "rejected": "No idea."},
        {"instruction": "What is NLP?", "chosen": "Natural language processing.", "rejected": "Hmm."},
    ]
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    for path in [train_path, valid_path]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    return str(train_path), str(valid_path)


@pytest.fixture
def tmp_math_data(tmp_path):
    """Create a small temporary math dataset."""
    data = [
        {"problem": "What is 2+2?", "solution": "4"},
        {"problem": "What is 3*5?", "solution": "15"},
        {"problem": "What is 10-7?", "solution": "3"},
    ]
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    for path in [train_path, valid_path]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    return str(train_path), str(valid_path)


@pytest.fixture
def tmp_code_data(tmp_path):
    """Create a small temporary code dataset."""
    data = [
        {
            "problem": "Read an integer and print it doubled.",
            "solution": "n = int(input())\nprint(n*2)",
            "test": "assert f(2) == 4",
        },
    ]
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    for path in [train_path, valid_path]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    return str(train_path), str(valid_path)


# ──────────────────────────────────────────────────────────────────────
#  Test: SFT Trainer
# ──────────────────────────────────────────────────────────────────────

class TestSFTTrainer:
    """Tests for the TRL SFT trainer module."""

    def test_import(self):
        """Can we import the SFT trainer module?"""
        from trl_trainer.sft_trainer import (
            build_chat_dataset,
            build_reasoning_dataset,
            load_prompt_template,
            ALPACA_TEMPLATE,
            MATH_TEMPLATE,
            CODE_TEMPLATE,
        )

    def test_load_prompt_template_defaults(self):
        """Default templates are returned for each mode."""
        from trl_trainer.sft_trainer import load_prompt_template
        assert "{instruction}" in load_prompt_template("chat")
        assert "{problem}" in load_prompt_template("reasoning", "math")

    def test_build_chat_dataset(self, tmp_ultrachat_data):
        """Chat dataset has the correct 'text' column."""
        from trl_trainer.sft_trainer import build_chat_dataset
        train_path, _ = tmp_ultrachat_data
        ds = build_chat_dataset(train_path, ALPACA_TEMPLATE)
        assert "text" in ds.column_names
        assert len(ds) == 3
        # Check that template is applied
        assert "### Instruction:" in ds[0]["text"]
        assert "### Response:" in ds[0]["text"]

    def test_build_reasoning_dataset_math(self, tmp_math_data):
        """Reasoning (math) dataset has correct 'text' column."""
        from trl_trainer.sft_trainer import build_reasoning_dataset
        train_path, _ = tmp_math_data
        ds = build_reasoning_dataset(train_path, MATH_TEMPLATE, "math")
        assert "text" in ds.column_names
        assert len(ds) == 3
        # Check that the prompt template is applied
        assert "User:" in ds[0]["text"]
        assert "<think>" in ds[0]["text"]

    def test_build_reasoning_dataset_code(self, tmp_code_data):
        """Reasoning (code) dataset has correct 'text' column."""
        from trl_trainer.sft_trainer import build_reasoning_dataset
        train_path, _ = tmp_code_data
        ds = build_reasoning_dataset(train_path, CODE_TEMPLATE, "code")
        assert "text" in ds.column_names
        assert len(ds) == 1

    def test_sft_config_instantiation(self, tmp_path):
        """SFTConfig can be created with expected parameters."""
        from trl import SFTConfig
        config = SFTConfig(
            output_dir=str(tmp_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            max_length=256,
            bf16=False,
            report_to="none",
        )
        assert config.num_train_epochs == 1
        assert config.per_device_train_batch_size == 2
        assert config.max_length == 256


# ──────────────────────────────────────────────────────────────────────
#  Test: DPO Trainer
# ──────────────────────────────────────────────────────────────────────

class TestDPOTrainer:
    """Tests for the TRL DPO trainer module."""

    def test_import(self):
        """Can we import the DPO trainer module?"""
        from trl_trainer.dpo_trainer import build_dpo_dataset, RLHF_TEMPLATE

    def test_build_dpo_dataset(self, tmp_rlhf_data):
        """DPO dataset has prompt, chosen, rejected columns."""
        from trl_trainer.dpo_trainer import build_dpo_dataset
        train_path, _ = tmp_rlhf_data
        ds = build_dpo_dataset(train_path, RLHF_TEMPLATE)
        assert "prompt" in ds.column_names
        assert "chosen" in ds.column_names
        assert "rejected" in ds.column_names
        assert len(ds) == 3

    def test_dpo_dataset_content(self, tmp_rlhf_data):
        """DPO dataset has correct content formatting."""
        from trl_trainer.dpo_trainer import build_dpo_dataset
        train_path, _ = tmp_rlhf_data
        ds = build_dpo_dataset(train_path, RLHF_TEMPLATE)
        # Prompt should contain instruction but not response
        assert "### Instruction:" in ds[0]["prompt"]
        # Chosen/rejected should be full formatted text
        assert "### Response:" in ds[0]["chosen"]
        assert "### Response:" in ds[0]["rejected"]
        # Chosen and rejected should differ
        assert ds[0]["chosen"] != ds[0]["rejected"]

    def test_dpo_config_instantiation(self, tmp_path):
        """DPOConfig can be created with expected parameters."""
        from trl import DPOConfig
        config = DPOConfig(
            output_dir=str(tmp_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            beta=0.2,
            max_length=256,
            bf16=False,
            report_to="none",
        )
        assert config.beta == 0.2
        assert config.max_length == 256


# ──────────────────────────────────────────────────────────────────────
#  Test: GRPO Trainer
# ──────────────────────────────────────────────────────────────────────

class TestGRPOTrainer:
    """Tests for the TRL GRPO trainer module."""

    def test_import(self):
        """Can we import the GRPO trainer module?"""
        from trl_trainer.grpo_trainer import (
            build_grpo_dataset,
            make_reward_fn,
            get_reward_fn,
            MATH_TEMPLATE,
            CODE_TEMPLATE,
        )

    def test_build_grpo_dataset_math(self, tmp_math_data):
        """GRPO math dataset has prompt and solution columns."""
        from trl_trainer.grpo_trainer import build_grpo_dataset
        train_path, _ = tmp_math_data
        ds = build_grpo_dataset(train_path, MATH_TEMPLATE, "math")
        assert "prompt" in ds.column_names
        assert "solution" in ds.column_names
        assert len(ds) == 3
        assert "User:" in ds[0]["prompt"]

    def test_reward_fn_wrapper_math(self):
        """Math reward function wrapper returns list of floats."""
        from utils.rewards import dsr1_reward_fn

        trl_fn = make_trl_reward_fn(dsr1_reward_fn)

        # Test with a correct answer
        prompts = ["What is 2+2?"]
        completions = ["<think>Let me calculate.</think><answer>4</answer>"]
        solutions = ["4"]

        rewards = trl_fn(prompts=prompts, completions=completions, solution=solutions)
        assert isinstance(rewards, list)
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)
        assert rewards[0] == 1.0

    def test_reward_fn_wrapper_wrong_answer(self):
        """Math reward function returns 0 for wrong answers."""
        from utils.rewards import dsr1_reward_fn

        trl_fn = make_trl_reward_fn(dsr1_reward_fn)

        prompts = ["What is 2+2?"]
        completions = ["<think>Hmm.</think><answer>5</answer>"]
        solutions = ["4"]

        rewards = trl_fn(prompts=prompts, completions=completions, solution=solutions)
        assert rewards[0] == 0.0

    def test_reward_fn_wrapper_bad_format(self):
        """Math reward function returns 0 for badly formatted answers."""
        from utils.rewards import dsr1_reward_fn

        trl_fn = make_trl_reward_fn(dsr1_reward_fn)

        prompts = ["What is 2+2?"]
        completions = ["The answer is 4"]  # Missing <think>/<answer> tags
        solutions = ["4"]

        rewards = trl_fn(prompts=prompts, completions=completions, solution=solutions)
        assert rewards[0] == 0.0

    def test_reward_fn_wrapper_no_solution(self):
        """Reward function returns 0 when no solution is provided."""
        from utils.rewards import dsr1_reward_fn

        trl_fn = make_trl_reward_fn(dsr1_reward_fn)

        rewards = trl_fn(prompts=["x"], completions=["y"], solution=None)
        assert rewards == [0.0]

    def test_grpo_config_instantiation(self, tmp_path):
        """GRPOConfig can be created with expected parameters."""
        from trl import GRPOConfig
        # generation_batch_size must be divisible by num_generations
        config = GRPOConfig(
            output_dir=str(tmp_path),
            max_steps=10,
            per_device_train_batch_size=4,
            num_generations=4,
            beta=0.01,
            epsilon=0.2,
            max_completion_length=256,
            bf16=False,
            report_to="none",
        )
        assert config.num_generations == 4
        assert config.beta == 0.01
        assert config.epsilon == 0.2


# ──────────────────────────────────────────────────────────────────────
#  Test: PPO Trainer
# ──────────────────────────────────────────────────────────────────────

class TestPPOTrainer:
    """Tests for the TRL PPO trainer module."""

    def test_import(self):
        """Can we import the PPO trainer module?"""
        from trl_trainer.ppo_trainer import (
            VerifiableRewardModel,
            DummyBackbone,
            build_ppo_dataset,
            get_reward_fn,
            MATH_TEMPLATE,
        )

    def test_build_ppo_dataset(self, tmp_math_data):
        """PPO dataset has query column and ground truths dict."""
        from trl_trainer.ppo_trainer import build_ppo_dataset
        train_path, _ = tmp_math_data
        ds, gt = build_ppo_dataset(train_path, MATH_TEMPLATE, "math")
        assert "query" in ds.column_names
        assert len(ds) == 3
        assert isinstance(gt, dict)
        assert len(gt) == 3

    def test_verifiable_reward_model_instantiation(self):
        """VerifiableRewardModel can be instantiated."""
        from trl_trainer.ppo_trainer import VerifiableRewardModel
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        from utils.rewards import dsr1_reward_fn
        ground_truths = {"What is 2+2?": "4"}

        model = VerifiableRewardModel(tokenizer, dsr1_reward_fn, ground_truths)
        assert hasattr(model, "backbone")
        assert hasattr(model, "score")
        assert model.base_model_prefix == "backbone"

    def test_ppo_config_instantiation(self, tmp_path):
        """PPOConfig can be created with expected parameters."""
        from trl import PPOConfig
        config = PPOConfig(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            num_ppo_epochs=2,
            kl_coef=0.1,
            cliprange=0.2,
            vf_coef=0.5,
            response_length=256,
            bf16=False,
            report_to="none",
        )
        assert config.num_ppo_epochs == 2
        assert config.kl_coef == 0.1
        assert config.cliprange == 0.2


# ──────────────────────────────────────────────────────────────────────
#  Cross-cutting tests
# ──────────────────────────────────────────────────────────────────────

class TestRewardFunctions:
    """Test that reward functions from utils work correctly."""

    def test_dsr1_reward_correct(self):
        """DSR1 reward function gives 1.0 for correct formatted answer."""
        from utils.rewards import dsr1_reward_fn
        results = dsr1_reward_fn(
            ["<think>calculating</think><answer>42</answer>"],
            ["42"],
        )
        assert results[0]["reward"] == 1.0
        assert results[0]["format_reward"] == 1.0
        assert results[0]["answer_reward"] == 1.0

    def test_dsr1_reward_wrong(self):
        """DSR1 reward function gives 0.0 for wrong answer."""
        from utils.rewards import dsr1_reward_fn
        results = dsr1_reward_fn(
            ["<think>hmm</think><answer>99</answer>"],
            ["42"],
        )
        assert results[0]["reward"] == 0.0
        assert results[0]["format_reward"] == 1.0
        assert results[0]["answer_reward"] == 0.0

    def test_dsr1_reward_bad_format(self):
        """DSR1 reward function gives 0.0 for bad format."""
        from utils.rewards import dsr1_reward_fn
        results = dsr1_reward_fn(
            ["The answer is 42"],
            ["42"],
        )
        assert results[0]["reward"] == 0.0
        assert results[0]["format_reward"] == 0.0

    def test_gsmk_reward_correct(self):
        """GSM8K reward function gives 1.0 for correct answer."""
        from utils.rewards import gsmk_reward_fn
        results = gsmk_reward_fn(
            ["Let me calculate... #### 42"],
            ["#### 42"],
        )
        assert results[0]["reward"] == 1.0

    def test_gsmk_reward_wrong(self):
        """GSM8K reward function gives 0.0 for wrong answer."""
        from utils.rewards import gsmk_reward_fn
        results = gsmk_reward_fn(
            ["Let me think... #### 99"],
            ["#### 42"],
        )
        assert results[0]["reward"] == 0.0


class TestPromptTemplateConsistency:
    """Verify that TRL trainer templates match the original .prompt files."""

    def test_math_template_has_key_elements(self):
        assert "{problem}" in MATH_TEMPLATE
        assert "<think>" in MATH_TEMPLATE
        assert "<answer>" in MATH_TEMPLATE
        assert "User:" in MATH_TEMPLATE
        assert "Assistant:" in MATH_TEMPLATE

    def test_code_template_has_key_elements(self):
        assert "{problem}" in CODE_TEMPLATE
        assert "<think>" in CODE_TEMPLATE
        assert "python" in CODE_TEMPLATE.lower()
        assert "User:" in CODE_TEMPLATE
        assert "Assistant:" in CODE_TEMPLATE

    def test_alpaca_template_has_key_elements(self):
        assert "{instruction}" in ALPACA_TEMPLATE
        assert "{response}" in ALPACA_TEMPLATE
        assert "### Instruction:" in ALPACA_TEMPLATE
        assert "### Response:" in ALPACA_TEMPLATE

    def test_rlhf_template_has_key_elements(self):
        assert "{instruction}" in RLHF_TEMPLATE
        assert "{response}" in RLHF_TEMPLATE
        assert "### Instruction:" in RLHF_TEMPLATE
        assert "### Response:" in RLHF_TEMPLATE
