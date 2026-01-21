import os
import tqdm
import json
import torch
from tqdm import tqdm
from unittest.mock import patch
from typing import Callable, List

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer


def init_policy(model_id='Qwen/Qwen2.5-Math-1.5B', device=None, debug=False):
    ''' Initialize the policy model and tokenizer '''
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.to(device)
    return model, tokenizer


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85, debug=False):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """ Load the weights from a HuggingFace model into a vLLM instance."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],  # questions with prompt template
    answers: List[str],  # ground truth answers
    eval_sampling_params: SamplingParams,
    output_filepath: str = ""
) -> float:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    # generate outputs for each example using the vLLM engine
    print(f"\nGenerating responses for {len(prompts)} problems...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    print("✅ Generation complete.")

    # prepare to compute metrics
    format_correct = 0
    answer_correct = 0
    total_correct = 0 
    results = []

    print("Evaluating responses and computing metrics...")
    for i, output in enumerate(tqdm(outputs, desc="Evaluating")):
        prompt_with_question = output.prompt
        # get the generated text and ground truth
        generated_text = output.outputs[0].text.strip()
        ground_truth_answer = answers[i]
        # use reward function to check for semantic equivalence
        reward_dict = reward_fn(generated_text, ground_truth_answer)
        format_correct += int(reward_dict.get("format_reward", 0))
        answer_correct += int(reward_dict.get("answer_reward", 0))
        total_correct  += int(reward_dict.get("reward", 0))
        # store detailed results for this example
        results.append({
            "prompt": prompt_with_question,       # 
            "response": generated_text,           # 
            "ground_truth": ground_truth_answer,  # 
            "format_reward": reward_dict.get("format_reward", 0),
            "answer_reward": reward_dict.get("answer_reward", 0),
            "is_correct": reward_dict.get("reward", 0)
        })
    
    # calculate overall accuracy
    format_accuracy = (format_correct / len(outputs)) * 100 if outputs else 0
    answer_accuracy = (answer_correct / len(outputs)) * 100 if outputs else 0
    accuracy = (total_correct / len(outputs)) * 100 if outputs else 0
    print(f"\n✅ Evaluation complete.")
    print(f"📈 Format Accuracy: {format_accuracy:.2f}% ({int(format_correct)}/{len(outputs)})")
    print(f"📈 Answer Accuracy: {answer_accuracy:.2f}% ({int(answer_correct)}/{len(outputs)})")
    print(f"📈 Overall Accuracy: {accuracy:.2f}% ({int(total_correct)}/{len(outputs)})")

    # serialize the detailed results to a JSON file for later analysis
    if output_filepath is not None and output_filepath != "":
        print(f"\nSerializing results to {output_filepath}...")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w") as f:
            for result in results:
                json.dump(result, f)
                f.write("\n") 
        print(f"✅ Results saved successfully to {output_filepath}.")

    return accuracy
