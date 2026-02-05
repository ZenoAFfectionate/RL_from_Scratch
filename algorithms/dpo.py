import torch
import torch.nn.functional as F

from typing import Dict
from transformers import PreTrainedModel, PreTrainedTokenizer



def format_with_template(
    instruction: str, 
    response: str, 
    tokenizer: PreTrainedTokenizer,
    template: str,
) -> str:
    """
    Format the instruction and response using the provided template,
    and add the EOS token at the end.
    
    Args:
        instruction: The instruction/prompt string
        response: The response string
        tokenizer: Tokenizer to get the EOS token
        template: Template string with {instruction} and {response} placeholders
    
    Returns:
        Formatted string with EOS token appended
    """
    formatted = template.format(instruction=instruction, response=response)
    formatted += tokenizer.eos_token
    return formatted


def compute_sequence_log_prob(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the log-probability of a sequence under a model.
    
    Args:
        model: The language model
        input_ids: shape (batch_size, seq_len) - token ids
        attention_mask: shape (batch_size, seq_len) - attention mask
    
    Returns:
        torch.Tensor: shape (batch_size,) - log-probability of each sequence
    """
    # Get model outputs (logits)
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    
    # Shift logits and labels for next-token prediction
    # logits[:, :-1, :] predicts input_ids[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    # Compute log-probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log-probs for actual tokens
    # (batch_size, seq_len-1)
    token_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask out padding tokens and sum
    token_log_probs = token_log_probs * shift_mask.float()
    sequence_log_probs = token_log_probs.sum(dim=-1)  # (batch_size,)
    
    return sequence_log_probs


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
    """
    Compute the per-instance DPO loss.
    
    The DPO loss is defined as:
        L = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))
    
    Since log p(y|x) = log p(x,y) - log p(x), and log p(x) cancels out when computing
    log π(y_w|x) - log π(y_l|x), we can simply compute unconditional log-probabilities:
        log π_θ(y_w|x) - log π_θ(y_l|x) = log π_θ(x⊕y_w) - log π_θ(x⊕y_l)
    
    Args:
        policy_model: The policy model π_θ being optimized
        reference_model: The reference model π_ref (frozen)
        tokenizer: Tokenizer for encoding text
        instruction: The prompt/instruction string
        chosen_response: The preferred response y_w
        rejected_response: The rejected response y_l
        template: Template string with {instruction} and {response} placeholders
        beta: Temperature parameter for DPO (default: 0.1)
    
    Returns:
        torch.Tensor: Scalar tensor containing the DPO loss on policy_model's device
    """
    # Get devices for both models
    policy_device = next(policy_model.parameters()).device
    ref_device = next(reference_model.parameters()).device
    
    # Format both responses using the provided template
    chosen_text = format_with_template(instruction, chosen_response, tokenizer, template)
    rejected_text = format_with_template(instruction, rejected_response, tokenizer, template)
    
    # Tokenize both sequences
    chosen_encoding = tokenizer(
        chosen_text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=2048
    )
    rejected_encoding = tokenizer(
        rejected_text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move to policy device for policy model computation
    chosen_ids_policy = chosen_encoding["input_ids"].to(policy_device)
    chosen_mask_policy = chosen_encoding["attention_mask"].to(policy_device)
    rejected_ids_policy = rejected_encoding["input_ids"].to(policy_device)
    rejected_mask_policy = rejected_encoding["attention_mask"].to(policy_device)
    
    # Compute log-probabilities under the policy model
    policy_log_prob_chosen = compute_sequence_log_prob(
        policy_model, chosen_ids_policy, chosen_mask_policy
    )
    policy_log_prob_rejected = compute_sequence_log_prob(
        policy_model, rejected_ids_policy, rejected_mask_policy
    )
    
    # Move to reference device for reference model computation
    chosen_ids_ref = chosen_encoding["input_ids"].to(ref_device)
    chosen_mask_ref = chosen_encoding["attention_mask"].to(ref_device)
    rejected_ids_ref = rejected_encoding["input_ids"].to(ref_device)
    rejected_mask_ref = rejected_encoding["attention_mask"].to(ref_device)
    
    # Compute log-probabilities under the reference model (no gradient needed)
    with torch.no_grad():
        ref_log_prob_chosen = compute_sequence_log_prob(
            reference_model, chosen_ids_ref, chosen_mask_ref
        )
        ref_log_prob_rejected = compute_sequence_log_prob(
            reference_model, rejected_ids_ref, rejected_mask_ref
        )
    
    # Move reference log-probs to policy device
    ref_log_prob_chosen = ref_log_prob_chosen.to(policy_device)
    ref_log_prob_rejected = ref_log_prob_rejected.to(policy_device)
    
    # Compute the DPO loss:
    # L = -log σ(β * ((log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))))
    # = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
    log_ratio_chosen = policy_log_prob_chosen - ref_log_prob_chosen
    log_ratio_rejected = policy_log_prob_rejected - ref_log_prob_rejected
    
    # The reward margin
    reward_margin = beta * (log_ratio_chosen - log_ratio_rejected)
    
    # DPO loss is -log sigmoid(reward_margin)
    # Using logsigmoid for numerical stability
    loss = -F.logsigmoid(reward_margin)
    
    return loss.squeeze()


def dpo_batch_loss(
    policy_model: PreTrainedModel,
    reference_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    instructions: list[str],
    chosen_responses: list[str],
    rejected_responses: list[str],
    template: str,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute the DPO loss for a batch of examples.
    
    Args:
        policy_model: The policy model π_θ being optimized
        reference_model: The reference model π_ref (frozen)
        tokenizer: Tokenizer for encoding text
        instructions: List of instruction strings
        chosen_responses: List of preferred responses
        rejected_responses: List of rejected responses
        template: Template string with {instruction} and {response} placeholders
        beta: Temperature parameter for DPO
    
    Returns:
        tuple: (mean_loss, metadata_dict)
    """
    # Get devices
    policy_device = next(policy_model.parameters()).device
    ref_device = next(reference_model.parameters()).device
    
    # Format all texts using the provided template
    chosen_texts = [
        format_with_template(inst, resp, tokenizer, template) 
        for inst, resp in zip(instructions, chosen_responses)
    ]
    rejected_texts = [
        format_with_template(inst, resp, tokenizer, template) 
        for inst, resp in zip(instructions, rejected_responses)
    ]
    
    # Batch tokenize
    chosen_encoding = tokenizer(
        chosen_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    rejected_encoding = tokenizer(
        rejected_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move to policy device
    chosen_ids_policy = chosen_encoding["input_ids"].to(policy_device)
    chosen_mask_policy = chosen_encoding["attention_mask"].to(policy_device)
    rejected_ids_policy = rejected_encoding["input_ids"].to(policy_device)
    rejected_mask_policy = rejected_encoding["attention_mask"].to(policy_device)
    
    # Compute policy log-probs
    policy_log_prob_chosen = compute_sequence_log_prob(
        policy_model, chosen_ids_policy, chosen_mask_policy
    )
    policy_log_prob_rejected = compute_sequence_log_prob(
        policy_model, rejected_ids_policy, rejected_mask_policy
    )
    
    # Move to reference device
    chosen_ids_ref = chosen_encoding["input_ids"].to(ref_device)
    chosen_mask_ref = chosen_encoding["attention_mask"].to(ref_device)
    rejected_ids_ref = rejected_encoding["input_ids"].to(ref_device)
    rejected_mask_ref = rejected_encoding["attention_mask"].to(ref_device)
    
    # Compute reference log-probs (no gradient)
    with torch.no_grad():
        ref_log_prob_chosen = compute_sequence_log_prob(
            reference_model, chosen_ids_ref, chosen_mask_ref
        )
        ref_log_prob_rejected = compute_sequence_log_prob(
            reference_model, rejected_ids_ref, rejected_mask_ref
        )
    
    # Move to policy device
    ref_log_prob_chosen = ref_log_prob_chosen.to(policy_device)
    ref_log_prob_rejected = ref_log_prob_rejected.to(policy_device)
    
    # Compute log ratios
    log_ratio_chosen = policy_log_prob_chosen - ref_log_prob_chosen
    log_ratio_rejected = policy_log_prob_rejected - ref_log_prob_rejected
    
    # Reward margin and loss
    reward_margin = beta * (log_ratio_chosen - log_ratio_rejected)
    losses = -F.logsigmoid(reward_margin)
    
    # Compute accuracy (how often chosen is preferred)
    with torch.no_grad():
        accuracy = (reward_margin > 0).float().mean()
    
    # Metadata for logging
    metadata = {
        "loss": losses.mean().item(),
        "accuracy": accuracy.item(),
        "reward_margin_mean": reward_margin.mean().item(),
        "reward_margin_std": reward_margin.std().item(),
        "chosen_log_ratio_mean": log_ratio_chosen.mean().item(),
        "rejected_log_ratio_mean": log_ratio_rejected.mean().item(),
    }
    
    return losses.mean(), metadata


def evaluate_dpo(
    policy_model,
    reference_model,
    tokenizer,
    val_loader,
    template: str,
    beta: float,
) -> dict:
    """Evaluate DPO model on validation set."""
    policy_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            _, metadata = dpo_batch_loss(
                policy_model=policy_model,
                reference_model=reference_model,
                tokenizer=tokenizer,
                instructions=batch['instructions'],
                chosen_responses=batch['chosen'],
                rejected_responses=batch['rejected'],
                template=template,
                beta=beta,
            )
            total_loss += metadata['loss']
            total_accuracy += metadata['accuracy']
            num_batches += 1

    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'accuracy': total_accuracy / num_batches if num_batches > 0 else 0.0,
    }
