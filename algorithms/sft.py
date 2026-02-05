import torch
import torch.nn.functional as F

from typing import List, Dict

from transformers import PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    """
    # tokenize each prompt and output, then combine them
    batch_ids, prompt_lens, output_lens = [], [], []
    for prompt, output in zip(prompt_strs, output_strs):
        # tokenize the prompt and output into ids
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        output_ids = tokenizer(output, add_special_tokens=False).input_ids
        # 
        batch_ids.append(prompt_ids + output_ids + [tokenizer.eos_token_id])
        prompt_lens.append(len(prompt_ids))
        output_lens.append(len(output_ids))

    # create input_ids and labels tensors, padding to the max length
    target_len = max(p_len + o_len for p_len, o_len in zip(prompt_lens, output_lens)) - 1
    input_rows, label_rows = [], []
    for seq in batch_ids:
        inputs = seq[:-1]  # len: p_len + o_len
        labels = seq[+1:]  # len: p_len + o_len
        # truncate if too long (this is the most tricky part...)
        inputs = inputs[:target_len]
        labels = labels[:target_len]

        cur_len = len(inputs)
        inputs = inputs + [tokenizer.pad_token_id] * (target_len - cur_len)
        labels = labels + [tokenizer.pad_token_id] * (target_len - cur_len)
        
        input_rows.append(inputs)  # 
        label_rows.append(labels)  # 

    input_ids = torch.tensor(input_rows, dtype=torch.long)  # 
    labels    = torch.tensor(label_rows, dtype=torch.long)  # 

    # create response mask
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, (p_len, o_len, seq) in enumerate(zip(prompt_lens, output_lens, batch_ids)):
        start = max(p_len - 1, 0)   # 
        end = start + o_len         # 
        if end > start: response_mask[i, start:end] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
            containing unnormalized logits.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    with torch.no_grad():
        # stabilize logits by subtracting max logit
        logits = logits - logits.amax(dim=-1, keepdim=True)
        # compute the log-softmax of logits for numerical stability
        probs = logits - torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
        # compute and return the entropy
        entropy = -torch.sum(probs * probs.exp(), dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
            response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
            compute_entropy.

    Returns:
        dict[str, torch.Tensor].

        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
            log p_theta(x_t | x_<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
            for each position (present only if return_token_entropy=True).
    """
    # calculate the logits from the model (batch_size, seq_len, vocab_size)
    logits = model(input_ids).logits

    # compute the log-probabilities for all possible next tokens
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}  # store log_probs in result dict

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all dimensions.

    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don't contribute to the sum.
    """
    summed_tensor = torch.sum(tensor * mask, dim=dim)  # sum over specific dim
    return summed_tensor / normalize_constant          # normalize by constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute a forward-and-backward pass on a microbatch.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps The number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].

        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """
    # compute the (negative) log likelihood loss on the response tokens
    microbatch_loss = masked_normalize(
        tensor=-policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=-1
    )

    # scale the loss for gradient accumulation and do backward
    scaled_loss = microbatch_loss.mean() / gradient_accumulation_steps
    scaled_loss.backward()

    metadata = {}  # add more metadata
    return scaled_loss, metadata


def compute_validation_loss(
    model: PreTrainedModel,
    val_loader,
    device: str,
) -> tuple[float, float]:
    """
    Compute average validation loss and perplexity over a validation dataset.
    
    Args:
        model: PreTrainedModel HuggingFace model to evaluate.
        val_loader: DataLoader yielding batches with 'input_ids' and 'labels' keys.
        device: Device string (e.g., 'cuda:0') to move tensors to.
    
    Returns:
        tuple[float, float]: (average_loss, perplexity)
    """
    from tqdm import tqdm
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass using HuggingFace's built-in loss computation
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Accumulate loss weighted by number of tokens
            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity