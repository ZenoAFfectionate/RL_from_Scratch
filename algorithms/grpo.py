import torch
import numpy as np
from typing import Literal


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    """
    # compute raw rewards based on reward function and reshape to group-wise
    rewards = [reward_fn(r, g) for r, g in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([res["reward"] for res in rewards], dtype=torch.float32)
    # compute mean for each group
    rewards_matrix = raw_rewards.view(-1, group_size)
    group_means = rewards_matrix.mean(dim=1, keepdim=True)

    # normalize rewards within each group
    if normalize_by_std:
        group_stds = rewards_matrix.std(dim=1, keepdim=True) + advantage_eps
        normalized_rewards = (rewards_matrix - group_means) / group_stds
    else:
        normalized_rewards = rewards_matrix - group_means
    
    # get advantage by flattening group-wise normalized rewards
    advantages = normalized_rewards.flatten()

    # record metadata for logging
    format_rewards = [res.get("format_reward", 0.0) for res in rewards]
    answer_rewards = [res.get("answer_reward", 0.0) for res in rewards]
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
        "format_reward_mean": np.mean(format_rewards) if format_rewards else 0.0,
        "answer_reward_mean": np.mean(answer_rewards) if answer_rewards else 0.0,
    }
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-token policy-gradient loss, where raw_rewards_or_advantages
    is either the raw reward or  an already-normalized advantage.
    """
    # trigger broadcast to match policy_log_probs shape
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes the per-token GRPO-Clip loss.
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    # compute unclipped and clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
    loss = -torch.min(surr1, surr2)

    # count how many tokens' gradients were clipped
    with torch.no_grad(): clip_fraction = (surr2 < surr1).float().mean()
    meta_info = {
        "loss": loss.detach().mean(),
        "clip_fraction": clip_fraction.detach(),
        "ratio_mean": ratio.detach().mean(),
    }
    return loss, meta_info


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for no_baseline loss"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards, policy_log_probs
        )
        meta_info = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for reinforce_with_baseline loss"
        loss = compute_naive_policy_gradient_loss(
            advantages, policy_log_probs
        )
        meta_info = {} 
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for grpo_clip loss"
        assert old_log_probs is not None, "old_log_probs must be provided for grpo_clip loss"
        assert cliprange is not None, "cliprange must be provided for grpo_clip loss"
        loss, meta_info = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return loss, meta_info


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, 
    considering only those elements where mask == 1.
    """
    masked_tensor = tensor * mask
    sum_masked = masked_tensor.sum(dim=dim)
    count_mask = mask.sum(dim=dim)  # no need clamp
    return sum_masked / count_mask  # normalization


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute GRPO loss for a micro-batch and return mean loss and meta-info.
    """
    # compute per-token policy-gradient loss
    loss, meta_info = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )

    # to consider only response tokens
    masked_loss = loss * response_mask
    # compute mean loss over the micro-batch
    mean_loss = masked_mean(
        masked_loss,
        response_mask,
    ) / gradient_accumulation_steps
    
    # backward pass
    mean_loss.backward()

    return mean_loss.detach(), meta_info
