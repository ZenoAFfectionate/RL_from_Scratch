import torch
import torch.nn.functional as F
from typing import List, Dict
from torch.utils.data import Dataset


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer,
) -> List[Dict[str, torch.Tensor]]:
    """
    Tokenize prompt+output pairs. Returns a list of per-sample
    dicts with variable-length tensors (no global padding).
    Sequence length is naturally bounded by the vLLM generation limit.
    """
    samples = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        output_ids = tokenizer(output, add_special_tokens=False).input_ids

        full_ids = prompt_ids + output_ids + [tokenizer.eos_token_id]
        input_ids = full_ids[:-1]  # 
        labels = full_ids[1:]      # 
        seq_len = len(input_ids)

        # build response mask
        p_len = len(prompt_ids)
        response_mask = torch.zeros(seq_len, dtype=torch.bool)
        start = max(p_len - 1, 0)
        end = min(start + len(output_ids) + 1, seq_len)
        if end > start: response_mask[start:end] = True

        samples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "response_mask": response_mask,
        })

    return samples


def pad_collate(samples, pad_token_id):
    """Pad a list of variable-length sample dicts to the max length in the batch."""
    max_len = max(s["input_ids"].size(0) for s in samples)

    def _pad(t, value):
        return F.pad(t, (0, max_len - t.size(0)), value=value)

    batch = {
        "input_ids": torch.stack([_pad(s["input_ids"], pad_token_id) for s in samples]),
        "labels": torch.stack([_pad(s["labels"], pad_token_id) for s in samples]),
        "response_mask": torch.stack([_pad(s["response_mask"], 0) for s in samples]),
    }
    if "advantage" in samples[0]:
        batch["advantage"] = torch.stack([s["advantage"] for s in samples])
    if "old_log_probs" in samples[0]:
        batch["old_log_probs"] = torch.stack([
            _pad(s["old_log_probs"], 0.0) for s in samples
        ])
    return batch


class ListDataset(Dataset):
    """Simple dataset wrapping a list of dicts for use with DataLoader."""
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]
