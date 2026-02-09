import torch
from typing import List, Dict


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer,
    max_seq_len: int = 2048,
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
    # each seq has length (p_len + o_len + 1) for EOS; after shift,
    # inputs/labels each have length (p_len + o_len), so pad to that.
    target_len = max(p_len + o_len for p_len, o_len in zip(prompt_lens, output_lens))
    # target_len = min(target_len, max_seq_len)
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
    labels = torch.tensor(label_rows, dtype=torch.long)  #

    # create response mask
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, (p_len, o_len, seq) in enumerate(zip(prompt_lens, output_lens, batch_ids)):
        # first position where label is a response token
        start = max(p_len - 1, 0)
        end = min(start + o_len + 1, target_len)  # +1 to include the EOS label
        if end > start:
            response_mask[i, start:end] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
