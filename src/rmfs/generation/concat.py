"""High-level generation helpers: prompt string in, answer text out."""

import torch

from reasoning_from_scratch.qwen3 import Qwen3Model, Qwen3Tokenizer

from rmfs.generation.greedy import generate_text_basic_stream_cache


def generate_text_stream_concat(
    model: Qwen3Model,
    tokenizer: Qwen3Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int | None,
    verbose: bool = False,
) -> str:
    # Tokenize the prompt into a batch of input ids
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    generated_ids = []
    # Greedy decoding with the KV cache from chapter 2
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        # Optionally print tokens as they are generated
        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )

    # Concatenate all generated tokens into the full response
    return tokenizer.decode(generated_ids)
