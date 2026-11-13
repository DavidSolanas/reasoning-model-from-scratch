"""Greedy decoding strategies (streaming, with and without KV cache)."""

import torch

from reasoning_from_scratch.qwen3 import KVCache, Qwen3Model


@torch.inference_mode()  # Disable gradient tracking
def generate_text_basic_stream(
    model: Qwen3Model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
):
    # Switch model to eval mode to enable deterministic behavior (best practice)
    model.eval()

    for _ in range(max_new_tokens):
        # Get the scores of last token
        out = model(token_ids)[:, -1]
        # Greedy-decode
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        # Stop if all sequences in the batch have generated EOS
        if (eos_token_id is not None and
                torch.all(next_token == eos_token_id)):
            break

        # Yield each token as soon as it's generated
        yield next_token

        token_ids = torch.cat([token_ids, next_token], dim=1)


@torch.inference_mode()
def generate_text_basic_stream_cache(
    model: Qwen3Model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
):
    model.eval()

    # Initialize KV-cache
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # In the first round, the whole input is provided to the model to store KV-cache
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        # Consequent iterations only feed the next_token to the input
        out = model(next_token, cache=cache)[:, -1]
