"""Sampling-based decoding: temperature scaling and top-p filtering."""

import torch

from reasoning_from_scratch.qwen3 import KVCache, Qwen3Model, Qwen3Tokenizer


def scale_logits_by_temperature(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    # Dividing by the temperature flattens the distribution for T > 1
    # (more random) and sharpens it for T < 1 (more deterministic)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return logits / temperature


def softmax_with_temperature(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    # Softmax over temperature-scaled logits
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


def count_samples(
    probas: torch.Tensor,
    num_samples: int = 1000,
    threshold: int = 1,
    tokenizer: Qwen3Tokenizer | None = None,
):
    # Draw samples from the probability distribution
    samples = torch.multinomial(
        probas.cpu(), num_samples=num_samples, replacement=True
    )

    # Count how often each vocabulary index was selected
    counts = torch.bincount(samples.squeeze(0), minlength=1)

    # Report the indices drawn more than `threshold` times
    for i, c in enumerate(counts):
        if c > threshold:
            if tokenizer is None:
                print(f"Vocab index {i}: {c.item()}x")
            else:
                print(f"'{tokenizer.decode([i])}': {c.item()}x")


@torch.inference_mode()
def generate_text_temp_stream_cache(
    model: Qwen3Model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float | None = 0.0,
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Prefill the cache with the whole input; keep the last logits
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):

        # torch.multinomial below samples on CPU only, so remember the
        # original device to move the sampled token back afterwards
        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            # A temperature of zero falls back to greedy decoding
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            # Scale logits, convert to probabilities, and sample from them
            logits = scale_logits_by_temperature(out, temperature)
            probas = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]


def top_p_filter(probas: torch.Tensor, top_p: float | None) -> torch.Tensor:
    # No filtering needed when top-p is disabled or covers everything
    if top_p is None or top_p >= 1.0:
        return probas

    # Sort probabilities in descending order
    sorted_probas, sorted_idx = torch.sort(probas, dim=1, descending=True)

    # Cumulative probability mass along the sorted tokens
    cumprobas = torch.cumsum(sorted_probas, dim=1)

    # Keep tokens whose cumulative mass *before* each token is < top_p,
    # which keeps the token that crosses the cutoff instead of dropping it
    prefix = cumprobas - sorted_probas
    keep = prefix < top_p

    # Always keep at least the most probable token, also when top_p <= 0
    keep[:, 0] = True

    # Zero out probabilities beyond the cutoff
    kept_sorted = torch.where(
        keep, sorted_probas,
        torch.zeros_like(sorted_probas)
    )

    # Map the remaining probabilities back to the original order
    filtered = torch.zeros_like(probas).scatter(1, sorted_idx, kept_sorted)

    # Renormalize so the remaining probabilities sum to one again
    denom = torch.sum(filtered, dim=1, keepdim=True).clamp_min(1e-12)
    return filtered / denom


@torch.inference_mode()
def generate_text_top_p_stream_cache(
    model: Qwen3Model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float | None = 0.0,
    top_p: float | None = None,
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    # Prefill the cache with the whole input; keep the last logits
    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):

        # torch.multinomial below samples on CPU only, so remember the
        # original device to move the sampled token back afterwards
        orig_device = token_ids.device

        if temperature is None or temperature == 0.0:
            # A temperature of zero falls back to greedy decoding
            next_token = torch.argmax(out, dim=-1, keepdim=True)

        else:
            # Scale logits and convert to probabilities
            logits = scale_logits_by_temperature(out, temperature)

            probas = torch.softmax(logits, dim=-1)

            # Apply top-p filtering, then sample from what remains
            probas = top_p_filter(probas, top_p)

            next_token = torch.multinomial(probas.cpu(), num_samples=1)
            next_token = next_token.to(orig_device)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]
