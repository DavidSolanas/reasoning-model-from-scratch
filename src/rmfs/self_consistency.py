"""Self-consistency: majority answer over sampled generations."""

from collections import Counter

import torch

from reasoning_from_scratch.qwen3 import Qwen3Model, Qwen3Tokenizer

from rmfs.evaluation.answers import extract_final_candidate
from rmfs.generation.concat import generate_text_stream_concat
from rmfs.generation.sampling import generate_text_top_p_stream_cache


def self_consistency_vote(
    model: Qwen3Model,
    tokenizer: Qwen3Tokenizer,
    prompt: str,
    device: torch.device,
    num_samples: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int | None = 2048,
    show_progress: bool = True,
    show_long_answer: bool = False,
    seed: int | None = None,
) -> dict:
    full_answers, short_answers = [], []

    # Sample multiple answers with temperature/top-p sampling
    for i in range(num_samples):
        if seed is not None:
            # Each sample gets its own deterministic seed
            torch.manual_seed(seed + i + 1)

        answer = generate_text_stream_concat(
            model=model, tokenizer=tokenizer, prompt=prompt, device=device,
            max_new_tokens=max_new_tokens, verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature, top_p=top_p,
        )

        # Extract the final (short) answer from each response
        short = extract_final_candidate(
            answer, fallback="number_then_full"
        )
        full_answers.append(answer)
        short_answers.append(short)
        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")

    # Choose the most frequent final answer (the self-consistency vote)
    counts = Counter(short_answers)
    groups: dict[str, list[int]] = {s: [] for s in counts}
    for idx, s in enumerate(short_answers):
        groups[s].append(idx)

    mc = counts.most_common()
    if not mc:
        # No samples at all
        majority_winners, final_answer = [], None
    else:
        top_freq = mc[0][1]
        majority_winners = [s for s, f in mc if f == top_freq]
        # Only report a winner if the vote is not tied
        final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }
