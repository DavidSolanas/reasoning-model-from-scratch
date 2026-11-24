"""Self-refinement: critique and revise answers in a loop."""

from collections.abc import Callable

import torch

from reasoning_from_scratch.qwen3 import Qwen3Model, Qwen3Tokenizer

from rmfs.evaluation.answers import extract_final_candidate
from rmfs.generation.concat import generate_text_stream_concat
from rmfs.generation.sampling import generate_text_top_p_stream_cache
from rmfs.prompts import render_prompt


def make_critique_prompt(raw_prompt: str, draft: str) -> str:
    """Ask a reviewer model to find errors in a draft answer."""
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\n"
        "Critique:"
    )


def make_refine_prompt(raw_prompt: str, draft: str, critique: str) -> str:
    """Ask the model to revise a draft answer following the critique."""
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{raw_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revised answer:"
    )


def self_refinement_loop(
    model: Qwen3Model,
    tokenizer: Qwen3Tokenizer,
    raw_prompt: str,
    device: torch.device,
    iterations: int = 2,
    max_response_tokens: int | None = 2048,
    max_critique_tokens: int | None = 256,
    score_fn: Callable[..., float] | None = None,
    prompt_renderer: Callable[[str], str] = render_prompt,
    prompt_suffix: str = "",
    verbose: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict:
    """Improve an answer by alternating critique and revision steps.

    Starts from a sampled draft answer, then repeatedly asks the model to
    critique the current answer and rewrite it. A revision is only kept
    when `score_fn` (called as `score_fn(answer=..., prompt=...)`) does
    not get worse. Returns the final answer plus one record per
    iteration for inspection.
    """
    steps = []

    # Initial response (draft)
    prompt = prompt_renderer(raw_prompt) + prompt_suffix
    current_full = generate_text_stream_concat(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_response_tokens,
        verbose=False,
        generate_func=generate_text_top_p_stream_cache,
        temperature=temperature,
        top_p=top_p,
    )

    current_extracted = extract_final_candidate(
        current_full, fallback="number_then_full"
    )
    if score_fn is not None:
        current_score = score_fn(answer=current_full, prompt=prompt)
    else:
        current_score = 0.0

    for it in range(iterations):
        draft_before_full = current_full
        draft_before_extracted = current_extracted
        score_before = current_score

        # Critique the current answer
        critique_prompt = make_critique_prompt(
            raw_prompt, draft_before_full
        )
        critique_full = generate_text_stream_concat(
            model=model,
            tokenizer=tokenizer,
            prompt=critique_prompt,
            device=device,
            max_new_tokens=max_critique_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )

        # Refine the answer using the critique
        refine_prompt = make_refine_prompt(
            raw_prompt, draft_before_full, critique_full
        )
        revised_full = generate_text_stream_concat(
            model=model,
            tokenizer=tokenizer,
            prompt=refine_prompt,
            device=device,
            max_new_tokens=max_response_tokens,
            verbose=False,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )

        revised_extracted = extract_final_candidate(
            revised_full, fallback="number_then_full"
        )
        if score_fn is not None:
            # Score revised answers against the original prompt
            revised_score = score_fn(answer=revised_full, prompt=prompt)
        else:
            revised_score = 0.0

        step = {
            "iteration": it + 1,
            "draft_full": draft_before_full,
            "draft_extracted": draft_before_extracted,
            "critique": critique_full,
            "revised_full": revised_full,
            "revised_extracted": revised_extracted,
            "score_before": score_before,
            "score_after": revised_score,
        }
        steps.append(step)

        if verbose:
            print(
                f"[Refinement {it+1}/{iterations}]"
                f"\nCurrent: {draft_before_extracted}"
                f"\nRevised: {revised_extracted}"
                f"\nScore before: {score_before:.3f}"
                f"\nScore after: {revised_score:.3f}"
                f"\n{'=' * 25}\n"
            )

        # Accept the revision if it did not make the score worse
        if revised_score >= current_score:
            current_full = revised_full
            current_extracted = revised_extracted
            current_score = revised_score

    return {
        "final_full": current_full,
        "final_extracted": current_extracted,
        "steps": steps,
    }
