"""Chapter 4: temperature and top-p sampling, plus self-consistency voting.

Reproduces the chapter 4 notebook: loads the pretrained Qwen3 0.6B model
and compares answers generated greedily with and without a chain-of-thought
instruction, then introduces temperature scaling and top-p filtering
(first on toy probability distributions, then during generation), and
finally aggregates sampled answers into a single majority answer via
self-consistency voting.
"""

import torch
from matplotlib import pyplot as plt

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer

from rmfs.generation import (
    generate_text_stream_concat,
    generate_text_temp_stream_cache,
    generate_text_top_p_stream_cache,
    scale_logits_by_temperature,
)
from rmfs.paths import QWEN3_DIR
from rmfs.prompts import render_prompt
from rmfs.self_consistency import self_consistency_vote

WHICH_MODEL = "base"  # "base" for the pretrained, "instruct" for the tuned variant
MAX_NEW_TOKENS = 2048

RAW_PROMPT = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)

# Suffix that asks the model to reason step by step before answering
COT_SUFFIX = "\n\nExplain step by step."

# Toy logits used to illustrate temperature scaling and top-p filtering
TOY_LOGITS = torch.tensor(
    [-0.7, -3.0, 0.1, -1.2, 2.0, -1.0, -0.5, -2.0, 0.3, 1.5]
)

# Top-p cutoff used in the toy walkthrough below
TOY_TOP_P = 0.8


def demo_greedy_answer(model, tokenizer, device, prompt):
    """Generate an answer with greedy decoding, streaming tokens out."""
    generate_text_stream_concat(
        model, tokenizer, prompt, device,
        max_new_tokens=MAX_NEW_TOKENS, verbose=True,
    )


def demo_cot_answer(model, tokenizer, device, prompt):
    """Same question, with an instruction to explain step by step."""
    prompt_cot = prompt + COT_SUFFIX
    generate_text_stream_concat(
        model, tokenizer, prompt_cot, device,
        max_new_tokens=MAX_NEW_TOKENS, verbose=True,
    )


def demo_single_token(model, tokenizer, device):
    """Predict one token of a factual statement (no reasoning needed)."""
    prompt = "The capital of Germany is"
    generate_text_stream_concat(
        model, tokenizer, prompt, device,
        max_new_tokens=1, verbose=True,
    )


def demo_temperature_generation(model, tokenizer, device, prompt):
    """Generate with temperature sampling instead of greedy decoding."""
    torch.manual_seed(123)  # For reproducibility of the sampled tokens
    generate_text_stream_concat(
        model, tokenizer, prompt, device,
        max_new_tokens=MAX_NEW_TOKENS, verbose=True,
        generate_func=generate_text_temp_stream_cache,
        temperature=1.1,
    )


def demo_temperature_toy_distribution():
    """Visualize a toy distribution after temperature scaling."""
    # A temperature of 1.0 leaves the logits (and probabilities) unchanged
    toy_logits_scaled = scale_logits_by_temperature(TOY_LOGITS, 1.0)
    toy_probas = torch.softmax(toy_logits_scaled, dim=-1)

    plt.bar(
        torch.arange(len(toy_probas)), toy_probas,
        alpha=0.5
    )

    plt.ylim([0, 1])
    plt.xlabel("Vocabulary index")
    plt.ylabel("Probability")
    plt.show()


def demo_top_p_toy_distribution():
    """Walk through top-p filtering step by step on a toy distribution."""
    toy_probas = torch.softmax(TOY_LOGITS, dim=-1)

    # Sort probabilities in descending order and accumulate their mass
    sorted_probas, sorted_idx = torch.sort(toy_probas, descending=True)
    cumsum = torch.cumsum(sorted_probas, dim=-1)

    # Simple rule: keep only tokens covered by the cumulative sum
    keep_mask = cumsum <= TOY_TOP_P
    n_kept = torch.sum(keep_mask).item()
    print("Cumulative sum:", cumsum)
    print("Tokens kept:", n_kept)

    # Common variant: also keep the token that crosses the cutoff, i.e.
    # tokens whose cumulative mass *before* them is below top_p
    keep_mask = (cumsum - sorted_probas) < TOY_TOP_P
    n_kept = keep_mask.sum().item()
    print("Tokens kept:", n_kept)

    plt.bar(
        torch.arange(len(sorted_probas)), sorted_probas,
        alpha=0.5, label="Sorted probabilities"
    )
    plt.step(
        torch.arange(len(cumsum)), cumsum, where="mid",
        color="darkorange", label="Cumulative sum"
    )

    plt.axhline(
        TOY_TOP_P, color="red", linestyle="--",
        label=f"top_p = {TOY_TOP_P}"
    )
    plt.axvline(
        n_kept - 0.5, color="gray", linestyle=":",
        label=f"Top-p cutoff at {n_kept} tokens"
    )

    plt.xlabel("Token rank (sorted by probability)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)
    plt.show()

    # Zero out the probabilities beyond the cutoff, map them back to the
    # original order, and renormalize so they sum to one again
    kept_sorted = torch.where(
        keep_mask, sorted_probas,
        torch.zeros_like(sorted_probas)
    )
    filtered = torch.zeros_like(toy_probas).scatter(0, sorted_idx, kept_sorted)
    print("Filtered:", filtered)

    denom = torch.sum(filtered).clamp_min(1e-12)
    renormalized = filtered / denom
    print("Renormalized:", renormalized)


def demo_top_p_generation(model, tokenizer, device, prompt):
    """Generate with top-p filtered temperature sampling."""
    torch.manual_seed(123)  # For reproducibility of the sampled tokens
    generate_text_stream_concat(
        model, tokenizer, prompt, device,
        max_new_tokens=MAX_NEW_TOKENS, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=1,
        top_p=0.85,
    )


def demo_self_consistency(model, tokenizer, device, prompt):
    """Sample several answers and pick the majority answer by vote."""
    results = self_consistency_vote(
        model,
        tokenizer,
        prompt,
        device=device,
        num_samples=5,
        temperature=0.8,
        top_p=0.9,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=123,
        show_progress=True,
    )
    print(results["full_answers"][0])


def main():
    device = get_device()
    # device = torch.device("cpu")  # to force evaluation on CPU instead

    model, tokenizer = load_model_and_tokenizer(
        which_model=WHICH_MODEL,
        device=device,
        use_compile=False,
        local_dir=QWEN3_DIR,
    )

    prompt = render_prompt(RAW_PROMPT)  # Apply the prompt template
    print(prompt)

    print("\n\n=== Greedy decoding ===")
    demo_greedy_answer(model, tokenizer, device, prompt)

    print("\n\n=== Chain-of-thought instruction ===")
    demo_cot_answer(model, tokenizer, device, prompt)

    print("\n\n=== Single-token prediction (factual recall) ===")
    demo_single_token(model, tokenizer, device)

    print("\n\n=== Temperature sampling (temperature=1.1) ===")
    demo_temperature_generation(model, tokenizer, device, prompt)

    print("\n\n=== Temperature scaling on a toy distribution ===")
    demo_temperature_toy_distribution()

    print("\n\n=== Top-p filtering on a toy distribution ===")
    demo_top_p_toy_distribution()

    print("\n\n=== Top-p sampling (temperature=1, top_p=0.85) ===")
    demo_top_p_generation(model, tokenizer, device, prompt)

    print("\n\n=== Self-consistency voting (plain prompt) ===")
    demo_self_consistency(model, tokenizer, device, prompt)

    print("\n\n=== Self-consistency voting (chain-of-thought prompt) ===")
    demo_self_consistency(model, tokenizer, device, prompt + COT_SUFFIX)


if __name__ == "__main__":
    main()
