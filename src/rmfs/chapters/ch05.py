"""Chapter 5: scoring answers and improving them with self-refinement.

Reproduces the chapter 5 notebook: loads the pretrained Qwen3 0.6B model,
scores a sampled chain-of-thought answer first with a heuristic (boxed
answers, brevity) and then with model-likelihood scores, from per-token
probabilities to the average log-probability of an answer, and finally
improves an answer by letting the model critique and revise it in a
self-refinement loop that only keeps revisions that do not worsen the
score.
"""

from functools import partial

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer

from rmfs.generation import (
    generate_text_stream_concat,
    generate_text_top_p_stream_cache,
)
from rmfs.paths import QWEN3_DIR
from rmfs.prompts import render_prompt
from rmfs.scoring import (
    avg_logprob_answer,
    calc_next_token_logprobas,
    calc_next_token_probas,
    heuristic_score,
)
from rmfs.self_refinement import (
    make_critique_prompt,
    make_refine_prompt,
    self_refinement_loop,
)

WHICH_MODEL = "base"  # "base" for the pretrained, "instruct" for the tuned variant
MAX_NEW_TOKENS = 2048
MAX_CRITIQUE_TOKENS = 256

RAW_PROMPT = (
    "Half the value of $3x-9$ is $x+37$. "
    "What is the value of $x$?"
)

# Suffix that asks the model to reason step by step before answering
COT_SUFFIX = "\n\nExplain step by step."

# Factual statements used to illustrate likelihood scoring
FACTUAL_PROMPTS = [
    "The capital of Germany is Berlin",
    "The capital of Germany is Bridge",
]

# Question and answer used to illustrate answer-only scoring
EXAMPLE_PROMPT = "What is the capital of Germany?"
EXAMPLE_ANSWER = " The capital of Germany is Berlin."
WRONG_ANSWER = " The capital of Germany is Bridge."


def demo_cot_answer(model, tokenizer, device, prompt):
    """Sample a chain-of-thought answer, later used to try out scoring."""
    torch.manual_seed(0)  # For reproducibility of the sampled tokens
    return generate_text_stream_concat(
        model, tokenizer, prompt, device,
        max_new_tokens=MAX_NEW_TOKENS, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.9,
        top_p=0.9,
    )


def demo_next_token_probas(model, tokenizer, device):
    """Compare how likely the model finds two factual statements."""
    # Show small probabilities in scientific notation
    torch.set_printoptions(precision=4, sci_mode=True)

    for text in FACTUAL_PROMPTS:
        probas, joint = calc_next_token_probas(
            model, tokenizer, text, device
        )
        print(f"\nPrompt: {text!r}")
        print("Next-token probabilities:", probas)
        print("Joint probability:", joint)


def demo_next_token_logprobas(model, tokenizer, device):
    """Same comparison in log space, where products become sums."""
    for text in FACTUAL_PROMPTS:
        logprobas, joint = calc_next_token_logprobas(
            model, tokenizer, text, device
        )
        print(f"\nPrompt: {text!r}")
        print("Next-token log-probabilities:", logprobas)
        print("Joint log-probability:", joint)


def demo_avg_logprob(model, tokenizer, device):
    """Score only the answer tokens by their average log-probability."""
    # The full sequence's log-probabilities include the prompt; the
    # trailing ones belong to the answer tokens
    logprobas, _ = calc_next_token_logprobas(
        model, tokenizer, EXAMPLE_PROMPT + EXAMPLE_ANSWER, device
    )
    n_answer_tokens = len(tokenizer.encode(EXAMPLE_ANSWER))
    print("Answer tokens:", n_answer_tokens)

    answer_logprobas = logprobas[-n_answer_tokens:]
    print("Answer log-probabilities:", answer_logprobas)
    print("Mean:", torch.mean(answer_logprobas))

    for answer in (EXAMPLE_ANSWER, WRONG_ANSWER):
        score = avg_logprob_answer(
            model, tokenizer, EXAMPLE_PROMPT, answer, device
        )
        print(f"\nAnswer: {answer!r}")
        print(f"Average log-probability: {score:.4f}")


def demo_initial_answer(model, tokenizer, device, prompt):
    """Generate the draft answer that the refinement demos start from."""
    torch.manual_seed(123)  # For reproducibility of the sampled tokens
    return generate_text_stream_concat(
        model, tokenizer, prompt, device,
        max_new_tokens=MAX_NEW_TOKENS, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.7,
        top_p=0.9,
    )


def demo_critique_and_revision(model, tokenizer, device, raw_prompt, draft):
    """Run one manual critique-and-revise pass."""
    torch.manual_seed(123)  # For reproducibility of the sampled tokens
    critique_prompt = make_critique_prompt(raw_prompt, draft)
    critique = generate_text_stream_concat(
        model, tokenizer, critique_prompt, device,
        max_new_tokens=MAX_NEW_TOKENS, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.7,
        top_p=0.9,
    )

    torch.manual_seed(123)  # For reproducibility of the sampled tokens
    refine_prompt = make_refine_prompt(raw_prompt, draft, critique)
    generate_text_stream_concat(
        model, tokenizer, refine_prompt, device,
        max_new_tokens=MAX_NEW_TOKENS, verbose=True,
        generate_func=generate_text_top_p_stream_cache,
        temperature=0.7,
        top_p=0.9,
    )


def demo_self_refinement(model, tokenizer, device, raw_prompt):
    """Run the full loop with average-logprob scoring."""
    torch.manual_seed(1)  # For reproducibility of the sampled tokens

    # Score answers by the average log-probability the model assigns them
    score_fn = partial(
        avg_logprob_answer,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    return self_refinement_loop(
        model=model,
        tokenizer=tokenizer,
        raw_prompt=raw_prompt,
        device=device,
        iterations=2,
        max_response_tokens=MAX_NEW_TOKENS,
        max_critique_tokens=MAX_CRITIQUE_TOKENS,
        score_fn=score_fn,
        verbose=True,
        temperature=0.7,
        top_p=0.9,
    )


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
    prompt_cot = prompt + COT_SUFFIX

    print("\n\n=== Chain-of-thought answer (temperature=0.9, top_p=0.9) ===")
    response = demo_cot_answer(model, tokenizer, device, prompt_cot)

    print("\n\n=== Heuristic score of the answer ===")
    print("Score:", round(heuristic_score(response), 3))

    print("\n\n=== Next-token probabilities of factual statements ===")
    demo_next_token_probas(model, tokenizer, device)

    print("\n\n=== Next-token log-probabilities of factual statements ===")
    demo_next_token_logprobas(model, tokenizer, device)

    print("\n\n=== Average answer log-probability ===")
    demo_avg_logprob(model, tokenizer, device)

    print("\n\n=== Draft answer (temperature=0.7, top_p=0.9) ===")
    draft = demo_initial_answer(model, tokenizer, device, prompt)

    print("\n\n=== Critique and single revision ===")
    demo_critique_and_revision(model, tokenizer, device, RAW_PROMPT, draft)

    print("\n\n=== Self-refinement loop (avg logprob scoring) ===")
    results = demo_self_refinement(model, tokenizer, device, RAW_PROMPT)

    print("\n\n=== Self-refinement result ===")
    print("Final answer:", results["final_extracted"])
    print("Iterations:", len(results["steps"]))


if __name__ == "__main__":
    main()
