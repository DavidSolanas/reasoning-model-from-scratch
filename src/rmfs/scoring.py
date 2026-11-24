"""Answer scoring: heuristic bonuses and model-likelihood scores."""

import math

import torch

from reasoning_from_scratch.qwen3 import Qwen3Model, Qwen3Tokenizer

from rmfs.evaluation.answers import extract_final_candidate


def heuristic_score(
    answer: str,
    prompt: str | None = None,
    brevity_bonus: float = 500.0,
    boxed_bonus: float = 2.0,
    extract_bonus: float = 1.0,
    fulltext_bonus: float = 0.0,
) -> float:
    """Score an answer without a reference solution.

    Rewards answers that end with a boxed result, with weaker rewards if
    only a number or just the full text can be extracted, and adds a
    brevity term that decays with the answer length. `prompt` is accepted
    (and ignored) so the function can be used wherever a
    `score_fn(answer=..., prompt=...)` is expected.
    """
    score = 0.0

    # Strongest reward: the answer ends with a boxed value
    cand = extract_final_candidate(answer, fallback=None)
    if cand:
        score += boxed_bonus

    else:
        # Weaker reward: at least a number could be extracted
        cand = extract_final_candidate(answer, fallback="number_only")
        if cand:
            score += extract_bonus
        else:
            # Weakest reward: only the full text could be extracted
            cand = extract_final_candidate(
                answer, fallback="number_then_full"
            )
            if cand:
                score += fulltext_bonus

    # Brevity reward that decays with the answer length
    score += 1.5 * math.exp(-len(answer) / brevity_bonus)
    return score


@torch.inference_mode()
def calc_next_token_probas(
    model: Qwen3Model,
    tokenizer: Qwen3Tokenizer,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Probability the model assigned to each next token in `prompt`.

    Returns the per-position probabilities of the tokens that actually
    follow, plus their product (the joint probability of the sequence
    under the model).
    """
    # Token ids of the whole prompt; shape (seq_len,)
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    # Add and drop the batch dimension: (1, seq_len) -> (seq_len, vocab)
    logits = model(token_ids.unsqueeze(0)).squeeze(0)

    # Softmax over the vocabulary: row i is the next-token distribution
    # after the model has seen tokens 0..i
    all_probas = torch.softmax(logits, dim=-1)

    # Row indices [0, ..., seq_len-2]; the last position's next token
    # lies outside the prompt. device= matters: indexing an MPS/CUDA
    # tensor with CPU index tensors raises an error
    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)

    # Column indices: the token that actually follows each row
    next_ids = token_ids[1:]

    # Pairwise indexing: result[k] = all_probas[t_idx[k], next_ids[k]],
    # e.g. for tokens [A, B, C, D] -> [P(B|A), P(C|AB), P(D|ABC)]
    next_token_probas = all_probas[t_idx, next_ids]

    # Joint probability p(x) = p(x2|x1) * p(x3|x1,x2) * ...
    joint_probability = torch.prod(next_token_probas)

    return next_token_probas, joint_probability


@torch.inference_mode()
def calc_next_token_logprobas(
    model: Qwen3Model,
    tokenizer: Qwen3Tokenizer,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-space version of `calc_next_token_probas`.

    Returns the per-position next-token log-probabilities and their sum
    (the joint log-probability of the sequence under the model).
    """
    token_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    logits = model(token_ids.unsqueeze(0)).squeeze(0)
    # Log-softmax turns the product of probabilities into a stable sum
    all_logprobas = torch.log_softmax(logits, dim=-1)

    t_idx = torch.arange(0, token_ids.shape[0] - 1, device=device)
    next_ids = token_ids[1:]
    next_token_logprobas = all_logprobas[t_idx, next_ids]

    sum_next_token_logprobas = torch.sum(next_token_logprobas)

    return next_token_logprobas, sum_next_token_logprobas


@torch.inference_mode()
def avg_logprob_answer(
    model: Qwen3Model,
    tokenizer: Qwen3Tokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
) -> float:
    """Average log-probability the model assigns to `answer` given `prompt`.

    Runs prompt and answer through the model in a single pass and averages
    the log-probabilities of the answer tokens only. Higher (closer to
    zero) means the model finds the answer more likely. Returns a plain
    float so it can be used directly as a `score_fn`.
    """
    # Encode prompt and answer separately to locate the answer tokens later
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)
    if not answer_ids:
        raise ValueError("answer must contain at least one token")

    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)

    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    logprobs = torch.log_softmax(logits, dim=-1)

    # Positions whose next token is an answer token; the position right
    # before the first answer token already predicts it
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    t_idx = torch.arange(start, end, device=device)
    next_tokens = full_ids[start + 1 : end + 1]
    next_token_logps = logprobs[t_idx, next_tokens]

    # Average over the answer token scores
    return torch.mean(next_token_logps).item()
