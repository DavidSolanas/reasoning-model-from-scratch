"""Final code for "Build a Reasoning Model (From Scratch)", developed chapter by chapter.

The `notebooks/` folder holds the exploratory work; this package holds the
cleaned-up, importable versions of the same code, organized by concern
(generation, prompts, data, evaluation, training, ...) so it can be reused
across chapters.
"""

from rmfs.data import load_math500_test
from rmfs.evaluation import (
    equality_check,
    evaluate_math500_stream,
    extract_final_candidate,
    get_last_boxed,
    grade_answer,
    normalize_text,
    split_into_parts,
    sympy_parser,
)
from rmfs.generation import (
    count_samples,
    generate_stats,
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
    generate_text_stream_concat,
    generate_text_temp_stream_cache,
    generate_text_top_p_stream_cache,
    scale_logits_by_temperature,
    softmax_with_temperature,
    top_p_filter,
)
from rmfs.prompts import render_prompt
from rmfs.scoring import (
    avg_logprob_answer,
    calc_next_token_logprobas,
    calc_next_token_probas,
    heuristic_score,
)
from rmfs.self_consistency import self_consistency_vote
from rmfs.self_refinement import (
    make_critique_prompt,
    make_refine_prompt,
    self_refinement_loop,
)

__all__ = [
    "avg_logprob_answer",
    "calc_next_token_logprobas",
    "calc_next_token_probas",
    "count_samples",
    "equality_check",
    "evaluate_math500_stream",
    "extract_final_candidate",
    "generate_stats",
    "generate_text_basic_stream",
    "generate_text_basic_stream_cache",
    "generate_text_stream_concat",
    "generate_text_temp_stream_cache",
    "generate_text_top_p_stream_cache",
    "get_last_boxed",
    "grade_answer",
    "heuristic_score",
    "load_math500_test",
    "make_critique_prompt",
    "make_refine_prompt",
    "normalize_text",
    "render_prompt",
    "scale_logits_by_temperature",
    "self_consistency_vote",
    "self_refinement_loop",
    "softmax_with_temperature",
    "split_into_parts",
    "sympy_parser",
    "top_p_filter",
]
