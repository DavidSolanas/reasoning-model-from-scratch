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
    generate_stats,
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
    generate_text_stream_concat,
)
from rmfs.prompts import render_prompt

__all__ = [
    "equality_check",
    "evaluate_math500_stream",
    "extract_final_candidate",
    "generate_stats",
    "generate_text_basic_stream",
    "generate_text_basic_stream_cache",
    "generate_text_stream_concat",
    "get_last_boxed",
    "grade_answer",
    "load_math500_test",
    "normalize_text",
    "render_prompt",
    "split_into_parts",
    "sympy_parser",
]
