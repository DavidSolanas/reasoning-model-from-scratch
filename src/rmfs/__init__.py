"""Final code for "Build a Reasoning Model (From Scratch)", developed chapter by chapter.

The `notebooks/` folder holds the exploratory work; this package holds the
cleaned-up, importable versions of the same code, organized by concern
(generation, data, training, ...) so it can be reused across chapters.
"""

from rmfs.generation import (
    generate_stats,
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
)

__all__ = [
    "generate_stats",
    "generate_text_basic_stream",
    "generate_text_basic_stream_cache",
]
