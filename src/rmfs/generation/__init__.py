"""Text generation: decoding strategies and generation statistics."""

from rmfs.generation.greedy import (
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
)
from rmfs.generation.stats import generate_stats

__all__ = [
    "generate_stats",
    "generate_text_basic_stream",
    "generate_text_basic_stream_cache",
]
