"""Text generation: decoding strategies and generation statistics."""

from rmfs.generation.concat import (
    generate_text_stream_concat,
    generate_text_stream_concat_flex,
)
from rmfs.generation.greedy import (
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
)
from rmfs.generation.sampling import (
    count_samples,
    generate_text_temp_stream_cache,
    generate_text_top_p_stream_cache,
    scale_logits_by_temperature,
    softmax_with_temperature,
    top_p_filter,
)
from rmfs.generation.stats import generate_stats

__all__ = [
    "count_samples",
    "generate_stats",
    "generate_text_basic_stream",
    "generate_text_basic_stream_cache",
    "generate_text_stream_concat",
    "generate_text_stream_concat_flex",
    "generate_text_temp_stream_cache",
    "generate_text_top_p_stream_cache",
    "scale_logits_by_temperature",
    "softmax_with_temperature",
    "top_p_filter",
]
