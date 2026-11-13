"""Chapter 2: text generation with the pretrained Qwen3 0.6B base model.

Reproduces the chapter 2 notebook: downloads and loads the model,
demonstrates a single next-token prediction, then compares streaming
generation without KV cache, with KV cache, and with a compiled model.
"""

import time

import torch

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.qwen3 import (
    QWEN_CONFIG_06_B,
    Qwen3Model,
    Qwen3Tokenizer,
    download_qwen3_small,
)

from rmfs.generation import (
    generate_stats,
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
)
from rmfs.paths import QWEN3_DIR

MODEL_PATH = QWEN3_DIR / "qwen3-0.6B-base.pth"
TOKENIZER_PATH = QWEN3_DIR / "tokenizer-base.json"


def load_model(model_path, device) -> Qwen3Model:
    model = Qwen3Model(QWEN_CONFIG_06_B)  # Instantiate a Qwen3 model with random weights as placeholders
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


def demo_next_token_prediction(model: Qwen3Model, tokenizer: Qwen3Tokenizer,
                               device: torch.device):
    prompt = "Explain large language models."
    input_token_ids_list = tokenizer.encode(prompt)
    print(f"Number of input tokens: {len(input_token_ids_list)}")

    input_tensor_fmt = torch.tensor(input_token_ids_list).unsqueeze(0)
    input_tensor_fmt = input_tensor_fmt.to(device)

    with torch.inference_mode():
        output_tensor_fmt = model(input_tensor_fmt)

    output_tensor_fmt = output_tensor_fmt.squeeze(0)
    print(f"Formatted Output tensor shape: {output_tensor_fmt.shape}")

    last_token = output_tensor_fmt[-1]
    last_token = torch.argmax(last_token, dim=-1, keepdim=True)
    next_token = tokenizer.decode(list(last_token))
    print(f"Predicted next token: {next_token!r}")


def run_generation(generate_fn, model, tokenizer, input_token_ids,
                   max_new_tokens):
    start_time = time.time()
    generated_ids = []

    for token in generate_fn(
        model=model,
        token_ids=input_token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id)

    end_time = time.time()
    output_token_ids = torch.cat(generated_ids, dim=0)
    generate_stats(output_token_ids, tokenizer, start_time, end_time)


def main():
    download_qwen3_small(kind="base", tokenizer_only=False, out_dir=QWEN3_DIR)

    tokenizer = Qwen3Tokenizer(tokenizer_file_path=TOKENIZER_PATH)
    device = get_device()
    model = load_model(MODEL_PATH, device)

    demo_next_token_prediction(model, tokenizer, device)

    prompt = "Explain large language models in a single sentence."

    input_token_ids_tensor = torch.tensor(
        tokenizer.encode(prompt),
        device=device
    ).unsqueeze(0)

    max_new_tokens = 100

    print("\n\n=== Basic generation (no KV cache) ===")
    run_generation(
        generate_text_basic_stream,
        model,
        tokenizer,
        input_token_ids_tensor,
        max_new_tokens,
    )

    print("\n\n=== Generation with KV cache ===")
    run_generation(
        generate_text_basic_stream_cache,
        model,
        tokenizer,
        input_token_ids_tensor,
        max_new_tokens,
    )

    # Model compilation - Inference speed-up technique
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 8):
        # This avoids retriggering model recompilations
        # in PyTorch 2.8 and newer
        # if the model contains code like self.pos = self.pos + 1
        torch._dynamo.config.allow_unspec_int_on_nn_module = True

    model_compiled = torch.compile(model)

    print("\n\n=== Generation with KV cache (compiled model) ===")
    run_generation(
        generate_text_basic_stream_cache,
        model_compiled,
        tokenizer,
        input_token_ids_tensor,
        max_new_tokens,
    )


if __name__ == "__main__":
    main()
