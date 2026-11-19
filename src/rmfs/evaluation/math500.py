"""Streamed MATH-500 evaluation: generate, extract, and grade each problem."""

import json
import time
from pathlib import Path

from rmfs.evaluation.answers import extract_final_candidate
from rmfs.evaluation.grading import grade_answer
from rmfs.generation.concat import generate_text_stream_concat
from rmfs.prompts import render_prompt
from rmfs.utils import eta_progress_message


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path: Path | None = None,
    max_new_tokens: int = 512,
    verbose: bool = False,
):
    # Default the output file name to the device (e.g., math500-cuda-0.jsonl)
    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    # Stream results to a JSONL log as they are produced
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])  # Apply the prompt template
            gen_text = generate_text_stream_concat(  # Generate a response
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )

            extracted = extract_final_candidate(gen_text)  # Extract and normalize
            is_correct = grade_answer(extracted, row["answer"])  # Grade
            num_correct += int(is_correct)

            record = {  # Record saved for later inspection
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)

            # Show per-example results while the eval is running
            if verbose:
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

    # Print summary information
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc
