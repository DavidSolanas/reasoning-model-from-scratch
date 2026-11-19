"""Chapter 3: answer extraction and grading for a pretrained LLM.

Reproduces the chapter 3 notebook: loads the pretrained Qwen3 0.6B model,
generates worked answers for a math problem, builds the answer-grading
pipeline (boxed-answer extraction, text normalization, SymPy equivalence
checks), and evaluates accuracy on the MATH-500 benchmark, streaming
results to a JSONL log as they are produced.
"""

from pprint import pprint

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer

from rmfs.data import load_math500_test
from rmfs.evaluation import (
    evaluate_math500_stream,
    extract_final_candidate,
    get_last_boxed,
    grade_answer,
    normalize_text,
    sympy_parser,
)
from rmfs.generation import generate_text_stream_concat
from rmfs.paths import QWEN3_DIR
from rmfs.prompts import render_prompt

WHICH_MODEL = "base"  # "base" for the pretrained, "instruct" for the tuned variant
MAX_NEW_TOKENS = 2048

TEXTBOOK_PROMPT = (
    r"If $a+b=3$ and $ab=\tfrac{13}{6}$, "
    r"what is the value of $a^2+b^2$?"
)

# Example model response showing a boxed final answer
MODEL_ANSWER = (
    r"""... some explanation...
**Final Answer:**

\[
\boxed{\dfrac{14}{3}}
\]
"""
)

# Sanity checks for the grading pipeline
GRADING_TESTS = [
    ("check_1", "3/4", r"\frac{3}{4}", True),
    ("check_2", "(3)/(4)", r"3/4", True),
    ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
    ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
    ("check_5", "(1, 2)", r"(1,2)", True),
    ("check_6", "(2, 1)", "(1, 2)", False),
    ("check_7", "(1, 2, 3)", "(1, 2)", False),
    ("check_8", "0.5", "1/2", True),
    ("check_9", "0.3333333333", "1/3", False),
    ("check_10", "1,234/2", "617", True),
    ("check_11", r"\text{2/3}", "2/3", True),
    ("check_12", "50%", "1/2", False),
    ("check_13", r"2\cdot 3/4", "3/2", True),
    ("check_14", r"90^\circ", "90", True),
    ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
    ("check_16", "2²", "2**2", True),
]


def demo_answer_generation(model, tokenizer, device, prompt=TEXTBOOK_PROMPT):
    """Generate a worked answer, streaming tokens as they are produced."""
    generated_text = generate_text_stream_concat(
        model, tokenizer, prompt, device,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose=True,
    )
    return generated_text


def demo_answer_extraction(model_answer=MODEL_ANSWER):
    """Extract, normalize, and parse the candidate answer from a response."""
    extracted_answer = get_last_boxed(model_answer)
    print("Last boxed answer:", extracted_answer)

    extracted = extract_final_candidate(model_answer)
    print("\nExtracted final candidate:", extracted)
    print("Normalized candidate:", normalize_text(extracted))
    print("Normalized LaTeX text:", normalize_text(r"\text{\[\frac{14}{3}\]}"))

    print("\nParsed candidate:", sympy_parser(normalize_text(extracted)))
    print("Parsed fraction:", sympy_parser("28/6"))


def run_demos_table(tests=GRADING_TESTS):
    """Run the grading sanity checks and print a summary table."""
    header = ("Test", "Expect", "Got", "Status")
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth)  # Compare prediction with ground truth
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))

    data = [header] + rows

    # Compute the max width of each column to align the table nicely
    col_widths = [
        max(len(row[i]) for row in data)
        for i in range(len(header))
    ]

    # Print the table row by row
    for row in data:
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(header))
        )
        print(line)

    # Print a summary of the passed checks
    passed = sum(r[3] == "PASS" for r in rows)
    print(f"\nPassed {passed}/{len(rows)}")


def mini_eval_demo(model, tokenizer, device):
    """End-to-end demo on a single example: prompt, generate, extract, grade."""
    ex = {  # Test example with "problem" and "answer" fields
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3"
    }
    prompt = render_prompt(ex["problem"])     # Apply the prompt template
    gen_text = generate_text_stream_concat(  # Generate a response
        model, tokenizer, prompt, device,
        max_new_tokens=64,
    )
    pred_answer = extract_final_candidate(gen_text)  # Extract and normalize answer
    is_correct = grade_answer(                        # Grade answer
        pred_answer, ex["answer"]
    )
    print(f"Device: {device}")
    print(f"Prediction: {pred_answer}")
    print(f"Ground truth: {ex['answer']}")
    print(f"Correct: {is_correct}")


def main():
    device = get_device()
    # device = torch.device("cpu")  # to force evaluation on CPU instead

    model, tokenizer = load_model_and_tokenizer(
        which_model=WHICH_MODEL,
        device=device,
        use_compile=False,
        local_dir=QWEN3_DIR,
    )

    print("\n\n=== Answer generation for a math problem ===")
    demo_answer_generation(model, tokenizer, device)

    print("\n\n=== Extracting and normalizing answers ===")
    demo_answer_extraction()

    print("\n\n=== Grading sanity checks ===")
    run_demos_table()

    print("\n\n=== MATH-500 test dataset ===")
    math_data = load_math500_test()
    print("Number of entries:", len(math_data))
    pprint(math_data[0])

    print("\n\n=== Answer generation with the formatted prompt ===")
    prompt_fmt = render_prompt(TEXTBOOK_PROMPT)
    demo_answer_generation(model, tokenizer, device, prompt=prompt_fmt)

    print("\n\n=== Mini evaluation demo ===")
    mini_eval_demo(model, tokenizer, device)

    print("\n\n=== Full MATH-500 evaluation ===")
    print("Model:", WHICH_MODEL)
    print("Device:", device)
    evaluate_math500_stream(
        model, tokenizer, device,
        math_data=math_data,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose=False,
    )


if __name__ == "__main__":
    main()
