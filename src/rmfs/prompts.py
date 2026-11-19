"""Prompt templates for asking the model to solve math problems."""


def render_prompt(prompt: str) -> str:
    # Chat-style instruction template used for the MATH-500 evaluation
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template
