# reasoning-model-from-scratch

Code following [*Build a Reasoning Model (From Scratch)*](https://github.com/rasbt/reasoning-from-scratch), developed chapter by chapter.

The `notebooks/` folder holds the exploratory work; the `src/rmfs` package holds
cleaned-up, importable versions of the same code, organized by concern
(generation, prompts, data, evaluation, ...) so it can be reused across chapters.
A reference copy of the companion `reasoning_from_scratch` package is kept at the
repository root.

## What it does

- **Chapters** (`src/rmfs/chapters/`) — one runnable script per chapter.
  Chapter 2 loads the pretrained Qwen3 0.6B base model, demonstrates single
  next-token prediction, and compares streaming generation without KV cache,
  with KV cache, and with a compiled model. Chapter 3 builds the
  answer-grading pipeline and evaluates the model on MATH-500. Chapter 4
  compares greedy answers with and without a chain-of-thought instruction,
  adds temperature and top-p sampling, and aggregates sampled answers via
  self-consistency voting. Chapter 5 scores answers first with a heuristic
  (boxed answers, brevity) and then by model likelihood (per-token
  probabilities, average answer log-probability), and improves them with a
  critique-and-revise self-refinement loop that only accepts revisions the
  score does not worsen.
- **Generation** (`src/rmfs/generation/`) — streaming text generation
  (with and without KV cache): greedy decoding, temperature and top-p
  sampling, the prompt-to-text helper that ties them together, plus
  token/throughput statistics.
- **Prompts** (`src/rmfs/prompts.py`) — chat-style instruction template used
  for the MATH-500 evaluation; asks for the final result as `\boxed{ANSWER}`.
- **Data** (`src/rmfs/data/`) — loads the MATH-500 test set, downloading it
  from the upstream repository and caching a local copy if none exists.
- **Evaluation** (`src/rmfs/evaluation/`) — streamed MATH-500 evaluation:
  generates an answer per problem, extracts and normalizes the candidate,
  grades it against the ground truth (including SymPy-based equality checks),
  and logs each result to JSONL with progress/ETA reporting.
- **Self-consistency** (`src/rmfs/self_consistency.py`) — samples several
  answers with temperature/top-p sampling and votes for the most frequent
  extracted answer.
- **Scoring** (`src/rmfs/scoring.py`) — reference-free answer scores: a
  heuristic favoring boxed, short answers, plus model-likelihood scores
  (next-token probabilities and log-probabilities of a sequence, average
  answer log-probability).
- **Self-refinement** (`src/rmfs/self_refinement.py`) — critique-and-revise
  loop that samples a draft, asks the model for a critique, rewrites the
  answer, and keeps a revision only when the score does not get worse.

## Requirements

- Python 3.12 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) for dependency management
- PyTorch and matplotlib (installed transitively via the
  `reasoning-from-scratch` dependency, pinned in `uv.lock`)

## Setup

```bash
uv sync
```

This creates a `.venv` and installs the project together with its dependencies.

## Usage

Run a chapter script, for example chapter 2:

```bash
uv run python -m rmfs.chapters.ch02
```

Chapter 4 additionally opens matplotlib figures for the toy temperature
and top-p walkthroughs.

The first run downloads the Qwen3 0.6B base model and tokenizer into
`models/qwen3/` (`models/` is gitignored). The MATH-500 dataset is likewise
downloaded and cached under `data/math500/` on first use.

Evaluation results are written as JSONL files named after the device (e.g.
`math500-mps.jsonl` on Apple Silicon); an example log from a notebook run is
kept in `notebooks/math500-mps.jsonl`.

## Project structure

```text
.
├── data/                   # cached datasets (MATH-500)
├── models/                 # downloaded model weights (gitignored)
├── notebooks/              # exploratory chapter notebooks
├── reasoning_from_scratch/ # reference copy of the companion package
├── src/
│   └── rmfs/               # cleaned-up, importable code
│       ├── chapters/       # one runnable script per chapter
│       ├── data/           # dataset loading (MATH-500)
│       ├── evaluation/     # answer extraction, grading, MATH-500 eval loop
│       ├── generation/     # streaming generation (greedy, temperature, top-p)
│       ├── paths.py        # shared filesystem locations
│       ├── prompts.py      # prompt template
│       ├── scoring.py      # heuristic and likelihood answer scoring
│       ├── self_consistency.py # majority answer over sampled generations
│       ├── self_refinement.py  # critique-and-revise refinement loop
│       └── utils.py        # progress/ETA helpers
├── pyproject.toml
└── uv.lock
```

## License

MIT — see [LICENSE](LICENSE).
