"""Experiment 06 — ONE swap, MANY functions: flexible generalization
(paper §3.4, Fig. 18).

The same France→China lens swap, defined once with no reference to any task,
redirects *different* downstream computations — capital, language, continent,
currency — exactly as a global-workspace account predicts: many specialist
consumers all read the same broadcast variable.

    python experiments/06_flexible_generalization.py
    python experiments/06_flexible_generalization.py --alpha 2
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import config
from jlens import (load_model, chat_ids, single_token_id, JLens,
                   next_token_logits, swap_edits)
from jlens.utils import top_str

# Qwen3 is instruction-tuned: bare completions make it answer with
# fill-in-the-blank tokens instead of the fact, leaving no confident
# intermediate to redirect.  Ask it properly (--raw for the old behaviour).
TEMPLATES = [
    "What is the capital of France? Reply with just the name.",
    "What language do most people in France speak? Reply with one word.",
    "Which continent is France on? Reply with one word.",
    "What currency is used in France? Reply with one word.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="France")
    ap.add_argument("--target", default="China")
    ap.add_argument("--raw", action="store_true",
                    help="bare completion instead of the chat template")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="the paper uses alpha=2 for this experiment")
    args = ap.parse_args()

    model, tok = load_model()
    lens = JLens(model, tok)
    band = config.workspace_layers(model)

    src = single_token_id(tok, args.source)
    tgt = single_token_id(tok, args.target)
    edits = swap_edits(lens, src, tgt, band, positions="all", alpha=args.alpha)

    print(f"One swap ({args.source} -> {args.target}, alpha={args.alpha}), "
          f"four different downstream functions:\n")
    for t in TEMPLATES:
        ids = (tok(t, return_tensors="pt").input_ids if args.raw
               else chat_ids(tok, t))
        clean = next_token_logits(model, ids)
        swapped = next_token_logits(model, ids, edits)
        print(f"{t!r}")
        print(f"  clean  : {top_str(tok, clean, 3)}")
        print(f"  swapped: {top_str(tok, swapped, 3)}\n")


if __name__ == "__main__":
    main()
