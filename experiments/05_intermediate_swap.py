"""Experiment 05 — SWAP an INTERMEDIATE step of a multi-hop computation and
watch the answer change consistently (paper §3.3, Fig. 12–14).

"The number of legs on the animal that spins webs is" → the model internally
resolves *spider*, then answers 8.  Swapping the spider lens coordinate for
*ant* mid-stream makes the same prompt answer 6: downstream computation
consumes the edited workspace content.

    python experiments/05_intermediate_swap.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import config
from jlens import (load_model, chat_ids, single_token_id, JLens, forward_hidden,
                   next_token_logits, clamp_swap_edits)
from jlens.utils import top_str, best_rank

# Qwen3 is an instruct model: fed bare completion text it answers with
# fill-in-the-blank tokens ("__", "______") instead of solving the question, and
# then there is no confident intermediate to swap.  So ask it properly.  Use
# --raw to feed the question as a plain completion instead.
#        source     target   question                                    clean→swapped
CASES = [
    ("spider", "ant",
     "How many legs does the animal that spins webs have? "
     "Reply with just the number.", "8 -> 6"),
    ("France", "Italy",
     "What is the capital city of the country famous for the Eiffel Tower? "
     "Reply with just the name.", "Paris -> Rome"),
    ("Egypt", "France",
     "On which continent is the country famous for the pyramids of Giza? "
     "Reply with just the name.", "Africa -> Europe"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true",
                    help="feed the question as a bare completion instead of "
                         "using the chat template (much weaker on an "
                         "instruction-tuned model)")
    args = ap.parse_args()

    model, tok = load_model()
    lens = JLens(model, tok)
    band = config.workspace_layers(model)

    for source, target, prompt, expect in CASES:
        try:
            src, tgt = single_token_id(tok, source), single_token_id(tok, target)
        except ValueError as e:
            print(f"skip ({e})")
            continue
        ids = (tok(prompt, return_tensors="pt").input_ids if args.raw
               else chat_ids(tok, prompt))

        hs, clean_logits = forward_hidden(model, ids)
        r, l = best_rank(lens, hs, src, band, pos=-1)
        print(f"\n=== {prompt!r} ===")
        print(f"intermediate {source!r}: best lens rank {r} at layer {l} "
              f"(read at the last position — the word never appears in the text)")
        print(f"clean   next token: {top_str(tok, clean_logits)}")

        # Fig. 13 uses a CLAMPED swap: the two lens coordinates are pinned to
        # their swapped clean-pass values at every position and layer, so
        # downstream layers cannot rewrite the original concept back in.
        edits = clamp_swap_edits(lens, src, tgt, band, hs)
        swapped = next_token_logits(model, ids, edits)
        print(f"swap {source} -> {target}: {top_str(tok, swapped)}")
        print(f"(expected: {expect})")


if __name__ == "__main__":
    main()
