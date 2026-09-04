"""Experiment 02 — SWAP a silently-held thought and change the verbal report
(paper §3.1, Fig. 5–6).

We ask the model to "think of a {category}" and answer with only its name.
The J-lens reads the choice off the residual stream *before* it is spoken;
swapping that lens coordinate for another word's makes the model report the
other word.

    python experiments/02_verbal_report_swap.py
    python experiments/02_verbal_report_swap.py --category "citrus fruit" \\
        --target-word Lime
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import config
from jlens import (load_model, chat_ids, single_token_id, JLens,
                   forward_hidden, generate, next_token_logits, swap_edits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="sport")
    ap.add_argument("--target-word", default="Rugby",
                    help="what the swapped thought should become")
    ap.add_argument("--source-word", default=None,
                    help="override the source (default: the model's own answer)")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    model, tok = load_model()
    lens = JLens(model, tok)
    band = config.workspace_layers(model)

    prompt = (f"Think of a {args.category}. Do not explain your reasoning. "
              f"Reply with only its name.")
    ids = chat_ids(tok, prompt)

    baseline = generate(model, tok, ids, max_new_tokens=8)
    print(f"Prompt   : {prompt}")
    print(f"Baseline : {baseline!r}")

    # §3.1 swaps "the model's spontaneously chosen item": use the model's own
    # top-1 next-token id, so the swap hits the exact vocab entry it decides
    # over (a leading-space variant of the same word is a different token).
    top10 = next_token_logits(model, ids).topk(10).indices.tolist()
    src = (single_token_id(tok, args.source_word) if args.source_word
           else top10[0])
    source_word = tok.decode([src])

    # Match the target's spacing to the source token, and require (§3.1) that
    # the target is NOT already in the model's top-10 possible outputs.
    forms = ([args.target_word, " " + args.target_word]
             if not source_word.startswith(" ")
             else [" " + args.target_word, args.target_word])
    tgt = None
    for w in forms:
        enc = tok.encode(w, add_special_tokens=False)
        if len(enc) == 1:
            tgt = enc[0]
            break
    if tgt is None:
        print(f"\n{args.target_word!r} is not a single token; pick another "
              f"--target-word.")
        return
    if tgt in top10:
        print(f"\n{args.target_word!r} is already in the model's top-10 next "
              f"tokens here, so a successful swap would prove nothing. "
              f"Pick another --target-word.")
        return

    # -- read: where in the stream is the choice, before any answer token? --
    hs, _ = forward_hidden(model, ids)
    print(f"\nJ-lens rank of {source_word!r} at the LAST PROMPT position "
          f"(before a single answer token exists):")
    for l in band[::3]:
        print(f"  layer {l:>2}: rank {lens.rank(hs[l][0, -1], l, src)}")

    # -- intervene: swap source <-> target lens coordinates in the band -----
    edits = swap_edits(lens, src, tgt, band, positions="all", alpha=args.alpha)
    swapped = generate(model, tok, ids, edits, max_new_tokens=8)
    print(f"\nSwap {source_word!r} -> {args.target_word!r} "
          f"(layers {band[0]}–{band[-1]}, alpha={args.alpha}):")
    print(f"Swapped  : {swapped!r}")


if __name__ == "__main__":
    main()
