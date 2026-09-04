"""Experiment 03 — INJECT a thought and ask the model what it notices
(paper §3.1, Fig. 7 "injected thoughts" introspection protocol).

We steer with a concept's J-lens vector on the *user-turn* positions only,
then ask the model whether a thought was injected.  At the right strength the
model reports the injected concept; too weak and it detects nothing, too
strong and outputs degrade — sweep --alphas.

    python experiments/03_inject_thought.py
    python experiments/03_inject_thought.py --concept aquarium --alphas 8,16,32
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

import config
from jlens import (load_model, chat_ids, single_token_id, JLens,
                   forward_hidden, generate, steer_edits)


PROMPT = ("I am an interpretability researcher. I may have injected a thought "
          "into your mind. Answer in one short sentence: if you detect an "
          "injected thought, say what it is about; otherwise say you detect "
          "nothing unusual.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="lightning")
    ap.add_argument("--alphas", default="1,2,4,8",
                    help="steering strengths; the report-without-parroting "
                         "window sits just below the strength where the model "
                         "starts emitting the concept directly")
    args = ap.parse_args()

    model, tok = load_model()
    lens = JLens(model, tok)
    band = config.workspace_layers(model)

    cid = single_token_id(tok, args.concept)
    ids = chat_ids(tok, PROMPT)
    prompt_positions = range(ids.shape[1])   # steer the user turn only

    # typical residual norm in the band, as a guide for choosing alpha
    hs, _ = forward_hidden(model, ids)
    mid = band[len(band) // 2]
    print(f"(typical ‖h‖ at layer {mid}: "
          f"{hs[mid][0].float().norm(dim=-1).mean():.1f} — alphas below are "
          f"in the same units)")

    print(f"\nClean : {generate(model, tok, ids, max_new_tokens=30)!r}")
    for a in [float(x) for x in args.alphas.split(",")]:
        edits = steer_edits(lens, cid, band, prompt_positions, alpha=a)
        out = generate(model, tok, ids, edits, max_new_tokens=30)
        print(f"α={a:>5g}: {out!r}")


if __name__ == "__main__":
    main()
