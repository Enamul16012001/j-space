"""Experiment 01 — READ the residual stream with the J-lens (paper §2.4, Fig. 4B).

Shows (a) a layer × position grid of the top-1 lens token — the "workspace
transcript" — and (b) at one position, the J-lens vs logit-lens top-k across
layers, where you can see the J-lens verbalizing intermediate content that the
logit lens misses in mid layers.

    python experiments/01_read_lens.py
    python experiments/01_read_lens.py --prompt "Je pense donc je" --pos -1
    python experiments/01_read_lens.py --chat --prompt "Name a citrus fruit."
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse

from jlens import load_model, chat_ids, JLens, forward_hidden, logit_lens_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The number of legs on the animal that "
                                        "spins webs is")
    ap.add_argument("--pos", default="-1",
                    help="position(s) for the detailed readout, comma-separated "
                         "(negative counts from the end): --pos 6,7,8")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--cosine", action="store_true",
                    help="rank by cos(v, h) instead of the raw lens logit "
                         "(§2.5). Suppresses rare long-vector tokens like "
                         "'____' that win on magnitude rather than meaning.")
    ap.add_argument("--chat", action="store_true",
                    help="wrap the prompt in the chat template")
    args = ap.parse_args()

    model, tok = load_model()
    lens = JLens(model, tok)

    if args.chat:
        ids = chat_ids(tok, args.prompt)
    else:
        ids = tok(args.prompt, return_tensors="pt").input_ids
    hs, _ = forward_hidden(model, ids)
    T = ids.shape[1]
    positions_detail = [int(p) % T for p in args.pos.split(",")]

    # ---- (a) layer x position grid of top-1 lens tokens -------------------
    positions = list(range(max(0, T - 24), T))       # cap width for readability
    print("\n=== Top-1 J-lens token per (layer, position) ===")
    header = " " * 7 + "".join(f"{tok.decode([ids[0, p]])[:9]:>10}" for p in positions)
    print(header)
    for l in range(2, lens.target_layer + 1, 3):
        row = f"L{l:>3} | "
        for p in positions:
            t = lens.readout(hs[l][0, p], l, k=1, cosine=args.cosine)[0][0]
            row += f"{t.strip()[:9]:>10}"
        print(row)

    # ---- (b) J-lens vs logit lens at the chosen positions ------------------
    for pos in positions_detail:
        print(f"\n=== Detailed readout at position {pos} "
              f"(token {tok.decode([ids[0, pos]])!r}) ===")
        for l in range(2, lens.target_layer + 1, 2):
            h = hs[l][0, pos]
            jl = [s for s, _, _ in lens.readout(h, l, k=args.topk, cosine=args.cosine)]
            vals, idx = logit_lens_logits(model, h).topk(args.topk)
            ll = [tok.decode([i]) for i in idx.tolist()]
            print(f"L{l:>3}  J-lens: {jl}")
            print(f"      logit : {ll}")


if __name__ == "__main__":
    main()
