"""Experiment 10 — J-space decomposition and OCCUPANCY (paper §4.2, Fig. 30).

Part 1 sparse-decomposes one activation into J-lens atoms ("what is in the
workspace right now"); part 2 plots occupancy by layer (K where J-lens atoms
stop beating a same-size random dictionary) and excess variance explained.

Writes occupancy.png.

    python experiments/10_occupancy.py
    python experiments/10_occupancy.py --n-random 50000     # low-memory GPUs
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import statistics

import config
from jlens import (load_model, JLens, forward_hidden,
                   sparse_decompose, RandomDict, random_pursuit_curve, occupancy)
from jlens.corpus import load_texts

K_MAX = 50   # pursuit budget; must sit above the occupancy plateau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The number of legs on the animal "
                                        "that spins webs is")
    ap.add_argument("--n-random", type=int, default=None,
                    help="random-dictionary size (default: vocab size, as in "
                         "the paper; ~750 MB fp16)")
    args = ap.parse_args()

    model, tok = load_model()
    lens = JLens(model, tok)
    band = config.workspace_layers(model)
    mid = band[len(band) // 2]
    d = model.config.hidden_size

    # ---------------- Part 1: J-space representation -----------------------
    ids = tok(args.prompt, return_tensors="pt").input_ids
    hs, _ = forward_hidden(model, ids)
    h = hs[mid][0, -1]
    support, c, r = sparse_decompose(lens, h, mid, k=25)
    total = h.float().pow(2).sum().item()
    print(f"\nJ-space representation of the last position of\n  {args.prompt!r}"
          f"\nat layer {mid}  (h ≈ Σ coeff · v̂_token + residual):\n")
    for t, coef in sorted(zip(support, c.tolist()), key=lambda x: -x[1]):
        print(f"  {coef:8.1f} · v̂[{tok.decode([t])!r}]")
    print(f"\n  variance reconstructed by these 25 atoms: "
          f"{1 - r.pow(2).sum().item() / total:.1%} (raw — the paper's <10% "
          f"figure is the EXCESS over a same-size random control, below)")

    # ---------------- Part 2: occupancy by layer (Fig. 30) ------------------
    n_rand = args.n_random or lens.vocab
    rand = RandomDict(n_rand, d, model.device)
    texts = load_texts(2)
    probe_layers = list(range(2, lens.target_layer + 1, 3))
    curves = {l: [] for l in probe_layers}       # (curve_j, curve_r, ||h||^2)

    for text in texts:
        tids = tok(text, return_tensors="pt", truncation=True,
                   max_length=96).input_ids
        ths, _ = forward_hidden(model, tids)
        T = tids.shape[1]
        for p in range(T // 12, T, max(1, T // 12)):
            for l in probe_layers:
                hp = ths[l][0, p]
                _, _, _, cj = sparse_decompose(lens, hp, l, k=K_MAX,
                                               return_curve=True)
                cr = random_pursuit_curve(rand, hp, k_max=K_MAX)
                curves[l].append((cj, cr, hp.float().pow(2).sum().item()))

    occ, q1, q3, excess = {}, {}, {}, {}
    for l in probe_layers:
        ks = [occupancy(cj, cr) for cj, cr, _ in curves[l]]
        occ[l] = statistics.median(ks)
        q1[l], _, q3[l] = statistics.quantiles(ks, n=4)
        K = min(int(occ[l]), K_MAX)              # paper: evaluate at median K
        excess[l] = statistics.median(
            (cr[min(K, len(cr) - 1)] - cj[min(K, len(cj) - 1)]) / tot
            for cj, cr, tot in curves[l])

    print(f"\nOccupancy by layer (vs {n_rand}-atom random control; "
          f"median [IQR] over positions, excess variance at K=median):")
    for l in probe_layers:
        mark = " <- workspace band" if l in band else ""
        print(f"  layer {l:>2}: {occ[l]:4.0f}  [{q1[l]:.0f}-{q3[l]:.0f}]   "
              f"excess var {excess[l]:6.1%}{mark}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
        ls = probe_layers
        a.fill_between(ls, [q1[l] for l in ls], [q3[l] for l in ls], alpha=0.25)
        a.plot(ls, [occ[l] for l in ls], marker="o", ms=3)
        a.axvspan(band[0], band[-1], alpha=0.12, color="gray")
        a.set_xlabel("residual layer"); a.set_ylabel("occupancy (atoms)")
        a.set_title("J-space occupancy (median, IQR)")
        wl = [l for l in ls if l in band]
        b.bar([str(l) for l in wl], [excess[l] for l in wl])
        b.axhline(0.10, ls="--", lw=1, color="gray")
        b.set_xlabel("workspace layer")
        b.set_ylabel("excess variance explained at K = median occupancy")
        b.set_title("J-space share of activation variance")
        fig.tight_layout()
        fig.savefig("occupancy.png", dpi=150)
        print("Wrote occupancy.png (Fig. 30's two panels)")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
