"""Experiment 09 — WHERE is the workspace? Structural metrics per layer
(paper §4.1, Fig. 27–28).

Four signatures, and what each one means (§4.1):

  1. top-k accuracy — the J-lens's top-5 contains the model's next token.
     Near zero through the early layers, ticks up at the workspace start, then
     jumps STEEPLY in the last few layers.  The steep jump marks the END of the
     workspace: there the lens reads "motor" output, not held content.
  2. excess kurtosis — the readout is sharply peaked on a few tokens.  Near
     zero for the first third, rises at the workspace ONSET, falls at the end.
     Caveat: computed on raw lens logits per the paper; on a small model a few
     rare tokens with 2-3x-length lens vectors fatten the tails at EVERY
     layer, which can wash out or invert this signature.  Trust persistence,
     effective dimension and the accuracy take-off over kurtosis here.
  3. persistence — the top-1 lens token repeats at the next position, as a
     Δ log probability over a position-shuffled null.  Near null early, rises
     at the onset, peaks across the band, falls back at the end.  This is
     content being *held* rather than recomputed per token.
  4. effective dimension — the fraction of residual dimensions needed for 90%
     of the variance across the J-lens vectors W_U J_l.  SMALL in the early
     layers (the lens collapses to a small subspace) and rising sharply at the
     onset, as the lens vectors fan out to span the residual stream.

So the workspace band is where kurtosis, persistence and effective dimension
are all high while top-5 accuracy is still low.

The band is derived from the model's depth by default (config.WORKSPACE_BAND,
the paper's 38–92%).  Use this script's output to calibrate it for your own
checkpoint, then pin it by setting config.WORKSPACE_LAYERS to an explicit list.
Writes workspace_layers.png.

    python experiments/09_workspace_layers.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config   # noqa: E402 - must precede torch; see the note in config.py

import math
import random

import torch
from jlens import load_model, JLens, forward_hidden
from jlens.corpus import load_texts


def excess_kurtosis(x):
    xm = x - x.mean()
    return (xm.pow(4).mean() / xm.pow(2).mean().pow(2) - 3).item()


def main():
    model, tok = load_model()
    lens = JLens(model, tok)
    texts = load_texts(6)
    layers = list(range(2, lens.target_layer + 1))
    band = config.workspace_layers(model)      # only used to shade the plot

    acc = {l: [0, 0] for l in layers}
    kurt = {l: [] for l in layers}
    top1 = {l: [] for l in layers}          # per-text top-1 sequences

    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=96).input_ids
        hs, _ = forward_hidden(model, ids)
        with torch.no_grad():
            model_next = model(ids.to(model.device)).logits[0].argmax(-1)
        for l in layers:
            lg = lens.logits(hs[l][0], l)                   # [T, vocab]
            top5 = lg.topk(5).indices
            acc[l][0] += (top5 == model_next[:, None]).any(-1).sum().item()
            acc[l][1] += ids.shape[1]
            kurt[l].append(sum(excess_kurtosis(lg[t]) for t in
                               range(0, lg.shape[0], 8)) / len(range(0, lg.shape[0], 8)))
            top1[l].append(lg.argmax(-1).tolist())

    # persistence: log P(top1[t] == top1[t+1]) minus the same under a
    # position-shuffled null, i.e. the paper's "Δ log probability" (panel c).
    persist = {}
    for l in layers:
        same = n = null = 0
        for seq in top1[l]:
            same += sum(a == b for a, b in zip(seq, seq[1:]))
            n += len(seq) - 1
            for _ in range(5):
                sh = seq[:]
                random.shuffle(sh)
                null += sum(a == b for a, b in zip(sh, sh[1:]))
        p, q = max(same / n, 1e-6), max(null / (5 * n), 1e-6)
        persist[l] = math.log(p) - math.log(q)

    # effective dimensionality of the lens dictionary at each layer: how many
    # residual dimensions carry 90% of the variance *across* the J-lens vectors
    # (so the population is mean-centered first).
    effdim = {}
    g = torch.Generator().manual_seed(0)
    sample = torch.randint(0, lens.vocab, (4096,), generator=g)
    for l in layers:
        V = lens.vectors(sample, l)
        V = V - V.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(V)     # on-device; CPU takes ~30 s/layer
        var = s.pow(2) / s.pow(2).sum()
        effdim[l] = int((var.cumsum(0) < 0.90).sum().item() + 1)

    d = model.config.hidden_size
    print(f"\n{'layer':>5} {'top5 acc':>9} {'kurtosis':>9} "
          f"{'persistence':>12} {'effdim/d':>9}")
    for l in layers:
        a = acc[l][0] / acc[l][1]
        k = sum(kurt[l]) / len(kurt[l])
        print(f"{l:>5} {a:>9.2f} {k:>9.1f} {persist[l]:>12.3f} "
              f"{effdim[l] / d:>9.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
        rows = [("top-5 lens accuracy\n(jumps at the END)",
                 [acc[l][0] / acc[l][1] for l in layers]),
                ("excess kurtosis", [sum(kurt[l]) / len(kurt[l]) for l in layers]),
                ("top-1 persistence\n(Δ log p vs null)", [persist[l] for l in layers]),
                ("effective dim / d", [effdim[l] / d for l in layers])]
        for ax, (name, ys) in zip(axes, rows):
            ax.plot(layers, ys, marker="o", ms=3)
            ax.set_ylabel(name, fontsize=8)
            ax.axvspan(band[0], band[-1], alpha=0.15)
        axes[-1].set_xlabel("residual layer (shaded = the configured band)")
        fig.tight_layout()
        fig.savefig("workspace_layers.png", dpi=150)
        print("\nWrote workspace_layers.png — the workspace band starts where "
              "kurtosis, persistence and effective dimension rise together, "
              "and ends where top-5 accuracy takes off.")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
