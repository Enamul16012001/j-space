"""Experiment 11 — BROADCAST (paper §4.3.1, Fig. 32): each layer's MLP path
amplifies unit J-lens vectors far more than random or MLP-neuron directions.

Writes broadcast_gain.png.

    python experiments/11_broadcast_gain.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config   # noqa: E402 - must precede torch; see the note in config.py

import torch
from jlens import load_model, JLens


@torch.no_grad()
def mlp_gain(block, V, model):
    """Median ‖mlp(post_attention_layernorm(v))‖ over unit rows of V."""
    V = (V / V.norm(dim=1, keepdim=True)).to(model.device, model.dtype)
    out = block.mlp(block.post_attention_layernorm(V))
    return out.float().norm(dim=1).median().item()


def main():
    model, tok = load_model()
    lens = JLens(model, tok)
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    n = 2000                                   # as in the paper (Fig. 32)

    vocab_sample = torch.randint(0, lens.vocab, (n,), generator=g)
    rand_dirs = torch.randn(n, d, generator=g)

    layers = config.workspace_layers(model)[::2]
    rows = []
    for l in layers:
        block = model.model.layers[l]          # block l consumes residual layer l
        Vj = lens.vectors(vocab_sample, l, unit=True)
        gj = mlp_gain(block, Vj, model)
        gr = mlp_gain(block, rand_dirs, model)
        # neuron-output directions of the previous block's MLP
        Wn = model.model.layers[l - 1].mlp.down_proj.weight  # [d, d_mlp]
        cols = torch.randint(0, Wn.shape[1], (n,), generator=g)
        gn = mlp_gain(block, Wn[:, cols].T.float(), model)
        # Fig. 32 normalizes so that isotropic random directions have gain 1
        rows.append((l, gj / gr, gn / gr))
        print(f"layer {l:>2}:  J-lens gain {gj / gr:5.1f}x   "
              f"neuron gain {gn / gr:5.1f}x   (random = 1 by construction)")

    print("\nExpected: J-lens directions are amplified ~an order of magnitude "
          "more than random or neuron directions — the substrate actively "
          "BROADCASTS workspace content to downstream consumers.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, name in [(1, "J-lens vectors"), (2, "neuron output dirs")]:
            ax.plot([r[0] for r in rows], [r[i] for r in rows],
                    marker="o", label=name)
        ax.axhline(1.0, ls="--", lw=1, color="gray", label="random = 1")
        ax.set_xlabel("residual layer")
        ax.set_ylabel("MLP gain (random-normalized, Fig. 32)")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig("broadcast_gain.png", dpi=150)
        print("Wrote broadcast_gain.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
