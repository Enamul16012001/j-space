"""Step 0 — estimate the J-lens for every layer of Qwen3-4B and save it.

Run this ONCE before any other experiment:

    python experiments/00_compute_jlens.py

The backward passes run in config.LENS_DTYPE (float32 by default): the Jacobian
is an average of gradients propagated through all 36 blocks, so this is the one
step where low precision really costs accuracy.  Set config.LENS_DTYPE to
torch.bfloat16 (or lower ROWS_PER_BACKWARD) if it does not fit.

Cost: with the defaults in config.py (16 prompts × 128 tokens), expect very
roughly 1–2 hours on an A100-class GPU and ~944 MB on disk.  The paper averages
over 1000 prompts but shows the J-lens already beats the logit and tuned lenses
with as few as 10 (Fig. 59); raise config.N_PROMPTS for fidelity.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config   # noqa: E402 - must precede torch; see the note in config.py

import torch
from jlens import load_model, compute_jlens
from jlens.corpus import load_texts


def main():
    model, tok = load_model(dtype=config.LENS_DTYPE)
    texts = load_texts(config.N_PROMPTS)
    print(f"Estimating J-lens on {len(texts)} prompts in {config.LENS_DTYPE} "
          f"(target = {config.TARGET} residual stream)…")
    print(f"Checkpointing to {config.JLENS_PATH} every {config.SAVE_EVERY} "
          f"prompts — you can stop early and still have a usable lens.")
    data = compute_jlens(model, tok, texts, save_path=config.JLENS_PATH)
    torch.save(data, config.JLENS_PATH)
    d = model.config.hidden_size
    print(f"\nSaved {config.JLENS_PATH}: {len(data['layers'])} layers, "
          f"J_l is [{d} x {d}] fp32, averaged over {data['n_prompts']} prompts.")
    print("You can now run experiments 01–12.")


if __name__ == "__main__":
    main()
