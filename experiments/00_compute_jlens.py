"""Step 0 — estimate the J-lens for every layer and save it.  Run ONCE
before any other experiment:

    python experiments/00_compute_jlens.py

Slow (hours-days at N_PROMPTS=1000); checkpoints regularly and resumes
automatically.  Lower ROWS_PER_BACKWARD if it does not fit in memory.
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
