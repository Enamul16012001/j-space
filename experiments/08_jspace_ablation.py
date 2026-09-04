"""Experiment 08 — ABLATE THE WHOLE J-SPACE (paper §3.5.2, Fig. 22).

At every position in a band of layers, remove the span of the top-k=10 active
J-lens vectors (excluding each position's own top-10 next-token predictions
from the clean pass, so the model's immediate output machinery is spared).

Prediction from the paper: multi-hop reasoning — which must pass intermediate
results through the workspace — collapses, while routine next-token prediction
on ordinary text barely degrades.

    python experiments/08_jspace_ablation.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config   # noqa: E402 - must precede torch; see the note in config.py

import torch
from jlens import (load_model, chat_ids, JLens, apply_edits,
                   next_token_logits, TopKJSpaceAblate)
from jlens.corpus import load_texts

# Asked through the chat template: Qwen3 is instruction-tuned and answers bare
# completions with fill-in-the-blank tokens, which would make the "clean"
# multi-hop baseline meaningless.
ONE_WORD = " Reply with just the answer, one word."
MULTIHOP = [
    ("How many legs does the animal that spins webs have?" + ONE_WORD, "8"),
    ("What is the capital of the country famous for the Eiffel Tower?" + ONE_WORD, "Paris"),
    ("What is the capital of the country whose flag is a red circle on white?" + ONE_WORD, "Tokyo"),
    ("What colour is the planet fourth from the Sun?" + ONE_WORD, "red"),
    ("How many legs does the insect that makes honey have?" + ONE_WORD, "6"),
    ("What language is spoken in the country famous for tacos and mariachi?" + ONE_WORD, "Spanish"),
    ("On which continent do kangaroos live in the wild?" + ONE_WORD, "Australia"),
]


def clean_exclusions(model, ids, k=10):
    """Per-position top-k next-token predictions from the clean pass."""
    with torch.no_grad():
        logits = model(ids.to(model.device)).logits[0].float()
    return {p: logits[p].topk(k).indices.tolist() for p in range(ids.shape[1])}


def band_edits(lens, layers, exclude):
    return [TopKJSpaceAblate(lens, k=10, layers=[l], exclude_per_pos=exclude)
            for l in layers]


def main():
    model, tok = load_model()
    lens = JLens(model, tok)
    W = config.workspace_layers(model)
    third = max(1, len(W) // 3)
    BANDS = {
        "light (middle third of band)": W[third:2 * third],
        "medium (full workspace band)": W,
        "heavy (band ± 3 layers)":
            [l for l in range(W[0] - 3, W[-1] + 4)
             if 1 <= l <= lens.target_layer],
    }

    # ---------------- multi-hop reasoning under ablation -------------------
    # Score against every single-token spelling of the answer ("8" and " 8",
    # "Paris" and " Paris" are different vocab entries).
    def answer_ids(word):
        out = set()
        for f in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            e = tok.encode(f, add_special_tokens=False)
            if len(e) == 1:
                out.add(e[0])
        return out

    cases = [(p, answer_ids(a)) for p, a in MULTIHOP if answer_ids(a)]

    def accuracy(edits_fn):
        hit = 0
        for prompt, ans in cases:
            ids = chat_ids(tok, prompt)
            edits = edits_fn(ids)
            lg = next_token_logits(model, ids, edits)
            hit += int(bool(ans & set(lg.topk(5).indices.tolist())))
        return hit / len(cases)

    print(f"\nMulti-hop accuracy (answer in top-5), {len(cases)} prompts:")
    print(f"  clean                          : {accuracy(lambda ids: []):.0%}")
    for name, layers in BANDS.items():
        acc = accuracy(lambda ids, L=layers: band_edits(
            lens, L, clean_exclusions(model, ids)))
        print(f"  ablated, {name:<30}: {acc:.0%}")

    # ---------------- routine next-token prediction ------------------------
    print("\nRoutine next-token prediction (top-1 agreement with the clean "
          "model on ordinary text):")
    texts = load_texts(4)
    for name, layers in BANDS.items():
        agree = total = 0
        for text in texts:
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=96).input_ids
            excl = clean_exclusions(model, ids)
            with torch.no_grad():
                clean = model(ids.to(model.device)).logits[0].argmax(-1)
            with apply_edits(model, band_edits(lens, layers, excl)), torch.no_grad():
                abl = model(ids.to(model.device)).logits[0].argmax(-1)
            agree += (clean == abl).sum().item()
            total += ids.shape[1]
        print(f"  {name:<38}: {agree / total:.0%}")

    print("\nExpected: multi-hop collapses under the full-band ablation while "
          "top-1 agreement on ordinary text stays high — the workspace "
          "carries the *intermediate results*, not routine prediction.")


if __name__ == "__main__":
    main()
