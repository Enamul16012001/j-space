"""Experiment 12 — is the J-SPACE PRIVILEGED for verbal report?
(paper §3.1, Fig. 8.)

Swapping J-lens vectors changes what the model reports — but that alone does
not show the *J-space* is special: some direction outside it might encode the
same concept just as causally.  The paper's control:

  1. Build a concept vector for a word: the residual stream just before the
     Assistant answers "Tell me about {word}", mean-subtracted over a baseline
     set of other concepts.
  2. Split it (gradient pursuit, k = 16) into a J-space component and a
     non-J-space remainder.  The J-space part holds only ~6–7% of the vector's
     variance.
  3. Re-run the "think of a {category}" swap using each part in place of the
     J-lens vectors, every perturbation rescaled to the same magnitude.

The paper finds the swap target reaches the model's top-5 on 88% of trials with
pure J-lens vectors, 59% with the J-space component, and 5% with the
non-J-space remainder — i.e. the thin J-space slice carries the causal effect.

    python experiments/12_jspace_privilege.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import statistics

import config   # noqa: E402 - must precede torch; see the note in config.py

import torch
from jlens import (load_model, chat_ids, JLens, forward_hidden, generate,
                   next_token_logits, jspace_split, Swap, swap_edits)

CATEGORIES = {
    "sport":   ["Rugby", "Tennis", "Golf", "Cricket", "Boxing", "Soccer"],
    "animal":  ["Otter", "Falcon", "Panda", "Whale", "Camel", "Tiger"],
    "color":   ["Purple", "Orange", "Yellow", "Green", "Brown", "Pink"],
    "country": ["Brazil", "Norway", "Kenya", "Japan", "Canada", "Egypt"],
}

# The baseline set the concept vectors are mean-subtracted over (§3.1 uses 100
# other concepts).  These need not be single tokens — only the mean is used.
BASELINE_CONCEPTS = """
bridge harvest lantern compass verdict granite orchestra plumbing satellite
meadow archive tunnel vinegar telescope hurricane parliament ceramic anchor
glacier notebook marathon sculpture pension railway bacteria umbrella cathedral
sediment trumpet warehouse almanac keyboard monsoon quarry trombone vineyard
wallpaper zeppelin blanket circuit dentist envelope fossil gallery hammock
insulin jasmine kettle ladder magnet nectar obelisk parachute quilt ribbon
saddle thermos ukulele velvet walnut xylophone yogurt zodiac balcony canyon
driftwood eclipse ferry gondola hedge iceberg jetty kiln lagoon mosaic
nutmeg orchard pottery quiver reservoir sandal tapestry utensil violin
windmill anvil bramble cobbler dormitory embankment furnace gazette
harbour ivory jubilee kerosene lattice marmalade nightingale observatory
""".split()
assert len(BASELINE_CONCEPTS) == 100


def concept_vectors(model, tok, words, band):
    """v_w(l) = h(l, last prompt position of "Tell me about w") − baseline mean."""
    def acts(w):
        ids = chat_ids(tok, f"Tell me about {w}")
        hs, _ = forward_hidden(model, ids)
        return {l: hs[l][0, -1].float() for l in band}

    print(f"Building concept vectors ({len(BASELINE_CONCEPTS)} baseline "
          f"concepts + {len(words)} targets)…")
    mean = {l: torch.zeros(model.config.hidden_size, device=model.device)
            for l in band}
    for b in BASELINE_CONCEPTS:
        a = acts(b)
        for l in band:
            mean[l] += a[l]
    for l in band:
        mean[l] /= len(BASELINE_CONCEPTS)

    out = {}
    for w in words:
        a = acts(w)
        out[w] = {l: a[l] - mean[l] for l in band}
    return out


def unit_swap_edits(vs, vt, band, alpha=1.0):
    """Swap along two arbitrary per-layer directions, each unit-normalized so
    every condition perturbs by the same magnitude."""
    return [Swap(vs[l] / vs[l].norm(), vt[l] / vt[l].norm(), alpha, [l])
            for l in band]


def main():
    model, tok = load_model()
    lens = JLens(model, tok)
    band = config.workspace_layers(model)

    # ---- pick, per category, the model's own answer and a swap target ------
    trials = []
    for cat, candidates in CATEGORIES.items():
        prompt = (f"Think of a {cat}. Do not explain your reasoning. "
                  f"Reply with only its name.")
        ids = chat_ids(tok, prompt)
        top10 = next_token_logits(model, ids).topk(10).indices.tolist()
        src = top10[0]
        # §3.1: the target must NOT be in the model's top-10 possible outputs.
        # Match the target token's spacing to the source token (a leading-space
        # variant of the same word is a different vocab entry).
        space = tok.decode([src]).startswith(" ")
        tgt = None
        for w in candidates:
            for form in ((" " + w, w) if space else (w, " " + w)):
                enc = tok.encode(form, add_special_tokens=False)
                if len(enc) == 1 and enc[0] not in top10:
                    tgt = enc[0]
                    break
            if tgt is not None:
                break
        if tgt is None:
            print(f"skip {cat}: no single-token candidate outside the top-10")
            continue
        # The chosen ITEM may span several tokens ('Basket'+'ball'); build the
        # concept vector from the full word the model writes, while the swap
        # itself acts on the first token, which is what decides the output.
        src_word = generate(model, tok, ids, max_new_tokens=4).split()[0].strip(".,!\"'")
        tgt_word = w
        trials.append((cat, ids, src, tgt, src_word, tgt_word))
        print(f"{cat:>8}: model says {src_word!r} (first token "
              f"{tok.decode([src])!r}) -> swap in {tgt_word!r}")

    if not trials:
        print("no usable trials")
        return

    # ---- concept vectors, split into J-space / non-J-space -----------------
    words = sorted({w for *_, sw, tw in trials for w in (sw, tw)})
    cvec = concept_vectors(model, tok, words, band)

    comp, rem, shares = {}, {}, []
    for w, v in cvec.items():
        comp[w], rem[w] = {}, {}
        for l in band:
            c, r = jspace_split(lens, v[l], l, k=16)
            comp[w][l], rem[w][l] = c, r
            shares.append((c.pow(2).sum() / v[l].pow(2).sum()).item())
    print(f"\nJ-space component holds {statistics.median(shares):.1%} of the "
          f"concept vector's variance (paper: 6–7%) — the rest is outside it.")

    # ---- the same swap, three ways ----------------------------------------
    def variants(word):
        """Every single-token spelling of `word` ('Purple' and ' Purple' are
        different vocab entries; a hit through either counts)."""
        out = set()
        for f in (word, " " + word):
            e = tok.encode(f, add_special_tokens=False)
            if len(e) == 1:
                out.add(e[0])
        return out

    hits = {"J-lens vector": 0, "J-space component": 0, "non-J-space": 0}
    for cat, ids, src, tgt, sw, tw in trials:
        conds = {
            "J-lens vector": swap_edits(lens, src, tgt, band),
            "J-space component": unit_swap_edits(comp[sw], comp[tw], band),
            "non-J-space": unit_swap_edits(rem[sw], rem[tw], band),
        }
        print(f"\n=== {cat} ({sw!r} -> {tw!r}) ===")
        for name, edits in conds.items():
            lg = next_token_logits(model, ids, edits)
            top5 = lg.topk(5).indices.tolist()
            hit = bool(variants(tw) & set(top5))
            hits[name] += hit
            print(f"  {name:<18}: top-5 {[tok.decode([i]) for i in top5]} "
                  f"{'HIT' if hit else 'miss'}")

    print(f"\nSwap target reached top-5 ({len(trials)} categories):")
    for name, h in hits.items():
        print(f"  {name:<18}: {h / len(trials):.0%}")
    print("\nExpected ordering (paper Fig. 8): J-lens > J-space component >> "
          "non-J-space — the causal effect lives in the thin J-space slice, "
          "not in the ~93% of the concept vector outside it.")


if __name__ == "__main__":
    main()
