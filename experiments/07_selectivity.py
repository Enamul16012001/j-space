"""Experiment 07 — SELECTIVITY of workspace edits (paper §3.5.1, Fig. 20).

Swapping the "Spanish" lens coordinate for "French" ACROSS THE QUESTION TOKENS
(the paper's protocol — the passage itself is left untouched) flips tasks that
*consult the verbalized language variable*:

  explicit report     "What language is this?"          Spanish -> French
  flexible inference  "Say 'hello' in that language."   Hola    -> Bonjour

but does NOT move the automatic tasks, even though they depend on the very
same variable:

  continuation        still continues in Spanish
  anomaly detection   still spots the French sentence spliced into the passage

Low-level language machinery runs outside the workspace; only report and
flexible inference route through it.

    python experiments/07_selectivity.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from jlens import (load_model, chat_ids, single_token_id, JLens,
                   generate, swap_edits)

# An original Spanish passage (any Spanish text works).
PASSAGE = ("La lluvia caía despacio sobre los tejados del pueblo. María cerró "
           "la ventana, encendió una vela y se sentó junto a la mesa de "
           "madera. Afuera, las calles estaban vacías y el aire olía a "
           "tierra mojada.")

# The same passage with one French sentence spliced in (anomaly detection).
PASSAGE_SPLICED = PASSAGE.replace(
    "Afuera,",
    "Le vent soufflait fort et faisait claquer les volets. Afuera,")

#         condition            passage          question
TASKS = {
    "explicit report":    (PASSAGE,
        "What language is this passage written in? Answer with one word."),
    "flexible inference": (PASSAGE,
        "Say the word for 'hello' in the language of this passage. "
        "Answer with one word."),
    "continuation":       (PASSAGE,
        "Continue this passage with one more sentence, in the same style."),
    "anomaly detection":  (PASSAGE_SPLICED,
        "Does this passage switch languages anywhere? Answer yes or no."),
}


def question_positions(tok, passage, user):
    """Chat ids for the full prompt, plus the absolute positions of the
    QUESTION tokens: everything after the longest shared prefix with the
    passage-only prompt.  §3.5.1 applies the swap across the question tokens
    only, so the passage's own processing is untouched."""
    full = chat_ids(tok, user)
    pref = chat_ids(tok, passage)
    n, m = 0, min(full.shape[1], pref.shape[1])
    while n < m and full[0, n] == pref[0, n]:
        n += 1
    return full, list(range(n, full.shape[1]))


def main():
    model, tok = load_model()
    lens = JLens(model, tok)
    band = config.workspace_layers(model)

    src = single_token_id(tok, "Spanish")
    tgt = single_token_id(tok, "French")

    for name, (passage, task) in TASKS.items():
        user = f"{passage}\n\n{task}"
        ids, qpos = question_positions(tok, passage, user)
        edits = swap_edits(lens, src, tgt, band, positions=qpos, alpha=1.0)
        clean = generate(model, tok, ids, max_new_tokens=16)
        swapped = generate(model, tok, ids, edits, max_new_tokens=16)
        print(f"\n=== {name} (swap on {len(qpos)} question tokens) ===")
        print(f"clean  : {clean!r}")
        print(f"swapped: {swapped!r}")

    print("\nExpected: report says French and 'hello' becomes 'bonjour', while "
          "the continuation is STILL Spanish and the spliced French sentence "
          "is still detected — the swap edits the verbalizable variable, not "
          "the low-level language machinery.")


if __name__ == "__main__":
    main()
