"""Experiment 04 — HOLD A THOUGHT in mind (paper §3.2, Fig. 9-10).

The model copies a fixed sentence (teacher-forced, so output is identical)
while silently thinking of citrus / computing 3^2-2 / nothing; probe words
are scored by best lens rank over all (workspace layer, copy position) cells.

    python experiments/04_hold_in_mind.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from jlens import load_model, chat_ids, single_token_id, JLens, forward_hidden

SENTENCE = "The old painting hung crookedly on the wall."

CONDITIONS = {
    "baseline":  "Copy the following sentence exactly, and do nothing else:",
    "citrus":    ("While you copy the following sentence exactly, silently "
                  "concentrate on citrus fruits. Do not mention them:"),
    "math":      ("While you copy the following sentence exactly, silently "
                  "compute 3^2 - 2. Do not mention the answer:"),
}

PROBES = {
    "citrus": ["orange", "lemon", "citrus"],
    "math":   ["nine", "seven", "math"],
}


def main():
    model, tok = load_model()
    lens = JLens(model, tok)
    band = config.workspace_layers(model)

    probe_ids = {w: single_token_id(tok, w)
                 for ws in PROBES.values() for w in ws}

    for name, instr in CONDITIONS.items():
        user = f"{instr}\n\n{SENTENCE}"
        # teacher-force the copy: the assistant turn IS the sentence
        ids = chat_ids(tok, user, prefill=SENTENCE)
        n_prompt = chat_ids(tok, user).shape[1]

        hs, _ = forward_hidden(model, ids)
        # best rank of each probe over every (band layer, copy position) cell
        best = {w: (10**9, None, None) for w in probe_ids}
        for l in band:
            S = lens.scores(hs[l][0, n_prompt:], l)          # [T_copy, vocab]
            for w, t in probe_ids.items():
                r = (S > S[:, t].unsqueeze(1)).sum(-1) + 1   # rank per position
                m = int(r.min())
                if m < best[w][0]:
                    best[w] = (m, l, int(r.argmin()) + n_prompt)

        print(f"\n=== condition: {name} ===")
        print("best lens rank over the copied sentence "
              "(any workspace layer x position, Fig. 10 metric):")
        for group, words in PROBES.items():
            for w in words:
                r, l, p = best[w]
                print(f"  {w:>7} ({group:>6}): rank {r:>6}  "
                      f"(layer {l}, token {tok.decode([ids[0, p]])!r})")

    print("\nExpected: the citrus words rank far higher in the 'citrus' "
          "condition, the number words in the 'math' condition — even though "
          "the text being produced is exactly the same sentence.")


if __name__ == "__main__":
    main()
