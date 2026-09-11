"""Pretraining-like corpus for averaging the Jacobians (paper §2.1):
50% C4 web text + 50% WikiText-103, ~25% short / 75% long passages, both
streams shuffled.  Cached on disk so resume always sees the identical list.
The fetch runs in a subprocess because `datasets`' streaming reader can
abort the interpreter at shutdown (exit 134); a built-in fallback keeps the
code runnable offline."""
import json
import os
import subprocess
import sys
import pathlib
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config

_FETCH_SRC = textwrap.dedent("""
    import json, random, sys
    from datasets import load_dataset

    n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    SOURCES = [(0.5, "allenai/c4", "en"),
               (0.5, "Salesforce/wikitext", "wikitext-103-raw-v1")]
    SHORT, LONG_MIN, SHORT_SHARE = (160, 500), 500, 0.25

    texts = []
    for share, name, cfg in SOURCES:
        quota = max(1, round(n * share))
        want_s = round(quota * SHORT_SHARE)
        want_l = quota - want_s
        ds = load_dataset(name, cfg, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        got_s = got_l = 0
        for row in ds:
            t = row["text"].strip()
            if got_s < want_s and SHORT[0] <= len(t) < SHORT[1]:
                texts.append(t); got_s += 1
            elif got_l < want_l and len(t) >= LONG_MIN:
                texts.append(t); got_l += 1
            if got_s >= want_s and got_l >= want_l:
                break
    random.Random(seed).shuffle(texts)
    with open(out, "w") as f:
        json.dump(texts[:n], f)
""")


# Original fallback paragraphs (varied topics, written for this repo).
FALLBACK_TEXTS = [
    "The first transcontinental railroads changed the economics of the interior "
    "almost overnight. Towns that had been week-long wagon rides from any port "
    "suddenly sat a day or two from tidewater, and grain that once rotted in "
    "local silos could be priced against harvests on other continents. "
    "Surveyors argued for years about gradients and passes, because a single "
    "percentage point of slope decided how much coal a locomotive burned and "
    "how many cars it could pull through the mountains in winter.",

    "Coral reefs are built by animals but engineered by light. The polyps "
    "secrete calcium carbonate skeletons, yet most of their energy arrives from "
    "photosynthetic algae living inside their tissues, which is why healthy "
    "reefs crowd the sunlit shallows and grow toward the surface. When water "
    "warms past a threshold, the partnership breaks down: the algae are "
    "expelled, the coral pales, and unless temperatures fall again quickly the "
    "colony starves within weeks.",

    "A sourdough starter is less a recipe than a small managed ecosystem. Wild "
    "yeasts provide the gas that lifts the loaf, while lactic acid bacteria "
    "supply the sourness and, just as importantly, defend the culture against "
    "invaders by keeping the pH low. Bakers feed the starter on a schedule "
    "because each feeding dilutes the acids and gives the yeast a fresh meal of "
    "starch; skip too many feedings and the balance tips toward the bacteria, "
    "producing a batter that smells of vinegar and barely rises.",

    "Early astronomers had no way to measure the distance to the stars "
    "directly, so they reasoned from parallax: if the Earth truly circled the "
    "Sun, nearby stars should appear to shift against the background sky over "
    "the course of a year. For centuries no such shift could be detected, which "
    "skeptics took as evidence against the moving Earth. The real explanation "
    "was scale. The shifts existed, but they were smaller than an arcsecond, "
    "and instruments capable of catching them did not arrive until the 1830s.",

    "Central banks manage expectations as much as they manage money. When a "
    "bank announces a target for inflation, it is trying to coordinate millions "
    "of private decisions about wages, prices, and loans around a shared "
    "forecast. If the public believes the target, the belief itself does much "
    "of the work: firms set prices modestly, workers moderate wage demands, and "
    "inflation lands near the target without drastic policy. Credibility, once "
    "lost, is expensive to buy back.",

    "High-altitude mountaineering is mostly logistics punctuated by brief "
    "climbing. Expeditions spend weeks ferrying loads between camps, not to "
    "move equipment so much as to move bodies: each trip to a higher camp and "
    "back down forces the blood to grow more oxygen-carrying cells. The summit "
    "push itself is timed to a weather window measured in hours, and the most "
    "dangerous part of the day is usually the descent, when concentration fades "
    "and the afternoon sun loosens the ice.",

    "The library card catalog was an information technology with rules as "
    "strict as any database schema. Every card carried the same fields in the "
    "same order, and filing rules dictated how to alphabetize names with "
    "prefixes, how to treat numerals, and where a title beginning with an "
    "article belonged. Librarians drilled these rules because the catalog only "
    "worked if every card was findable by a stranger who had never seen it "
    "before.",

    "Fermentation preserved food long before anyone understood microbes. Salt "
    "drew water out of vegetables and selected for bacteria that could tolerate "
    "brine, and those bacteria in turn acidified the barrel until spoilage "
    "organisms could not survive. Villages that pickled cabbage or fish through "
    "the winter were running a microbiology experiment with their food supply, "
    "and the recipes that survived are the ones whose chemistry happened to be "
    "sound.",
]


def load_texts(n):
    """Return n diverse passages, cached on disk (see the module docstring)."""
    cache = config.ROOT / f".corpus-mix-{config.CORPUS_SEED}-{n}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    if not os.environ.get("HF_HUB_OFFLINE"):
        try:
            r = subprocess.run([sys.executable, "-c", _FETCH_SRC, str(n),
                                str(config.CORPUS_SEED), str(cache)],
                               check=False, timeout=3600,
                               capture_output=True, text=True)
            if cache.exists():
                texts = json.loads(cache.read_text())
                if len(texts) < n:
                    print(f"[corpus] gathered only {len(texts)} of {n} texts")
                return texts
            print(f"[corpus] fetch failed:\n{r.stderr[-400:]}")
        except Exception as e:
            print(f"[corpus] fetch failed ({type(e).__name__}: {e})")

    if n > len(FALLBACK_TEXTS):
        print(f"[corpus] only {len(FALLBACK_TEXTS)} built-in texts available "
              f"({n} requested)")
    return FALLBACK_TEXTS[:n]
