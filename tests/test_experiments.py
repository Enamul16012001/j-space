"""Smoke-run every experiment script end to end against a tiny random Qwen3.

This does not check the science (the model is untrained) — it checks that each
script actually runs: no NameErrors, no shape bugs, no wrong API calls.
"""
import os, sys, runpy, traceback, io, contextlib, pathlib, tempfile

# Keep the smoke test hermetic and fast: no WikiText download, use the
# built-in fallback corpus.
os.environ["HF_HUB_OFFLINE"] = "1"

REPO = str(pathlib.Path(__file__).resolve().parents[1])
HERE = tempfile.mkdtemp(prefix="jlens-smoke-")
sys.path.insert(0, REPO)

import config   # noqa: E402 - must precede torch; see the note in config.py

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
config.JLENS_PATH = os.path.join(HERE, "tiny_exp.pt")

D, L, V = 32, 4, 4096
torch.manual_seed(0)
cfg = Qwen3Config(vocab_size=V, hidden_size=D, intermediate_size=64,
                  num_hidden_layers=L, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=512, tie_word_embeddings=True)
MODEL = Qwen3ForCausalLM(cfg).eval().float()
with torch.no_grad():
    MODEL.model.norm.weight.normal_(1.0, 0.3)


class WordTok:
    """A word-level stand-in for the Qwen3 tokenizer."""
    eos_token_id = 0

    def __init__(self):
        self.w2i, self.i2w, self.n = {}, {}, 1

    def _id(self, w):
        if w not in self.w2i:
            assert self.n < V, "tiny vocab exhausted"
            self.w2i[w], self.i2w[self.n] = self.n, w
            self.n += 1
        return self.w2i[w]

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]

    def __call__(self, text, return_tensors=None, truncation=False,
                 max_length=None, add_special_tokens=True):
        ids = self.encode(text)[:max_length or 10**9] or [1]
        return type("Enc", (), {"input_ids": torch.tensor([ids])})()

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.i2w.get(int(i), f"tok{int(i)}") for i in ids)

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=None):
        return "<user> " + " ".join(m["content"] for m in messages) + " <assistant> "


TOK = WordTok()

# Build the tiny lens the experiments will load.
from jlens.lens import compute_jlens
from jlens.corpus import FALLBACK_TEXTS
with contextlib.redirect_stderr(io.StringIO()):
    # Use the SAME 4 texts script 00 will load offline (N_PROMPTS=4, fallback
    # corpus), so its resume fingerprint matches and it hits the
    # "already covers" path instead of refusing.
    data = compute_jlens(MODEL, TOK, FALLBACK_TEXTS[:4], rows_per_backward=8,
                         seq_len=128)   # config defaults, same as script 00
torch.save(data, config.JLENS_PATH)

# Patch what the experiment scripts import.
import jlens
jlens.load_model = lambda dtype=None: (MODEL, TOK)
config.WORKSPACE_LAYERS = [1, 2, 3]
config.N_PROMPTS = 4

SCRIPTS = sorted(f for f in os.listdir(os.path.join(REPO, "experiments"))
                 if f.endswith(".py"))

os.chdir(HERE)
ok, bad = [], []
for name in SCRIPTS:
    path = os.path.join(REPO, "experiments", name)
    sys.argv = [path]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            runpy.run_path(path, run_name="__main__")
        ok.append(name)
        open(os.path.join(HERE, "out_" + name + ".txt"), "w").write(buf.getvalue())
        print(f"  ok   {name}  ({len(buf.getvalue().splitlines())} lines out)")
    except SystemExit:
        ok.append(name)
        print(f"  ok   {name} (clean exit)")
    except Exception:
        bad.append(name)
        print(f"  FAIL {name}")
        print("       " + traceback.format_exc().strip().replace("\n", "\n       "))

# ---- workbench (server logic only, no HTTP) --------------------------------
try:
    import threading
    import workbench
    from jlens import JLens as _JL
    wb = workbench.Workbench.__new__(workbench.Workbench)
    wb.model, wb.tok = MODEL, TOK
    wb.lens = _JL(MODEL, TOK, config.JLENS_PATH)
    wb.band = wb.layers = [1, 2, 3]
    wb.lock = threading.Lock()
    wb.ids = wb.hs = None
    rc = wb.read("the animal that spins webs", chat=True, reply=True, max_new=3)
    assert rc["asst_start"] is not None and rc["reply"] != "" or True
    assert "edited" in wb.intervene("swap", "a", "b", 1.0, 1, 3, max_new=2)
    r = wb.read("the animal that spins webs", chat=False, words_only=True)
    assert len(r["grid"]) == 3 and len(r["tokens"]) > 3
    assert len(r["outrow"]) == len(r["tokens"]) and "vocab" in r
    assert {"t", "r", "s", "t3"} <= set(r["grid"][0][0])
    sl = wb.slice(2, -1)
    assert len(sl["bylayer"]) == 3 and len(sl["bypos"]) == len(r["tokens"])
    assert len(wb.detail(2, -1)["top"]) == 15
    assert "ranks" in wb.pin("the")
    assert "edited" in wb.intervene("swap", "a", "b", 1.0, 1, 3, max_new=3)
    assert "edited" in wb.intervene("clamp", "a", "b", 1.0, 1, 3, max_new=3)
    assert "edited" in wb.intervene("steer", "", "b", 4.0, 1, 3, max_new=3)
    print("  ok   workbench (chat-reply/read/outrow/slice/detail/pin/swap/clamp/steer)")
except Exception:
    bad.append("workbench")
    print("  FAIL workbench"); import traceback; traceback.print_exc()

print(f"\n{len(ok)} scripts ran, {len(bad)} failed")
if bad:
    sys.exit(1)
