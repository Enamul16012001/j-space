"""Central configuration.

Every knob lives in the `.env` file next to this one — edit that, not this.
Real environment variables win over `.env`, so you can also do a one-off:

    MODEL_NAME=Qwen/Qwen3-0.6B python experiments/00_compute_jlens.py
"""
import os
import pathlib
import sysconfig

ROOT = pathlib.Path(__file__).resolve().parent


def _load_env(path=ROOT / ".env"):
    """Minimal KEY=value reader (# comments, quotes optional). No dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env()

_str = lambda k, d: os.environ.get(k, d).strip()
_int = lambda k, d: int(_str(k, str(d)))
_flag = lambda k, d: _str(k, str(d)).lower() in ("1", "true", "yes", "on")

# Recent torch routes some ops (RoPE's batched matmul among them) through
# Triton, which JIT-compiles a small C shim and so needs the Python development
# headers.  Where those are missing (no python3-dev, no root) the first forward
# pass dies with "fatal error: Python.h"; fall back to torch's own kernels.
# This must happen BEFORE `import torch`, because the Triton kernels are
# registered at torch-import time -- which is why every entry point in this
# repo imports `config` before it imports torch.
if not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")):
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch   # noqa: E402 - must follow the guard above

# ---------------- model ----------------
MODEL_NAME = _str("MODEL_NAME", "Qwen/Qwen3-4B")
DEVICE = _str("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = getattr(torch, _str("DTYPE", "bfloat16" if DEVICE == "cuda" else "float32"))

# ---------------- J-lens computation (paper §2.1, §A.7) ----------------
N_PROMPTS = _int("N_PROMPTS", 1000)         # the paper averages 1000 sequences
SEQ_LEN = _int("SEQ_LEN", 128)              # tokens per averaging prompt
CORPUS_SEED = _int("CORPUS_SEED", 0)        # shuffle seed for the WikiText sample
ROWS_PER_BACKWARD = _int("ROWS_PER_BACKWARD", 16)   # memory dial for step 00
SAVE_EVERY = _int("SAVE_EVERY", 25)         # checkpoint interval, in prompts
RESUME = _flag("RESUME", True)              # continue from a compatible checkpoint
SKIP_FIRST = _int("SKIP_FIRST", 16)         # official-reference default
TARGET = _str("TARGET", "penultimate")      # or "final" (the §A.7 variant)

# Step 00 runs its backward passes in this dtype.  The Jacobian is an average of
# gradients propagated through every layer, so it is the one place where low
# precision really costs accuracy: float32 is the default.
LENS_DTYPE = getattr(torch, _str("LENS_DTYPE", "float32"))

# TF32 keeps 10 mantissa bits where float32 keeps 23, and runs ~2.3x faster on
# recent NVIDIA cards.  Still far more precise than bfloat16's 8 bits.
ALLOW_TF32 = _flag("ALLOW_TF32", False)
if ALLOW_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True

# One lens per checkpoint, so different model sizes never overwrite each other.
# Anchored to the repo root, not the working directory.
JLENS_PATH = _str("JLENS_PATH",
                  str(ROOT / f"jlens_{MODEL_NAME.split('/')[-1].lower()}.pt"))

# ---------------- residual-stream index convention ----------------------
# Everywhere in this repo:
#     layer 0 = embedding output
#     layer i = output of transformer block i
#
# The "workspace band" (§4.1): coherent J-space content lives between ~38% and
# ~92% of the model's depth.  We map those proportions onto whatever checkpoint
# is loaded.  Set WORKSPACE_LAYERS in .env (e.g. "14-33") to pin a calibrated
# band instead; run experiments/09_workspace_layers.py to find it.
WORKSPACE_BAND = tuple(float(x) for x in _str("WORKSPACE_BAND", "0.38,0.92").split(","))

_band = _str("WORKSPACE_LAYERS", "auto")
WORKSPACE_LAYERS = None
if _band.lower() not in ("", "auto", "none"):
    _lo, _hi = (int(x) for x in _band.split("-"))
    WORKSPACE_LAYERS = list(range(_lo, _hi + 1))


def workspace_layers(model):
    """The workspace layer band for a loaded model (§4.1)."""
    if WORKSPACE_LAYERS is not None:
        return list(WORKSPACE_LAYERS)
    n = model.config.num_hidden_layers
    lo, hi = WORKSPACE_BAND
    return list(range(round(lo * n), round(hi * n) + 1))
