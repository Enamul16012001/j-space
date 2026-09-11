"""Central configuration.  Edit `.env`, not this file; real environment
variables win over `.env` for one-offs like MODEL_NAME=... python ..."""
import os
import pathlib
import sysconfig

ROOT = pathlib.Path(__file__).resolve().parent


def _load_env(path=ROOT / ".env"):
    """Minimal KEY=value reader (# comments, quotes optional)."""
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

# Without Python dev headers, torch's Triton JIT dies compiling its C shim.
# The flag is read at torch-import time, so this must run BEFORE `import
# torch` — every entry point imports `config` first for this reason.
if not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")):
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import torch   # noqa: E402 - must follow the guard above

# ---------------- model ----------------
MODEL_NAME = _str("MODEL_NAME", "Qwen/Qwen3-4B")
DEVICE = _str("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = getattr(torch, _str("DTYPE", "bfloat16" if DEVICE == "cuda" else "float32"))

# ---------------- J-lens computation (paper §2.1, §A.7) ----------------
N_PROMPTS = _int("N_PROMPTS", 1000)
SEQ_LEN = _int("SEQ_LEN", 128)
CORPUS_SEED = _int("CORPUS_SEED", 0)
ROWS_PER_BACKWARD = _int("ROWS_PER_BACKWARD", 16)   # memory dial for step 00
SAVE_EVERY = _int("SAVE_EVERY", 25)
RESUME = _flag("RESUME", True)
SKIP_FIRST = _int("SKIP_FIRST", 16)
TARGET = _str("TARGET", "penultimate")      # or "final" (the §A.7 variant)

# float32 by default: the Jacobian averages gradients through every layer,
# the one place where low precision really costs accuracy.
LENS_DTYPE = getattr(torch, _str("LENS_DTYPE", "float32"))

ALLOW_TF32 = _flag("ALLOW_TF32", False)     # ~2.3x faster, 10 mantissa bits
if ALLOW_TF32:
    torch.backends.cuda.matmul.allow_tf32 = True

JLENS_PATH = _str("JLENS_PATH",
                  str(ROOT / f"jlens_{MODEL_NAME.split('/')[-1].lower()}.pt"))

# Layer indexing everywhere in this repo:
#     layer 0 = embedding output, layer i = output of transformer block i
# WORKSPACE_LAYERS (e.g. "14-33") pins a calibrated band; otherwise the
# §4.1 proportions in WORKSPACE_BAND are mapped onto the model's depth.
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
