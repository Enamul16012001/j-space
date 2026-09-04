"""Behavioural tests for the J-lens/J-space maths, on a tiny random Qwen3.

Runs on CPU in seconds, needs no model download.  The headline check is that
compute_jlens reproduces a naive one-output-dimension-at-a-time reference
Jacobian exactly.

    python tests/test_math.py
"""
import sys, types, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config   # noqa: E402 - must precede torch; see the note in config.py

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
config.JLENS_PATH = tempfile.mkstemp(suffix=".pt")[1]

from jlens.lens import compute_jlens, corpus_fingerprint, JLens
from jlens.jspace import sparse_decompose, jspace_split, RandomDict, random_pursuit_curve, occupancy
from jlens.interventions import (Steer, ProjectOut, Swap, ClampSwap, TopKJSpaceAblate,
                                 apply_edits, clamp_swap_edits,
                                 forward_hidden, generate)

OK, FAIL = [], []
def check(name, cond, extra=""):
    (OK if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + ("  " + extra if extra else ""))

torch.manual_seed(0)
D, L, T, V = 32, 4, 40, 64
cfg = Qwen3Config(vocab_size=V, hidden_size=D, intermediate_size=64,
                  num_hidden_layers=L, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                  tie_word_embeddings=True)
model = Qwen3ForCausalLM(cfg).eval().float()
# make the final RMSNorm gain non-trivial, so folding gamma in actually matters
with torch.no_grad():
    model.model.norm.weight.normal_(1.0, 0.3)

class Tok:
    eos_token_id = 0
    def __call__(self, text, return_tensors=None, truncation=False,
                 max_length=None, add_special_tokens=True):
        ids = [(ord(c) % (V - 1)) + 1 for c in text][:max_length or 10**9]
        return types.SimpleNamespace(input_ids=torch.tensor([ids]))
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + (int(i) % 26)) for i in ids)
tok = Tok()

TEXT = "x" * T
ids = tok(TEXT).input_ids

# ---------------------------------------------------------------- compute_jlens
print("\n[1] compute_jlens vs a naive one-output-dim-at-a-time reference")
data = compute_jlens(model, tok, [TEXT], rows_per_backward=5, skip_first=0, seq_len=T)
check("target_layer == L-1 for 'penultimate'", data["target_layer"] == L - 1,
      f"got {data['target_layer']}")
check("source layers are 0..L-1", data["layers"] == list(range(L)),
      f"got {data['layers']}")

tgt = L - 1
for l in (0, 2, L - 1):
    J_ref = torch.zeros(D, D)
    for i in range(D):
        out = model(ids, output_hidden_states=True, use_cache=False)
        s = out.hidden_states[tgt].sum(dim=1)[0, i]
        g, = torch.autograd.grad(s, out.hidden_states[l])
        J_ref[i] = g[0].mean(dim=0)
    err = (data["J"][l] - J_ref).abs().max().item()
    check(f"J[{l}] matches naive reference (max abs err {err:.2e})", err < 1e-4)

check("J[target] is the identity",
      torch.allclose(data["J"][tgt], torch.eye(D), atol=1e-4),
      f"max dev {(data['J'][tgt] - torch.eye(D)).abs().max():.2e}")

print("\n[2] target='final' hits the RAW (pre-norm) final residual")
dfin = compute_jlens(model, tok, [TEXT], target="final", rows_per_backward=5,
                     skip_first=0, seq_len=T)
check("source layers capped at L-1 (hs[L] is post-norm)",
      dfin["layers"] == list(range(L)), f"got {dfin['layers']}")
grab = {}
h = model.model.norm.register_forward_pre_hook(lambda m, a: grab.__setitem__("z", a[0]))
o = model(ids, output_hidden_states=True, use_cache=False)
h.remove()
check("norm pre-hook input is pre-norm final residual",
      torch.allclose(model.model.norm(grab["z"]), o.hidden_states[-1], atol=1e-5))
J_ref = torch.zeros(D, D)
for i in range(D):
    grab2 = {}
    hh = model.model.norm.register_forward_pre_hook(lambda m, a: grab2.__setitem__("z", a[0]))
    out = model(ids, output_hidden_states=True, use_cache=False)
    hh.remove()
    s = grab2["z"].sum(dim=1)[0, i]
    g, = torch.autograd.grad(s, out.hidden_states[2])
    J_ref[i] = g[0].mean(dim=0)
err = (dfin["J"][2] - J_ref).abs().max().item()
check(f"final-target J[2] matches reference (max abs err {err:.2e})", err < 1e-4)
check("penultimate and final targets differ",
      not torch.allclose(dfin["J"][2], data["J"][2], atol=1e-3))

print("\n[2b] resume: an interrupted run continues to an identical result")
T4 = [chr(97 + i) * T for i in range(4)]              # 4 distinct prompts
ref4 = compute_jlens(model, tok, T4, rows_per_backward=5, skip_first=0,
                     seq_len=T, resume=False)
pth = tempfile.mktemp(suffix=".pt")
part = compute_jlens(model, tok, T4[:2], rows_per_backward=5, skip_first=0,
                     seq_len=T, resume=False)          # the first 2 prompts
part["corpus"] = corpus_fingerprint(T4)                # forge "stopped at 2 of 4"
torch.save(part, pth)
done = compute_jlens(model, tok, T4, rows_per_backward=5, skip_first=0,
                     seq_len=T, save_path=pth, resume=True)
err4 = max((done["J"][l] - ref4["J"][l]).abs().max().item() for l in ref4["layers"])
check(f"resumed run == uninterrupted run (max err {err4:.1e})",
      done["n_prompts"] == 4 and err4 < 1e-5)
part["corpus"] = "not-this-corpus"
torch.save(part, pth)
try:
    compute_jlens(model, tok, T4, rows_per_backward=5, skip_first=0,
                  seq_len=T, save_path=pth, resume=True)
    check("checkpoint from a different corpus refuses to resume", False)
except RuntimeError:
    check("checkpoint from a different corpus refuses to resume", True)
import os as _os; _os.remove(pth)

torch.save(data, config.JLENS_PATH)

# ------------------------------------------------------------------------ JLens
print("\n[3] JLens readout == the paper's W_U . norm(J h)")
lens = JLens(model, tok, config.JLENS_PATH)
hs, _ = forward_hidden(model, ids)
hv = hs[2][0, 5].float()

ref = model.lm_head(model.model.norm(lens.propagate(hv, 2))).float()
got = lens.logits(hv, 2)
check("logits == lm_head(norm(J h))", torch.allclose(got, ref, atol=1e-3),
      f"max err {(got-ref).abs().max():.2e}")

sc = lens.scores(hv, 2)
ratio = (got / sc)
check("scores is logits up to one positive scalar",
      (ratio.max() - ratio.min()).abs() < 1e-3 and ratio.mean() > 0,
      f"spread {(ratio.max()-ratio.min()).abs():.2e}")

vv = lens.vectors(torch.arange(V), 2)
check("scores[v] == <vector_v, h> for all v",
      torch.allclose(sc, vv @ hv, atol=1e-3),
      f"max err {(sc - vv @ hv).abs().max():.2e}")
check("vector() == vectors()[0]",
      torch.allclose(lens.vector(7, 2), lens.vectors([7], 2)[0]))
check("atom_norms == ||vectors||",
      torch.allclose(lens.atom_norms(2), vv.norm(dim=-1), atol=1e-4))
check("unit vectors have norm 1",
      torch.allclose(lens.vectors([3, 9], 2, unit=True).norm(dim=-1),
                     torch.ones(2), atol=1e-5))

print("\n[4] batched vs single-vector readout")
H = hs[2][0].float()                       # [T, d]
Lg = lens.logits(H, 2)
check("logits([T,d]) rows == logits(h) each",
      all(torch.allclose(Lg[t], lens.logits(H[t], 2), atol=1e-4) for t in range(T)))
Sc = lens.scores(H, 2)
check("scores([T,d]) rows == scores(h) each",
      all(torch.allclose(Sc[t], lens.scores(H[t], 2), atol=1e-4) for t in range(T)))
check("rank() agrees with an explicit sort",
      lens.rank(hv, 2, 11) == 1 + int((sc > sc[11]).sum()))

print("\n[5] gamma really is folded in (not a no-op)")
naive = lens.J[2].T @ model.get_output_embeddings().weight[7].float()
check("vector != J^T W_U[v] (gamma matters)",
      not torch.allclose(lens.vector(7, 2), naive, atol=1e-4))

# ----------------------------------------------------------------------- J-space
print("\n[6] sparse decomposition (nonnegative gradient pursuit)")
sup, c, r, curve = sparse_decompose(lens, hv, 2, k=8, return_curve=True)
check("coefficients are nonnegative", bool((c >= 0).all()))
check("no atom selected twice", len(set(sup)) == len(sup))
check("residual curve is nonincreasing",
      all(curve[i] >= curve[i + 1] - 1e-6 for i in range(len(curve) - 1)))
Vsup = lens.vectors(sup, 2, unit=True).T
check("h == V c + r", torch.allclose(hv, Vsup @ c + r, atol=1e-4),
      f"max err {(hv - (Vsup @ c + r)).abs().max():.2e}")
check("curve[-1] == ||r||^2", abs(curve[-1] - r.pow(2).sum().item()) < 1e-4)

comp, rem = jspace_split(lens, hv, 2, k=6)
check("jspace_split: component + remainder == vec",
      torch.allclose(comp + rem, hv, atol=1e-5))

rand = RandomDict(200, D, "cpu")
cr = random_pursuit_curve(rand, hv, k_max=8)
check("random pursuit curve is nonincreasing",
      all(cr[i] >= cr[i + 1] - 1e-6 for i in range(len(cr) - 1)))
K = occupancy(curve, cr)
check("occupancy in range", 0 <= K <= 8, f"K={K}")

# ------------------------------------------------------------------ interventions
print("\n[7] Swap matches h + a*V(sigma(c) - c)  (paper 2.5)")
vs, vt = lens.vector(3, 2, unit=True), lens.vector(9, 2, unit=True)
sw = Swap(vs, vt, 1.0, [2])
h1 = sw.apply(hv.unsqueeze(0), 2, torch.tensor([0]))[0]
Vm = torch.stack([vs, vt], dim=1)
cc = torch.linalg.pinv(Vm) @ hv
ref = hv + Vm @ (cc.flip(0) - cc)
check("swap formula", torch.allclose(h1, ref, atol=1e-5))
perp = torch.linalg.svd(Vm.T, full_matrices=True)[2][2:]       # basis of the complement
check("component orthogonal to span{vs,vt} unchanged",
      torch.allclose(perp @ h1, perp @ hv, atol=1e-5))
c_new = torch.linalg.pinv(Vm) @ h1
check("lens coordinates are exchanged",
      torch.allclose(c_new, cc.flip(0), atol=1e-4), f"{cc.tolist()} -> {c_new.tolist()}")

print("\n[7b] ClampSwap pins coordinates to swapped clean values (3.3)")
cedits = clamp_swap_edits(lens, 3, 9, [2], hs)
ce = [e for e in cedits if 2 in e.layers][0]
hc = ce.apply(hv.unsqueeze(0), 2, torch.tensor([5]))[0]
c_clean = torch.linalg.pinv(Vm) @ hv
c_after = torch.linalg.pinv(Vm) @ hc
check("clamped coords equal swapped clean coords",
      torch.allclose(c_after, c_clean.flip(0), atol=1e-4))
check("clamp on an ALREADY-EDITED h still lands on clean-swapped coords",
      torch.allclose(torch.linalg.pinv(Vm) @ ce.apply((hv + 3 * vs).unsqueeze(0), 2,
                                                      torch.tensor([5]))[0],
                     c_clean.flip(0), atol=1e-4))
check("component orthogonal to the pair unchanged by clamp",
      torch.allclose(perp @ hc, perp @ hv, atol=1e-5))

print("\n[8] ProjectOut / TopKJSpaceAblate remove exactly the right span")
po = ProjectOut([vs, vt], [2])
h2 = po.apply(hv.unsqueeze(0), 2, torch.tensor([0]))[0]
check("result is orthogonal to both vectors",
      abs(torch.dot(h2, vs)) < 1e-4 and abs(torch.dot(h2, vt)) < 1e-4)

abl = TopKJSpaceAblate(lens, k=3, layers=[2], exclude_per_pos={0: [5]})
Hin = hs[2][0, :3].float()
Hout = abl.apply(Hin, 2, torch.tensor([0, 1, 2]))
corr = lens.scores(Hin, 2) / lens.atom_norms(2).clamp_min(1e-8)
corr[0, 5] = float("-inf")
top = corr.topk(3, dim=-1).indices
check("excluded token is never ablated at its position", 5 not in top[0].tolist())
resid_ok = True
for i in range(3):
    Vi = lens.vectors(top[i], 2, unit=True)
    if (Vi @ Hout[i]).abs().max() > 1e-3:
        resid_ok = False
check("ablated activation is orthogonal to its top-k atoms", resid_ok)

print("\n[9] apply_edits: layer and absolute-position bookkeeping")
big = Steer(torch.ones(D), 50.0, [2], positions={4})
hs2, _ = forward_hidden(model, ids, [big])
check("edit at layer 2 leaves layer 2's input (layer 1) untouched",
      torch.allclose(hs2[1], hs[1], atol=1e-4))
check("edit changes layer 2 at position 4",
      (hs2[2][0, 4] - hs[2][0, 4]).abs().max() > 1.0)
check("edit leaves layer 2 at other positions untouched",
      torch.allclose(hs2[2][0, :4], hs[2][0, :4], atol=1e-4))

seen = []
class Recorder(Steer):
    def apply(self, h, layer, abs_pos):
        seen.extend(abs_pos.tolist())
        return h
rec = Recorder(torch.zeros(D), 0.0, [2], positions={T, T + 1})
generate(model, tok, ids, [rec], max_new_tokens=3)
check("edit fires at the right absolute positions during generation",
      sorted(seen) == [T, T + 1], f"saw {sorted(seen)}")

print(f"\n{len(OK)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
