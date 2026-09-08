"""The Jacobian lens (paper §2.1, §2.4, §2.5; pseudocode in §A.7).

For every layer ℓ we estimate the corpus-averaged Jacobian

    J_ℓ = E_{prompt, t} [ Σ_{t' ≥ t}  ∂ z_t' / ∂ h_ℓ,t ]

where z is the residual stream at the target layer — by default the
*penultimate* one, the paper's default recipe ("omitting the last transformer
block from the backward pass").

Σ vs E over t':  §2.1 writes the aggregation over target positions as an
expectation, but the §A.7 pseudocode seeds a one-hot gradient at *every* target
position and averages only over source positions t — i.e. a sum over t'.  We
follow the pseudocode.  (The two differ: a sum weights early source positions
more, since they have more future positions to affect.)

Reading (Fig. 4B):

    lens(h_ℓ) = softmax( W_U · norm( J_ℓ h_ℓ ) )

The J-lens *vector* of vocabulary token v at layer ℓ — a direction in layer-ℓ
residual space whose inner product with h gives token v's lens logit — is

    vec_v(ℓ) = J_ℓᵀ (γ ⊙ W_U[v])

where γ is the final RMSNorm gain.  §2.1 defines these vectors as the rows of
W_U J_ℓ; we fold γ in because the model's own readout is
logits = W_U (γ ⊙ z) / rms(z), so only with γ included is ⟨vec_v, h⟩ exactly
proportional to token v's lens logit — the property §2.5 relies on when it uses
the same vectors as probes and as intervention directions.  The leftover
1 / rms(z) is a positive scalar and never changes rankings.
"""
import hashlib
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config   # noqa: E402 - must precede torch; see the note in config.py

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Computing the lens
# ---------------------------------------------------------------------------

def corpus_fingerprint(texts):
    """Stable id of an exact corpus (content AND order).  Stored inside every
    checkpoint so that resume can refuse to continue onto a different corpus."""
    return hashlib.md5("\n".join(texts).encode()).hexdigest()[:12]


def compute_jlens(model, tok, texts, layers=None, target=config.TARGET,
                  rows_per_backward=config.ROWS_PER_BACKWARD,
                  skip_first=config.SKIP_FIRST, seq_len=config.SEQ_LEN,
                  save_path=None, save_every=config.SAVE_EVERY,
                  resume=config.RESUME):
    """Estimate J_ℓ for every requested layer; returns a dict ready to torch.save.

    Implementation of the §A.7 pseudocode.  Trick used for efficiency: the
    forward pass runs on B identical copies of the prompt, so one backward pass
    (with a different one-hot output dimension per copy) yields B rows of every
    J_ℓ at once.

    With `save_path`, the running average is checkpointed atomically every
    `save_every` prompts.  A checkpoint is a complete, usable lens — just
    averaged over fewer prompts — so a run that dies at prompt 900 of 1000
    still leaves you a lens.  With `resume`, a checkpoint from the SAME model,
    target and corpus (verified by fingerprint) is picked up where it left
    off; a mismatched checkpoint raises instead of being silently mixed in.
    """
    d = model.config.hidden_size
    L = model.config.num_hidden_layers
    tgt = L - 1 if target == "penultimate" else L
    if layers is None:
        layers = list(range(tgt + 1))
    # hidden_states[i] is the residual at layer i only for i <= L-1: transformers
    # replaces the last entry with the *normalized* final residual.  So source
    # layers stop at L-1, and a "final" target is read from a hook on the final
    # norm, whose input is the raw final residual.
    layers = [l for l in layers if l <= min(tgt, L - 1)]

    # Drop prompts too short to average over (< 32 tokens) BEFORE the
    # fingerprint/resume logic: list position must always equal prompts done,
    # or an interrupted run would resume against the wrong texts.
    texts = [t for t in texts
             if tok(t, return_tensors="pt", truncation=True,
                    max_length=seq_len).input_ids.shape[1] >= max(32, skip_first + 2)]
    fp = corpus_fingerprint(texts)

    raw_final = {}
    hook = model.model.norm.register_forward_pre_hook(
        lambda mod, args: raw_final.__setitem__("z", args[0]))

    J = {l: torch.zeros(d, d) for l in layers}  # float32 accumulators on CPU
    n_used = 0

    # Resume: a checkpoint stores the *average*, so multiply back out to get the
    # running sum and skip the prompts already folded in (the corpus is cached
    # on disk, so `texts` is the same list in the same order across runs — and
    # the fingerprint check makes that assumption safe rather than hopeful).
    if resume and save_path and os.path.exists(save_path):
        ck = torch.load(save_path, map_location="cpu", weights_only=True)
        compatible = (ck["model"] == config.MODEL_NAME and ck["target"] == target
                      and list(ck["layers"]) == layers
                      and ck.get("corpus") == fp
                      and ck.get("skip_first") == skip_first
                      and ck.get("seq_len") == seq_len)
        if compatible and ck["n_prompts"] >= len(texts):
            print(f"[resume] {save_path} already covers {ck['n_prompts']} "
                  f"prompts; delete it to recompute")
            return ck
        if compatible:
            n_used = ck["n_prompts"]
            J = {l: ck["J"][l].float() * n_used for l in layers}
            texts = texts[n_used:]
            print(f"[resume] continuing from {n_used} prompts already averaged; "
                  f"{len(texts)} to go")
        else:
            raise RuntimeError(
                f"{save_path} exists but was built from a different "
                f"model/target/corpus, so it cannot be resumed safely. "
                f"Move or delete that file, or set RESUME=false to overwrite it.")

    def pack(n):
        return {"model": config.MODEL_NAME, "target": target, "target_layer": tgt,
                "layers": layers, "n_prompts": n, "corpus": fp,
                "skip_first": skip_first, "seq_len": seq_len,
                "J": {l: J[l] / n for l in layers}}

    try:
        for text in tqdm(texts, desc="J-lens prompts"):
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=seq_len).input_ids
            B = rows_per_backward
            batch = ids.repeat(B, 1).to(model.device)
            with torch.enable_grad():
                # run the bare decoder: the LM head's [B, T, vocab] logits are
                # never needed here (the official reference does the same)
                out = model.model(batch, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states      # hs[0]=embeddings, hs[i]=block-i output
                z = hs[tgt] if tgt < L else raw_final["z"]
                # One-hot cotangent in dim i at every VALID target position:
                # positions [skip_first, T-1), matching the official reference
                # (anthropics/jacobian-lens): the first positions are attention
                # sinks with atypical statistics, and the final position has no
                # next-token target.  The source mean uses the same window.
                s = z[:, skip_first:-1].sum(dim=1)          # [B, d]
                sources = [hs[l] for l in layers]
                for start in range(0, d, B):
                    rows = torch.arange(start, min(start + B, d))
                    loss = s[torch.arange(len(rows)), rows.to(s.device)].sum()
                    grads = torch.autograd.grad(loss, sources,
                                                retain_graph=start + B < d)
                    for l, g in zip(layers, grads):          # g: [B, T, d]
                        gm = g[:len(rows), skip_first:-1].float().mean(dim=1)  # mean over t
                        J[l][rows] += gm.cpu()
            del out, hs, z, s, sources
            n_used += 1
            if save_path and n_used % save_every == 0:
                # write-then-rename, so Ctrl-C can never leave a half-written file
                torch.save(pack(n_used), save_path + ".tmp")
                os.replace(save_path + ".tmp", save_path)
    finally:
        hook.remove()

    assert n_used > 0, "no usable corpus prompts"
    return pack(n_used)


# ---------------------------------------------------------------------------
# Reading with the lens
# ---------------------------------------------------------------------------

class JLens:
    """Read / vector / probe interface over precomputed J matrices (§2.5).

    Every readout is computed in float32: the vocabulary is ~150k tokens, and
    rankings among near-ties are not reliable in the model's bf16.
    """

    def __init__(self, model, tok, path=config.JLENS_PATH):
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.model, self.tok = model, tok
        self.target_layer = data["target_layer"]
        self.n_prompts = data.get("n_prompts")
        self.path = str(path)
        self.J = {l: m.to(model.device).float() for l, m in data["J"].items()}
        self.W_U = model.get_output_embeddings().weight        # [vocab, d]
        self.gamma = model.model.norm.weight.float()           # final RMSNorm gain
        self.eps = model.model.norm.variance_epsilon
        self.vocab = self.W_U.shape[0]
        self._norms = {}

    # ---- reading ---------------------------------------------------------
    @torch.no_grad()
    def propagate(self, H, layer):
        """Map layer-ℓ residual vectors into target-layer coordinates: J_ℓ h.
        H is [d] or [n, d].  For layers at/after the Jacobian target J = I (this
        is the logit lens, which the paper shows agrees with the J-lens there)."""
        H = H.float().to(self.model.device)
        return H @ self.J[layer].T if layer in self.J else H

    @torch.no_grad()
    def _dict_scores(self, Z, chunk=16384):
        """(γ ⊙ W_U) Z for target-layer vectors Z, without ever materializing
        the [vocab × d] dictionary.  Z: [d] or [n, d] -> [vocab] or [n, vocab]."""
        flat = Z.dim() == 1
        Zg = (Z.unsqueeze(0) if flat else Z) * self.gamma
        out = torch.empty(Zg.shape[0], self.vocab, device=Zg.device)
        for i in range(0, self.vocab, chunk):
            out[:, i:i + chunk] = Zg @ self.W_U[i:i + chunk].float().T
        return out[0] if flat else out

    def scores(self, H, layer):
        """Single-token probe form (§2.5): ⟨vec_v, h⟩ for every vocabulary token
        v at once.  Proportional to `logits`, so it ranks identically."""
        return self._dict_scores(self.propagate(H, layer))

    def logits(self, H, layer):
        """Full lens readout (Fig. 4B): W_U · norm(J_ℓ h), a score per vocab
        token, using the model's own final RMSNorm and unembedding."""
        Z = self.propagate(H, layer)
        rms = Z.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self._dict_scores(Z) / rms

    def cosine(self, H, layer):
        """Length-normalised readout: cos(vec_v, h) for every vocabulary token.

        §2.5 offers this alongside the raw score as the probe form.  It matters
        in practice because a handful of rare tokens have J-lens vectors 2-3x
        longer than the median, so they top the raw ranking on magnitude rather
        than on direction and crowd out the real content.
        """
        H = H.float().to(self.model.device)
        norms = self.atom_norms(layer).clamp_min(1e-8)
        return self.scores(H, layer) / norms / H.norm(dim=-1, keepdim=True)

    def readout(self, h, layer, k=10, cosine=False):
        """Top-k lens tokens for one activation: [(token_str, token_id, score)]."""
        scores = self.cosine(h, layer) if cosine else self.logits(h, layer)
        vals, idx = scores.topk(k)
        return [(self.tok.decode([i]), i, v)
                for i, v in zip(idx.tolist(), vals.tolist())]

    def rank(self, h, layer, token_id):
        """Rank (1 = top) of a token in the lens readout of activation h."""
        lg = self.scores(h, layer)
        return int((lg > lg[token_id]).sum().item()) + 1

    # ---- vectors ---------------------------------------------------------
    @torch.no_grad()
    def vectors(self, token_ids, layer, unit=False):
        """The J-lens vectors of several tokens at `layer` (§2.1): [n, d]."""
        ids = torch.as_tensor(token_ids, device=self.W_U.device)
        W = self.gamma * self.W_U[ids].float()
        V = W @ self.J[layer] if layer in self.J else W        # rows of (γ⊙W_U)J
        return V / V.norm(dim=-1, keepdim=True) if unit else V

    def vector(self, token_id, layer, unit=False):
        """The J-lens vector of a single token at `layer`: [d]."""
        return self.vectors([token_id], layer, unit)[0]

    @torch.no_grad()
    def atom_norms(self, layer, chunk=16384):
        """‖vec_v‖ for every vocab token (cached per layer). Used to normalize
        dictionary atoms for sparse decomposition."""
        if layer not in self._norms:
            out = torch.empty(self.vocab, device=self.model.device)
            for i in range(0, self.vocab, chunk):
                ids = torch.arange(i, min(i + chunk, self.vocab))
                out[i:i + chunk] = self.vectors(ids, layer).norm(dim=-1)
            self._norms[layer] = out
        return self._norms[layer]


@torch.no_grad()
def logit_lens_logits(model, h):
    """Baseline logit lens (§2.4): unembed the raw residual stream (J = I)."""
    return model.lm_head(model.model.norm(h.to(model.device, model.dtype))).float()
