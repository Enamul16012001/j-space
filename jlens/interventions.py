"""Causal interventions in J-lens coordinates (paper §2.5).

    Steer            h <- h + α·v̂                       ("add a concept")
    ProjectOut       h <- h − proj_span(V) h             ("remove a concept")
    Swap             swap two lens coordinates (Fig. 4C) ("replace a concept")
    TopKJSpaceAblate remove the top-k active lens vectors per position (§3.5.2)

Edits are installed with the `apply_edits` context manager, which hooks the
transformer blocks.  Residual-stream index convention (see config.py):
layer i = output of block i, so an edit at layer i hooks model.model.layers[i-1].

Positions are *absolute* token positions in the full sequence (prompt +
generated).  A small hook on the embedding layer keeps track of which absolute
positions each forward pass covers, so edits keep working during generation
with a KV cache (where each step only sees one new token).
"""
import contextlib

import torch


# ---------------------------------------------------------------------------
# Edit types
# ---------------------------------------------------------------------------

class Edit:
    """Base class.  `layers`: residual layers to act on.  `positions`: "all"
    or an iterable of absolute token positions."""

    def __init__(self, layers, positions="all"):
        self.layers = set(layers)
        self.positions = positions if positions == "all" else set(positions)

    def local_idx(self, span, device):
        """Indices *within the current forward pass* that this edit touches.
        span = (start, end) absolute positions covered by the pass."""
        start, end = span
        if self.positions == "all":
            return torch.arange(end - start, device=device)
        hit = [p - start for p in self.positions if start <= p < end]
        return torch.tensor(hit, dtype=torch.long, device=device)

    def apply(self, h, layer, abs_pos):
        """h: [n, d] float32 activations at the selected positions.
        abs_pos: [n] their absolute positions.  Returns the edited h."""
        raise NotImplementedError


class Steer(Edit):
    """h <- h + α · v̂   (unit-normalized steering vector)."""

    def __init__(self, v, alpha, layers, positions="all"):
        super().__init__(layers, positions)
        self.v = (v / v.norm()).float()
        self.alpha = alpha

    def apply(self, h, layer, abs_pos):
        return h + self.alpha * self.v.to(h.device)


class ProjectOut(Edit):
    """Remove the span of the given vectors:  h <- h − V V⁺ h."""

    def __init__(self, vectors, layers, positions="all"):
        super().__init__(layers, positions)
        V = torch.stack([v.float() for v in vectors], dim=1)       # [d, m]
        self.P = (V @ torch.linalg.pinv(V))                        # [d, d]

    def apply(self, h, layer, abs_pos):
        P = self.P.to(h.device)
        return h - h @ P.T


class Swap(Edit):
    """Swap the coefficients of two lens vectors, in lens coordinates (Fig. 4C).

    With V = [v_source, v_target] and c = V⁺ h the (least-squares) coordinates
    of h on the two atoms, we move h by α · (σ(c) − c) expressed back in
    residual space, where σ swaps the two entries.  At α = 1 the source
    coordinate becomes the target coordinate and vice versa; everything
    orthogonal to the two vectors is untouched.
    """

    def __init__(self, v_source, v_target, alpha, layers, positions="all"):
        super().__init__(layers, positions)
        V = torch.stack([v_source.float(), v_target.float()], dim=1)  # [d, 2]
        self.V = V
        self.pinv = torch.linalg.pinv(V)                              # [2, d]
        self.alpha = alpha

    def apply(self, h, layer, abs_pos):
        V, pinv = self.V.to(h.device), self.pinv.to(h.device)
        c = h @ pinv.T                                                # [n, 2]
        delta = (c.flip(-1) - c) @ V.T                                # [n, d]
        return h + self.alpha * delta


class ClampSwap(Edit):
    """Clamped lens-coordinate swap (§3.3, Fig. 13; also Fig. 8's re-entry
    control): at every (layer, position) the activation's coordinates on
    [v_source, v_target] are SET to the swapped clean-pass values, instead of
    exchanged in place.  Pinning them keeps downstream layers from writing the
    original concept back into the stream.

    Build with `clamp_swap_edits`, which computes the clean coordinates.
    """

    def __init__(self, V, c_swapped, layers, positions):
        super().__init__(layers, positions)
        self.V = V                                   # [d, 2] unit lens vectors
        self.pinv = torch.linalg.pinv(V)             # [2, d]
        self.c = c_swapped                           # [T, 2] target coordinates

    def apply(self, h, layer, abs_pos):
        V, pinv = self.V.to(h.device), self.pinv.to(h.device)
        c_cur = h @ pinv.T                           # [n, 2]
        c_tgt = self.c.to(h.device)[abs_pos]         # [n, 2]
        return h + (c_tgt - c_cur) @ V.T


class TopKJSpaceAblate(Edit):
    """Whole-J-space ablation (§3.5.2): at every position, find the k lens
    vectors with the largest positive correlation with the activation and
    project out their span.

    `exclude_per_pos` maps an absolute position to token ids that must NOT be
    ablated there — the paper excludes each position's top-10 next-token
    predictions from the clean forward pass, so that the ablation removes
    *workspace content* rather than the model's immediate output.
    """

    def __init__(self, lens, k, layers, positions="all", exclude_per_pos=None):
        super().__init__(layers, positions)
        self.lens, self.k = lens, k
        self.exclude = exclude_per_pos or {}

    def apply(self, h, layer, abs_pos):
        lens = self.lens
        corr = lens.scores(h, layer) / lens.atom_norms(layer).clamp_min(1e-8)
        for i, p in enumerate(abs_pos.tolist()):
            for t in self.exclude.get(p, ()):
                corr[i, t] = float("-inf")
        top = corr.topk(self.k, dim=-1).indices                # [n, k]
        out = h.clone()
        for i in range(h.shape[0]):
            V = lens.vectors(top[i], layer, unit=True).T       # [d, k]
            out[i] = h[i] - V @ (torch.linalg.pinv(V) @ h[i])
        return out


# ---------------------------------------------------------------------------
# Installing edits on the model
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def apply_edits(model, edits):
    """Context manager: while active, every forward pass through `model`
    applies the edits at their layers/positions (works under generation)."""
    edits = list(edits)
    handles = []
    span = {}   # absolute (start, end) covered by the current forward pass

    def embed_hook(module, args, output):
        n = output.shape[1]
        start = span.get("end", 0)
        span["start"], span["end"] = start, start + n
        return output

    handles.append(model.model.embed_tokens.register_forward_hook(embed_hook))

    layers_needed = sorted({l for e in edits for l in e.layers})

    def make_hook(layer):
        def hook(module, args, output):
            tup = isinstance(output, tuple)
            h = output[0] if tup else output           # [B, T, d]
            cur = (span["start"], span["end"])
            for e in edits:
                if layer not in e.layers:
                    continue
                idx = e.local_idx(cur, h.device)
                if idx.numel() == 0:
                    continue
                abs_pos = idx + cur[0]
                for b in range(h.shape[0]):
                    sel = h[b, idx].float()
                    h[b, idx] = e.apply(sel, layer, abs_pos).to(h.dtype)
            return (h,) + output[1:] if tup else h
        return hook

    for l in layers_needed:
        assert l >= 1, "edits act on block outputs (residual layers >= 1)"
        handles.append(model.model.layers[l - 1].register_forward_hook(make_hook(l)))

    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


# ---------------------------------------------------------------------------
# Convenience factories (one edit per layer, since lens vectors are per-layer)
# ---------------------------------------------------------------------------

def steer_edits(lens, token_id, layers, positions="all", alpha=8.0):
    """Add the (unit) J-lens vector of `token_id` at every layer in `layers`."""
    return [Steer(lens.vector(token_id, l, unit=True), alpha, [l], positions)
            for l in layers]


def swap_edits(lens, source_id, target_id, layers, positions="all", alpha=1.0):
    """Swap the lens coordinates of source and target token at each layer."""
    return [Swap(lens.vector(source_id, l, unit=True),
                 lens.vector(target_id, l, unit=True),
                 alpha, [l], positions) for l in layers]


def clamp_swap_edits(lens, source_id, target_id, layers, hidden_states):
    """Clamped swap (§3.3): pin the source/target lens coordinates at every
    prompt position and layer to their swapped CLEAN-pass values.
    `hidden_states` is the clean forward pass (from `forward_hidden`)."""
    edits = []
    for l in layers:
        V = torch.stack([lens.vector(source_id, l, unit=True),
                         lens.vector(target_id, l, unit=True)], dim=1)
        H = hidden_states[l][0].float().to(V.device)          # [T, d]
        c_clean = H @ torch.linalg.pinv(V).T                  # [T, 2]
        edits.append(ClampSwap(V, c_clean.flip(-1), [l],
                               positions=range(H.shape[0])))
    return edits


# ---------------------------------------------------------------------------
# Running the model with (or without) edits
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tok, input_ids, edits=(), max_new_tokens=40):
    """Greedy generation with edits active; returns the newly generated text."""
    input_ids = input_ids.to(model.device)
    attn = torch.ones_like(input_ids)
    with apply_edits(model, edits):
        out = model.generate(input_ids, attention_mask=attn,
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def next_token_logits(model, input_ids, edits=()):
    """Logits for the next token after the prompt, with edits active."""
    input_ids = input_ids.to(model.device)
    with apply_edits(model, edits):
        out = model(input_ids)
    return out.logits[0, -1].float()


@torch.no_grad()
def forward_hidden(model, input_ids, edits=()):
    """One forward pass; returns (hidden_states, last-position logits).
    hidden_states[l][0, t] is the residual at layer l, position t."""
    input_ids = input_ids.to(model.device)
    with apply_edits(model, edits):
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states, out.logits[0, -1].float()
