"""The J-space (paper §2.3): activations expressible as sparse nonnegative
combinations of J-lens vectors, identified by gradient pursuit (Blumensath &
Davies 2008) with a nonnegativity clamp (§2.5, §4.2).  Atom correlations are
computed as (γ⊙W_U)(J r) / ‖atom‖ so the [vocab × d] dictionary is never
materialized."""
import torch

CHUNK = 16384


@torch.no_grad()
def sparse_decompose(lens, h, layer, k=25, return_curve=False):
    """Nonnegative gradient pursuit: pick the most-correlated atom, take one
    line-searched gradient step on ½‖h − Vc‖² over the support, clamp c ≥ 0.
    Returns (token_ids, coeffs, residual), plus the per-step squared-residual
    curve if return_curve=True."""
    h = h.float().to(lens.model.device)
    r = h.clone()
    norms = lens.atom_norms(layer).clamp_min(1e-8)
    banned = torch.zeros(lens.vocab, dtype=torch.bool, device=h.device)
    support, atoms = [], []
    c = h.new_zeros(0)
    curve = [r.pow(2).sum().item()]
    for _ in range(k):
        corr = lens.scores(r, layer) / norms          # ⟨v̂_v, r⟩ for unit atoms
        corr[banned] = float("-inf")
        t = int(corr.argmax())
        if corr[t] <= 0:          # nonnegativity: stop when no atom correlates
            break
        banned[t] = True
        support.append(t)
        atoms.append(lens.vector(t, layer, unit=True))
        V = torch.stack(atoms, dim=1)                 # [d, m]
        c = torch.cat([c, c.new_zeros(1)])
        g = V.T @ r                                   # −∇_c ½‖h−Vc‖² = Vᵀr
        Vg = V @ g
        step = torch.dot(r, Vg) / Vg.pow(2).sum().clamp_min(1e-12)
        c = (c + step * g).clamp_min(0)               # nonnegative coefficients
        r = h - V @ c
        curve.append(r.pow(2).sum().item())

    if return_curve:
        return support, c, r, curve
    return support, c, r


@torch.no_grad()
def jspace_split(lens, vec, layer, k=16):
    """Split a vector into its J-space component and non-J-space remainder
    (§3.1, Fig. 8: the J-space part carries the causal effect on verbal report;
    the paper uses k = 16 here).

    Returns (component, remainder) with vec = component + remainder.
    """
    _, _, r = sparse_decompose(lens, vec, layer, k=k)
    return vec.float().to(r.device) - r, r


# ---------------------------------------------------------------------------
# Random-direction control (for occupancy, §4.2)
# ---------------------------------------------------------------------------

class RandomDict:
    """Fixed dictionary of random unit directions — the §4.2 occupancy
    control, vocabulary-sized by default (~750 MB fp16 on GPU)."""

    def __init__(self, n_atoms, d, device, seed=0):
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(n_atoms, d, generator=g)
        A = A / A.norm(dim=1, keepdim=True)
        self.A = A.to(device, torch.float16)     # stored fp16, used in fp32


@torch.no_grad()
def random_pursuit_curve(rand, h, k_max):
    """The same nonnegative gradient pursuit, over the random dictionary.
    Returns the per-step squared-residual curve."""
    A = rand.A
    h = h.float().to(A.device)
    r = h.clone()
    banned, atoms = set(), []
    c = h.new_zeros(0)
    curve = [r.pow(2).sum().item()]
    for _ in range(k_max):
        corr = torch.empty(A.shape[0], device=r.device)
        for i in range(0, A.shape[0], CHUNK):     # fp32, as for the J-lens atoms
            corr[i:i + CHUNK] = A[i:i + CHUNK].float() @ r
        if banned:
            corr[list(banned)] = float("-inf")
        t = int(corr.argmax())
        if corr[t] <= 0:
            break
        banned.add(t)
        atoms.append(A[t].float())
        V = torch.stack(atoms, dim=1)
        c = torch.cat([c, c.new_zeros(1)])
        g = V.T @ r
        Vg = V @ g
        step = torch.dot(r, Vg) / Vg.pow(2).sum().clamp_min(1e-12)
        c = (c + step * g).clamp_min(0)
        r = h - V @ c
        curve.append(r.pow(2).sum().item())
    return curve


def occupancy(curve_j, curve_r):
    """Occupancy (§4.2): the number of J-lens atoms K at which the marginal
    reconstruction improvement first falls below that of the same-size random
    control."""
    n = min(len(curve_j), len(curve_r)) - 1
    for i in range(n):
        dj = curve_j[i] - curve_j[i + 1]
        dr = curve_r[i] - curve_r[i + 1]
        if dj < dr:
            return i          # i atoms were "real"; the (i+1)-th was not
    return n
