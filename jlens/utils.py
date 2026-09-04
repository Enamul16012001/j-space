"""Small helpers shared by the experiment scripts."""
import torch


def top_str(tok, logits, k=5):
    """'token (p=…), token (p=…), …' for the top-k of a logit vector."""
    probs = torch.softmax(logits, dim=-1)
    vals, idx = probs.topk(k)
    return ", ".join(f"{tok.decode([i])!r} ({v:.2f})"
                     for i, v in zip(idx.tolist(), vals.tolist()))


def best_rank(lens, hidden_states, token_id, layers, pos):
    """Best (lowest) lens rank of `token_id` at position `pos`, over `layers`.
    Returns (rank, layer)."""
    best = (10**9, None)
    for l in layers:
        r = lens.rank(hidden_states[l][0, pos], l, token_id)
        if r < best[0]:
            best = (r, l)
    return best

