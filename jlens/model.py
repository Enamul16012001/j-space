"""Model loading, chat prompting, and small token helpers for Qwen3."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config   # noqa: E402 - must precede torch; see the note in config.py

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(dtype=None):
    """Load the model in `dtype` (default: config.DTYPE).

    Step 00 passes config.LENS_DTYPE (float32) because the Jacobian is an
    average of gradients propagated through every layer; everything else runs
    in bf16, where only the forward pass matters.
    """
    tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=dtype or config.DTYPE,
        attn_implementation="sdpa",
    ).to(config.DEVICE)
    model.eval()
    return model, tok


def chat_ids(tok, user, prefill=""):
    """Token ids for a chat prompt, with an optional assistant prefill.

    Qwen3 has a "thinking" mode; we disable it (enable_thinking=False) because,
    as in the paper, we study the model's *silent* computation, not its
    chain-of-thought.
    """
    messages = [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:  # tokenizer without the enable_thinking kwarg
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    return tok(text + prefill, return_tensors="pt", add_special_tokens=False).input_ids


def single_token_id(tok, word):
    """Vocabulary id of `word` as a single token (leading-space form preferred).

    The J-lens can only name concepts that are single tokens (paper §9.1), so
    experiment scripts check this up front.
    """
    for w in (" " + word, word):
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"{word!r} is not a single token in this vocabulary; "
                     f"pick a different word")
