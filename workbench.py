"""Interactive J-lens workbench: read the residual stream in the browser,
pin tokens to see their rank across every (layer, position) cell, and apply
live interventions (clamped swap / swap / steer) to watch the model's answer
change.

    python workbench.py            # then open http://localhost:7860
    python workbench.py --port 7861

Uses only the Python standard library for the server; the page is plain HTML
and vanilla JS served from workbench.html next to this file.  Everything runs
through this repo's own jlens package.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import config

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from jlens import (load_model, chat_ids, JLens, forward_hidden, generate,
                   next_token_logits, swap_edits, clamp_swap_edits, steer_edits)

ROOT = pathlib.Path(__file__).resolve().parent


class Workbench:
    """Holds the model, the lens, and the last prompt's activations."""

    def __init__(self):
        print("Loading model and lens…", flush=True)
        self.model, self.tok = load_model()
        self.lens = JLens(self.model, self.tok)
        self.band = config.workspace_layers(self.model)
        self.layers = list(range(2, self.lens.target_layer + 1))
        self.lock = threading.Lock()        # one GPU, one request at a time
        self.ids = None                     # last prompt's token ids
        self.hs = None                      # and hidden states

    # ---- reading -----------------------------------------------------------
    def read(self, prompt, chat=False, cosine=True):
        ids = (chat_ids(self.tok, prompt) if chat
               else self.tok(prompt, return_tensors="pt", truncation=True,
                             max_length=192).input_ids)
        hs, _ = forward_hidden(self.model, ids)
        self.ids, self.hs = ids, hs

        tokens = [self.tok.decode([t]) for t in ids[0].tolist()]
        grid, scores = [], []
        for l in self.layers:
            S = (self.lens.cosine(hs[l][0], l) if cosine
                 else self.lens.logits(hs[l][0], l))
            top3 = S.topk(3, dim=-1)
            grid.append([[self.tok.decode([i]) for i in row.tolist()]
                         for row in top3.indices])
            scores.append(top3.values[:, 0].tolist())
        return {"model": config.MODEL_NAME, "tokens": tokens,
                "layers": self.layers, "band": [self.band[0], self.band[-1]],
                "grid": grid, "scores": scores}

    def detail(self, layer, pos, cosine=True, k=15):
        h = self.hs[layer][0, pos]
        S = self.lens.cosine(h, layer) if cosine else self.lens.logits(h, layer)
        vals, idx = S.topk(k)
        return {"top": [{"tok": self.tok.decode([i]), "score": round(v, 4)}
                        for i, v in zip(idx.tolist(), vals.tolist())]}

    def pin(self, word):
        """Rank of `word` (best single-token spelling) at every cell."""
        variants = []
        for f in (word, " " + word):
            e = self.tok.encode(f, add_special_tokens=False)
            if len(e) == 1:
                variants.append(e[0])
        if not variants:
            return {"error": f"{word!r} is not a single token"}
        ranks = []
        for l in self.layers:
            S = self.lens.scores(self.hs[l][0], l)                # [T, vocab]
            r = None
            for t in variants:
                rt = (S > S[:, t].unsqueeze(1)).sum(-1) + 1
                r = rt if r is None else torch.minimum(r, rt)
            ranks.append(r.tolist())
        return {"word": word, "ranks": ranks}

    # ---- intervening -------------------------------------------------------
    def _resolve(self, text):
        """Token id for a grid token (exact) or a typed word (spacing variants)."""
        for f in (text, " " + text.strip(), text.strip()):
            e = self.tok.encode(f, add_special_tokens=False)
            if len(e) == 1:
                return e[0]
        raise ValueError(f"{text!r} is not a single token")

    def intervene(self, kind, source, target, alpha, lo, hi, max_new=24):
        band = [l for l in range(int(lo), int(hi) + 1)
                if 1 <= l <= self.lens.target_layer]
        tgt = self._resolve(target)
        if kind == "steer":
            edits = steer_edits(self.lens, tgt, band, alpha=float(alpha))
            label = f"steer +{alpha}·v[{self.tok.decode([tgt])!r}]"
        else:
            src = self._resolve(source)
            if kind == "clamp":
                edits = clamp_swap_edits(self.lens, src, tgt, band, self.hs)
            else:
                edits = swap_edits(self.lens, src, tgt, band, alpha=float(alpha))
            label = (f"{kind} {self.tok.decode([src])!r} -> "
                     f"{self.tok.decode([tgt])!r}")

        def side(ed):
            lg = next_token_logits(self.model, self.ids, ed)
            p = torch.softmax(lg, -1)
            vals, idx = p.topk(5)
            top = [{"tok": self.tok.decode([i]), "p": round(v, 4)}
                   for i, v in zip(idx.tolist(), vals.tolist())]
            text = generate(self.model, self.tok, self.ids, ed,
                            max_new_tokens=int(max_new))
            return {"top": top, "gen": text}

        return {"label": label, "layers": [band[0], band[-1]],
                "clean": side(()), "edited": side(edits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    wb = Workbench()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (ROOT / "workbench.html").read_bytes(),
                           "text/html; charset=utf-8")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            route = self.path.removeprefix("/api/")
            try:
                with wb.lock:
                    if route == "read":
                        out = wb.read(req["prompt"], req.get("chat", False),
                                      req.get("cosine", True))
                    elif route in ("detail", "pin", "intervene"):
                        if wb.hs is None:
                            raise ValueError("read a prompt first")
                        if route == "detail":
                            out = wb.detail(int(req["layer"]), int(req["pos"]),
                                            req.get("cosine", True))
                        elif route == "pin":
                            out = wb.pin(req["word"])
                        else:
                            out = wb.intervene(req["kind"], req.get("source", ""),
                                               req["target"], req.get("alpha", 1.0),
                                               req["lo"], req["hi"],
                                               req.get("max_new", 24))
                    else:
                        raise ValueError(f"unknown route {route!r}")
                self._send(200, out)
            except Exception as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a):        # quiet server log
            pass

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Workbench ready: http://localhost:{args.port}  "
          f"({config.MODEL_NAME}, band {wb.band[0]}-{wb.band[-1]})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
