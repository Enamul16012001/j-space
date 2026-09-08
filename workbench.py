"""Interactive J-lens workbench: read the residual stream in the browser,
pin words to heat-map their rank across every (layer, position) cell, follow
rank trajectories in charts, and apply live interventions (clamped swap /
swap / steer) to watch the model's answer change.

    python workbench.py            # then open http://localhost:7860
    python workbench.py --port 7861

Standard-library server; the page is plain HTML + vanilla JS served from
workbench.html next to this file.  Everything runs through this repo's own
jlens package.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import config

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from jlens import (load_model, chat_ids, JLens, generate,
                   next_token_logits, swap_edits, clamp_swap_edits, steer_edits)

ROOT = pathlib.Path(__file__).resolve().parent
WORDLIKE = re.compile(r"[^\W_]", re.UNICODE)     # any letter or digit


class Workbench:
    """Holds the model, the lens, and the last prompt's activations."""

    def __init__(self):
        print("Loading model and lens…", flush=True)
        self.model, self.tok = load_model()
        self.lens = JLens(self.model, self.tok)
        self.band = config.workspace_layers(self.model)
        self.layers = list(range(2, self.lens.target_layer + 1))
        self.lock = threading.Lock()        # one GPU, one request at a time
        self.ids = None                     # full analysed sequence
        self.gen_from = None                # in chat mode: ids generation starts from
        self.hs = None                      # hidden states
        self.out_logits = None              # model's real per-position logits

    def _readout(self, H, layer, cosine):
        return (self.lens.cosine(H, layer) if cosine
                else self.lens.logits(H, layer))

    # ---- reading -----------------------------------------------------------
    def read(self, prompt, chat=False, cosine=True, words_only=False,
             reply=False, prefill="", max_new=48):
        """Chat mode (`reply=True`): send the message, let the model answer,
        then run the lens over the WHOLE exchange — so the grid shows the
        workspace during the model's own reply."""
        self.gen_from = None
        reply_text, asst_start = "", None
        if reply:
            gen_from = chat_ids(self.tok, prompt, prefill=prefill)
            reply_text = generate(self.model, self.tok, gen_from, (),
                                  max_new_tokens=int(max_new))
            asst_start = chat_ids(self.tok, prompt).shape[1]
            tail = self.tok(reply_text, return_tensors="pt",
                            add_special_tokens=False).input_ids
            ids = torch.cat([gen_from, tail], dim=1)
            self.gen_from = gen_from
        elif chat:
            ids = chat_ids(self.tok, prompt)
        else:
            ids = self.tok(prompt, return_tensors="pt", truncation=True,
                           max_length=192).input_ids
        with torch.no_grad():
            out = self.model(ids.to(self.model.device),
                             output_hidden_states=True, use_cache=False)
        self.ids, self.hs = ids, out.hidden_states
        self.out_logits = out.logits[0].float()

        tokens = [self.tok.decode([t]) for t in ids[0].tolist()]
        grid = []                       # grid[layer][pos] = [tok, rank, top3…]
        for l in self.layers:
            S = self._readout(self.hs[l][0], l, cosine)
            vals, idx = S.topk(24 if words_only else 3, dim=-1)
            row = []
            for p in range(len(tokens)):
                toks = [self.tok.decode([i]) for i in idx[p].tolist()]
                pick = 0
                if words_only:
                    pick = next((i for i, t in enumerate(toks)
                                 if WORDLIKE.search(t)), 0)
                row.append({"t": toks[pick], "r": pick + 1,
                            "s": round(vals[p, pick].item(), 3),
                            "t3": toks[pick:pick + 3]})
            grid.append(row)

        # the model's actual next-token prediction per position (final row)
        p = torch.softmax(self.out_logits, -1)
        pv, pi = p.max(-1)
        outrow = [{"t": self.tok.decode([i]), "p": round(v, 3)}
                  for i, v in zip(pi.tolist(), pv.tolist())]

        return {"model": config.MODEL_NAME, "tokens": tokens,
                "layers": self.layers, "band": [self.band[0], self.band[-1]],
                "vocab": self.lens.vocab, "grid": grid, "outrow": outrow,
                "reply": reply_text, "asst_start": asst_start}

    def detail(self, layer, pos, cosine=True, k=15):
        S = self._readout(self.hs[layer][0, pos], layer, cosine)
        vals, idx = S.topk(k)
        return {"top": [{"tok": self.tok.decode([i]), "score": round(v, 4)}
                        for i, v in zip(idx.tolist(), vals.tolist())]}

    def slice(self, layer, pos, cosine=True, k=6):
        """Readout across all layers at `pos`, and across all positions at
        `layer` (the paper's Fig. 5 side panels)."""
        bylayer = []
        for l in self.layers:
            S = self._readout(self.hs[l][0, pos], l, cosine)
            vals, idx = S.topk(k)
            bylayer.append([self.tok.decode([i]) for i in idx.tolist()])
        S = self._readout(self.hs[layer][0], layer, cosine)
        vals, idx = S.topk(k, dim=-1)
        bypos = [[self.tok.decode([i]) for i in r.tolist()] for r in idx]
        return {"bylayer": bylayer, "bypos": bypos}

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

        base = self.gen_from if self.gen_from is not None else self.ids

        def side(ed):
            lg = next_token_logits(self.model, base, ed)
            p = torch.softmax(lg, -1)
            vals, idx = p.topk(5)
            top = [{"tok": self.tok.decode([i]), "p": round(v, 4)}
                   for i, v in zip(idx.tolist(), vals.tolist())]
            text = generate(self.model, self.tok, base, ed,
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
                                      req.get("cosine", True),
                                      req.get("words_only", False),
                                      req.get("reply", False),
                                      req.get("prefill", ""),
                                      req.get("max_new", 48))
                    elif route in ("detail", "slice", "pin", "intervene"):
                        if wb.hs is None:
                            raise ValueError("read a prompt first")
                        if route == "detail":
                            out = wb.detail(int(req["layer"]), int(req["pos"]),
                                            req.get("cosine", True))
                        elif route == "slice":
                            out = wb.slice(int(req["layer"]), int(req["pos"]),
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
