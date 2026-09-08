# J-lens & J-space on Qwen3-4B

A simple, faithful, single-GPU implementation of
**[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)**
(Gurnee et al., 2026; [arXiv:2607.15495](https://arxiv.org/abs/2607.15495))
for the open [Qwen3](https://huggingface.co/Qwen/Qwen3-4B) model family.
New to the topic? [HOW_IT_WORKS.md](HOW_IT_WORKS.md) explains every idea and
every number from scratch.

The authors' own
[reference implementation](https://github.com/anthropics/jacobian-lens)
covers lens **fitting, reading and visualization**; this repo additionally
implements the **J-space** sparse decomposition (§2.3), the full
**intervention suite** (§2.5, §3.3) and runnable versions of the paper's
experiments (Figs. 4–32).

The paper's core objects, all implemented here:

- **J-lens** (§2.1, §A.7) — for each layer ℓ, the corpus-averaged Jacobian
  `J_ℓ = E[∂h_target,t′ / ∂h_ℓ,t]` maps a residual-stream activation into the
  (penultimate) target layer, where the model's own final norm + unembedding
  read it as tokens: `lens(h) = softmax(W_U · norm(J_ℓ h))`.
- **J-lens vectors** (§2.1) — `v̂_token(ℓ) = J_ℓᵀ(γ ⊙ W_U[token])`, one
  interpretable direction per vocabulary token per layer. Used as probes and
  as intervention directions.
- **J-space** (§2.3) — sparse **nonnegative** combinations of J-lens vectors,
  found by gradient pursuit (k ≈ 25). This is the human-readable
  "contents of the workspace": a few tokens with coefficients, carrying <10%
  of activation variance beyond a random control.
- **Interventions** (§2.5, §3.3) — steer (**add**), project out (**remove**),
  swap in lens coordinates (**replace**), the **clamped** swap of Fig. 13 that
  pins coordinates to swapped clean-pass values, and whole-J-space top-k
  ablation.

## Install & run

Requires Python ≥ 3.10 and a CUDA GPU (CPU works, slowly). Then:

```bash
pip install -r requirements.txt
cp .env.example .env                       # optional: all settings, documented
python experiments/00_compute_jlens.py     # REQUIRED first: computes the lens
python experiments/01_read_lens.py         # then any of 01–12, in any order
bash run_all.sh                            # or everything
```

**All configuration lives in `.env`** — model, corpus size, precision, layer
band. Edit that file and nothing else; `config.py` reads it and falls back to
sensible defaults when it is absent. Real environment variables override it,
so a one-off is just:

```bash
MODEL_NAME=Qwen/Qwen3-0.6B N_PROMPTS=32 python experiments/00_compute_jlens.py
```

Hardware: experiments 01–12 run the model in bf16 (~8 GB for 4B), so a 16 GB
GPU is plenty. **Step 00 is the demanding one** and is worth understanding
before you start it:

- It runs the backward passes in **float32** (`config.LENS_DTYPE`) — the
  Jacobian is an average of gradients propagated through every block, and it is
  the one place where bf16 noticeably degrades the result. For Qwen3-4B that is
  ~16 GB of weights plus activations.
- Cost per prompt is `hidden_size / ROWS_PER_BACKWARD` backward passes over the
  whole model. `ROWS_PER_BACKWARD` is therefore both the memory dial *and* the
  speed dial: each backward re-reads every weight, so bigger batches mean fewer
  passes over the model. Raise it to 32–64 if you have the memory; drop it to
  4–8 on a 24 GB card.
- `N_PROMPTS` defaults to the paper's **1000**, which is a multi-day run for 4B
  on one GPU. Work per prompt is fixed at `d_model` sequence-backward-passes, so
  batching changes utilisation but not the total. The lens is checkpointed
  atomically every `SAVE_EVERY` (25) prompts, and every checkpoint is a
  complete, usable lens — just averaged over fewer prompts — so the intended
  workflow is to start it, watch the tqdm rate, and **Ctrl-C when you have
  enough**. Fig. 59 says even 10–100 prompts is a solid lens.

Everything runs on CPU too, just slowly.

## Layout

```
.env                       ALL settings live here
config.py                  reads .env, derives the rest
jlens/model.py             load model, chat templating, single-token lookup
jlens/corpus.py            corpus sampling: C4 web text + WikiText, mixed lengths
jlens/lens.py              compute_jlens (§A.7) + JLens (read/vector/probe)
jlens/jspace.py            sparse decomposition, occupancy, random control
jlens/interventions.py     Steer / ProjectOut / Swap / ClampSwap / TopKJSpaceAblate
jlens/utils.py             printing helpers
experiments/00–12          the paper's experiments (see table)
workbench.py + .html       interactive browser UI: read / pin / intervene
tests/test_math.py         J-lens/J-space maths vs reference (CPU, no download)
tests/test_experiments.py  smoke-runs every experiment on a tiny random model
```

Both test files run in seconds on CPU and need no model download:

```bash
python tests/test_math.py          # 43 checks
python tests/test_experiments.py   # runs experiments 00-12 end to end
```

Residual-stream indexing everywhere: **layer 0 = embeddings, layer i = output
of block i** (Qwen3-4B has 36 blocks; the default Jacobian target is the
penultimate residual, layer 35).

## Interactive workbench

```bash
python workbench.py            # open http://localhost:7860
```

A browser UI over the lens (this repo's own code; stdlib server, vanilla JS,
no extra dependencies):

- **Read**: the layer × position grid of top lens tokens for any prompt,
  cosine or raw readout, workspace band highlighted, top-3 tooltips.
- **Pin**: type any word (or click 📌 in a cell's detail) to heat-map its
  rank across every cell — the paper's Figure 5 view.
- **Intervene**: pick a source and target (the A/B buttons fill them from a
  cell's readout), choose clamped swap / swap / steer and a layer range, and
  compare the clean vs edited next-token distribution and generation live.

Over VS Code Remote-SSH the port is forwarded automatically. The GPU serves
one request at a time; a full read takes a few seconds per prompt.

## Experiment ↔ paper map

| script | paper | what it shows |
|---|---|---|
| 00_compute_jlens | §2.1, §A.7 | estimate and save J_ℓ for all layers |
| 01_read_lens | §2.4, Fig. 4B | read the stream; J-lens vs logit lens |
| 02_verbal_report_swap | §3.1, Fig. 5–6 | swap a silently held choice → report changes |
| 03_inject_thought | §3.1, Fig. 7 | steer a concept in → model introspects on it |
| 04_hold_in_mind | §3.2, Fig. 9 | "concentrate on citrus" while copying text — lens reads it |
| 05_intermediate_swap | §3.3, Fig. 12–14 | swap the *intermediate* step (spider→ant ⇒ 8→6) |
| 06_flexible_generalization | §3.4, Fig. 18 | one France→China swap redirects many functions |
| 07_selectivity | §3.5.1, Fig. 20 | report/inference flip; low-level continuation doesn't |
| 08_jspace_ablation | §3.5.2, Fig. 22 | ablate all workspace content: multi-hop dies, routine prediction survives |
| 09_workspace_layers | §4.1, Fig. 27–28 | locate the band: accuracy, kurtosis, persistence, eff. dim |
| 10_occupancy | §4.2, Fig. 30 | J-space representation per token + capacity (median ≈ 25 atoms) |
| 11_broadcast_gain | §4.3.1, Fig. 32 | MLPs amplify J-lens directions ~10× over random |
| 12_jspace_privilege | §3.1, Fig. 8 | the thin J-space component of a concept vector, not the other ~93%, drives verbal report |

## Other Qwen3 sizes

Change `MODEL_NAME` in `.env` — nothing else. The lens filename and the
workspace band are both derived from the checkpoint, so different sizes never
clash. All six dense Qwen3 models work:

| model | blocks | d_model | lens file | step-00 fp32 weights | relative cost/prompt |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 1024 | 117 MB | 2.4 GB | 1× |
| Qwen3-1.7B | 28 | 2048 | 470 MB | 6.8 GB | 6× |
| Qwen3-4B | 36 | 2560 | 944 MB | 16 GB | 17× |
| Qwen3-8B | 36 | 4096 | 2.4 GB | 32 GB | 53× |
| Qwen3-14B | 40 | 5120 | 4.2 GB | 56 GB | 117× |
| Qwen3-32B | 64 | 5120 | 6.7 GB | 128 GB | 267× |

Cost per prompt scales as `params × d_model`, because bigger models need both
more work per backward pass and more passes (`d_model / ROWS_PER_BACKWARD` of
them). **Qwen3-0.6B is ~17× cheaper than 4B**, which makes it the right place
to shake out a pipeline before committing to a long run — but expect the
workspace structure itself to be weak there, since the paper reports these
effects strengthen with scale. Qwen3-32B needs `LENS_DTYPE = torch.bfloat16` to
fit in 128 GB at all.

Not supported: the **MoE** checkpoints (Qwen3-30B-A3B, 235B-A22B). The lens
itself would be fine, but experiment 11 assumes each block has a single dense
`block.mlp`.

## Troubleshooting

**`fatal error: Python.h: No such file or directory`** during the first forward
pass. Recent torch routes some ops (e.g. RoPE's batched matmul) through Triton,
which JIT-compiles a small C shim and therefore needs the Python development
headers. Either install them:

```bash
sudo apt-get install -y python3.12-dev     # or python3-dev
```

or skip the Triton path entirely, at some cost in speed:

```bash
TORCH_DISABLE_NATIVE_JIT=1 python experiments/00_compute_jlens.py
```

**`HfUriError: Repository id must be 'namespace/name', got 'wikitext'`** means
an old `jlens/corpus.py`; the dataset id must be `Salesforce/wikitext` for
`datasets >= 3`.

## Faithfulness notes & knobs

- The lens recipe follows the authors' reference implementation
  ([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)):
  one-hot backward seeds at every valid target position, sum over targets and
  mean over sources restricted to positions `[SKIP_FIRST, T-1)` (the first 16
  positions are attention sinks; the last has no next-token target), no
  stop-grads, and the model's own final RMSNorm + unembedding for readout.
  One deliberate difference: we default to the **penultimate**-layer target
  (the paper's stated default, §A.7) where the reference defaults to the final
  layer; both are supported via `TARGET` in `.env`.
- **Σ vs E over t′.** §2.1 writes the aggregation over target positions as an
  expectation; the §A.7 pseudocode seeds a one-hot gradient at every target
  position and averages only over source positions — a *sum* over t′. We follow
  the pseudocode. (The two differ: a sum weights early source positions more.)
- **γ in the J-lens vector.** §2.1 defines the vectors as the rows of `W_U J_ℓ`;
  we fold in the final RMSNorm gain γ. The model's readout is
  `logits = W_U (γ ⊙ z) / rms(z)`, so only with γ included is `⟨v̂, h⟩` exactly
  proportional to that token's lens logit — the property §2.5 relies on when it
  reuses these vectors as probes and as intervention directions. The remaining
  `1/rms(z)` is a positive scalar and never changes rankings.
- Readouts, probes and sparse-decomposition correlations are all computed in
  float32. The vocabulary is ~150k tokens and rankings among near-ties are not
  reliable in the model's bf16.
- `N_PROMPTS` is the paper's 1000. Fig. 59 shows the J-lens already beats the
  logit and tuned lenses with as few as 10 sequences and improves only modestly
  after that, so 64 is a good fast setting and 16 is fine for a smoke test.
  The corpus is a 50/50 mix of C4 web text and WikiText-103 with ~25% short
  passages, shuffled, cached on disk, and fingerprinted into every checkpoint —
  an interrupted run can only resume onto the identical corpus.
  **Install `datasets`** — without it the corpus falls back to 8 built-in
  paragraphs.
- The workspace band is derived from the checkpoint's depth using the paper's
  ~38–92% (`config.WORKSPACE_BAND`); for Qwen3-4B's 36 blocks that gives layers
  14–33. Run experiment 09 and set `config.WORKSPACE_LAYERS` to an explicit
  list to pin a calibrated band for your model.
- Chat experiments use `enable_thinking=False` — the paper studies *silent*
  workspace computation, not chain-of-thought.
- Qwen3-4B is smaller than the models in the paper; the paper reports these
  effects strengthen with scale (e.g. Fig. 10), so expect qualitatively
  matching but weaker/noisier results, and expect to tune `--alpha`.

## Not implemented (out of scope by design)

- Counterfactual **reflection training** (§7) — requires an SFT pipeline.
- Broadcast **attention-head** analysis (§4.3.2), ambiguous-input
  **ignition** dynamics (§4.1.1), the **model-organism** studies (§5.4–5.5),
  and the multi-token **template lens** (§A.9).
- Anything requiring **sparse autoencoders** (§A.18, the SAE strata in Fig. 32).
- The **character-counting / linewrap** protocol (Fig. 9 third example, Fig. 21)
  and the paired-question implicit-modulation protocol (Fig. 11).

Everything else in the paper's main line — reading, per-token J-space
representations, the J-space privilege control, and the add / remove / replace
/ modulate intervention suite — is here.

## Citing

This repository implements the methods of:

```bibtex
@article{gurnee2026verbalizable,
  author  = {Gurnee, Wes and Sofroniew, Nicholas and Pearce, Adam and
             Piotrowski, Mateusz and Kauvar, Isaac and Chen, Runjin and
             Soligo, Anna and Bogdan, Paul and Ong, Euan and Wang, Rowan and
             Thompson, Ben and Abrahams, David and Kantamneni, Subhash and
             Ameisen, Emmanuel and Batson, Joshua and Lindsey, Jack},
  title   = {Verbalizable Representations Form a Global Workspace in Language Models},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/workspace/index.html}
}
```

The paper PDF is not distributed with this repository; use the links above.

## License

MIT — see [LICENSE](LICENSE). Qwen3 model weights and the C4 / WikiText-103
datasets are governed by their own licenses.
