# cell-priors

Efficient, diverse **priors** for virtual-cell foundation-model pretraining — with a
uniform interface so priors can be benchmarked and compared against each other and
against real data.

The first prior is a from-scratch **JAX reimplementation of [SERGIO]** (the single-cell
expression simulator), built so the whole generative process runs *inside* a JAX
computation graph: a prior + model can train together on the GPU with no host
round-trip, in the spirit of [purejaxrl]. It is numerically validated against the Rust
reference [`sergio_rs`].

[SERGIO]: https://github.com/PayamDibaeinia/SERGIO
[`sergio_rs`]: https://pypi.org/project/sergio-rs/
[purejaxrl]: https://github.com/luchris429/purejaxrl

---

## Why JAX SERGIO

SERGIO models gene expression as a stochastic differential equation driven by a gene
regulatory network (GRN). Each cell type's expression of gene *i* evolves as

```
dx_i = (P_i(x) - λ_i x_i) dt  +  s · (sqrt(P_i) dW_p + sqrt(λ_i x_i) dW_d) · sqrt(dt)
```

where the production `P_i` is a sum of Hill functions over gene *i*'s regulators
(activation or repression), and master regulators (genes with no regulators) have a
fixed basal production rate per cell type.

This reimplementation is engineered for the prior+model training loop, where the prior's
speed directly bounds training throughput:

- **Sparse edge list, not dense adjacency.** Interactions are stored as `(reg, tar)`
  index arrays, so each integration step costs `O(E · C)` (edges × cell types) via a
  single `segment_sum`, instead of `O(G² · C)`.
- **The expensive init is a fixed point, not a graph walk.** SERGIO estimates per-edge
  half-responses and the steady state with a sequential pass over topological levels.
  We recast that as a `lax.scan` fixed-point iteration that converges to the *exact* same
  values on a DAG — no Python-level traversal, fully `jit`-able and `vmap`-able.
- **One `scan` for the whole SDE.** The time integration is a single `lax.scan`; the
  trajectory is stacked once and sampled with a gather.
- **Everything is a pytree of arrays.** `SergioParams` and interventions are pure data,
  so `prior → model → loss → grad` fuses into one compiled graph (see
  `benchmark_e2e.py`).

### Numerically validated against `sergio_rs`

With `noise_s = 0` the SDE is deterministic and converges to a fixed point. The test
suite runs `sergio_rs` to that fixed point, recovers the master-regulator production
rates from the converged state, feeds them into the JAX core, and asserts every gene
matches (`max abs diff < 1e-3` across several DAGs and seeds). This exercises the Hill
function, the half-response/steady-state estimation, and the SDE integration together.

```bash
uv run pytest          # 21 tests, including sergio_rs parity
```

---

## Install

Uses [uv](https://docs.astral.sh/uv/). Python 3.12.

```bash
uv sync                # CPU JAX + everything needed for tests/benchmarks
uv sync --extra cuda   # add GPU JAX (jax[cuda12])
```

`sergio_rs` is pulled as a prebuilt wheel (x86_64 / aarch64). A ready-to-use dev
container lives in `.devcontainer/`, and GitHub Actions build & publish it to GHCR on
version tags.

---

## The uniform prior interface

Every prior implements `cell_priors.base.Prior`:

```python
import jax
from cell_priors.priors.sergio import SergioPrior, SergioConfig
from cell_priors.base import InterventionKind

cfg = SergioConfig(num_cells=200, num_cell_types=2, noise_s=1.0)
prior = SergioPrior(cfg)

params = prior.sample_params(jax.random.PRNGKey(0), num_genes=100)   # a GRN (pytree)

obs = prior.observational(params, jax.random.PRNGKey(1))             # (cells, genes)

# Intervene on gene 7 and sample the perturbed distribution
ko  = prior.interventional(params, jax.random.PRNGKey(2), [7], kind=InterventionKind.KNOCKOUT)
kd  = prior.interventional(params, jax.random.PRNGKey(2), [7],
                           kind=InterventionKind.KNOCKDOWN, strength=0.5)
```

Because `sample_params`, `observational`, `intervene` and `interventional` are pure
functions of `(params, key)`, the prior composes with a JAX model in a single `jit`/`vmap`:

```python
@jax.jit
def step(model, key):
    expr = prior.observational(params, key)   # simulated on-device
    return loss_fn(model, expr)               # model trains on it, same graph
```

### Hard knockouts vs. soft CRISPRi knockdowns

The SERGIO prior supports both perturbation styles so you can compare them directly:

| | mechanism | causal graph | targeted gene |
|---|---|---|---|
| **`KNOCKOUT`** (hard) | remove the gene and its outgoing edges | edges deleted; orphaned targets become master regulators | silenced |
| **`KNOCKDOWN`** (soft) | scale the gene's production by `1 − strength` | **intact** — the attenuated gene keeps regulating | reduced |

`strength=1.0` fully silences production, but unlike a hard knockout the edges remain, so
downstream genes respond differently. This is the principled difference between ablating
a node and dialing down its transcription, and it lets you study how a model trained on
one generalizes to the other.

---

## Generate datasets (MapPFN `.h5ad` format)

```python
from cell_priors.io import generate_anndata, write_h5ad

adata = generate_anndata(prior, jax.random.PRNGKey(0),
                         num_contexts=8, num_genes=100, add_noise=True)
write_h5ad(adata, "sergio.h5ad")
```

Output matches MapPFN: `adata.X` is counts `(cells, genes)`, `adata.var_names` are
`GENE0000…`, and `adata.obs` has `context` (GRN id) and `treatment` (perturbed gene id or
`"control"`). Technical noise (outlier / library-size / dropout / UMI) uses the SERGIO
paper's DS1–DS14 profiles.

---

## Benchmarks & comparison (scripts, not notebooks)

All scripts share a clean CLI and select priors by name.

**Prior speed — JAX vs. `sergio_rs`, across dimensionalities:**

```bash
uv run python -m cell_priors.eval.benchmark_speed --genes 20,50,100,200 --cells 200
uv run python -m cell_priors.eval.benchmark_speed --genes 100 --batch 16      # vmapped
```

**End-to-end prior + model in one graph (throughput):**

```bash
uv run python -m cell_priors.eval.benchmark_e2e --genes 50 --batch 8 --steps 50
```

Each step samples a fresh batch of networks from the prior, simulates them, feeds the
expression into a small permutation-invariant JAX model, and backprops the model — all
inside a single `jit`, no host transfer.

**Compare distributions, DE genes, and real data:**

```bash
# Health & distributional stats for any source
uv run python -m cell_priors.eval.compare stats sergio

# Differential expression of a perturbation vs control (scanpy)
uv run python -m cell_priors.eval.compare de sergio --gene 0

# Compare two sources distributionally (KS + Wasserstein, with a figure)
uv run python -m cell_priors.eval.compare distribution sergio hf:marvinsxtr/MapPFN/frangieh.h5ad
```

A *source* is `sergio` (generate fresh), a local `.h5ad` path, or
`hf:<repo>/<file>` to pull from a Hugging Face dataset (e.g. the real
[`marvinsxtr/MapPFN`](https://huggingface.co/datasets/marvinsxtr/MapPFN) datasets:
`frangieh.h5ad`, `papalexi.h5ad`, `sergio.h5ad`).

---

## Utilities

```python
from cell_priors.utils import summarize, assert_healthy            # sparsity/NaN/dead-gene checks
from cell_priors.utils import infer_grn_correlation, infer_grn_regression, edge_auroc
```

- **Diagnostics** (`utils.stats`): one-call health/sparsity summary, per-gene moments,
  `assert_healthy` for pipelines.
- **GRN inference** (`utils.grn_infer`): marginal-correlation and GENIE3-style Lasso edge
  scorers plus edge-recovery AUROC — for generating *comparable* synthetic data (infer a
  GRN from real data, re-simulate it) and for controllable overfitting experiments.

---

## Project layout

```
src/cell_priors/
  base.py                     # uniform Prior interface + InterventionKind
  priors/sergio/
    grn.py                    # SergioParams pytree, random DAG generator
    core.py                   # Hill, fixed-point init, scan SDE loop
    noise.py                  # technical noise (DS1–DS14)
    interventions.py          # hard knockout / soft CRISPRi knockdown
    prior.py                  # SergioPrior (uniform interface)
  io/h5ad.py                  # MapPFN-format AnnData export
  utils/                      # GRN inference + diagnostics
  eval/                       # benchmark_speed, benchmark_e2e, compare
tests/                        # parity vs sergio_rs, hill, interface, noise, utils
.devcontainer/  .github/workflows/
```

## License

See the upstream SERGIO and `sergio_rs` projects for the simulator's lineage; this is an
independent JAX reimplementation for research use.
