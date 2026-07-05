"""End-to-end benchmark: prior + JAX model in a single computational graph.

Demonstrates the key requirement that the prior and the model run together,
on-device, with no host round-trip: each training step samples a fresh batch of
networks from the prior, simulates them, feeds the expression into a small JAX
model and backprops the model -- all inside one ``jit``.

A tiny permutation-invariant encoder stands in for a real foundation model: it
embeds genes, mean-pools over cells, and predicts per-gene mean expression
(a self-supervised target). The point is throughput, not the model::

    uv run python -m cell_priors.eval.benchmark_e2e --genes 50 --batch 8 --steps 20
"""

from __future__ import annotations

import time

import click
import jax
import jax.numpy as jnp

from ._common import build_prior


def _init_model(key, num_genes, hidden):
    k1, k2, k3 = jax.random.split(key, 3)
    scale = 0.1
    return {
        "embed": scale * jax.random.normal(k1, (num_genes, hidden)),
        "w": scale * jax.random.normal(k2, (hidden, hidden)),
        "head": scale * jax.random.normal(k3, (hidden, num_genes)),
    }


def _model_loss(model, expr):
    """Self-supervised loss: predict per-gene mean from a pooled cell summary."""
    x = jnp.log1p(expr)  # (cells, genes)
    # Per-cell embedding: project genes, pool, nonlinearity.
    h = jnp.tanh(x @ model["embed"])  # (cells, hidden)
    h = jnp.tanh(h @ model["w"])
    pooled = h.mean(axis=0)  # (hidden,)
    pred = pooled @ model["head"]  # (genes,)
    target = x.mean(axis=0)
    return jnp.mean((pred - target) ** 2)


@click.command()
@click.option("--genes", default=50)
@click.option("--cells", default=128)
@click.option("--cell-types", default=1)
@click.option("--hidden", default=64)
@click.option("--batch", default=8, help="Networks sampled per step.")
@click.option("--steps", default=20)
@click.option("--lr", default=1e-2)
@click.option("--safety-iter", default=100)
@click.option("--scale-iter", default=5)
@click.option("--simulator", default="sergio", type=click.Choice(["sergio", "mappfn", "grn_paper", "boolode"]))
def main(genes, cells, cell_types, hidden, batch, steps, lr, safety_iter, scale_iter, simulator):
    """Run the prior+model training loop and report throughput."""
    if simulator in ("sergio", "mappfn"):
        prior = build_prior(
            simulator, num_cells=cells, num_cell_types=cell_types, safety_iter=safety_iter, scale_iter=scale_iter
        )
    else:
        prior = build_prior(simulator, num_cells=cells)

    key = jax.random.PRNGKey(0)
    params = prior.sample_params(jax.random.PRNGKey(1), num_genes=genes)
    model = _init_model(jax.random.PRNGKey(2), genes, hidden)

    def batch_loss(model, key):
        keys = jax.random.split(key, batch)
        # Prior simulation and model fully fused; vmap over the batch of networks.
        exprs = jax.vmap(lambda k: prior.observational(params, k))(keys)  # (batch, cells, genes)
        return jnp.mean(jax.vmap(lambda e: _model_loss(model, e))(exprs))

    @jax.jit
    def step(model, key):
        loss, grads = jax.value_and_grad(batch_loss)(model, key)
        model = jax.tree.map(lambda m, g: m - lr * g, model, grads)
        return model, loss

    # Compile.
    key, sk = jax.random.split(key)
    model, loss = step(model, sk)
    jax.block_until_ready(loss)

    print(f"device={jax.devices()[0].platform}  genes={genes}  cells={cells}  batch={batch}")
    print(f"{'step':>5} {'loss':>12}")
    t0 = time.perf_counter()
    for s in range(steps):
        key, sk = jax.random.split(key)
        model, loss = step(model, sk)
        if s % max(steps // 10, 1) == 0:
            print(f"{s:>5} {float(loss):>12.5f}")
    jax.block_until_ready(loss)
    dt = time.perf_counter() - t0
    nets = steps * batch
    print(
        f"\n{steps} steps, {nets} networks in {dt:.2f}s -> "
        f"{steps / dt:.1f} steps/s, {nets / dt:.1f} networks/s (prior+model fused)"
    )


if __name__ == "__main__":
    main()
