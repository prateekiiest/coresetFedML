#!/usr/bin/env python
"""
Self-contained reconstruction of the vanilla Bayesian-coreset experiment
(paper Sec. 7.1, Fig. 2 and Fig. 3-left).

The original script depended on a patched ``bayesiancoresets`` fork, a
``model_linreg`` helper, ``../data/prices2018.npy`` and a bokeh/cairosvg
plotting stack -- none vendored.  This version keeps the *same* model
(Gaussian-conjugate Bayesian linear regression with RBF bases over 2-D
spatial locations) and the *same* A-IHT solver
(``src/bayesianCoresets/accelerated_iht.py``), but:

  * generates a synthetic 2-D spatial dataset by default
    (``--prices2018 path.npy`` uses the real [lat, lon, price] array instead);
  * builds coresets with RAND / A-IHT / A-IHT-II / GIGA (Frank-Wolfe);
  * measures forward KL( coreset posterior || true posterior ) in closed form;
  * renders both figures with matplotlib.

    python -m src.experiments.riemann_linear_regression.reproduce \
        --trials 5 --M 300 --outdir src/experiments/riemann_linear_regression/out
"""

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.bayesianCoresets.accelerated_iht import AcceleratedIHT  # noqa: E402


# --------------------------------------------------------------------- data
def synth_spatial(n, seed):
    rng = np.random.default_rng(seed)
    loc = rng.uniform(0, 1, size=(n, 2))
    centres = rng.uniform(0, 1, size=(6, 2))
    amp = rng.normal(0, 1, size=6)
    field = sum(
        a * np.exp(-((loc - c) ** 2).sum(1) / (2 * 0.15**2)) for a, c in zip(amp, centres)
    )
    y = field + rng.normal(0, 0.2, size=n)
    return np.column_stack([loc, y])


def load_data(args):
    if args.prices2018 and os.path.exists(args.prices2018):
        x = np.load(args.prices2018)
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(x.shape[0])[: args.n_subsample]
        x = x[idx].astype(np.float64)
        x[:, 2] = np.log10(x[:, 2])
        return x
    return synth_spatial(args.n_subsample, args.seed)


def rbf_features(x, n_bases, seed):
    rng = np.random.default_rng(seed + 7)
    scales = np.array([0.1, 0.2, 0.4, 0.8, 1.6, 100.0])
    per = max(1, n_bases // len(scales))
    locs, sc = [], []
    for s in scales:
        sel = rng.choice(x.shape[0], size=per, replace=False)
        locs.append(x[sel, :2])
        sc.append(np.full(per, s))
    locs = np.vstack(locs)
    sc = np.concatenate(sc)
    X = np.exp(-((x[:, None, :2] - locs[None]) ** 2).sum(-1) / (2 * sc[None] ** 2))
    return X, x[:, 2]


# ------------------------------------------------------- conjugate posterior
def weighted_post(mu0, Sig0inv, sigsq, X, Y, w):
    Xw = X * w[:, None]
    A = Sig0inv + Xw.T @ X / sigsq
    b = Sig0inv @ mu0 + Xw.T @ Y / sigsq
    Sig = np.linalg.inv(A)
    return Sig @ b, Sig


def gauss_kl(mu0, Sig0, mu1, Sig1):
    """KL( N(mu0,Sig0) || N(mu1,Sig1) )."""
    Sig1inv = np.linalg.inv(Sig1)
    d = mu0.shape[0]
    _, ld0 = np.linalg.slogdet(Sig0)
    _, ld1 = np.linalg.slogdet(Sig1)
    diff = mu1 - mu0
    return 0.5 * (np.trace(Sig1inv @ Sig0) - d + diff @ Sig1inv @ diff + ld1 - ld0)


# --------------------------------------------------------- coreset builders
def potentials(X, Y, sigsq, samples):
    """g[s, j] centred log-likelihood of point j under sample s, scaled 1/sqrt(S)."""
    resid = Y[None, :] - samples @ X.T             # [S, n]
    ll = -0.5 * resid**2 / sigsq
    ll -= ll.mean(0, keepdims=True)
    return ll / np.sqrt(samples.shape[0])


def build(name, A, y, k):
    n = A.shape[1]
    if name == "RAND":
        rng = np.random.default_rng(k)
        idx = rng.choice(n, size=k, replace=False)
        w = np.zeros(n)
        w[idx] = n / k
        return w
    if name in ("IHT", "IHT-2"):
        solver = AcceleratedIHT(y.reshape(-1, 1), A, np.zeros((n, 1)), K=k, max_iter_num=300)
        w, _ = (solver.a_iht_i() if name == "IHT" else solver.a_iht_ii())
        w = np.asarray(w).reshape(-1)
        w[w < 0] = 0
        return w
    if name == "GIGA":
        # Greedy forward selection with non-negative least-squares refit
        # (geodesic-ascent-style greedy baseline; Campbell & Broderick 2018).
        w = np.zeros(n)
        supp = []
        r = y.copy()
        for _ in range(k):
            scores = A.T @ r
            scores[supp] = -np.inf
            j = int(np.argmax(scores))
            supp.append(j)
            As = A[:, supp]
            wl, *_ = np.linalg.lstsq(As, y, rcond=None)
            wl = np.clip(wl, 0, None)
            r = y - As @ wl
        w[supp] = wl
        return w
    raise ValueError(name)


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--M", type=int, default=300)
    ap.add_argument("--n_subsample", type=int, default=1000)
    ap.add_argument("--n_bases", type=int, default=100)
    ap.add_argument("--proj_dim", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2020)
    ap.add_argument(
        "--prices2018",
        default="src/experiments/riemann_linear_regression/data/prices2018.npy",
        help="real [lat, lon, price] array; falls back to synthetic if missing "
        "(build it with `python scripts/download_datasets.py prices2018`)",
    )
    ap.add_argument("--outdir", default="src/experiments/riemann_linear_regression/out")
    ap.add_argument("--sizes", type=int, nargs="+", default=[220, 260, 300])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    methods = ["GIGA", "IHT", "IHT-2", "RAND"]
    grid = list(range(2, a.M + 1, 8))

    kl_curves = {m: np.full((a.trials, len(grid)), np.nan) for m in methods}
    last = {}

    for t in range(a.trials):
        x = load_data(argparse.Namespace(**{**vars(a), "seed": a.seed + t}))
        X, Y = rbf_features(x, a.n_bases, a.seed + t)
        n, d = X.shape
        sigsq = Y.var()
        mu0 = np.full(d, Y.mean())
        Sig0 = (Y.var() + Y.mean() ** 2) * np.eye(d)
        Sig0inv = np.linalg.inv(Sig0)
        mup, Sigp = weighted_post(mu0, Sig0inv, sigsq, X, Y, np.ones(n))
        samples = np.random.default_rng(a.seed + t).multivariate_normal(mup, Sigp, a.proj_dim)
        G = potentials(X, Y, sigsq, samples)
        y_full = G.sum(1)

        for mi, k in enumerate(grid):
            for name in methods:
                w = build(name, G, y_full, k)
                muw, Sigw = weighted_post(mu0, Sig0inv, sigsq, X, Y, w)
                kl_curves[name][t, mi] = gauss_kl(muw, Sigw, mup, Sigp)
        # keep a full-resolution sweep on trial 0 for the coreset-point figure
        if t == 0:
            wt = {}
            for k in a.sizes:
                wt[k] = build("IHT-2", G, y_full, k)
            last = dict(x=x, X=X, Y=Y, mu0=mu0, Sig0inv=Sig0inv, sigsq=sigsq,
                        mup=mup, wt=wt)
        print(f"[trial {t}] done")

    np.savez(os.path.join(a.outdir, "kl_curves.npz"),
             grid=grid, **{k: v for k, v in kl_curves.items()})
    _fig_kl(kl_curves, grid, a.outdir)
    _fig_points(last, a.sizes, a.outdir)


def _fig_kl(kl_curves, grid, outdir):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    styles = {"GIGA": "s-", "IHT": "o-", "IHT-2": "o--", "RAND": "^-"}
    labels = {"GIGA": "GIGA", "IHT": "A-IHT", "IHT-2": "A-IHT II", "RAND": "Uniform"}
    for name, curv in kl_curves.items():
        med = np.nanmedian(curv, 0)
        lo, hi = np.nanpercentile(curv, [25, 75], axis=0)
        ax.plot(grid, med, styles[name], ms=4, label=labels[name])
        ax.fill_between(grid, lo, hi, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("coreset size k")
    ax.set_ylabel(r"forward KL$(\hat\pi_w \,\|\, \pi)$")
    ax.set_title("Bayesian coreset quality vs size")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(outdir, "fig3_kl.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def _fig_points(last, sizes, outdir):
    if not last:
        return
    x, wt = last["x"], last["wt"]
    n_panels = len(sizes) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    for ax, k in zip(axes[:-1], sizes):
        w = wt[k]
        ax.scatter(x[:, 1], x[:, 0], s=4, c="0.8")
        sel = w > 0
        ax.scatter(x[sel, 1], x[sel, 0], s=120 * (w[sel] / w.max()) ** 0.4,
                   c="k", edgecolors="none")
        ax.set_title(f"A-IHT II coreset, k={k}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[-1].scatter(x[:, 1], x[:, 0], s=6, c=x[:, 2], cmap="viridis")
    axes[-1].set_title("data / true posterior mean field")
    axes[-1].set_xticks([]); axes[-1].set_yticks([])
    fig.tight_layout()
    out = os.path.join(outdir, "fig2_coreset_points.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
