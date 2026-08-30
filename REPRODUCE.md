# Reproducing *Bayesian Coreset Optimization for Personalized Federated Learning* (ICLR 2024)

This document accompanies a reconstruction of the codebase so that the paper's
figures and tables can be produced end-to-end. It also records the intended
reading of the ambiguous equations, since the original repository shipped
without a finished implementation of Algorithm 1 (the A-IHT solver was present
but never called; training mixed minibatch rows with a random `true_w`).

---

## 1. What each equation means (the three questions)

### 1.1 Definition of `P_{θ,w}(D^i)` (Eq. 3, 4, 5, 8, 9; Prop. 1-2)

One definition makes every equation consistent — the **Bayesian-coreset
weighted likelihood** (Campbell & Broderick 2019; Zhang et al. 2021):

```
log P_{θ,w}(D^i) := Σ_j w_j · log p_θ(D^i_j),      w ≥ 0,  ‖w‖₀ ≤ n_k
```

equivalently `P_{θ,w}(D^i) = Π_j p_θ(D^i_j)^{w_j}`.

* In the norm-form equations (3, 4, Prop. 2), "`P_θ(D^i)`" inside `‖·‖_{π̂,2}`
  is the log-likelihood **function** `θ ↦ Σ_j log p_θ(D^i_j)` viewed as a
  vector in `L²(π̂)`. Eq. 4 matches `Σ_j ĝ_j` (unit weights) against
  `Σ_j w_j ĝ_j`, where `ĝ_j` is the centred, Monte-Carlo-projected per-example
  log-likelihood potential. Linear in `w` — in the **potentials**.
  Code: [`src/bayesianCoresets/coreset.py`](src/bayesianCoresets/coreset.py) `loglik_potentials`.
* In Eq. 5 / 9, `−E_q[log P_{θ,w}(D^i)] = −E_q[Σ_j w_j log p_θ(D^i_j)]` — the
  weighted data log-likelihood.
  Code: [`src/model.py`](src/model.py) `federatedBNN._nll` (weighted branch) and `elbo`.
* **Eq. 9 as printed drops the `w_j` inside the sum** — a typo. The estimator
  is `−(n/b)(1/K) Σ_{j∈Λ} Σ_k w_j log p^i_{h(v_w,g_k)}(D^i_j) + ζ·KL`.
* Optimal `w` puts ≈ `n` units of mass on `n_k` points, so coreset `w_j ≫ 1`.
  `_nll` therefore self-normalises the minibatch term by `n / Σ_{batch} w`
  instead of multiplying by `n/b`.

### 1.2 Is `KL(q̂^i(θ,w) ‖ q̂^i(θ))` optimised through `q̂^i(w)`'s dependence on `w`?

* **Theory:** yes, bilevel. Proposition 1 (Appendix Eq. 19-29) is the
  implicit-function-theorem derivation through the inner stationarity condition
  of Eq. 5.
* **Algorithm 1 / this code:** no — **alternating (block-coordinate)**. A-IHT
  (Algorithm 2) is a quadratic solver with exact line search; it minimises only
  `‖Σĝ − Σ w ĝ‖²`. The `KL` term is frozen at the posteriors from the last
  `ClientUpdate` during A-IHT's inner iterations and re-enters only through the
  outer `repeat`. `KL(q_w ‖ q)` is tracked as the diagnostic of Fig. 3a / 4 and
  used as the outer stop criterion.
  Code: [`src/clientModels/clientModelClass.py`](src/clientModels/clientModelClass.py) `local_train`.

### 1.3 `ClientUpdate` per A-IHT update, or per communication round?

Neither extreme. Per communication round `t`, per client:

```
w_i ← 0 (or warm-start from t-1)
repeat  coreset_outer_steps  times  (until supp(w_i) / KL(q_w‖q) plateaus):
    ClientUpdate      # local_rounds passes of reparam-SGD on Eq. 9 (q_w) + Eq. 1 (q) + Eq. 2 (z)
    w_i ← A-IHT(f(w)) # A-IHT iterates internally on the FROZEN quadratic
upload z-track
```

One `ClientUpdate` per **A-IHT invocation** (not per internal A-IHT iterate),
and only a few outer steps per round (usually 1-2 before the stop criterion).
Code: [`src/clientModels/clientModelClass.py`](src/clientModels/clientModelClass.py) `local_train`;
server loop in [`src/serverModels/serverpFedbayes.py`](src/serverModels/serverpFedbayes.py) `train`.

---

## 2. Equation → code map

| Paper | Code |
|---|---|
| Eq. 1  client full-data objective `F_i(z)` | `federatedBNN.elbo(..., weights=None)` |
| Eq. 2  server objective (agg. KL) | z-track update in `ClientModelClass._local_rounds` step (3) + `Server.aggregate` |
| Eq. 3-4, Prop. 2  coreset matching `‖Σĝ − Σwĝ‖²` | `coreset.loglik_potentials` + `AcceleratedIHT.a_iht_ii` |
| Eq. 5 / 9  weighted client objective `F^w_i(z)` | `federatedBNN.elbo(..., weights=wb)` |
| Eq. 7-8  `{w_i*} = argmin_w KL(q_w‖q) + ‖·‖²` | alternation in `ClientModelClass.local_train` (A-IHT on the quadratic; KL as stop criterion) |
| Alg. 1  `v^{t+1} = (1-β)v^t + (β/S)Σ v_i` | `Server.aggregate` |
| Alg. 2  Accelerated-IHT II | `src/bayesianCoresets/accelerated_iht.py` (`a_iht_ii`, unchanged) |
| Def. 4-6  submodular diversity funcs | `submodlib` calls in `src/experiments/medical/run_medical.py` |

Reparameterisation `θ = μ + softplus(ρ)·ε`, `ε ~ N(0,1)`:
`federatedBNN.sample_params`.

---

## 3. Running everything

```bash
pip install -r requirements.txt
```

### 3.1 Benchmark method — Table 1 / 3, Fig. 3-5  (MNIST / FashionMNIST / CIFAR-10)

```bash
# CPU sanity (~30 s): data gen + 3 methods + figures + summary
python scripts/run_benchmark.py --dataset Mnist --preset smoke

# paper-scale (10 clients, ζ=10, lr=1e-3, 200 rounds) — use a GPU box
python scripts/run_benchmark.py --dataset Mnist --preset paper --sweep
```

Outputs in `results/`:
`fig_accuracy_<ds>.png`, `fig_kl_<ds>.png` (Fig. 3a/4),
`fig_comm_rounds_<ds>.png` (Fig. 5, with `--sweep`), `summary_<ds>.txt` (Table 3).

Single run: `python main.py --method {coreset,pfedbayes,randomsubset} --dataset Mnist ...`
(`python main.py -h` for all flags).

### 3.2 Medical datasets — Table 2

```bash
# OCTMNIST (free download via medmnist)
python -m utils.medical_data --per_class 800
python -m src.experiments.medical.run_medical --seeds 3
```

`--methods full random logdet dispsum dispmin coreset`. Output:
`results/table2_octmnist.txt` (class-wise mean ± std).
For COVID-19 Radiography / APTOS 2019 (gated Kaggle downloads) pass
`--image_root <ImageFolder-dir>` to `utils.medical_data` — same pipeline.

### 3.3 Vanilla Bayesian coresets — Fig. 2, Fig. 3-left

```bash
python -m src.experiments.riemann_linear_regression.reproduce --trials 10 --M 300
```

Outputs `src/experiments/riemann_linear_regression/out/`:
`fig3_kl.png` (forward KL vs coreset size: GIGA / A-IHT / A-IHT II / Uniform),
`fig2_coreset_points.png` (coreset points sized by weight for k∈{220,260,300}).
Synthetic 2-D spatial regression by default; `--prices2018 prices2018.npy`
uses the real UK housing array
(<https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads>).

---

## 4. What changed vs. the original repository

* `src/model.py` — replaced the duplicated `coreset_mus/rhos` "second BNN" with
  a clean mean-field Gaussian posterior + weighted ELBO; `ε ~ N(0,1)` (was
  `std=1e-3`, i.e. near-deterministic).
* `src/clientModels/clientModelClass.py` — **new** Algorithm 1 client
  (the previous `train` used a random `true_w` to linearly mix minibatch rows,
  updated the global model in the "KL" step, and never called A-IHT).
* `src/bayesianCoresets/coreset.py` — **new**: potentials + A-IHT wrapper +
  random-subset baseline. `accelerated_iht.py` itself is unchanged.
* `src/serverModels/*` — Algorithm 1 loop with β-mixing and client subsampling.
* `utils/argparse.py` — deleted (it silently ignored every CLI flag); `main.py`
  now uses stdlib `argparse`.
* `utils/data_gen.py` — **new**: non-i.i.d. client sharding → LEAF JSON.
* `utils/plot_utils.py` — rewrote; the old `simple_read_data` ignored its `alg`
  argument and always opened one hard-coded `.h5`.
* `requirements.txt` — real dependency list (was an un-installable conda dump
  with `@ file://` paths and no torch/numpy/matplotlib).
* `src/experiments/.../reproduce.py`, `src/experiments/medical/run_medical.py`,
  `utils/medical_data.py`, `scripts/run_benchmark.py` — **new** drivers.
  The original `riemann_linear_regression/{main,plot_kl,plot_coreset_pts}.py`
  are kept for reference but need a patched `bayesiancoresets` fork + bokeh +
  `prices2018.npy` that were never vendored.

## 5. Known gaps for full paper-scale reproduction

* Numbers in the smoke presets are not the paper's; use `--preset paper`
  (benchmark) / larger `--per_class`, `--seeds` (medical) on a GPU.
* Table 1 external baselines (FedAvg, BNFed, pFedMe, perFedAvg) are not
  re-implemented here — only PFedBayes, RandomSubset and CoreSet-PFedBayes.
* Medical Table 2 defaults to OCTMNIST; COVID/APTOS need manual Kaggle download.
* Fig. 2/3 use a synthetic spatial dataset unless `--prices2018` is supplied.
