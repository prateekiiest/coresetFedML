# Bayesian Coreset Optimization for Personalized Federated Learning
### _Prateek Chanda, Shrey Modi, Ganesh Ramakrishnan_

### Accepted at International Conference on Learning Representations (ICLR) 2024

[Project Page](https://coresetfederatedlearning.github.io/) | [OpenReview](https://openreview.net/forum?id=uz7d2N2zul) | [Paper](https://openreview.net/pdf?id=uz7d2N2zul) | [Poster](https://iclr.cc/media/PosterPDFs/ICLR%202024/17557.png?t=1713590458.7719705) | [Slides](https://iclr.cc/media/iclr-2024/Slides/17557.pdf)

------------------------------

### Abstract
In a distributed machine learning setting like Federated Learning where there are multiple clients involved which update their individual weights to a single central server, often training on the entire individual client's dataset for each client becomes cumbersome. To address this issue we propose CORESET-PFEDBAYES: a personalized coreset weighted federated learning setup where the training updates for each individual clients are forwarded to the central server based on only individual client coreset based representative data points instead of the entire client data. Through theoretical analysis we present how the average generalization error is minimax optimal up to logarithm bounds - upper bounded by $\mathcal{O}(n_k^{-\frac{2 \beta}{2 \beta+\boldsymbol{\Lambda}}} \log ^{2 \delta^{\prime}}(n_k))$ and lower bounds of $\mathcal{O}(n_k^{-\frac{2 \beta}{2 \beta+\boldsymbol{\Lambda}}})$, and how the overall generalization error on the data likelihood differs from a vanilla Federated Learning setup as a closed form function ${\boldsymbol{\Im}}(\boldsymbol{w}, n_k)$ of the coreset weights $\boldsymbol{w}$ and coreset sample size $n_k$.

------------------------------

## Repository contents

This repository contains a runnable reconstruction of **Algorithm 1
(CoreSet-PFedBayes)** together with drivers for the three experiment families
in the paper:

| Paper | Driver | Output |
|---|---|---|
| Benchmark FL — Table 1 / 3, Fig. 3–5 (MNIST, FashionMNIST, CIFAR-10) | `scripts/run_benchmark.py`, `main.py` | accuracy / KL / comm-round figures, summary table |
| Medical datasets — Table 2 (OCTMNIST; COVID-19 Radiography / APTOS 2019 via a folder path) | `src/experiments/medical/run_medical.py` | class-wise accuracy table |
| Vanilla Bayesian coresets — Fig. 2, Fig. 3-left | `src/experiments/riemann_linear_regression/reproduce.py` | coreset-point + KL-vs-size figures |

The precise reading of the ambiguous equations (`P_{θ,w}(D^i)`, the KL term in
Eq. 7–8, the `ClientUpdate` cadence), the full equation → code map and the list
of changes vs. the first commit are in **[`REPRODUCE.md`](REPRODUCE.md)**.

### Layout

```
main.py                                  # benchmark FL entry point (one method / dataset)
scripts/run_benchmark.py                 # data-gen + 3 methods + figures + table
scripts/run_paper_cpu.sh                 # 10-client, 40-round CPU config used for the paper-scale runs
src/model.py                             # mean-field Gaussian BNN + weighted ELBO (Eq. 1 / 5 / 9)
src/bayesianCoresets/accelerated_iht.py  # Accelerated-IHT II (Algorithm 2, unchanged)
src/bayesianCoresets/coreset.py          # log-likelihood potentials + A-IHT wrapper + random-subset baseline
src/clientModels/clientModelClass.py     # Algorithm 1 client (q, q_w, z; A-IHT outer loop)
src/serverModels/serverpFedbayes.py      # Algorithm 1 server loop (β-mixing, client subsampling)
utils/data_gen.py                        # non-i.i.d. client sharding  ->  LEAF-style JSON
utils/medical_data.py                    # ResNet-18 embeddings for the 2-client medical setup
utils/plot_utils.py                      # figures + Table-3-style summary
src/experiments/medical/run_medical.py   # Table 2 (coreset vs random vs submodular subset selection)
src/experiments/riemann_linear_regression/reproduce.py   # Fig. 2 + Fig. 3-left
```

------------------------------

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.9–3.11
pip install -r requirements.txt
```

CPU is sufficient (and, for this small BNN, faster than Apple MPS). A CUDA GPU
helps only at large `--num_users` / high `--mc`.

------------------------------

## 1. Benchmark experiments — Table 1 / 3, Fig. 3–5

**Quick sanity check** (CPU, ≈30 s): generate data, run all three methods,
write figures + summary.

```bash
python scripts/run_benchmark.py --dataset Mnist --preset smoke
```

**Paper-scale** (10 clients, 3 classes/client, 1000 train each, lr `1e-3`,
`ζ=10`; App. 10.6.1):

```bash
python scripts/run_benchmark.py --dataset Mnist  --preset paper --sweep
python scripts/run_benchmark.py --dataset FMnist --preset paper --sweep
python scripts/run_benchmark.py --dataset Cifar  --preset paper --sweep
# or, running the three methods concurrently on CPU:
bash scripts/run_paper_cpu.sh Mnist 40
```

Outputs (in `results/`):

* `fig_accuracy_<ds>.png` — personal / global test accuracy vs communication round
* `fig_kl_<ds>.png` — `D_KL(q̂ⁱ(θ;w) ‖ q̂ⁱ(θ))` vs round (Fig. 3a / Fig. 4)
* `fig_comm_rounds_<ds>.png` — accuracy vs round for `k ∈ {10,15,30,50}%` (Fig. 5, with `--sweep`)
* `summary_<ds>.txt` — final accuracies + rounds-to-target (Table 3 style)

**Single run:**

```bash
python -m utils.data_gen --dataset Mnist --num_users 10 --n_train 1000 --n_test 400 --classes_per_user 3
python main.py --method coreset      --dataset Mnist --num_glob_iters 40 --num_users 10 --clients_per_round 10
python main.py --method pfedbayes    --dataset Mnist --num_glob_iters 40 --num_users 10 --clients_per_round 10
python main.py --method randomsubset --dataset Mnist --num_glob_iters 40 --num_users 10 --clients_per_round 10
```

`python main.py -h` lists every flag (`--coreset_frac`, `--coreset_S`,
`--coreset_outer_steps`, `--local_rounds`, `--zeta`, `--beta`, …).

------------------------------

## 2. Medical datasets — Table 2

```bash
# OCTMNIST — free download via the medmnist package
python -m utils.medical_data --per_class 800
python -m src.experiments.medical.run_medical --seeds 3
```

`--methods full random logdet dispsum dispmin coreset` selects which rows to
produce; output goes to `results/table2_octmnist.txt` (class-wise mean ± std).

For **COVID-19 Radiography Database** / **APTOS 2019 Blindness Detection**
(gated Kaggle downloads), arrange the images as an `ImageFolder`
(`<root>/<class_name>/*.png`) and point the loader at it — the rest of the
pipeline is unchanged:

```bash
python -m utils.medical_data --name covid --image_root /path/to/covid_imagefolder
python -m src.experiments.medical.run_medical --name covid --seeds 3
```

------------------------------

## 3. Vanilla Bayesian coresets — Fig. 2, Fig. 3-left

```bash
python -m src.experiments.riemann_linear_regression.reproduce --trials 10 --M 300
```

Outputs in `src/experiments/riemann_linear_regression/out/`:

* `fig3_kl.png` — forward `KL(π̂_w ‖ π)` vs coreset size for GIGA / A-IHT / A-IHT II / Uniform
* `fig2_coreset_points.png` — selected coreset points sized by weight for `k ∈ {220,260,300}`

A synthetic 2-D spatial regression is used by default. To use the UK
house-price data from the paper, supply a preprocessed `[lat, lon, price]`
array: `--prices2018 prices2018.npy`
(source: <https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads>).

------------------------------

## Method ↔ paper (summary; full detail in `REPRODUCE.md`)

* `P_{θ,w}(Dⁱ)` is the Bayesian-coreset weighted likelihood
  `log P_{θ,w}(Dⁱ) = Σ_j w_j log p_θ(Dⁱ_j)`, `w ≥ 0`, `‖w‖₀ ≤ n_k`. Eq. 9 as
  printed omits the `w_j` inside the sum (typo).
* The coreset weights are optimised by an **alternating** scheme (Algorithm 1):
  A-IHT II on the quadratic coreset-matching term, with the posteriors held
  fixed; `KL(q_w ‖ q)` is the diagnostic (Fig. 3a) and the outer stop criterion.
  The bilevel implicit gradient of Proposition 1 is the justification, not the
  practical update.
* One `ClientUpdate` (R local reparam-SGD rounds) per A-IHT invocation; a few
  outer `(ClientUpdate → A-IHT)` steps per communication round.

------------------------------

### Citation
```
@inproceedings{
      chanda2024bayesian,
      title={Bayesian Coreset Optimization for Personalized Federated Learning},
      author={Prateek Chanda and Shrey Modi and Ganesh Ramakrishnan},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=uz7d2N2zul}
      }
```
