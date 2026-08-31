# Bayesian Coreset Optimization for Personalized Federated Learning

[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202024-8C1B13)](https://openreview.net/forum?id=uz7d2N2zul)
[![Paper](https://img.shields.io/badge/Paper-PDF-B21A1B)](https://openreview.net/pdf?id=uz7d2N2zul)
[![Project Page](https://img.shields.io/badge/Project-Page-1F6FEB)](https://coresetfederatedlearning.github.io/)
[![Poster](https://img.shields.io/badge/ICLR%202024-Poster-4B4B77)](https://iclr.cc/media/PosterPDFs/ICLR%202024/17557.png?t=1713590458.7719705)
[![Slides](https://img.shields.io/badge/ICLR%202024-Slides-4B4B77)](https://iclr.cc/media/iclr-2024/Slides/17557.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-3DA639)](./LICENSE)

**Authors**: **Prateek Chanda**$^{1}$ &nbsp; **Shrey Modi**$^{1}$ &nbsp; **Ganesh Ramakrishnan**$^{1}$

$^{1}$ Department of Computer Science, Indian Institute of Technology Bombay, India

Published at **ICLR 2024** &nbsp;·&nbsp; [OpenReview](https://openreview.net/forum?id=uz7d2N2zul) &nbsp;·&nbsp; [Conference page](https://iclr.cc/virtual/2024/poster/17557) &nbsp;·&nbsp; [Project page](https://coresetfederatedlearning.github.io/)

## Overview

**TL;DR** — A personalized coreset-weighted federated learning setup where each
client's training updates are computed from a small set of representative
(coreset) data points instead of its entire dataset.

We propose **CORESET-PFEDBAYES**: for each client `i` a weight vector `wᵢ`
(`‖wᵢ‖₀ ≤ n_k`) is chosen so that the coreset-weighted posterior stays close to
the full-data posterior while matching the coreset log-likelihood to the full
log-likelihood. Through theoretical analysis we show the averaged generalization
error is minimax optimal up to logarithmic terms — upper bounded by
$\mathcal{O}(n_k^{-\frac{2 \beta}{2 \beta+\boldsymbol{\Lambda}}} \log ^{2 \delta^{\prime}}(n_k))$
with a lower bound of
$\mathcal{O}(n_k^{-\frac{2 \beta}{2 \beta+\boldsymbol{\Lambda}}})$ — and that the
gap to a vanilla federated setup is a closed-form function
${\boldsymbol{\Im}}(\boldsymbol{w}, n_k)$ of the coreset weights and coreset
size.

------------------------------

## Repository contents

Implementation of **Algorithm 1 (CoreSet-PFedBayes)** with drivers for the three
experiment families in the paper:

| Paper | Driver | Output |
|---|---|---|
| Benchmark FL — Table 1 / 3, Fig. 3–5 (MNIST, FashionMNIST, CIFAR-10) | `scripts/run_benchmark.py`, `main.py` | accuracy / KL / comm-round figures, summary table |
| Medical datasets — Table 2 (OCTMNIST; COVID-19 Radiography / APTOS 2019 via a folder path) | `src/experiments/medical/run_medical.py` | class-wise accuracy table |
| Vanilla Bayesian coresets — Fig. 2, Fig. 3-left | `src/experiments/riemann_linear_regression/reproduce.py` | coreset-point + KL-vs-size figures |

### Layout

```
main.py                                  # benchmark FL entry point (one method / dataset)
scripts/run_benchmark.py                 # data-gen + 3 methods + figures + table
scripts/run_paper_cpu.sh                 # 10-client, 40-round CPU config used for the paper-scale runs
src/model.py                             # mean-field Gaussian BNN + weighted ELBO (Eq. 1 / 5 / 9)
src/bayesianCoresets/accelerated_iht.py  # Accelerated-IHT II (Algorithm 2)
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

## Datasets

| Dataset | Used for | How to get it |
|---|---|---|
| MNIST · FashionMNIST · CIFAR-10 | Table 1 / 3, Fig. 3–5 | `python scripts/download_datasets.py benchmark` (torchvision, auto) |
| OCTMNIST | Table 2 | `python scripts/download_datasets.py octmnist` (`medmnist`, auto) |
| UK Price Paid 2018 | Fig. 2, Fig. 3-left | `python scripts/download_datasets.py prices2018` (gov.uk CSV + postcodes.io geocoding, auto) |
| COVID-19 Radiography Database | Table 2 | `python scripts/download_datasets.py covid` — needs `~/.kaggle/kaggle.json` |
| APTOS 2019 Blindness Detection | Table 2 | `python scripts/download_datasets.py aptos` — needs a Kaggle token + accepted competition rules |

`python scripts/download_datasets.py all` fetches everything that needs no
credentials. The per-experiment sections below also generate their own data if
you skip this step.

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
(Kaggle token required):

```bash
python scripts/download_datasets.py covid          # -> data/medical_src/covid/...
python -m utils.medical_data --name covid --image_root data/medical_src/covid/COVID-19_Radiography_Dataset
python -m src.experiments.medical.run_medical --name covid --seeds 3
```

Any `ImageFolder` layout (`<root>/<class_name>/*.png`) works with
`--image_root`; the rest of the pipeline is unchanged.

------------------------------

## 3. Vanilla Bayesian coresets — Fig. 2, Fig. 3-left

```bash
python scripts/download_datasets.py prices2018          # UK Price Paid 2018 -> [lat, lon, price]
python -m src.experiments.riemann_linear_regression.reproduce --trials 10 --M 300
```

Outputs in `src/experiments/riemann_linear_regression/out/`:

* `fig3_kl.png` — forward `KL(π̂_w ‖ π)` vs coreset size for GIGA / A-IHT / A-IHT II / Uniform
* `fig2_coreset_points.png` — selected coreset points sized by weight for `k ∈ {220,260,300}`

`reproduce.py` uses
`src/experiments/riemann_linear_regression/data/prices2018.npy` if present
(built by the download script from the gov.uk
[Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
+ postcodes.io geocoding), otherwise it falls back to a synthetic 2-D spatial
regression. Pass `--prices2018 <path>` to point elsewhere.

------------------------------

### Scope

* `CoreSet-PFedBayes`, `PFedBayes` and `RandomSubset` are implemented; the other
  Table 1 baselines (FedAvg, BNFed, pFedMe, perFedAvg) are out of scope here.
* Medical experiments run on OCTMNIST out of the box; COVID-19 Radiography and
  APTOS 2019 are supported via `--image_root`.
* The Fig. 2 / 3 driver uses a synthetic 2-D spatial dataset by default, or the
  UK house-price array via `--prices2018`.
* Use the `paper` preset (not `smoke`) for accuracy numbers comparable to the
  paper.

------------------------------

## Citation

```bibtex
@inproceedings{chanda2024bayesian,
  title     = {Bayesian Coreset Optimization for Personalized Federated Learning},
  author    = {Prateek Chanda and Shrey Modi and Ganesh Ramakrishnan},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=uz7d2N2zul}
}
```
