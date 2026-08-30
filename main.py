#!/usr/bin/env python
"""
Entry point for the benchmark experiments (MNIST / FashionMNIST / CIFAR-10).

Example (CPU smoke)::

    python -m utils.data_gen --dataset Mnist --num_users 3 --n_train 600 \
        --n_test 200 --classes_per_user 3
    python main.py --method coreset     --dataset Mnist --num_glob_iters 30
    python main.py --method pfedbayes   --dataset Mnist --num_glob_iters 30
    python main.py --method randomsubset --dataset Mnist --num_glob_iters 30
"""

import argparse
import random

import numpy as np
import torch

from src.model import federatedBNN
from src.serverModels.serverpFedbayes import pFedBayes

INPUT_DIM = {"Mnist": 784, "FMnist": 784, "Cifar": 3072}


def build_cfg(a):
    tag = (
        f"{a.dataset}_{a.method}_frac{a.coreset_frac}_z{a.zeta}"
        f"_u{a.num_users}_b{a.batch_size}_seed{a.seed}"
    )
    return {
        "dataset": a.dataset, "method": a.method,
        "num_users": a.num_users, "clients_per_round": a.clients_per_round,
        "num_glob_iters": a.num_glob_iters, "local_rounds": a.local_rounds,
        "batch_size": a.batch_size, "mc": a.mc,
        "lr": a.learning_rate, "personal_lr": a.personal_learning_rate,
        "beta": a.beta, "zeta": a.zeta,
        "coreset_frac": a.coreset_frac, "coreset_S": a.coreset_S,
        "coreset_outer_steps": a.coreset_outer_steps, "coreset_iters": a.coreset_iters,
        "coreset_tol": a.coreset_tol,
        "output_dim": 10, "hidden_dim": a.hidden_dim,
        "seed": a.seed, "device": torch.device(a.device), "tag": tag,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Mnist", choices=list(INPUT_DIM))
    p.add_argument("--method", default="coreset",
                   choices=["coreset", "pfedbayes", "randomsubset"])
    p.add_argument("--num_users", type=int, default=3)
    p.add_argument("--clients_per_round", type=int, default=3)
    p.add_argument("--num_glob_iters", type=int, default=30)
    p.add_argument("--local_rounds", type=int, default=1, help="R in Algorithm 1")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--mc", type=int, default=2, help="MC samples for the ELBO")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--personal_learning_rate", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--zeta", type=float, default=10.0)
    p.add_argument("--coreset_frac", type=float, default=0.5, help="n_k / n")
    p.add_argument("--coreset_S", type=int, default=20, help="projection dim for potentials")
    p.add_argument("--coreset_outer_steps", type=int, default=2)
    p.add_argument("--coreset_iters", type=int, default=200, help="max A-IHT iterations")
    p.add_argument("--coreset_tol", type=float, default=1e-2)
    p.add_argument("--hidden_dim", type=int, default=100)
    p.add_argument("--weight_scale", type=float, default=0.1)
    p.add_argument("--rho_offset", type=float, default=-3.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    cfg = build_cfg(a)
    print("=" * 72)
    for k, v in cfg.items():
        print(f"  {k:20s}: {v}")
    print("=" * 72)

    template = federatedBNN(
        INPUT_DIM[a.dataset], a.hidden_dim, 10, cfg["device"],
        a.weight_scale, a.rho_offset, a.zeta,
    )
    global_model = federatedBNN(
        INPUT_DIM[a.dataset], a.hidden_dim, 10, cfg["device"],
        a.weight_scale, a.rho_offset, a.zeta,
    )
    global_model.load_from(*template.detached_params())

    server = pFedBayes(global_model, template, cfg)
    path = server.train()
    print(f"done -> {path}")
    return path


if __name__ == "__main__":
    main()
