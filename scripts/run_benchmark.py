#!/usr/bin/env python
"""
One-shot driver for the benchmark experiments: generate client data, run
CoreSet-PFedBayes + the PFedBayes / RandomSubset baselines (+ a coreset-size
sweep), then emit all figures and the summary table.

    python scripts/run_benchmark.py --dataset Mnist --preset smoke
    python scripts/run_benchmark.py --dataset Mnist --preset paper

``smoke`` is a CPU-sized sanity config; ``paper`` targets the settings in
Sec. 10.6 (10 clients, zeta=10, lr=1e-3).  Runtimes for ``paper`` on CPU are
long -- use a GPU box.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_gen, plot_utils  # noqa: E402

PRESETS = {
    "smoke": dict(
        num_users=3, classes_per_user=3, n_train=300, n_test=100,
        num_glob_iters=15, local_rounds=1, batch_size=64, mc=2,
        coreset_S=12, coreset_outer_steps=2, coreset_iters=80,
    ),
    "paper": dict(
        num_users=10, classes_per_user=3, n_train=1000, n_test=400,
        num_glob_iters=200, local_rounds=5, batch_size=100, mc=5,
        coreset_S=50, coreset_outer_steps=3, coreset_iters=300,
    ),
}
FRACS = [0.1, 0.15, 0.3, 0.5]


def run(method, dataset, frac, P, seed):
    tag = f"{dataset}_{method}_frac{frac}_z10.0_u{P['num_users']}_b{P['batch_size']}_seed{seed}"
    cmd = [
        sys.executable, "main.py", "--method", method, "--dataset", dataset,
        "--num_users", str(P["num_users"]), "--clients_per_round", str(P["num_users"]),
        "--num_glob_iters", str(P["num_glob_iters"]), "--local_rounds", str(P["local_rounds"]),
        "--batch_size", str(P["batch_size"]), "--mc", str(P["mc"]),
        "--coreset_frac", str(frac), "--coreset_S", str(P["coreset_S"]),
        "--coreset_outer_steps", str(P["coreset_outer_steps"]),
        "--coreset_iters", str(P["coreset_iters"]), "--seed", str(seed),
        "--zeta", "10.0", "--learning_rate", "1e-3", "--personal_learning_rate", "1e-3",
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Mnist", choices=["Mnist", "FMnist", "Cifar"])
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip_data", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="also run the coreset-size sweep")
    a = ap.parse_args()
    P = PRESETS[a.preset]

    if not a.skip_data:
        users, tr, te = data_gen.partition(
            a.dataset, P["num_users"], P["n_train"], P["n_test"],
            P["classes_per_user"], seed=2020, root="./_torchvision",
        )
        data_gen.write_leaf(a.dataset, users, tr, te)

    tags, labels = [], []
    for method, lab in [
        ("coreset", "CoreSet-PFedBayes (k=50%)"),
        ("pfedbayes", "PFedBayes (full)"),
        ("randomsubset", "RandomSubset (50%)"),
    ]:
        tags.append(run(method, a.dataset, 0.5, P, a.seed))
        labels.append(lab)

    plot_utils.plot_accuracy(tags, labels, a.dataset)
    plot_utils.plot_kl(tags, labels, a.dataset)
    plot_utils.summarize(tags, labels, a.dataset)

    if a.sweep:
        by_frac = {f: run("coreset", a.dataset, f, P, a.seed) for f in FRACS}
        plot_utils.plot_comm_rounds(by_frac, a.dataset)


if __name__ == "__main__":
    main()
