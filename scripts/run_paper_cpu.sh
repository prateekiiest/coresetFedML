#!/bin/bash
# Paper-ish CPU runs: 10 clients, 3 classes/client, 1000 train each,
# lr=1e-3, zeta=10 (App. 10.6.1).  ~40 rounds is plenty for the personalized
# task to converge.  Runs the 3 methods for a dataset concurrently.
set -e
DS=${1:-Mnist}
ROUNDS=${2:-40}
COMMON="--dataset $DS --device cpu --num_users 10 --clients_per_round 10 \
  --num_glob_iters $ROUNDS --local_rounds 5 --batch_size 100 --mc 3 \
  --coreset_S 50 --coreset_outer_steps 3 --coreset_iters 300 \
  --zeta 10 --learning_rate 1e-3 --personal_learning_rate 1e-3"

mkdir -p results/logs
for M in coreset pfedbayes randomsubset; do
  PYTHONUNBUFFERED=1 python3 main.py --method $M --coreset_frac 0.5 $COMMON \
    > results/logs/${DS}_${M}.log 2>&1 &
done
wait
python3 - "$DS" <<'PY'
import sys
from utils import plot_utils
ds = sys.argv[1]
u = 10; b = 100
tags = [f"{ds}_{m}_frac0.5_z10.0_u{u}_b{b}_seed1" for m in ("coreset","pfedbayes","randomsubset")]
labs = ["CoreSet-PFedBayes (k=50%)", "PFedBayes (full)", "RandomSubset (50%)"]
plot_utils.plot_accuracy(tags, labs, ds)
plot_utils.plot_kl(tags, labs, ds)
plot_utils.summarize(tags, labs, ds)
PY
