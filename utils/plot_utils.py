"""
Plotting / tabulation for the benchmark experiments.  Reads the per-run HDF5
files written by ``serverBaseClass.save_results`` (keys: round, per_acc,
glob_acc, train_acc, train_loss, kl_qw_q).

Figures produced (matplotlib, PNG):
  * accuracy vs communication round          -> results/fig_accuracy_<dataset>.png
  * KL(q_w || q) vs round  (Fig. 3a / 4)     -> results/fig_kl_<dataset>.png
  * test acc vs round for several n_k        -> results/fig_comm_rounds_<dataset>.png
And a text table (Table 3 style) to stdout / results/summary_<dataset>.txt
"""

import os

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = "results"


def load(tag, results=RESULTS):
    with h5py.File(os.path.join(results, f"{tag}.h5"), "r") as hf:
        return {k: np.asarray(hf[k]) for k in hf.keys()}


def _rounds_to_target(d, target):
    """First round index whose glob/per acc reaches ``target`` (else last)."""
    acc = np.maximum(d["per_acc"], d["glob_acc"])
    hit = np.where(acc >= target)[0]
    return int(d["round"][hit[0]]) if len(hit) else int(d["round"][-1])


def plot_accuracy(tags, labels, dataset, results=RESULTS):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tag, lab in zip(tags, labels):
        d = load(tag, results)
        ax[0].plot(d["round"], d["per_acc"], marker="o", ms=3, label=lab)
        ax[1].plot(d["round"], d["glob_acc"], marker="o", ms=3, label=lab)
    for a, t in zip(ax, ("Personal model", "Global model")):
        a.set_title(t); a.set_xlabel("communication round"); a.set_ylabel("test accuracy")
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout()
    out = os.path.join(results, f"fig_accuracy_{dataset}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[plot] {out}")


def plot_kl(tags, labels, dataset, results=RESULTS):
    fig, ax = plt.subplots(figsize=(6, 4))
    any_kl = False
    for tag, lab in zip(tags, labels):
        d = load(tag, results)
        kl = d["kl_qw_q"]
        m = ~np.isnan(kl)
        if m.any():
            any_kl = True
            ax.plot(d["round"][m], kl[m], marker="o", ms=3, label=lab)
    ax.set_xlabel("communication round")
    ax.set_ylabel(r"$D_{KL}(\hat q^i(\theta;w)\,\|\,\hat q^i(\theta))$")
    ax.set_title(f"Coreset / full posterior divergence ({dataset})")
    ax.grid(alpha=0.3)
    if any_kl:
        ax.legend()
    fig.tight_layout()
    out = os.path.join(results, f"fig_kl_{dataset}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[plot] {out}")


def plot_comm_rounds(tag_by_frac, dataset, results=RESULTS):
    fig, ax = plt.subplots(figsize=(6, 4))
    for frac, tag in sorted(tag_by_frac.items()):
        d = load(tag, results)
        ax.plot(d["round"], d["per_acc"], marker="o", ms=3, label=f"k = {int(frac*100)}%")
    ax.set_xlabel("communication round"); ax.set_ylabel("personal test accuracy")
    ax.set_title(f"Convergence vs coreset size ({dataset})")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out = os.path.join(results, f"fig_comm_rounds_{dataset}.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[plot] {out}")


def summarize(tags, labels, dataset, target=0.8, results=RESULTS):
    lines = [f"{'method':<28}{'per_acc':>10}{'glob_acc':>10}{'rounds@%.2f' % target:>14}"]
    lines.append("-" * len(lines[0]))
    for tag, lab in zip(tags, labels):
        d = load(tag, results)
        lines.append(
            f"{lab:<28}{d['per_acc'][-1]:>10.4f}{d['glob_acc'][-1]:>10.4f}"
            f"{_rounds_to_target(d, target):>14d}"
        )
    txt = "\n".join(lines)
    print(txt)
    out = os.path.join(results, f"summary_{dataset}.txt")
    with open(out, "w") as f:
        f.write(txt + "\n")
    print(f"[plot] {out}")
