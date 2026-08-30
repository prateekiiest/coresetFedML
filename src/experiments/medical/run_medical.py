#!/usr/bin/env python
"""
Table 2 reconstruction: model-centric coreset selection (CoreSet-PFedBayes) vs
model-agnostic subset selection (random + diversity submodular functions) on
medical-image embeddings, 2 clients + 1 shared class.

Pipeline per method:
  1. per client, select a ``--frac`` subset of the local train set:
       * ``full``          -> keep everything (Vanilla FedAvg / PFedBayes)
       * ``random``        -> uniform subset
       * ``logdet`` / ``dispsum`` / ``dispmin``  -> submodlib maximisation
       * ``coreset``       -> A-IHT on BNN log-likelihood potentials
  2. train a classifier federated across the 2 clients:
       * ``coreset`` / ``pfedbayes`` -> mean-field BNN, CoreSet-PFedBayes loop
       * everything else            -> plain MLP + FedAvg
  3. report class-wise test accuracy (mean +/- std over ``--seeds`` seeds).

    python -m src.experiments.medical.run_medical --data data/medical \
        --name octmnist --seeds 3
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.bayesianCoresets.coreset import build_weights
from src.model import federatedBNN, gaussian_kl

METHODS = ["full", "random", "logdet", "dispsum", "dispmin", "coreset"]


# --------------------------------------------------------------- data utils
def load_clients(data_dir, name):
    clients = []
    for cid in (0, 1):
        d = np.load(os.path.join(data_dir, f"{name}_client{cid}.npz"), allow_pickle=True)
        clients.append((d["x"].astype(np.float32), d["y"].astype(np.int64),
                        [str(c) for c in d["classes"]]))
    return clients


def split(x, y, seed, test_frac=0.3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_te = int(len(y) * test_frac)
    te, tr = idx[:n_te], idx[n_te:]
    return x[tr], y[tr], x[te], y[te]


# --------------------------------------------------------- subset selection
def select_subset(method, x, y, frac, seed, device):
    n = len(y)
    k = max(1, int(round(frac * n)))
    if method == "full":
        return np.arange(n)
    if method == "random":
        return np.random.default_rng(seed).choice(n, size=k, replace=False)
    if method in ("logdet", "dispsum", "dispmin"):
        import submodlib

        kw = dict(n=n, mode="dense", data=x.astype(np.float64), metric="euclidean")
        if method == "logdet":
            fn = submodlib.LogDeterminantFunction(lambdaVal=1.0, **kw)
        elif method == "dispsum":
            fn = submodlib.DisparitySumFunction(**kw)
        else:
            fn = submodlib.DisparityMinFunction(**kw)
        greedy = fn.maximize(budget=k, optimizer="NaiveGreedy", stopIfZeroGain=False,
                             stopIfNegativeGain=False, verbose=False)
        return np.array([i for i, _ in greedy])
    if method == "coreset":
        # potentials from a quickly-fit BNN on the full local data
        model = federatedBNN(x.shape[1], 64, int(y.max()) + 1, device)
        Xt = torch.as_tensor(x, device=device)
        Yt = torch.as_tensor(y, device=device)
        Yoh = F.one_hot(Yt, int(y.max()) + 1).float()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        z_mus, z_rhos = model.detached_params()
        for _ in range(60):
            opt.zero_grad()
            model.elbo(Xt, Yoh, z_mus, z_rhos, 1, None, 2).backward()
            opt.step()
        w, supp = build_weights(model, Xt, Yoh, k, n_proj=30, max_iter=200)
        return np.array(supp if supp else np.argsort(-w)[:k])
    raise ValueError(method)


# ----------------------------------------------------------------- models
class MLP(torch.nn.Module):
    def __init__(self, d_in, n_cls):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 128), torch.nn.ReLU(), torch.nn.Linear(128, n_cls)
        )

    def forward(self, x):
        return self.net(x)


def fedavg(client_data, n_cls, device, rounds=40, local_epochs=2, lr=1e-3):
    d_in = client_data[0][0].shape[1]
    glob = MLP(d_in, n_cls).to(device)
    for _ in range(rounds):
        states = []
        for (xtr, ytr, _, _) in client_data:
            loc = MLP(d_in, n_cls).to(device)
            loc.load_state_dict(glob.state_dict())
            opt = torch.optim.Adam(loc.parameters(), lr=lr)
            Xt = torch.as_tensor(xtr, device=device)
            Yt = torch.as_tensor(ytr, device=device)
            for _ in range(local_epochs):
                opt.zero_grad()
                F.cross_entropy(loc(Xt), Yt).backward()
                opt.step()
            states.append({k: v.detach() for k, v in loc.state_dict().items()})
        new = {k: sum(s[k] for s in states) / len(states) for k in states[0]}
        glob.load_state_dict(new)
    return lambda x: glob(torch.as_tensor(x, device=device)).argmax(1).cpu().numpy()


def coreset_pfedbayes(client_data, n_cls, device, rounds=40, local_rounds=2,
                      zeta=10.0, lr=1e-3, use_weights=True):
    d_in = client_data[0][0].shape[1]
    glob = federatedBNN(d_in, 64, n_cls, device, zeta=zeta)
    personals = [federatedBNN(d_in, 64, n_cls, device, zeta=zeta) for _ in client_data]
    for p in personals:
        p.load_from(*glob.detached_params())

    for _ in range(rounds):
        uploads = []
        for (xtr, ytr, wtr, _), p in zip(client_data, personals):
            p.load_from(*glob.detached_params())
            z = federatedBNN(d_in, 64, n_cls, device, zeta=zeta)
            z.load_from(*glob.detached_params())
            optp = torch.optim.Adam(p.parameters(), lr=lr)
            optz = torch.optim.Adam(z.parameters(), lr=lr)
            Xt = torch.as_tensor(xtr, device=device)
            Yoh = F.one_hot(torch.as_tensor(ytr, device=device), n_cls).float()
            wb = (torch.as_tensor(wtr, dtype=torch.float32, device=device)
                  if (use_weights and wtr is not None) else None)
            for _ in range(local_rounds):
                zm, zr = z.detached_params()
                optp.zero_grad()
                p.elbo(Xt, Yoh, zm, zr, 1, wb, 3).backward()
                optp.step()
                pm, pr = p.detached_params()
                optz.zero_grad()
                (z.zeta * gaussian_kl(pm, pr, z.mus, z.rhos)).backward()
                optz.step()
            uploads.append(z.detached_params())
        gm = [sum(u[0][k] for u in uploads) / len(uploads) for k in range(len(uploads[0][0]))]
        gr = [sum(u[1][k] for u in uploads) / len(uploads) for k in range(len(uploads[0][1]))]
        glob.load_from(gm, gr)

    def predict_for(cid):
        m = personals[cid]
        return lambda x: m.forward(torch.as_tensor(x, device=device), 8).argmax(1).cpu().numpy()

    return predict_for


# ----------------------------------------------------------------- driver
def run_once(method, clients, frac, seed, device):
    n_cls = int(max(c[1].max() for c in clients)) + 1
    prepared, test_sets = [], []
    for cid, (x, y, cls_names) in enumerate(clients):
        xtr, ytr, xte, yte = split(x, y, seed + cid)
        sel = select_subset(method, xtr, ytr, frac, seed + cid, device)
        w = None
        if method == "coreset":
            w_full = np.zeros(len(ytr))
            w_full[sel] = len(ytr) / len(sel)
            xtr, ytr, w = xtr[sel], ytr[sel], w_full[sel]
        elif method != "full":
            xtr, ytr = xtr[sel], ytr[sel]
        prepared.append((xtr, ytr, w, None))
        test_sets.append((xte, yte, cls_names))

    if method in ("coreset", "pfedbayes"):
        pf = coreset_pfedbayes(prepared, n_cls, device, use_weights=(method == "coreset"))
        preds = [pf(cid) for cid in range(len(clients))]
    else:
        f = fedavg(prepared, n_cls, device)
        preds = [f for _ in clients]

    per_class = {}
    for cid, (xte, yte, cls_names) in enumerate(test_sets):
        yp = preds[cid](xte)
        for c in np.unique(yte):
            m = yte == c
            per_class.setdefault(int(c), []).append((yp[m] == yte[m]).mean())
    return {c: float(np.mean(v)) for c, v in per_class.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/medical")
    ap.add_argument("--name", default="octmnist")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/table2_octmnist.txt")
    a = ap.parse_args()
    device = torch.device(a.device)
    clients = load_clients(a.data, a.name)
    class_names = {}
    for _, y, names in clients:
        for c, nm in zip(sorted(np.unique(y)), names):
            class_names[int(c)] = nm

    rows = []
    for method in a.methods:
        accs = [run_once(method, clients, a.frac, 100 * s, device) for s in range(a.seeds)]
        agg = {c: (np.mean([d[c] for d in accs]), np.std([d[c] for d in accs]))
               for c in accs[0]}
        rows.append((method, agg))
        cells = "  ".join(f"{class_names[c]}={m:.3f}+-{s:.3f}" for c, (m, s) in sorted(agg.items()))
        print(f"{method:<10} {cells}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        header = ["method"] + [class_names[c] for c in sorted(class_names)]
        fh.write("\t".join(header) + "\n")
        for method, agg in rows:
            fh.write(method + "\t" + "\t".join(
                f"{agg[c][0]:.3f}±{agg[c][1]:.3f}" if c in agg else "-"
                for c in sorted(class_names)) + "\n")
    print(f"[medical] table -> {a.out}")


if __name__ == "__main__":
    main()
