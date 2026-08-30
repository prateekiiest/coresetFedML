"""Slim FL server base: client registry, beta-mixed aggregation, result I/O."""

import os

import h5py
import numpy as np
import torch


class Server:
    def __init__(self, global_model, cfg):
        self.model = global_model
        self.cfg = cfg
        self.beta = cfg["beta"]
        self.users = []
        self.history = {"per_acc": [], "glob_acc": [], "train_acc": [], "train_loss": [],
                        "kl_qw_q": [], "round": []}

    # ----------------------------------------------------------- aggregation
    def broadcast(self):
        mus, rhos = self.model.detached_params()
        for u in self.users:
            u.set_global(mus, rhos)

    def aggregate(self, selected):
        """v^{t+1} = (1-beta) v^t + beta * sum_i (n_i / sum n) v^{t+1}_i   (Alg. 1)."""
        total = sum(u.train_samples for u in selected)
        prev_mus, prev_rhos = self.model.detached_params()
        agg_mus = [np.zeros_like(m.detach().cpu().numpy()) for m in self.model.mus]
        agg_rhos = [np.zeros_like(r.detach().cpu().numpy()) for r in self.model.rhos]
        for u in selected:
            mus, rhos = u.upload()
            ratio = u.train_samples / total
            for k in range(len(agg_mus)):
                agg_mus[k] += ratio * mus[k].cpu().numpy()
                agg_rhos[k] += ratio * rhos[k].cpu().numpy()
        new_mus = [
            (1 - self.beta) * pm + self.beta * torch.as_tensor(am, device=pm.device, dtype=pm.dtype)
            for pm, am in zip(prev_mus, agg_mus)
        ]
        new_rhos = [
            (1 - self.beta) * pr + self.beta * torch.as_tensor(ar, device=pr.device, dtype=pr.dtype)
            for pr, ar in zip(prev_rhos, agg_rhos)
        ]
        self.model.load_from(new_mus, new_rhos)

    def select(self, rng):
        s = self.cfg["clients_per_round"]
        if s >= len(self.users):
            return list(self.users)
        idx = rng.choice(len(self.users), size=s, replace=False)
        return [self.users[i] for i in idx]

    # ----------------------------------------------------------- evaluation
    def evaluate(self, round_idx):
        agg = {k: 0 for k in ("pc", "pn", "gc", "gn", "tc", "tn")}
        tot_nll = 0.0
        kls = []
        for u in self.users:
            r = u.evaluate()
            agg["pc"] += r["per_correct"]; agg["pn"] += r["per_n"]
            agg["gc"] += r["glob_correct"]; agg["gn"] += r["glob_n"]
            agg["tc"] += r["train_correct"]; agg["tn"] += r["train_n"]
            tot_nll += r["train_nll"]
            if not np.isnan(r["kl_qw_q"]):
                kls.append(r["kl_qw_q"])
        per = agg["pc"] / max(1, agg["pn"])
        glob = agg["gc"] / max(1, agg["gn"])
        tra = agg["tc"] / max(1, agg["tn"])
        loss = tot_nll / max(1, agg["tn"])
        kl = float(np.mean(kls)) if kls else float("nan")
        self.history["round"].append(round_idx)
        self.history["per_acc"].append(per)
        self.history["glob_acc"].append(glob)
        self.history["train_acc"].append(tra)
        self.history["train_loss"].append(loss)
        self.history["kl_qw_q"].append(kl)
        print(
            f"[round {round_idx:3d}] per_acc={per:.4f} glob_acc={glob:.4f} "
            f"train_acc={tra:.4f} train_loss={loss:.3f} kl(q_w||q)={kl:.3f}"
        )
        return per, glob

    # ----------------------------------------------------------- persistence
    def save_results(self, tag, out_dir="results"):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{tag}.h5")
        with h5py.File(path, "w") as hf:
            for k, v in self.history.items():
                hf.create_dataset(k, data=np.asarray(v, dtype=np.float64))
        print(f"[server] results -> {path}")
        return path
