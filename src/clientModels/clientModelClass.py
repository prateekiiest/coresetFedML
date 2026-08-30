"""
CoreSet-PFedBayes client  (paper Algorithm 1, "Client Side Objective" column,
plus the coreset outer loop that in practice runs per client).

Per communication round the client:

  1. receives the global model z and copies it into ``self.z``;
  2. repeats, for ``coreset_outer_steps`` iterations (Sec. 4.3 alternation):
       a. ClientUpdate: ``local_rounds`` passes of reparameterised SGD that
          jointly update
             - q_w  via Eq. 9  (weighted NLL + zeta/Nb * KL(q_w || z)),
             - q    via Eq. 1  (NLL          + zeta/Nb * KL(q   || z)),
             - z    via Eq. 2  (zeta/Nb * KL( stop-grad(q_or_qw) || z));
       b. rebuild coreset weights w_i with A-IHT on the potentials of q
          (skipped for the baselines);
       c. record KL(q_w || q) and stop early once it plateaus;
  3. uploads the local z-track.

Methods:
  * ``coreset``      : full Algorithm 1 (q_w + A-IHT, z regularised to q_w)
  * ``pfedbayes``    : w_i == 1, no A-IHT, single posterior, z regularised to q
  * ``randomsubset`` : w_i = random n_k mask, no A-IHT, z regularised to q_w
"""

import copy

import numpy as np
import torch

from src.bayesianCoresets.coreset import build_weights, random_subset_weights
from src.model import gaussian_kl


class ClientModelClass:
    def __init__(self, base, template_model, cfg):
        self.b = base
        self.cfg = cfg
        self.method = cfg["method"]
        self.device = base.device
        self.rng = np.random.default_rng(abs(hash(("client", base.id))) % (2**32))

        self.z = copy.deepcopy(template_model).to(self.device)
        self.q = copy.deepcopy(template_model).to(self.device)
        self.q_w = copy.deepcopy(template_model).to(self.device)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=cfg["personal_lr"])
        self.opt_qw = torch.optim.Adam(self.q_w.parameters(), lr=cfg["personal_lr"])
        self.opt_z = torch.optim.Adam(self.z.parameters(), lr=cfg["lr"])

        n = self.b.train_samples
        self.n_k = int(round(cfg["coreset_frac"] * n))
        self.w = np.ones(n, dtype=np.float64)
        if self.method == "randomsubset":
            self.w, _ = random_subset_weights(n, self.n_k, self.rng)
        self.kl_hist = []

    # ------------------------------------------------------------ round hooks
    def set_global(self, mus, rhos):
        self.z.load_from(mus, rhos)

    def upload(self):
        return self.z.detached_params()

    @property
    def train_samples(self):
        return self.b.train_samples

    # ------------------------------------------------------------ local train
    def _local_rounds(self, use_qw):
        cfg, nb = self.cfg, self.b.n_batches
        for _ in range(cfg["local_rounds"]):
            for Xb, Yb, idx in self.b.batches():
                z_mus, z_rhos = self.z.detached_params()
                wb = (
                    torch.as_tensor(self.w[idx], dtype=torch.float32, device=self.device)
                    if use_qw
                    else None
                )

                # (1) q on full data -- Eq. 1
                self.opt_q.zero_grad()
                self.q.elbo(Xb, Yb, z_mus, z_rhos, nb, None, cfg["mc"]).backward()
                self.opt_q.step()

                # (2) q_w on coreset-weighted data -- Eq. 9
                if use_qw:
                    self.opt_qw.zero_grad()
                    self.q_w.elbo(Xb, Yb, z_mus, z_rhos, nb, wb, cfg["mc"]).backward()
                    self.opt_qw.step()

                # (3) local z-track -- Eq. 2 : zeta/Nb * KL( stop-grad(src) || z )
                src = self.q_w if use_qw else self.q
                s_mus, s_rhos = src.detached_params()
                self.opt_z.zero_grad()
                loss_z = self.z.zeta / nb * gaussian_kl(
                    s_mus, s_rhos, self.z.mus, self.z.rhos
                )
                loss_z.backward()
                self.opt_z.step()

    def local_train(self):
        use_qw = self.method in ("coreset", "randomsubset")
        outer = self.cfg["coreset_outer_steps"] if self.method == "coreset" else 1

        for _ in range(outer):
            self._local_rounds(use_qw)

            if self.method == "coreset" and self.n_k < self.b.train_samples:
                self.w, _ = build_weights(
                    self.q, self.b.X, self.b.Y_onehot, self.n_k,
                    n_proj=self.cfg["coreset_S"], max_iter=self.cfg["coreset_iters"],
                )

            if use_qw:
                with torch.no_grad():
                    qw_m, qw_r = self.q_w.detached_params()
                    q_m, q_r = self.q.detached_params()
                    self.kl_hist.append(float(gaussian_kl(qw_m, qw_r, q_m, q_r)))
                if len(self.kl_hist) >= 2 and abs(
                    self.kl_hist[-1] - self.kl_hist[-2]
                ) < self.cfg["coreset_tol"] * max(1.0, abs(self.kl_hist[-2])):
                    break

    # ------------------------------------------------------------ evaluation
    def evaluate(self):
        personal = self.q_w if self.method in ("coreset", "randomsubset") else self.q
        pc, pn = self.b._accuracy(personal, self.b.Xte, self.b.Yte)
        gc, gn = self.b._accuracy(self.z, self.b.Xte, self.b.Yte)
        tr_c, tr_n = self.b._accuracy(personal, self.b.X, self.b.Y)
        return {
            "per_correct": pc, "per_n": pn,
            "glob_correct": gc, "glob_n": gn,
            "train_correct": tr_c, "train_n": tr_n,
            "train_nll": self.b._nll_sum(personal, self.b.X, self.b.Y_onehot),
            "kl_qw_q": self.kl_hist[-1] if self.kl_hist else float("nan"),
        }
