"""
Mean-field Gaussian Bayesian neural network used by CoreSet-PFedBayes and the
PFedBayes baseline (Zhang et al., 2022b).

A variational posterior is a fully-factorised Gaussian over the T network
weights,  q(theta) = prod_m N(mu_m, softplus(rho_m)^2),  sampled with the
reparameterisation trick  theta = mu + softplus(rho) * eps,  eps ~ N(0, 1).

The class stores ONE posterior (mus, rhos).  A client owns several instances:
``q`` (personal, full data, Eq. 1), ``q_w`` (personal, coreset-weighted,
Eq. 5) and ``z`` (local copy of the global model, Eq. 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal


def gaussian_kl(mus_p, rhos_p, mus_q, rhos_q):
    """sum_m KL( N(mu_p, sp(rho_p)^2) || N(mu_q, sp(rho_q)^2) ) over all weights."""
    total = 0.0
    for mp, rp, mq, rq in zip(mus_p, rhos_p, mus_q, rhos_q):
        sp = F.softplus(rp)
        sq = F.softplus(rq)
        total = total + torch.sum(kl_divergence(Normal(mp, sp), Normal(mq, sq)))
    return total


class federatedBNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        device=torch.device("cpu"),
        weight_scale=0.1,
        rho_offset=-3,
        zeta=10,
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = 1  # weights: in->hidden, hidden->out  (== [in,hidden,out])
        self.weight_scale = weight_scale
        self.rho_offset = rho_offset
        self.register_buffer("zeta", torch.tensor(float(zeta)))

        self.layer_param_shapes = self._shapes()
        self.mus = nn.ParameterList()
        self.rhos = nn.ParameterList()
        for shape in self.layer_param_shapes:
            self.mus.append(
                nn.Parameter(torch.normal(torch.zeros(shape), weight_scale * torch.ones(shape)))
            )
            self.rhos.append(nn.Parameter(rho_offset + torch.zeros(shape)))
        self.to(device)

    # ------------------------------------------------------------------ shapes
    def _shapes(self):
        shapes = []
        dims = [self.input_dim] + [self.hidden_dim] * self.num_hidden_layers + [self.output_dim]
        for a, b in zip(dims[:-1], dims[1:]):
            shapes.append((a, b))  # weight
            shapes.append((b,))    # bias
        return shapes

    # ------------------------------------------------------ reparameterisation
    def sigmas(self, rhos=None):
        rhos = self.rhos if rhos is None else rhos
        return [F.softplus(r) for r in rhos]

    def sample_epsilons(self):
        return [torch.randn(s, device=self.device) for s in self.layer_param_shapes]

    def sample_params(self, mus=None, rhos=None, epsilons=None):
        mus = self.mus if mus is None else mus
        rhos = self.rhos if rhos is None else rhos
        epsilons = self.sample_epsilons() if epsilons is None else epsilons
        sig = self.sigmas(rhos)
        return [m + s * e for m, s, e in zip(mus, sig, epsilons)]

    # --------------------------------------------------------------- forward
    def net(self, X, params):
        h = X
        n_layers = len(params) // 2
        for i in range(n_layers - 1):
            h = F.relu(torch.mm(h, params[2 * i]) + params[2 * i + 1])
        return torch.mm(h, params[-2]) + params[-1]

    def forward(self, X, n_samples=1):
        """Predictive logits averaged over ``n_samples`` posterior draws."""
        logits = 0.0
        for _ in range(n_samples):
            logits = logits + F.log_softmax(self.net(X, self.sample_params()), dim=1)
        return logits / n_samples

    # --------------------------------------------------------------- losses
    @staticmethod
    def _nll(logits, y_onehot, weights=None):
        """Negative categorical log-likelihood, summed over the batch.

        ``weights`` (shape [B]) are the per-example coreset weights w_j; the
        weighted sum is re-scaled to the full-batch total  n / sum_batch(w)
        (self-normalised) so it estimates  -E_q[ sum_j w_j log p_theta(D_j) ].
        """
        logp = F.log_softmax(logits, dim=1)
        per_ex = -(y_onehot * logp).sum(dim=1)  # [B]
        if weights is None:
            return per_ex.sum()
        wsum = weights.sum().clamp_min(1e-8)
        return (weights * per_ex).sum() * (per_ex.numel() / wsum)

    def elbo(self, X, y_onehot, z_mus, z_rhos, num_batches, weights=None, n_samples=1):
        """-E_q[log lik] + zeta/num_batches * KL(q || z).   (Eq. 1 / Eq. 5, 9)

        ``z_mus, z_rhos`` are the (detached) parameters of the model this
        posterior is regularised towards (the local copy of the global model).
        """
        nll = 0.0
        for _ in range(n_samples):
            nll = nll + self._nll(self.net(X, self.sample_params()), y_onehot, weights)
        nll = nll / n_samples
        kl = gaussian_kl(self.mus, self.rhos, z_mus, z_rhos)
        return nll + self.zeta / num_batches * kl

    def kl_to(self, other_mus, other_rhos):
        """KL( this || other ), used for the server term (Eq. 2) and the
        coreset-matching diagnostic KL(q_w || q) (Sec. 4.3, Fig. 3a/4)."""
        return gaussian_kl(self.mus, self.rhos, other_mus, other_rhos)

    # --------------------------------------------------------------- utils
    def detached_params(self):
        return (
            [m.detach().clone() for m in self.mus],
            [r.detach().clone() for r in self.rhos],
        )

    def load_from(self, mus, rhos):
        with torch.no_grad():
            for p, s in zip(self.mus, mus):
                p.copy_(s)
            for p, s in zip(self.rhos, rhos):
                p.copy_(s)
