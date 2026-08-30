"""
Per-client Bayesian coreset weight construction (paper Eq. 3-4, Prop. 2).

Given the current personal posterior ``q`` acting as the weighting distribution
pi-hat, we form the Monte-Carlo projected log-likelihood potentials

    g_hat_j = (1/sqrt(S)) * ( l_{1,j} - lbar_j, ..., l_{S,j} - lbar_j )    in R^S

where  l_{s,j} = log p_{theta_s}(D^i_j),  theta_s ~ q,  lbar_j = mean_s l_{s,j}.
Stacking columns gives  A = [g_hat_1 ... g_hat_n]  (S x n)  and the target is
the full-data sum  y = sum_j g_hat_j.  The coreset problem is

    argmin_w || y - A w ||_2^2   s.t.   ||w||_0 <= n_k,   w >= 0

solved with Accelerated-IHT II (Zhang et al., 2021; Algorithm 2).

``build_weights`` returns a length-n non-negative vector with <= n_k non-zeros.
"""

import numpy as np
import torch

from src.bayesianCoresets.accelerated_iht import AcceleratedIHT


@torch.no_grad()
def loglik_potentials(model, X, y_onehot, n_proj):
    """Return A (n_proj x n) and y (n_proj,) numpy arrays of centred potentials."""
    n = X.shape[0]
    ll = np.empty((n_proj, n), dtype=np.float64)
    for s in range(n_proj):
        logits = model.net(X, model.sample_params())
        logp = torch.log_softmax(logits, dim=1)
        ll[s] = (y_onehot * logp).sum(dim=1).cpu().numpy()  # l_{s,j}
    ll -= ll.mean(axis=0, keepdims=True)                     # centre over samples
    A = ll / np.sqrt(n_proj)
    return A, A.sum(axis=1)


def build_weights(model, X, y_onehot, n_k, n_proj=20, max_iter=300):
    """Coreset weights via A-IHT II.  Returns ``(w[np.float64, n], support[list])``."""
    n = X.shape[0]
    n_k = int(max(1, min(n_k, n)))
    if n_k >= n:
        return np.ones(n, dtype=np.float64), list(range(n))

    A, y = loglik_potentials(model, X, y_onehot, n_proj)
    solver = AcceleratedIHT(
        y=y.reshape(-1, 1), A=A, w=np.zeros((n, 1)), K=n_k, max_iter_num=max_iter
    )
    w, supp = solver.a_iht_ii()
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w[w < 0] = 0.0
    return w, list(supp)


def random_subset_weights(n, n_k, rng):
    """RANDOMSUBSET baseline: uniform mask of n_k points, weight n/n_k each."""
    n_k = int(max(1, min(n_k, n)))
    idx = rng.choice(n, size=n_k, replace=False)
    w = np.zeros(n, dtype=np.float64)
    w[idx] = n / n_k
    return w, sorted(idx.tolist())
