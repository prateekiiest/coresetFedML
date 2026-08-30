"""Slim client base: data handling, mini-batching and evaluation."""

import numpy as np
import torch
import torch.nn.functional as F


class Client:
    def __init__(self, cid, train_data, test_data, output_dim, batch_size, device):
        self.id = cid
        self.device = device
        self.output_dim = output_dim
        self.batch_size = batch_size

        Xtr = torch.stack([x for x, _ in train_data]).view(len(train_data), -1).to(device)
        Ytr = torch.tensor([int(y) for _, y in train_data], device=device)
        Xte = torch.stack([x for x, _ in test_data]).view(len(test_data), -1).to(device)
        Yte = torch.tensor([int(y) for _, y in test_data], device=device)

        self.X, self.Y = Xtr, Ytr
        self.Y_onehot = F.one_hot(Ytr, output_dim).float()
        self.Xte, self.Yte = Xte, Yte
        self.train_samples = Xtr.shape[0]
        self.test_samples = Xte.shape[0]
        self.input_dim = Xtr.shape[1]
        self.n_batches = max(1, self.train_samples // batch_size)

    # -------------------------------------------------------------- batching
    def batches(self, shuffle=True):
        """Yield ``(Xb, Yb_onehot, idx)`` mini-batches; idx indexes into self.X."""
        n = self.train_samples
        order = np.random.permutation(n) if shuffle else np.arange(n)
        for s in range(0, n, self.batch_size):
            idx = order[s : s + self.batch_size]
            t = torch.as_tensor(idx, device=self.device, dtype=torch.long)
            yield self.X[t], self.Y_onehot[t], idx

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def _accuracy(self, model, X, Y, n_samples=5):
        pred = model.forward(X, n_samples=n_samples).argmax(dim=1)
        return (pred == Y).sum().item(), Y.shape[0]

    @torch.no_grad()
    def _nll_sum(self, model, X, Y_onehot, n_samples=5):
        logits = model.forward(X, n_samples=n_samples)  # already log-softmax
        return float(-(Y_onehot * logits).sum().item())
