"""
Non-i.i.d. client data generation for the benchmark experiments
(MNIST / FashionMNIST / CIFAR-10), following the sharding strategy of
T. Dinh et al. (2020) "Personalized Federated Learning with Moreau Envelopes"
that the paper (Sec. 7.2) says it follows.

Each client receives samples from only ``classes_per_user`` classes.  Output is
written in the LEAF-style JSON layout that ``utils.model_utils.read_data``
already expects:

    data/<Dataset>/train/train.json
    data/<Dataset>/test/test.json

with ``{"users": [...], "user_data": {uid: {"x": [[...]], "y": [...]}}}``.

Run directly, e.g.::

    python -m utils.data_gen --dataset Mnist --num_users 3 --n_train 600 \
        --n_test 200 --classes_per_user 3
"""

import argparse
import json
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

DATASETS = {
    "Mnist": ("MNIST", (1, 28, 28)),
    "FMnist": ("FashionMNIST", (1, 28, 28)),
    "Cifar": ("CIFAR10", (3, 32, 32)),
}


def _load_raw(name, root):
    tv_name, _ = DATASETS[name]
    cls = getattr(torchvision.datasets, tv_name)
    tfm = T.ToTensor()
    train = cls(root=root, train=True, download=True, transform=tfm)
    test = cls(root=root, train=False, download=True, transform=tfm)

    def to_np(ds):
        xs = torch.stack([ds[i][0] for i in range(len(ds))]).view(len(ds), -1).numpy()
        ys = np.array([int(ds[i][1]) for i in range(len(ds))], dtype=np.int64)
        return xs.astype(np.float32), ys

    return to_np(train), to_np(test)


def _by_class(y, num_classes):
    idx = {c: np.where(y == c)[0] for c in range(num_classes)}
    for c in idx:
        np.random.shuffle(idx[c])
    return idx


def partition(name, num_users, n_train, n_test, classes_per_user, seed, root):
    """Return ``(users, train_data, test_data)`` in LEAF dict form."""
    np.random.seed(seed)
    (xtr, ytr), (xte, yte) = _load_raw(name, root)
    num_classes = int(ytr.max()) + 1

    tr_idx = _by_class(ytr, num_classes)
    te_idx = _by_class(yte, num_classes)
    tr_ptr = {c: 0 for c in range(num_classes)}
    te_ptr = {c: 0 for c in range(num_classes)}

    users, train_data, test_data = [], {}, {}
    per_class_tr = max(1, n_train // classes_per_user)
    per_class_te = max(1, n_test // classes_per_user)

    for u in range(num_users):
        uid = f"f_{u:05d}"
        users.append(uid)
        # rotate the class window so coverage is balanced across clients
        cls_start = (u * classes_per_user) % num_classes
        chosen = [(cls_start + k) % num_classes for k in range(classes_per_user)]

        def take(cls_list, src_idx, ptr, per_class, src_x, src_y):
            xs, ys = [], []
            for c in cls_list:
                avail = src_idx[c]
                s = ptr[c]
                e = min(s + per_class, len(avail))
                if e <= s:  # wrap around if this class is exhausted
                    np.random.shuffle(avail)
                    s, e = 0, min(per_class, len(avail))
                sel = avail[s:e]
                ptr[c] = e
                xs.append(src_x[sel])
                ys.append(src_y[sel])
            X = np.concatenate(xs, 0)
            Y = np.concatenate(ys, 0)
            perm = np.random.permutation(len(Y))
            return X[perm], Y[perm]

        Xtr, Ytr = take(chosen, tr_idx, tr_ptr, per_class_tr, xtr, ytr)
        Xte, Yte = take(chosen, te_idx, te_ptr, per_class_te, xte, yte)
        train_data[uid] = {"x": Xtr.tolist(), "y": Ytr.tolist()}
        test_data[uid] = {"x": Xte.tolist(), "y": Yte.tolist()}

    return users, train_data, test_data


def write_leaf(name, users, train_data, test_data, out_root="data"):
    base = os.path.join(out_root, name)
    os.makedirs(os.path.join(base, "train"), exist_ok=True)
    os.makedirs(os.path.join(base, "test"), exist_ok=True)
    with open(os.path.join(base, "train", "train.json"), "w") as f:
        json.dump({"users": users, "user_data": train_data}, f)
    with open(os.path.join(base, "test", "test.json"), "w") as f:
        json.dump({"users": users, "user_data": test_data}, f)
    print(f"[data_gen] wrote {name}: {len(users)} clients -> {base}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Mnist", choices=list(DATASETS))
    p.add_argument("--num_users", type=int, default=3)
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--classes_per_user", type=int, default=3)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--torch_root", default="./_torchvision")
    p.add_argument("--out_root", default="data")
    a = p.parse_args()
    users, tr, te = partition(
        a.dataset, a.num_users, a.n_train, a.n_test,
        a.classes_per_user, a.seed, a.torch_root,
    )
    write_leaf(a.dataset, users, tr, te, a.out_root)


if __name__ == "__main__":
    main()
