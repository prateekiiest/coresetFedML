"""
Medical-dataset loader for the Table 2 experiments.

Default dataset is **OCTMNIST** (via the ``medmnist`` package -- freely
downloadable, no login).  COVID-19 Radiography / APTOS 2019 are gated Kaggle
downloads; point ``--image_root`` at an ``ImageFolder``-style directory
(``<root>/<class_name>/*.png``) to use them with the same pipeline.

Two clients, three classes: one shared "normal" class plus one distinct class
each (paper Sec. 10.4 / Fig. 6).  Images are embedded once with an
ImageNet-pretrained ResNet-18 (penultimate 512-d features) and cached to
``data/medical/<name>_client{0,1}.npz`` with keys ``x`` (float32 [n,512]),
``y`` (int64 [n]), ``classes`` (list[str]).
"""

import argparse
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T


def _resnet18_embedder(device):
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    return net


@torch.no_grad()
def _embed(net, imgs, device, bs=128):
    """imgs: uint8/float tensor [n,3,H,W] already resized/normalised."""
    out = []
    for s in range(0, len(imgs), bs):
        out.append(net(imgs[s : s + bs].to(device)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


_NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _prep(x):
    """x: numpy [n,H,W] or [n,H,W,C] in [0,255] or [0,1] -> tensor [n,3,224,224]."""
    t = torch.as_tensor(np.asarray(x))
    if t.ndim == 3:
        t = t.unsqueeze(-1).repeat(1, 1, 1, 3)
    t = t.permute(0, 3, 1, 2).float()
    if t.max() > 1.5:
        t = t / 255.0
    t = torch.nn.functional.interpolate(t, size=224, mode="bilinear", align_corners=False)
    return _NORM(t)


def load_octmnist():
    import medmnist
    from medmnist import INFO

    info = INFO["octmnist"]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split="train", download=True, size=28)
    x = ds.imgs  # [n,28,28] uint8
    y = ds.labels.reshape(-1).astype(np.int64)
    names = [info["label"][str(i)] for i in range(len(info["label"]))]
    return x, y, names


def load_imagefolder(root):
    ds = torchvision.datasets.ImageFolder(root, transform=T.Compose([T.Resize((64, 64)), T.ToTensor()]))
    xs = torch.stack([ds[i][0] for i in range(len(ds))]).permute(0, 2, 3, 1).numpy()
    ys = np.array([ds[i][1] for i in range(len(ds))], dtype=np.int64)
    return (xs * 255).astype(np.uint8), ys, ds.classes


def build(name, image_root, out_dir, per_class, seed, device):
    rng = np.random.default_rng(seed)
    if image_root:
        x, y, names = load_imagefolder(image_root)
    else:
        x, y, names = load_octmnist()

    classes = sorted(np.unique(y).tolist())[:3]
    shared, a_cls, b_cls = classes[0], classes[1], classes[2]

    def sample(c, k):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        return idx[:k]

    net = _resnet18_embedder(device)
    os.makedirs(out_dir, exist_ok=True)
    layout = {
        0: [shared, a_cls],
        1: [shared, b_cls],
    }
    for cid, cls_list in layout.items():
        idx = np.concatenate([sample(c, per_class) for c in cls_list])
        rng.shuffle(idx)
        feats = _embed(net, _prep(x[idx]), device)
        path = os.path.join(out_dir, f"{name}_client{cid}.npz")
        np.savez(path, x=feats, y=y[idx].astype(np.int64),
                 classes=np.array([names[c] for c in cls_list]))
        print(f"[medical_data] {path}  x={feats.shape}  classes={[names[c] for c in cls_list]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="octmnist")
    ap.add_argument("--image_root", default="")
    ap.add_argument("--out_dir", default="data/medical")
    ap.add_argument("--per_class", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    build(a.name, a.image_root, a.out_dir, a.per_class, a.seed, torch.device(a.device))


if __name__ == "__main__":
    main()
