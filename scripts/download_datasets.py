#!/usr/bin/env python
"""
One-stop dataset setup for every experiment in the paper.

    python scripts/download_datasets.py all           # everything that needs no credentials
    python scripts/download_datasets.py benchmark     # MNIST / FashionMNIST / CIFAR-10 (torchvision)
    python scripts/download_datasets.py octmnist      # OCTMNIST (medmnist)
    python scripts/download_datasets.py prices2018    # UK house-price array for Fig. 2 / 3
    python scripts/download_datasets.py covid         # COVID-19 Radiography  (needs a Kaggle token)
    python scripts/download_datasets.py aptos         # APTOS 2019            (needs a Kaggle token)

Sources
-------
* MNIST / FashionMNIST / CIFAR-10 : torchvision, non-i.i.d. sharded by ``utils.data_gen``
* OCTMNIST                        : ``medmnist`` package (Zenodo)
* prices2018                      : gov.uk Price Paid Data + postcodes.io geocoding
* COVID-19 Radiography Database   : Kaggle ``tawsifurrahman/covid19-radiography-database``
* APTOS 2019 Blindness Detection  : Kaggle competition ``aptos2019-blindness-detection``

Kaggle datasets need ``~/.kaggle/kaggle.json`` (Account -> Create New Token) and
``pip install kaggle``; the competition also needs its rules accepted on the site.
"""

import argparse
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BENCH = [("Mnist", 784), ("FMnist", 784), ("Cifar", 3072)]
PRICES_URL = (
    "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2018.csv"
)  # 301 -> prod2.<...>, ~180 MB, text/csv
POSTCODES_API = "https://api.postcodes.io/postcodes"


# --------------------------------------------------------------- benchmark
def get_benchmark(num_users=10, n_train=1000, n_test=400, classes_per_user=3):
    from utils import data_gen

    for name, _ in BENCH:
        users, tr, te = data_gen.partition(
            name, num_users, n_train, n_test, classes_per_user,
            seed=2020, root="./_torchvision",
        )
        data_gen.write_leaf(name, users, tr, te)


# --------------------------------------------------------------- octmnist
def get_octmnist(per_class=800):
    subprocess.run(
        [sys.executable, "-m", "utils.medical_data", "--name", "octmnist",
         "--per_class", str(per_class)],
        check=True,
    )


# --------------------------------------------------------------- kaggle
def _kaggle(args, cwd):
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        sys.exit(
            "No ~/.kaggle/kaggle.json found.\n"
            "  1. https://www.kaggle.com/settings -> Account -> Create New Token\n"
            "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n"
            "  3. pip install kaggle"
        )
    try:
        subprocess.run(["kaggle", *args], check=True, cwd=cwd)
    except FileNotFoundError:
        sys.exit("kaggle CLI not installed:  pip install kaggle")


def get_covid(out="data/medical_src/covid"):
    os.makedirs(out, exist_ok=True)
    _kaggle(["datasets", "download", "-d",
             "tawsifurrahman/covid19-radiography-database", "--unzip"], cwd=out)
    print(f"[covid] extracted under {out}/  -- point --image_root at the class folders "
          f"(e.g. {out}/COVID-19_Radiography_Dataset)")


def get_aptos(out="data/medical_src/aptos"):
    os.makedirs(out, exist_ok=True)
    _kaggle(["competitions", "download", "-c", "aptos2019-blindness-detection"], cwd=out)
    for z in os.listdir(out):
        if z.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(out, z)) as zf:
                zf.extractall(out)
    print(f"[aptos] extracted under {out}/ (train_images/ + train.csv). "
          f"Build an ImageFolder by diagnosis grade before running run_medical.")


# --------------------------------------------------------------- prices2018
def get_prices2018(n_subsample=1500, out="src/experiments/riemann_linear_regression/data/prices2018.npy",
                   max_scan=250_000):
    import numpy as np

    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"[prices2018] streaming {PRICES_URL} (first {max_scan} rows)")
    rows = []
    req = urllib.request.Request(PRICES_URL, headers={"User-Agent": "coresetFedML/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        for i, raw in enumerate(io.TextIOWrapper(resp, encoding="utf-8", errors="ignore")):
            if i >= max_scan:
                break
            parts = raw.split('","')
            if len(parts) < 4:
                continue
            try:
                price = float(parts[1].strip('"'))
            except ValueError:
                continue
            pc = parts[3].strip('"').strip()
            if price > 0 and pc:
                rows.append((pc, price))
    print(f"[prices2018] {len(rows)} rows with a postcode; sampling {n_subsample}")
    rng = np.random.default_rng(2020)
    sample = [rows[i] for i in rng.permutation(len(rows))[:n_subsample]]

    out_rows = []
    for s in range(0, len(sample), 100):
        chunk = sample[s : s + 100]
        body = json.dumps({"postcodes": [pc for pc, _ in chunk]}).encode()
        req = urllib.request.Request(POSTCODES_API, data=body,
                                     headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=60).read())["result"]
        for (pc, price), r in zip(chunk, res):
            g = r.get("result")
            if g and g.get("latitude") is not None:
                out_rows.append((g["latitude"], g["longitude"], price))
        print(f"  geocoded {len(out_rows)}/{len(sample)}", end="\r")
    arr = np.asarray(out_rows, dtype=np.float64)
    np.save(out, arr)
    print(f"\n[prices2018] wrote {out}  shape={arr.shape}  "
          f"(columns: lat, lon, price -- reproduce.py applies log10 to price)")


# --------------------------------------------------------------- main
TARGETS = {
    "benchmark": get_benchmark, "octmnist": get_octmnist,
    "prices2018": get_prices2018, "covid": get_covid, "aptos": get_aptos,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=[*TARGETS, "all"])
    ap.add_argument("--n_subsample", type=int, default=1500)
    a = ap.parse_args()
    if a.target == "all":
        get_benchmark()
        get_octmnist()
        get_prices2018(a.n_subsample)
        print("\nKaggle datasets (covid, aptos) skipped -- run them explicitly with a token.")
    elif a.target == "prices2018":
        get_prices2018(a.n_subsample)
    else:
        TARGETS[a.target]()


if __name__ == "__main__":
    main()
