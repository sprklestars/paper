"""记录实测环境与 CIFAR 来源；不下载数据、不改变实验参数。

根目录运行：.venv/bin/python exp/tools/provenance.py --export main
数据核对：.venv-torch/bin/python exp/tools/provenance.py --export torch --data
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exp" / "provenance"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_record(a):
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": a.dtype.str, "order": "C",
            "sha256_bytes": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def environment():
    packages = dict(sorted((d.metadata["Name"], d.version)
                           for d in importlib.metadata.distributions()))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    data = dict(python=sys.version, executable=sys.executable,
                platform=platform.platform(), machine=platform.machine(),
                packages=packages, numpy_configuration=stream.getvalue(),
                thread_environment={k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    data["environment_id"] = hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()).hexdigest()
    return data


def data_manifest():
    import pyarrow.parquet as pq
    from PIL import Image
    base = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
    mirror = "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
    records = {}
    for split, count in (("train", 20000), ("test", 2000)):
        src = ROOT / "exp" / "data" / f"cifar10_{split}.parquet"
        cache = src.with_suffix(".npz")
        table = pq.read_table(src).slice(0, count)
        images = np.stack([np.array(Image.open(io.BytesIO(r["bytes"])).convert("RGB"))
                           for r in table.column("img").to_pylist()])
        labels = np.array(table.column("label").to_pylist(), dtype=np.int64)
        with np.load(cache) as z:
            assert np.array_equal(images, z["X"]), (split, "images")
            assert np.array_equal(labels, z["y"]), (split, "labels")
            arrays = {k: array_record(z[k]) for k in ("X", "y")}
        records[split] = dict(source_url=base + f"{split}-00000-of-00001.parquet",
            downloaded_from=mirror + f"{split}-00000-of-00001.parquet",
            source_revision="main; immutable upstream revision not recorded",
            parquet_sha256=file_hash(src), npz_container_sha256=file_hash(cache),
            row_rule=f"first {count} rows in stored parquet order; no shuffle",
            row_indices_zero_based=[0, count - 1], arrays=arrays,
            decoded_parquet_matches_npz=True)
    return dict(dataset="uoft-cs/cifar10", splits=records,
                partition_record="exp/results/e10_partition.json (actual indices written by E10)",
                caveat="Container hash may vary after recompression; compare array dtype, shape and byte hash separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--export", choices=("main", "torch"))
    parser.add_argument("--data", action="store_true")
    args = parser.parse_args()
    env = environment()
    if args.probe:
        print(json.dumps(env))
        return
    if not args.export:
        parser.error("需要 --probe 或 --export")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"environment-{args.export}.json").write_text(
        json.dumps(env, indent=2), encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                            check=True, capture_output=True, text=True).stdout
    (OUT / f"requirements-{args.export}.lock.txt").write_text(freeze, encoding="utf-8")
    if args.data:
        (OUT / "cifar10.json").write_text(json.dumps(data_manifest(), indent=2), encoding="utf-8")
    print(f"已记录 {args.export} 环境 {env['environment_id']}；数据核对={args.data}")


if __name__ == "__main__":
    main()
