"""Fetches CIFAR-10 once and decodes it into the npz e10_cifar_cnn.py reads.

Run in the torch venv:

    .venv-torch/bin/python3 exp/tools/fetch_cifar.py

该工具解码已下载的 uoft-cs/cifar10 parquet，不自行联网下载。
按存储顺序取前 20000 个训练样本、前 2000 个测试样本；不是随机抽样。
这些固定子集限定了实验范围，并未证明样本量不会影响结果。

运行 tools/provenance.py --export torch --data 可记录源文件、npz 容器和
数组内容的 SHA-256，并逐数组核对解码结果；不声称 parquet 字节与官方 tar 相同。
"""

import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = ("https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/"
        "plain_text/{}-00000-of-00001.parquet")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
KEEP = {"train": 20000, "test": 2000}


def main():
    os.makedirs(HERE, exist_ok=True)
    for split, n in KEEP.items():
        src = os.path.join(HERE, f"cifar10_{split}.parquet")
        if not os.path.exists(src):
            raise SystemExit(
                f"missing {src}\n  fetch it with:\n"
                f"  curl -L -o {src} {BASE.format(split)}")
        t = pq.read_table(src)
        rows = t.column("img").to_pylist()[:n]
        y = np.array(t.column("label").to_pylist()[:n], dtype=np.int64)
        X = np.stack([np.array(Image.open(io.BytesIO(r["bytes"])).convert("RGB"))
                      for r in rows])
        assert X.shape == (n, 32, 32, 3) and X.dtype == np.uint8, X.shape
        assert len(np.bincount(y)) == 10, "not all ten classes present"
        out = os.path.join(HERE, f"cifar10_{split}.npz")
        np.savez_compressed(out, X=X, y=y)
        print(f"{split}: {X.shape} -> {out}  class counts {np.bincount(y)}")


if __name__ == "__main__":
    main()
