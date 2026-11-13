"""共享秩诊断：精确并列值用平均秩，常量输入的相关保持未定义。"""
import numpy as np


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("秩输入必须为有限一维数组")
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranked[order[start:end]] = (start + end - 1) / 2.0
    return ranked


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError("相关输入必须为等长一维数组")
    ra, rb = average_ranks(a), average_ranks(b)
    if len(a) < 2 or np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(np.clip(ra @ rb / denominator, -1.0, 1.0))


def correlation_summary(values):
    """不将未定义值当作零，也不静默丢弃部分种子；JSON 使用 null。"""
    values = np.asarray(values, float)
    defined = np.isfinite(values)
    return dict(mean=float(values.mean()) if len(values) and defined.all() else None,
                defined=int(defined.sum()), total=len(values),
                per_seed=[float(v) if ok else None for v, ok in zip(values, defined)])


def format_correlation(value, precision=3):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:+.{precision}f}"
