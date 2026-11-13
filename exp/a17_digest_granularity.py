"""A17：区分函数容差、浮点输出和严格字节哈希。

固定 atol=rtol=1e-12，报告 Gauge、诚实重分批及重排序的绝对／相对差。
哪一种差异更大是当前环境的测量结果，不作跨平台断言。
SHA-256 比较没有“容差相等”；固定完整计算路径的协议可以要求字节重放。
"""

import hashlib

import numpy as np

from a14_auditgap_exact import (agg_mse, forward, scale_one, setup,
                               audit_compare, audit_negative_control, write_audit_report)

N_TEST = 1500


def digest(v):
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


def witness(clients, teacher, Xp, unit=0, target=372.0, iters=200):
    """Bisect the single-unit scale that puts aggregate damage at target."""
    clean = agg_mse(clients, teacher, Xp)

    def dmg(a):
        return agg_mse(clients, teacher, Xp,
                       scale_one(clients[0], unit, a)) / clean

    lo, hi = 1.0, 1e12
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if dmg(mid) < target:
            lo = mid
        else:
            hi = mid
    a = np.sqrt(lo * hi)
    return a, dmg(a)


def main():
    clients, teacher, Xp, ft = setup(seed=0)
    p = clients[0]
    full = forward(p, Xp)[0]

    print("=" * 78)
    print("A17.1  the gauge move seen as a raw output vector")
    print("=" * 78)
    a, got = witness(clients, teacher, Xp)
    q = scale_one(p, 0, a)
    fq = forward(q, Xp)[0]
    d = full - fq
    gauge_max = audit_compare(full, fq, "gauge_prediction")
    audit_negative_control(full)
    assert abs(got / 372.0 - 1) < 1e-6
    print(f"\nwitness a = {a:.6g}, aggregate damage {got:.4f}x clean")
    print(f"entries differing: {int(np.count_nonzero(d))} of {full.size}")
    print(f"max |diff| {gauge_max:.3e}   max relative "
          f"{float(np.abs(d / (np.abs(full) + 1e-300)).max()):.3e}")
    print(f"sha256 equal: {digest(full) == digest(fq)}")
    print("\n哈希是否相同由当前字节决定；下面比较诚实执行调度变化。")

    print("\n" + "=" * 78)
    print("A17.2  honest implementation variation, identical parameters")
    print("=" * 78)
    print("\nSame client, same weights, same inputs; only the evaluation")
    print("schedule changes; differences below are measured on this environment.\n")
    print(f"{'variation':>22} {'max |diff|':>12} {'entries':>9} "
          f"{'digest equal':>13}")
    print("-" * 60)
    honest = []
    for bs in [1500, 750, 500, 128, 1]:
        parts = [forward(p, Xp[i:i + bs])[0] for i in range(0, len(Xp), bs)]
        cat = np.concatenate(parts)
        mx = audit_compare(full, cat, f"honest_batch_{bs}")
        honest.append(mx)
        print(f"{'batch size ' + str(bs):>22} {mx:>12.3e} "
              f"{int(np.count_nonzero(cat - full)):>9} "
              f"{str(digest(cat) == digest(full)):>13}")
    r = np.random.default_rng(0)
    pi = r.permutation(len(Xp))
    out = forward(p, Xp[pi])[0][np.argsort(pi)]
    mx = audit_compare(full, out, "honest_reordering")
    honest.append(mx)
    print(f"{'permuted row order':>22} {mx:>12.3e} "
          f"{int(np.count_nonzero(out - full)):>9} "
          f"{str(digest(out) == digest(full)):>13}")

    print("\n" + "=" * 78)
    print("A17.3  the comparison that decides how to state the claim")
    print("=" * 78)
    hmax = max(honest)
    print(f"\ngauge move at {got:.0f}x damage : max |diff| {gauge_max:.3e}")
    print(f"honest re-batching, no attack : max |diff| {hmax:.3e}")
    ratio = hmax / gauge_max if gauge_max > 0 else None
    print(f"ratio honest / gauge          : {ratio if ratio is not None else 'N/A (zero gauge difference)'}")
    print(f"honest_max >= gauge_max in this run: {hmax >= gauge_max}")
    print("两类输出均按预声明容差检查；误差大小次序与哈希相等性不作通用保证。")
    write_audit_report("a17_revision.json", target=372.0, achieved=got,
                       gauge_max=gauge_max, honest_max=hmax, honest_gauge_ratio=ratio,
                       hash_reference=digest(full), hash_gauge=digest(fq),
                       batch_sizes=[1500, 750, 500, 128, 1], reorder_seed=0)


if __name__ == "__main__":
    main()
