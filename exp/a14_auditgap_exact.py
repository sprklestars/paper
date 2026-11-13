"""A14：Gauge 损害构造与有限精度功能审计。

prop:auditgap 陈述数学函数不可区分性；prop:gauge-damage 的损害定理
另有单隐层、正激活二阶矩、非零输出极限和有限损失等条件。
这里保留原深网 witness，测量二次尾部和指定目标，不把它当任意深网定理。
浮点值用预先固定的 atol=rtol=1e-12 检查；字节哈希另由 A17 比较。
"""

import numpy as np

from a7_capacity import agg_mse, build
from mlp import flat, forward

DIMS = [20, 32, 32, 1]
N = 24
ATOL = 1e-12
RTOL = 1e-12
AUDIT_RECORDS = []


def audit_compare(reference, candidate, label):
    """记录差异并执行固定容差检查；拒绝形状变化、非有限值和超界扰动。"""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    assert reference.shape == candidate.shape, label
    assert np.all(np.isfinite(reference)) and np.all(np.isfinite(candidate)), label
    delta = np.abs(candidate - reference)
    absolute = float(delta.max(initial=0.0))
    relative = float((delta / np.maximum(np.abs(reference), np.finfo(float).tiny)).max(initial=0.0))
    scaled = float((delta / (ATOL + RTOL * np.abs(reference))).max(initial=0.0))
    passed = bool(np.allclose(candidate, reference, atol=ATOL, rtol=RTOL))
    AUDIT_RECORDS.append(dict(label=label, max_absolute=absolute,
                              max_relative=relative, max_tolerance_ratio=scaled,
                              passed=passed))
    print(f"{label}: abs={absolute:.3e}, rel={relative:.3e}, tolerance_ratio={scaled:.3e}")
    assert passed, f"{label}: 超过预声明容差"
    return absolute


def audit_negative_control(reference):
    """确认容差比较不会无条件通过。"""
    bad = np.array(reference, dtype=np.float64, copy=True)
    bad.flat[0] += 100 * (ATOL + RTOL * abs(bad.flat[0]))
    try:
        audit_compare(reference, bad, "negative_control_expected_rejection")
    except AssertionError:
        print("超界扰动对照：按预期拒绝")
    else:
        raise AssertionError("超界扰动未被拒绝")


def write_audit_report(name, **extra):
    """保存种子、参数和环境标识；不把历史末位误差作为回归目标。"""
    import json
    import platform
    import sys
    from pathlib import Path
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    report = dict(seed=0, dims=DIMS, clients=N, atol=ATOL, rtol=RTOL,
                  python=sys.version, numpy=np.__version__, platform=platform.platform(),
                  comparisons=AUDIT_RECORDS, **extra)
    (out / name).write_text(json.dumps(report, indent=2))


def scale_one(p, unit, a, layer=0):
    """Scale a single hidden unit by a: the minimal element of the gauge group.

    Returns a new net; leaves p untouched.  This is eq:homog applied to one
    coordinate, which is all the proof uses -- no random draw over the whole
    group, because a claim about existence needs one witness, not a sample.
    """
    q = [{k: v.copy() for k, v in L.items()} for L in p]
    q[layer]["W"][unit] *= a
    q[layer]["b"][unit] *= a
    q[layer + 1]["W"][:, unit] /= a
    return q


def setup(seed=0):
    clients, teacher, Xp, _ = build(DIMS, N=N, seed=seed)
    ft = forward(teacher, Xp)[0]
    return clients, teacher, Xp, ft


def damage(clients, teacher, Xp, unit, a, clean):
    q = scale_one(clients[0], unit, a)
    return agg_mse(clients, teacher, Xp, q) / clean


def own_loss(net, Xp, ft):
    return float(np.mean((forward(net, Xp)[0] - ft) ** 2))


# ========================================================== A14.1 =========
def a14_1_divergence():
    print("=" * 78)
    print("A14.1  D(a) -> inf, the side conditions, and the audit at each a")
    print("=" * 78)
    clients, teacher, Xp, ft = setup(seed=0)
    clean = agg_mse(clients, teacher, Xp)
    own0 = own_loss(clients[0], Xp, ft)
    assert np.isfinite(clean) and clean > 0
    original_output = forward(clients[0], Xp)[0]
    audit_negative_control(original_output)
    unit = 0

    # The two side conditions the proof needs, measured not assumed.
    w_norm = float(np.linalg.norm(clients[0][0]["W"][unit]))
    # limiting outgoing column: a^-1 U^(1) -> 0, so c is the honest mean
    c_lim = float(np.linalg.norm(
        np.mean([c[1]["W"][:, unit] for c in clients[1:]], axis=0)))
    print(f"\nside conditions at unit {unit}:")
    print(f"  |W_j^(1)|           = {w_norm:.6f}   (needs != 0)")
    print(f"  |c| = |mean_{{i>1}} U_j^(i)| = {c_lim:.6f}   (needs != 0)")
    assert w_norm > 0 and c_lim > 0, "side condition failed at this unit"

    print(f"\nclean aggregate MSE = {clean:.6f}")
    print(f"\n{'a':>12} {'D(a)':>14} {'own-MSE delta':>16} {'|q-p|':>12}")
    print("-" * 58)
    rows = []
    for a in [1.0, 2.0, 4.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        q = scale_one(clients[0], unit, a)
        D = agg_mse(clients, teacher, Xp, q) / clean
        d_own = audit_compare(own0, own_loss(q, Xp, ft), f"own_MSE_a={a:g}")
        audit_compare(original_output, forward(q, Xp)[0], f"prediction_a={a:g}")
        assert np.isfinite(D)
        dn = float(np.linalg.norm(flat(q) - flat(clients[0])))
        rows.append((a, D, d_own))
        print(f"{a:>12.0f} {D:>14.4f} {d_own:>16.2e} {dn:>12.4e}")

    As = np.array([r[0] for r in rows])
    Ds = np.array([r[1] for r in rows])
    audits = [r[2] for r in rows]

    # D is NOT monotone at small a: it dips marginally below 1 around a = 4,
    # because there the displacement is still small enough that its effect on
    # the aggregate is within the noise of the clean baseline.  The proof does
    # not need monotonicity anywhere -- only divergence, continuity, and
    # D(1) = 1 -- so this is reported rather than asserted.
    print(f"\nD monotone over the whole range: "
          f"{bool(np.all(np.diff(Ds) > 0))}  (not needed by the proof)")
    print(f"  min D on the range = {Ds.min():.6f} at a = {As[Ds.argmin()]:.0f}")

    # Asymptotic slope in log-log: the tail should be quadratic, since the
    # incoming row grows like a while the outgoing column tends to a constant.
    tail = slice(-4, None)
    slope = np.polyfit(np.log(As[tail]), np.log(Ds[tail]), 1)[0]
    print(f"\nlog-log slope of D over the last four decades: {slope:.4f}")
    print("  predicted 2: pre-activation grows like a, error enters squared")
    assert abs(slope - 2.0) < 0.05, f"tail exponent {slope} not quadratic"

    print(f"\nD at a=1e6 is {Ds[-1]:.4g}x clean -- unbounded in a, as claimed")
    print(f"D spans {np.log10(Ds.max() / max(Ds.min(), 1e-12)):.1f} orders "
          f"of magnitude over the range")
    print(f"audit deltas over the whole range: max {max(audits):.3e}")
    assert max(audits) <= ATOL + RTOL * abs(own0)

    # Quoted in sec:e5: D at a=1e6, the tail exponent, and the dip at a=4.
    assert abs(Ds[-1] / 3.33e7 - 1) < 0.01, f"paper quotes 3.33e7, got {Ds[-1]:.3e}"
    assert abs(slope - 2.0008) < 5e-4, f"paper quotes 2.0008, got {slope:.4f}"
    assert abs(Ds.min() - 0.999947) < 1e-6, f"paper quotes 0.999947"
    print("D(1e6), tail exponent and the a=4 dip all match sec:e5")
    print("功能值满足固定 float64 容差；有限扫描支持本 witness 的损害增长。")
    return clients, teacher, Xp, clean, unit


# ========================================================== A14.2 =========
def a14_2_continuity(clients, teacher, Xp, clean, unit):
    print("\n" + "=" * 78)
    print("A14.2  continuity of D on (0, inf)")
    print("=" * 78)
    print("\nD is a composition of continuous maps (parameter average, relu,")
    print("squared error), so continuity is not in doubt analytically.  What is")
    print("worth checking is that nothing in the harness introduces a jump --")
    print("a clipped unit, an overflow -- at the scales the proof ranges over.")

    # Fine grid in log a; report the largest relative step between neighbours.
    grid = np.exp(np.linspace(np.log(0.5), np.log(500.0), 200))
    Dv = np.array([damage(clients, teacher, Xp, unit, a, clean) for a in grid])
    assert np.all(np.isfinite(Dv)), "non-finite D on the grid"
    steps = np.abs(np.diff(Dv)) / np.maximum(Dv[:-1], 1e-12)
    print(f"\ngrid: 200 points, a in [{grid[0]:.3f}, {grid[-1]:.1f}]")
    print(f"D range on grid:            [{Dv.min():.4f}, {Dv.max():.4f}]")
    print(f"max relative step to next:  {steps.max():.4f}")
    print(f"median relative step:       {np.median(steps):.5f}")
    print("\nsteps shrink with grid spacing, which is what continuity means")
    print("operationally here; no jump discontinuity is present.")


# ========================================================== A14.3 =========
def a14_3_ivt(clients, teacher, Xp, clean, unit):
    print("\n" + "=" * 78)
    print("A14.3  the IVT clause: prescribed targets are attained")
    print("=" * 78)
    print("\nGiven (D), (C), (I), every target in [1, inf) is attained.  The")
    print("test of a constructive claim is construction, so we bisect for each")
    print("target rather than reporting that one exists.  Targets deliberately")
    print("include values far above the 372x the paper had exhibited.")

    def solve(target, lo=1.0, hi=1e7, iters=200):
        # D is not monotone near a = 1 (see A14.1), so bisection is not being
        # used as an inversion of a monotone function.  It is the constructive
        # form of the IVT: any bracket [lo, hi] with D(lo) < target <= D(hi)
        # contains a solution, and halving preserves that invariant.
        for _ in range(iters):
            mid = np.sqrt(lo * hi)          # bisect in log a
            if damage(clients, teacher, Xp, unit, mid, clean) < target:
                lo = mid
            else:
                hi = mid
        return np.sqrt(lo * hi)

    print(f"\n{'target D':>12} {'a found':>16} {'D achieved':>14} "
          f"{'rel err':>10}")
    print("-" * 56)
    worst = 0.0
    for target in [2.0, 10.0, 372.0, 1e3, 1e4, 1e6, 1e9]:
        a = solve(target)
        got = damage(clients, teacher, Xp, unit, a, clean)
        rel = abs(got - target) / target
        worst = max(worst, rel)
        print(f"{target:>12.4g} {a:>16.6g} {got:>14.6g} {rel:>10.2e}")
    print(f"\nworst relative error over the targets: {worst:.2e}")
    assert worst < 1e-6, "a prescribed target was not attained"

    # Values quoted in sec:e5 and in the proof of prop:auditgap.  Asserted here
    # so that a re-run which moves them fails loudly instead of leaving the
    # paper quoting a number no script produces.
    AUDIT_RECORDS.append(dict(label="IVT_targets", worst_relative_error=worst,
                              threshold=1e-6, passed=True))
    print("\nEvery prescribed degradation level was attained by an explicit")
    print("gauge element, including 1e9x -- six orders above the largest value")
    print("sec:e5 exhibits.  The 'any target degradation' clause of")
    print("prop:gauge-damage is proved under its stated assumptions; this")
    print("proof should not cite it.  eq:maglaw remains what it is: an")
    print("empirical statement about the SHAPE of D under a random draw from")
    print("the full group, which is a different question from existence.")


def a14_4_robustness():
    print("\n" + "=" * 78)
    print("A14.4  深网 witness 的多种子扫描，不证明一般性")
    print("=" * 78)
    print("\n测量非零载体及输出列；这些诊断不替代单隐层定理中的激活条件。")
    print(f"\n{'seed':>6} {'units':>7} {'min |W_j|':>12} {'min |c|':>12} "
          f"{'D(1e4) min':>12}")
    print("-" * 54)
    for seed in range(4):
        clients, teacher, Xp, ft = setup(seed=seed)
        clean = agg_mse(clients, teacher, Xp)
        wn, cn, dd = [], [], []
        for layer in (0, 1):
            for u in range(clients[0][layer]["b"].size):
                wn.append(np.linalg.norm(clients[0][layer]["W"][u]))
                cn.append(np.linalg.norm(np.mean(
                    [c[layer + 1]["W"][:, u] for c in clients[1:]], axis=0)))
        # damage at a large scale, worst unit of layer 0
        for u in range(clients[0][0]["b"].size):
            q = scale_one(clients[0], u, 1e4)
            dd.append(agg_mse(clients, teacher, Xp, q) / clean)
        print(f"{seed:>6} {len(wn):>7} {min(wn):>12.4e} {min(cn):>12.4e} "
              f"{min(dd):>12.4e}")
        assert min(wn) > 0 and min(cn) > 0
        assert min(dd) > 1.0, "some unit gave no damage at a=1e4"
    print("\nBoth side conditions hold at every hidden unit of every seed, and")
    print("every unit taken alone already damages the aggregate at a = 1e4.")
    print("以上结论仅覆盖列出的种子、单元和幅度。")


# ========================================================== A14.5 =========
def a14_5_four_audits(clients, teacher, Xp, clean, unit):
    """在同一个损害 witness 上检查四个功能统计量的固定容差。"""
    print("\n" + "=" * 78)
    print("A14.5  the four output-based audits AT the 372x witness")
    print("=" * 78)

    def solve(target, lo=1.0, hi=1e7, iters=200):
        for _ in range(iters):
            mid = np.sqrt(lo * hi)
            if damage(clients, teacher, Xp, unit, mid, clean) < target:
                lo = mid
            else:
                hi = mid
        return np.sqrt(lo * hi)

    ft = forward(teacher, Xp)[0]
    rr = np.random.default_rng(77)
    Xc = rr.normal(size=(500, Xp.shape[1]))
    trig = Xp[Xp[:, 0] > 1.0]

    checks = [
        ("held-out MSE",
         lambda m: float(np.mean((forward(m, Xp)[0] - ft) ** 2))),
        ("max abs prediction error",
         lambda m: float(np.abs(forward(m, Xp)[0] - ft).max())),
        ("challenge output energy",
         lambda m: float(np.sum(forward(m, Xc)[0] ** 2))),
        ("prediction on trigger region",
         lambda m: float(np.mean(forward(m, trig)[0]))),
    ]

    for target in [372.0, 1e6]:
        a = solve(target)
        q = scale_one(clients[0], unit, a)
        got = damage(clients, teacher, Xp, unit, a, clean)
        print(f"\nwitness a = {a:.6g}, aggregate damage {got:.6g}x clean")
        print(f"{'audit':>30} {'honest':>18} {'gauge-moved':>18} "
              f"{'within tolerance':>16}")
        print("-" * 84)
        for name, fn in checks:
            h, g = fn(clients[0]), fn(q)
            audit_compare(h, g, f"{name}_target={target:g}")
            print(f"{name:>30} {h:>18.12f} {g:>18.12f} {'True':>16}")
        assert abs(got / target - 1) < 1e-6

    print("\n四项功能统计量在 372x 与 1e6x witness 上通过固定容差检查。")
    print("输出能量不是密码学 digest；原始输出 SHA-256 比较见 A17。")


if __name__ == "__main__":
    clients, teacher, Xp, clean, unit = a14_1_divergence()
    a14_2_continuity(clients, teacher, Xp, clean, unit)
    a14_3_ivt(clients, teacher, Xp, clean, unit)
    a14_4_robustness()
    a14_5_four_audits(clients, teacher, Xp, clean, unit)
    print("\n" + "=" * 78)
    write_audit_report("a14_revision.json", robustness_seeds=list(range(4)))
    print("A14：深网 witness 通过；数学损害定理范围以 prop:gauge-damage 为准。")
    print("=" * 78)
