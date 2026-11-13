"""
V5：M1 下的时间平均权重网格诊断，不是逐轮位移定理的假设核验。

V4.4 falsified the unconditional bound 24/24 at large a: as a grows the
bound tends to 1/(1+Pi_max) but the measured ratio stays near 0.83, because
with no variance floor Pi_max is not 1/N -- it is unbounded. The bound was
being evaluated with a Pi_max that the run did not enforce.

历史公式的数值检查：M1 相对地板为 phi 时，单个攻击者的逐轮权重满足

    w_adv <= (1/(phi*m)) / (1/(phi*m) + sum_h 1/sigmahat_h^2)

with m the median dispersion, so

    Pi_max = k / (phi * m * sum_h 1/sigmahat_h^2)

and the displacement bound becomes

    w_h^atk / w_h^clean  >=  (1/(1+lam tau/s_H))^2 * 1/(1+Pi_max).

Pi_max 依赖实际诚实分数及地板，不能仅由防御参数预先确定。
本脚本把时间平均权重与平均 Pi 代入表达式，共 1080 个样本；这不是
当前 thm:displacement 的逐轮证明，未核验其 score-ratio 和归一化条件。
"""

import json
from pathlib import Path
import numpy as np

d, N = 20, 30
fair = 1.0 / N


def run(atk, a, lam, phi, k=1, s_adv=0.35, delta=0.05,
        rounds=300, rho=0.2, s_anch=0.3, seed=0):
    rng = np.random.default_rng(seed)
    sig = np.random.default_rng(seed + 5000).uniform(0.5, 1.5, size=N)
    off = np.zeros((N, d))
    if atk:
        sig[:k] = s_adv
        off[:k, 0] = delta
    ev = np.ones(N)
    W, P = [], []
    for t in range(rounds):
        V = rng.normal(size=(N, d)) * sig[:, None] + off
        m = np.median(ev)
        fl = phi * m if phi > 0 else 1e-12
        evf = np.maximum(ev, fl)
        iv = 1.0 / evf
        w = iv / iv.sum()
        mu_p = (w[:, None] * V).sum(0)
        if a > 0:
            mu = lam * mu_p + (1 - lam) * rng.normal(size=(a, d)).mean(0) * s_anch
        else:
            mu = mu_p
        inst = np.linalg.norm(V - mu, axis=1) ** 2 / d
        ev = inst if t == 0 else (1 - rho) * ev + rho * inst
        W.append(w)
        # Pi_max as the defender can compute it: adversary capped at the floor
        P.append(k / fl / (1.0 / evf[k:]).sum())
    return np.mean(W[-100:], axis=0), float(np.mean(P[-100:]))


z, sA, sH = 3.5, 0.3, 1.0
SEEDS = 24

print("=" * 88)
print("V5.1  Averaged-weight diagnostic with M1; not a pointwise theorem check")
print("=" * 88)
print("  bound = (1/(1+lam*tau/s_H))^2 / (1+Pi_max),  tau = z*sigma_A/sqrt(a),")
print("  Pi_max = k/(phi*m*sum_h 1/sigmahat_h^2), evaluated from the run.\n")
print(f"{'a':>4} {'lam':>5} {'phi':>6} | {'Pi_max':>8} {'bound':>8} | "
      f"{'mean ratio':>11} {'min ratio':>10} | {'viol':>7} {'loose':>7}")

tot_v = tot_n = 0
records = []
for phi in [0.2, 0.5, 0.8]:
    for a in [1, 3, 8, 32, 64]:
        for lam in [1.0, 0.5, 0.25]:
            rs, pis = [], []
            for s in range(SEEDS):
                Wa, pi_a = run(True, a, lam, phi, seed=200 + s)
                Wc, _ = run(False, a, lam, phi, seed=200 + s)
                rs.append(np.min(Wa[1:] / Wc[1:]))
                pis.append(pi_a)
            rs = np.array(rs)
            Pi = float(np.mean(pis))
            tau = z * sA / np.sqrt(a)
            bound = (1.0 / (1 + lam * tau / sH)) ** 2 / (1 + Pi)
            v = int((rs < bound).sum())
            tot_v += v; tot_n += SEEDS
            records.append(dict(anchors=a, lam=lam, phi=phi,
                seeds=list(range(200, 200 + SEEDS)), mean_pi=Pi,
                diagnostic_bound=bound, min_weight_ratios=rs.tolist(), violations=v))
            print(f"{a:4d} {lam:5.2f} {phi:6.2f} | {Pi:8.4f} {bound:8.4f} | "
                  f"{rs.mean():11.4f} {rs.min():10.4f} | {v:3d}/{SEEDS} "
                  f"{rs.min() / bound:6.2f}x")

print(f"\n  TOTAL VIOLATIONS: {tot_v}/{tot_n}")
print(f"  Averaged-grid violations: {tot_v}/{tot_n}; theorem hypotheses not tested")
out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
(out / "v5_revision.json").write_text(json.dumps(dict(
    evidence_kind="averaged_weight_diagnostic", violations=tot_v, samples=tot_n,
    parameters=dict(d=d, N=N, rounds=300, rho=0.2, z=z, sigma_A=sA, sigma_H=sH),
    grid=records), indent=2))
assert tot_n == 1080 and tot_v == 0, (tot_v, tot_n)

print()
print("=" * 88)
print("V5.2  Attack surface as a function of the two defence knobs")
print("=" * 88)
print("  The design rule the paper should state. Rows: floor. Cols: lam.\n")
print(f"{'phi \\ lam':>10}", end="")
LAMS = [1.0, 0.75, 0.5, 0.25, 0.0]
for lam in LAMS:
    print(f"{lam:>9.2f}", end="")
print("   | utility cost")
for phi in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
    lbl = "none" if phi == 0 else f"{phi:.1f}"
    print(f"{lbl:>10}", end="")
    for lam in LAMS:
        Wa, _ = run(True, 3, lam, phi, seed=301)
        print(f"{Wa[0] / fair:8.2f}x", end="")
    Wh, _ = run(False, 3, 1.0, phi, seed=301)
    sg = np.random.default_rng(301 + 5000).uniform(0.5, 1.5, size=N)
    mse = float((Wh ** 2 * sg ** 2).sum())
    opt = float(1.0 / (1.0 / sg ** 2).sum())
    print(f"   | {mse / opt:11.3f}x")

print("""
  Reading: moving DOWN a column (raising the floor) collapses the capture.
  Moving ACROSS a row (adding anchors to the frame) barely moves it. M1 is
  the load-bearing mechanism; M2 addresses a different theorem.""")
