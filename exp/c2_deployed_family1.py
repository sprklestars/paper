"""
C2: the three DEPLOYED Family I scorers, per their own estimators.

Section III names three real systems as evidence that Family I is deployed
rather than hypothetical, and says our attack results apply to that family.
E1 and E1b never tested any of them: E1 tested a generic

    w_i = (1/sigmahat_i^2) / sum_j (1/sigmahat_j^2)

with sigmahat_i^2 an EMA of the client's own across-round movement.  A
reviewer is entitled to ask whether the capture survives the estimators the
named papers actually use, because two of them differ from the generic model
in ways that are not cosmetic:

  PWFL      Reyes et al., Computational Intelligence 41(6), e70150 (2025),
            正式 Eq.(5)、Sec.4.4。本地仍是 SGD 记录的离线二阶矩适配，
            未复现正式版的 Adam 训练路径。Weights are the inverse
            of "the variance of the maximum likelihood estimator of weight
            theta", estimated (their Sec. 4.4) by the RAW SECOND MOMENT --
            they say "uncentered variance" -- of the stochastic gradient
            taken from the Adam optimiser, which they note approximates the
            diagonal of the Fisher information.  A callback averages the
            estimator over the second half of the last epoch.
                theta <- sum_i (v_i)^-1 theta_i / sum_i (v_i)^-1
            UNCENTERED is the load-bearing word.  E[g^2] = Var(g) + E[g]^2,
            so a client carrying a persistent gradient bias inflates its own
            denominator.  Our adversary carries exactly such a bias (it
            trains toward w* + delta while the global model sits near w*).
            Whether the capture survives is therefore an open question and
            not a formality -- the estimator may defend itself by accident.

  FIVA-G    Ukaye et al., PMLR 298 (2025), Eqs. (2), (3), (5)-(8).  Welford
            over the T mini-batch gradients gives sigma^2_g, converted to a
            parameter variance by
                sigma^2_theta = sigma^2_theta,0 + T * eta^2 * sigma^2_g,
            then c_i = nhat_i / sigma^2_i with nhat_i = n_i / sum_j n_j, and
                theta_global = sum_i c_i theta_i / sum_i c_i.
            Welford is CENTERED, so this is the honest inverse-variance
            reading -- but the weights are per-parameter vectors, not scalars.
  FIVA-P    Same paper, Eq. (4): Welford applied to the iterates instead of
            the gradients.
            式 (7) 的历史精度仅进入全局方差，lambda=0.95；式 (8) 广播
            下一轮 G 的起始方差。均值权重不混入历史精度。正式版与
            arXiv:2508.09196v1 已核对的式 (1)--(8) 一致。

Two structural differences from E1, both measured here rather than assumed:

    per-parameter   All three weight EVERY COORDINATE separately.  E1's
                    scorer emits one scalar per client.  If coordinate-wise
                    precision averaged out the adversary's advantage, the
                    E1 result would not transfer to the deployed form, so
                    block C2.3 measures the scalar collapse directly.
    uncentered      PWFL only.  See above.

Axes are matched to c1_recent_scorers.py so the two tables can be read
together: reward share as a multiple of fair share k/N, MSE against the same
scorer's own no-attack run, and max detector excess over that scorer's own
false-positive rate.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from e2_silence import CFG, w_invvar, w_uniform
from e2b_sweep import detectors2
from b1_published_baselines import craft_fang_trim, craft_shej

# ---------------------------------------------------------------- constants

BATCH = 5           # 50 local samples / batch 5 = 10 mini-batch steps
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8
FIVA_LAM = 0.95     # their forgetting factor, Eq. (7)
FIVA_VAR0 = 1.0     # "parameter variances uniformly initialized to one"
VAR_FLOOR = 1e-12   # neither paper gives one; see C2.4

TSP_S_ADV, TSP_DELTA = 0.35, 0.05
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
TAIL = 20


def _norm_rows(C):
    """Column-stochastic per-coordinate weights: each coordinate sums to 1."""
    s = C.sum(axis=0, keepdims=True)
    bad = s <= 0
    if np.any(bad):
        C = np.where(np.broadcast_to(bad, C.shape), 1.0 / C.shape[0], C)
        s = C.sum(axis=0, keepdims=True)
    return C / s


# ---------------------------------------------------------------- estimators

def adam_second_moment(grads, lr):
    """PWFL Sec. 4.4: Adam's raw second moment, averaged over the second half
    of the run, per coordinate.  grads is (T, d)."""
    T, d = grads.shape
    v = np.zeros(d)
    keep = []
    for t in range(T):
        v = ADAM_B2 * v + (1 - ADAM_B2) * grads[t] ** 2
        vhat = v / (1 - ADAM_B2 ** (t + 1))
        if t >= T // 2:            # "second half of the last epoch"
            keep.append(vhat)
    return np.mean(keep, axis=0)


def welford(seq):
    """FIVA Eqs. (2)/(4): online Welford variance, per coordinate."""
    mean = np.zeros(seq.shape[1])
    M2 = np.zeros(seq.shape[1])
    for t, x in enumerate(seq, start=1):
        prev = mean
        mean = prev + (x - prev) / t
        M2 = M2 + (x - prev) * (x - mean)
    return M2 / len(seq)


def fiva_gradient_variance(grads, lr, global_var):
    """正式式 (3)/(8)：本轮起始方差来自上一轮全局广播。"""
    return global_var + len(grads) * lr ** 2 * welford(grads)


def fiva_weights(est, nhat, global_var, lam=FIVA_LAM,
                 floor=VAR_FLOOR, numerics="formula"):
    """式 (5)--(7)：均值只用当前精度，历史精度只更新全局方差。

    formula 使用声明的局部方差下限；official-epsilon 只对照官方 full
    模式的方差加 1e-5 及全局方差截断，不表示复现其完整训练系统。
    """
    est = np.asarray(est, dtype=float)
    nhat = np.asarray(nhat, dtype=float)
    global_var = np.asarray(global_var, dtype=float)
    if (est.ndim != 2 or nhat.shape != (len(est),)
            or global_var.shape != (est.shape[1],)):
        raise ValueError("FIVA 输入形状不一致")
    if (not all(np.all(np.isfinite(x)) for x in (est, nhat, global_var))
            or np.any(est < 0) or np.any(nhat < 0)
            or not np.isclose(nhat.sum(), 1.0) or np.any(global_var <= 0)
            or not np.isfinite(lam) or not 0 <= lam <= 1
            or not np.isfinite(floor) or floor < 0):
        raise ValueError("FIVA 方差、样本比例或遗忘系数非法")
    if numerics == "formula":
        v = np.maximum(est, floor)
    elif numerics == "official-epsilon":
        v = est + 1e-5
    else:
        raise ValueError(f"未知数值处理：{numerics}")
    if np.any(v <= 0):
        raise ValueError("零方差需要正的数值稳定项")
    precision = nhat[:, None] / v
    total = precision.sum(axis=0)
    weights = precision / total
    next_global_var = 1.0 / (lam / global_var + total)
    if numerics == "official-epsilon":
        next_global_var = np.maximum(next_global_var, 1e-5)
    if (not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(next_global_var))
            or np.any(next_global_var <= 0)):
        raise FloatingPointError("FIVA 递推产生非有限值或非正方差")
    return weights, next_global_var


# ---------------------------------------------------------------- simulation

def local_work(rng, w, tgt, sigma, n_loc, d, lr):
    """One client's local epoch.  Returns (update, per-step grads, iterates)."""
    X = rng.normal(size=(n_loc, d))
    y = X @ tgt + rng.normal(0, sigma, size=n_loc)
    theta = w.copy()
    grads, iters = [], []
    for s in range(0, n_loc, BATCH):
        Xb, yb = X[s:s + BATCH], y[s:s + BATCH]
        g = Xb.T @ (Xb @ theta - yb) / len(yb)
        theta = theta - lr * g
        grads.append(g)
        iters.append(theta.copy())
    return theta - w, np.array(grads), np.array(iters)


def sim(attack, scorer, cfg, seed, lam=FIVA_LAM, floor=VAR_FLOOR,
        collapse=False, var0=FIVA_VAR0, numerics="formula", trace=False):
    """One run.  scorer in {fedavg, invvar, pwfl, fiva-g, fiva-p}.

    collapse=True replaces each per-coordinate weight vector by its mean, so
    the same estimator is scored as a scalar rule.  That isolates whether
    per-parameter weighting matters (C2.3)."""
    rng = np.random.default_rng(seed)
    d, N, k = cfg["d"], cfg["N"], cfg["k_adv"]
    lr, n_loc = cfg["lr"], cfg["n_local"]
    adv, hon = np.arange(k), np.arange(k, N)

    w_star = rng.normal(size=d)
    delta = rng.normal(size=d)
    delta = TSP_DELTA * delta / np.linalg.norm(delta)

    sigma = np.empty(N)
    sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
    sigma[adv] = TSP_S_ADV if attack == "tsp-var" else \
        rng.uniform(cfg["s_lo"], cfg["s_hi"], size=k)

    Xte = rng.normal(size=(cfg["n_test"], d))
    yte = Xte @ w_star
    w = np.zeros(d)

    if not np.isfinite(var0) or var0 <= 0:
        raise ValueError("初始全局方差必须为正有限值")
    global_var = np.full(d, var0, dtype=float)
    prev_upd = np.zeros((N, d))
    ema = np.full((N, d), np.nan)
    nhat = np.full(N, 1.0 / N)               # equal local sample counts

    rec = {"aw": [], "hw": [], "mse": [], "alarms": [], "advvar": [],
           "honvar": [], "global_var": [], "trace": []}

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        est = np.empty((N, d))
        start_var = global_var.copy()
        for i in range(N):
            tgt = w_star + delta if i < k and attack == "tsp-var" else w_star
            upd, grads, iters = local_work(rng, w, tgt, sigma[i], n_loc, d, lr)
            V[i] = upd
            if scorer == "pwfl":
                est[i] = adam_second_moment(grads, lr)
            elif scorer == "fiva-g":
                est[i] = fiva_gradient_variance(grads, lr, start_var)
            elif scorer == "fiva-p":
                est[i] = welford(iters)
            else:
                est[i] = 0.0

        if attack == "fang-trim":
            V[adv] = craft_fang_trim(V[hon], k, rng)
        elif attack == "shej-minmax":
            V[adv] = craft_shej(V[hon], k, "minmax")

        # scalar EMA state, for the generic scorer and for the detectors
        if t > 0:
            inst = np.linalg.norm(V - prev_upd, axis=1) ** 2 / d
            ema = inst if t == 1 else (1 - cfg["rho"]) * ema + cfg["rho"] * inst
        prev_upd = V.copy()
        svar = np.full(N, 1.0) if t == 0 else np.asarray(ema)
        state = {"var": svar, "hist": V.copy(),
                 "anchor_ref": _anchor(rng, w, w_star, cfg)}

        if scorer == "fedavg":
            C = np.tile(w_uniform(V, state, cfg)[:, None], (1, d))
        elif scorer == "invvar":
            C = np.tile(w_invvar(V, state, cfg)[:, None], (1, d))
        elif scorer in ("fiva-g", "fiva-p"):
            C, global_var = fiva_weights(est, nhat, start_var, lam, floor, numerics)
            rec["global_var"].append(float(global_var.mean()))
        elif scorer == "pwfl":
            C = nhat[:, None] / np.maximum(est, floor)
        else:
            raise ValueError(f"未知评分器：{scorer}")
        C = _norm_rows(C)
        if collapse:
            C = _norm_rows(np.tile(C.mean(axis=1)[:, None], (1, d)))

        w = w + (C * V).sum(axis=0)
        if trace:
            rec["trace"].append(dict(start_var=start_var, local_var=est.copy(),
                                     weights=C.copy(), global_var=global_var.copy(),
                                     updates=V.copy(), model=w.copy()))

        rec["aw"].append(float(C[adv].sum(axis=0).mean()))
        rec["hw"].append(float(C[hon].sum(axis=0).mean() / len(hon)))
        rec["mse"].append(float(np.mean((Xte @ w - yte) ** 2)))
        rec["alarms"].append(detectors2(V, adv, state, cfg))
        if scorer not in ("fedavg", "invvar"):
            rec["advvar"].append(float(est[adv].mean()))
            rec["honvar"].append(float(est[hon].mean()))

    rec["sigma"] = sigma
    rec["final_C"] = C
    return rec


def _anchor(rng, w, w_star, cfg):
    A = []
    for _ in range(cfg["a_anchor"]):
        X = rng.normal(size=(cfg["n_anchor"], cfg["d"]))
        y = X @ w_star + rng.normal(0, cfg["s_anchor"], size=cfg["n_anchor"])
        A.append(-cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_anchor"]))
    return np.mean(A, axis=0)


def agg(attack, scorer, seeds=SEEDS, cfg=CFG, **kw):
    runs = [sim(attack, scorer, cfg, s, **kw) for s in seeds]
    aw = float(np.mean([np.mean(r["aw"][-TAIL:]) for r in runs]))
    hw = float(np.mean([np.mean(r["hw"][-TAIL:]) for r in runs]))
    mse = float(np.mean([np.mean(r["mse"][-TAIL:]) for r in runs]))
    keys = runs[0]["alarms"][0].keys()
    rate = {k: float(np.mean([np.mean([a[k] for a in r["alarms"][-TAIL:]])
                             for r in runs])) for k in keys}
    out = {"aw": aw, "hw": hw, "mse": mse, "rate": rate,
           "x_fair": aw / (cfg["k_adv"] / cfg["N"]),
           "per_seed": [dict(seed=s,
                             x_fair=float(np.mean(r["aw"][-TAIL:]) / (cfg["k_adv"] / cfg["N"])),
                             mse=float(np.mean(r["mse"][-TAIL:])))
                        for s, r in zip(seeds, runs)]}
    if runs[0]["global_var"]:
        out["global_var_first"] = float(np.mean([r["global_var"][0] for r in runs]))
        out["global_var_last"] = float(np.mean([r["global_var"][-1] for r in runs]))
    if runs[0]["advvar"]:
        out["advvar"] = float(np.mean([np.mean(r["advvar"][-TAIL:])
                                       for r in runs]))
        out["honvar"] = float(np.mean([np.mean(r["honvar"][-TAIL:])
                                       for r in runs]))
    return out


from tools.rank_metrics import spearman, correlation_summary, format_correlation


SCORERS = ["fedavg", "invvar", "pwfl", "fiva-g", "fiva-p"]
LABEL = {"fedavg": "FedAvg (control)", "invvar": "Inverse-variance (E1)",
         "pwfl": "PWFL '25 (SGD-record 2nd moment)",
         "fiva-g": "FIVA-G '25 (Welford grad)",
         "fiva-p": "FIVA-P '25 (Welford iter)"}

def run_experiments():
    """执行完整 C2；导入本模块不运行实验、不写结果。"""
    from test_fiva import run_checks
    run_checks()
    out = {}
    print("C2.0 局部估计器自检及跨轮公式回归", flush=True)
    g = np.random.default_rng(7).normal(2.0, 0.5, size=(4000, 3))
    wf = welford(g)
    am = adam_second_moment(g, 0.3)
    np.testing.assert_allclose(wf, g.var(axis=0), rtol=1e-10)
    assert am.mean() > g.var(axis=0).mean() * 5
    out["_welford_err"] = float(np.abs(wf - g.var(axis=0)).max())
    out["_adam_uncentered"] = float(am.mean())

    print("C2.1 全诚实对照（无目标偏移）；rho/CV 保持 seed=0 口径", flush=True)
    base = {}
    for s in SCORERS:
        r = sim("none", s, CFG, 0)
        base[s] = agg("none", s)
        C = r["final_C"].mean(axis=1)
        rho = spearman(C, 1.0 / r["sigma"] ** 2)
        cv = float(np.std(C) / np.mean(C))
        out[f"none|{s}"] = base[s]
        summary = correlation_summary([rho])
        out[f"_discrim|{s}"] = summary["mean"]
        out[f"_discrim_details|{s}"] = summary
        out[f"_cv|{s}"] = cv
        print(f"  {s:10} rho={format_correlation(rho, 5)} CV={cv:.5f}", flush=True)

    print("C2.2 攻击网格与各自全诚实基线", flush=True)
    for s in SCORERS:
        for a in ("none", "tsp-var", "fang-trim", "shej-minmax"):
            r = out.get(f"{a}|{s}") or agg(a, s)
            out[f"{a}|{s}"] = r
            r["excess"] = {k: r["rate"][k] - base[s]["rate"][k] for k in r["rate"]}
            r["mse_ratio"] = r["mse"] / base[s]["mse"]
            print(f"  {s:10} {a:12} reward={r['x_fair']:.6f} "
                  f"mse/base={r['mse_ratio']:.6f} excess={max(r['excess'].values()):+.6f}", flush=True)

    print("C2.3 逐坐标与标量折叠", flush=True)
    for s in ("pwfl", "fiva-g", "fiva-p"):
        pc = out[f"tsp-var|{s}"]["x_fair"]
        sc = agg("tsp-var", s, collapse=True)["x_fair"]
        out[f"_collapse|{s}"] = {"per_coord": pc, "scalar": sc}
        print(f"  {s}: {pc:.6f} / {sc:.6f}", flush=True)

    print("C2.4 遗忘系数与局部方差下限敏感性", flush=True)
    for lam in (0.0, 0.5, 0.95, 0.99):
        r = agg("tsp-var", "fiva-g", lam=lam)
        out[f"_lam|{lam}"] = r["x_fair"]
        print(f"  lambda={lam}: {r['x_fair']:.6f}", flush=True)
    for s in ("pwfl", "fiva-g", "fiva-p"):
        for fl in (1e-12, 1e-8, 1e-4):
            r = agg("tsp-var", s, floor=fl)
            key = f"_floor|{fl}" if s == "pwfl" else f"_floor|{s}|{fl}"
            out[key] = r["x_fair"]
            print(f"  {s} floor={fl}: {r['x_fair']:.6f}", flush=True)

    print("C2.5 初始方差扫描；后续各轮均广播动态全局方差", flush=True)
    for v0 in (1.0, 0.1, 1e-2, 1e-4, 1e-8):
        r = agg("tsp-var", "fiva-g", var0=v0)
        out[f"_fivag_v0|{v0}"] = dict(x_fair=r["x_fair"],
            ratio=r["advvar"] / r["honvar"],
            global_var_first=r["global_var_first"], global_var_last=r["global_var_last"])
        print(f"  var0={v0}: {out[f'_fivag_v0|{v0}']}", flush=True)

    print("C2.6 同评分器检测器对照", flush=True)
    for s in SCORERS:
        row = {a: max(out[f"{a}|{s}"]["excess"].values())
               for a in ("tsp-var", "fang-trim", "shej-minmax")}
        out[f"_control|{s}"] = row
        print(f"  {s}: {row}", flush=True)

    print("C2.7 官方 full 模式数值稳定项对照（不等于官方全系统复现）", flush=True)
    for s in ("fiva-g", "fiva-p"):
        clean = agg("none", s, numerics="official-epsilon")
        attack = agg("tsp-var", s, numerics="official-epsilon")
        attack["mse_ratio"] = attack["mse"] / clean["mse"]
        attack["excess"] = {k: attack["rate"][k] - clean["rate"][k] for k in clean["rate"]}
        out[f"_official_epsilon|{s}"] = dict(clean=clean, attack=attack)
        print(f"  {s}: reward={attack['x_fair']:.6f}, mse/base={attack['mse_ratio']:.6f}", flush=True)

    # 先验证再写结果，不把预期的攻击方向当作算法正确性断言。
    for key, v in out.items():
        if isinstance(v, dict) and "aw" in v:
            total = v["aw"] + v["hw"] * (CFG["N"] - CFG["k_adv"])
            assert abs(total - 1.0) < 1e-6, (key, total)
            assert 0 <= v["aw"] <= 1
    here = Path(__file__).resolve().parent
    sources = ("c2_deployed_family1.py", "test_fiva.py", "tools/rank_metrics.py", "e2_silence.py",
               "e2b_sweep.py", "b1_published_baselines.py")
    out["_meta"] = dict(version="fiva-equations-1-8-v3-ranks", seeds=SEEDS,
        rank_policy="average ties; constant undefined; JSON null",
        cfg=CFG, batch=BATCH, tail=TAIL, lam=FIVA_LAM, var0=FIVA_VAR0,
        floor=VAR_FLOOR, numerics="formula", quality_seed=0,
        honest_target_offset=0.0, python=sys.version, numpy=np.__version__,
        source_sha256={n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in sources})
    data = json.dumps(out, indent=1, default=float, allow_nan=False)
    target = here / "results/c2_deployed.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(data)
    print(f"C2 OK: {target}", flush=True)
    return out


if __name__ == "__main__":
    run_experiments()
