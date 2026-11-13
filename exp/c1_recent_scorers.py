"""
C1: 2026 contribution-evaluation methods in the same harness.

Sections V-VI compare TSP-Var against scorers published 2017-2021 (Krum,
trimmed mean, FoolsGold, FLTrust) and against attacks published 2020-2021
(b1_published_baselines.py).  A reviewer will ask the obvious question: the
contribution-evaluation literature did not stop in 2021, so does the L3
result survive against what people are proposing now?  This script answers
it by reimplementing four 2026 methods from their own definitions and
running them against the identical generator, seeds and detector suite.

  FP／EE    正式来源：ISWA 32 (2026) 200725，Defs.3--4，Eqs.(3)--(6)。
            FP 对 alpha=(LOO+IOI)/2 取正部；EE 对 beta、gamma 分别取正部。
            本地以非负 fractional shares 为聚合权重，不乘可能为负的 v(M)。
            FP 零分母先试 LOO，再试 IOI（Remark 2 允许的两个替代通道）；
            全部无正值时均匀回退是显式的本地补充，不是正式定义。
            EE 的未归一化量不含自己的报告；归一化分母仍有间接影响。
            Remark 3 的 min-shift 比较映射另作敏感性检查，不冒称原下游系统。
  FedTSV    Trajectory Shapley Value, arXiv 2605.30336 Eqs. (9)-(13).  The
            per-round utility scores a coalition by how closely its mean
            update tracks a server-side validation descent step:
                v_t(S) = (1 + ||Delta_S - Delta_val||^2 / sigma_t)^{-1},
                sigma_t = max(||Delta_val||^2, eps),
            Shapley values of v_t are accumulated across rounds and
            truncated at zero to form the aggregation weights.
  FedRandom arXiv 2602.05693 Sec. 3.2.  Not a per-client weighting rule: it
            samples the SERVER optimiser uniformly per round from
            {FedAvg, FedAvgM, FedAdagrad, FedAdam, FedYogi} so that
            contribution values can be averaged over an ensemble of
            federations.  Included because it is the 2026 answer to
            contribution *instability*, and because its per-client weights
            are uniform by construction -- which is itself a measurement.

Two axes, matched to the other baseline scripts:

    L3 harm     adversary weight share, as a multiple of fair share k/N
                (for FP/EE/FedTSV the weight IS the normalised contribution
                score, so this is simultaneously the reward share)
    L1 harm     test MSE / the same scorer's own no-attack MSE
    silence     max detector excess over that scorer's false-positive rate

Attacks:

    none            control, the denominator of every ratio
    tsp-var         ours.  The adversary fabricates nothing and misreports
                    nothing; it trains honestly on low-variance biased data.
    misreport       the manipulation FP is admittedly open to (the client
                    computes its own score, so it inflates the two marginals
                    it alone can evaluate) and, in the same move, the
                    residual attack the EE authors acknowledge in their
                    Sec. 5.4 (a client cannot touch its own raw EE score, but
                    the marginals it reports enter everybody else's).  These
                    are one attack, not two: raising p_a and lowering q_a
                    maximises FP(a) and minimises every EE(j != a)
                    simultaneously, which block C1.2b checks rather than
                    assumes.  Bounded by the honest range, so a range check
                    cannot see it (lambda=1 puts the report at the most
                    extreme honest value, not outside it).
    fang-trim       Fang et al., USENIX Security 2020, Sec. 5, for contrast.
    shej-minmax     Shejwalkar & Houmansadr, NDSS 2021, Alg. 1, for contrast.

效用适配：正式 FP／EE 实验使用负平均损失；本回归主实验使用
v(w) = 1 / (1 + MSE_val(w))，另以 -MSE 检查效用变换。
聚合使用非负份额，不把负的效率预算 v(M) 当成模型权重。
C1.4 用 v(w) = -MSE_val(w) 测量效用变换敏感性，不预设结果不变。

Run:  python3 -u c1_recent_scorers.py
"""

import hashlib
import json
import sys
from pathlib import Path

from tools.rank_metrics import spearman, correlation_summary, format_correlation

import numpy as np

from e2_silence import SCORERS, CFG, w_uniform, w_invvar
from e2b_sweep import detectors2
from b1_published_baselines import (craft_fang_trim, craft_shej,
                                    TSP_S_ADV, TSP_DELTA)

# FedTSV's Shapley estimate: permutation (Monte Carlo) sampling.  C1.0 checks
# this against 8x as many permutations before anything else is reported.
TSV_PERM = 32
TSV_EPS = 1e-12

# Misreport boldness.  1.0 = report the most extreme value any honest client
# reported this round, i.e. maximal manipulation that stays inside the honest
# range.  Larger would be trivially rejectable.
MISREPORT_LAM = 1.0

# Server-side validation set used by every value function here (FedTSV's
# Delta_val, FP/EE's v).  Same size and noise as FLTrust's root dataset in
# e2_silence, so no scorer gets more server data than another.
N_VAL = 150


# ------------------------------------------------------------------- utility

def make_value(state, cfg):
    """v(w) on the server validation set; bounded in (0,1] unless raw=True."""
    X, y = state["val_X"], state["val_y"]
    n = X.shape[0]
    raw = state.get("v_raw", False)

    def v(w):
        m = float(np.mean((X @ w - y) ** 2)) if n else 0.0
        if not np.isfinite(m):
            return 0.0
        return -m if raw else 1.0 / (1.0 + m)

    return v


def marginals(V, state, cfg):
    """正式 FP／EE 的可观测联盟族（ISWA 32, 200725）。

    U_i = V[i]/N is the pseudogradient of client i, M = M_0 + sum_i U_i.
    Returns v(M_0), v(M), and per client p_i = v(M_0 + U_i),
    q_i = v(M - U_i).
    """
    N = V.shape[0]
    v = make_value(state, cfg)
    w0 = state["w"]
    U = V / N
    wM = w0 + U.sum(axis=0)
    p = np.array([v(w0 + U[i]) for i in range(N)])
    q = np.array([v(wM - U[i]) for i in range(N)])
    return v(w0), v(wM), p, q


def apply_misreport(p, q, state):
    """Client-side misreport of the two marginals only that client can compute.

    Raising p_a = v(M_0 + U_a) and lowering q_a = v(M - U_a) raises FP(a)
    directly (self evaluation) and lowers EE(j) for every j != a, since a's
    reports appear in everyone's score but its own.  Clipped to the honest
    range；这只限制逐标量范围检测，不排除其他异常检测器。
    """
    adv = state.get("misreport")
    if adv is None or len(adv) == 0:
        return p, q
    p, q = p.copy(), q.copy()
    hon = np.setdiff1d(np.arange(len(p)), adv)
    if len(hon) == 0:
        return p, q
    p_hi, q_lo = p[hon].max(), q[hon].min()
    lam = MISREPORT_LAM
    p[adv] = p[adv] + lam * (p_hi - p[adv])
    q[adv] = q[adv] - lam * (q[adv] - q_lo)
    return p, q


def _normalise(score):
    """FedTSV Eq.(13) 的负值截断及全零均匀回退。"""
    s = np.maximum(score, 0.0)
    t = s.sum()
    if not np.isfinite(t) or t <= 0.0:
        return np.full(len(score), 1.0 / len(score))
    return s / t


def fp_ee_weights(v0, vM, p, q, kind, policy="fractional", events=None):
    """正式 Eq.(3)/(5) 的非负份额；退化处理与下游映射单独记录。"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim != 1 or not len(p) or q.shape != p.shape:
        raise ValueError("联盟效用必须为等长非空一维数组")
    if not np.all(np.isfinite(np.r_[v0, vM, p, q])):
        raise ValueError("联盟效用包含非有限值")
    if kind not in ("fp", "ee") or policy not in ("fractional", "minshift"):
        raise ValueError("未知评分器或下游映射")
    events = events if events is not None else {}

    def count(name):
        events[name] = events.get(name, 0) + 1

    count("evaluations")
    if kind == "fp":
        loo, ioi = vM - q, p - v0
        score = np.maximum((loo + ioi) / 2.0, 0.0)
        if not score.sum():
            for name, candidate in (("fp_loo_fallback", loo), ("fp_ioi_fallback", ioi)):
                positive = np.maximum(candidate, 0.0)
                if positive.sum() > 0:
                    score = positive
                    count(name)
                    break
    elif len(p) == 1:
        count("ee_single_client_local_fallback")
        return np.ones(1)
    else:
        tb, tg = vM - p, q - v0
        denominator = (len(p) - 1.0) ** 2
        beta = (tb.sum() - tb) / denominator
        gamma = (tg.sum() - tg) / denominator
        if np.any(((beta > 0) & (gamma < 0)) | ((beta < 0) & (gamma > 0))):
            count("ee_mixed_sign_rounds")
        score = (np.maximum(beta, 0.0) + np.maximum(gamma, 0.0)) / 2.0
    if policy == "minshift":
        # 比较 Remark 3 的平移；除均值后再归一化等价于这里直接除总和。
        score = score - score.min()
    total = score.sum()
    if not np.isfinite(total):
        raise FloatingPointError("评分总和非有限")
    if total <= 0:
        count("uniform_local_fallback")
        return np.full(len(p), 1.0 / len(p))
    return score / total


# ------------------------------------------------------------------- scorers

def w_fp(V, state, cfg):
    """正式 Def.3、Eqs.(3)--(4) 与 Remark 2。"""
    v0, vM, p, q = marginals(V, state, cfg)
    p, q = apply_misreport(p, q, state)
    return fp_ee_weights(v0, vM, p, q, "fp", state.get("score_policy", "fractional"),
                         state.setdefault("fp_ee_events", {}))


def w_ee(V, state, cfg):
    """正式 Def.4、Eqs.(5)--(6)：分别取正部，不能提前平均。"""
    v0, vM, p, q = marginals(V, state, cfg)
    p, q = apply_misreport(p, q, state)
    return fp_ee_weights(v0, vM, p, q, "ee", state.get("score_policy", "fractional"),
                         state.setdefault("fp_ee_events", {}))


def w_fedtsv(V, state, cfg):
    """Trajectory Shapley Value, 2605.30336 Eqs. (9)-(13).

    v_t(S) uses Delta_S = mean update over S (Eq. 9, Delta_empty = 0), so
    v_t(empty) = (1 + ||Delta_val||^2/sigma_t)^{-1} = 1/2 under the paper's
    own sigma_t -- it is not zero, and the Shapley marginals are taken with
    that value, per their Eq. (11).
    """
    N = V.shape[0]
    dval = state["val_step"]
    sig = max(float(dval @ dval), TSV_EPS)
    rng = state["tsv_rng"]
    m = state.get("tsv_perm", TSV_PERM)

    def util(delta):
        r = delta - dval
        return 1.0 / (1.0 + float(r @ r) / sig)

    v_empty = util(np.zeros_like(dval))
    phi = np.zeros(N)
    for _ in range(m):
        order = rng.permutation(N)
        run = np.zeros(V.shape[1])
        prev = v_empty
        for k, i in enumerate(order, start=1):
            run += V[i]
            cur = util(run / k)
            phi[i] += cur - prev
            prev = cur
    phi /= m

    state["tsv_phi"] = state["tsv_phi"] + phi   # Eq. (12), cumulative
    return _normalise(state["tsv_phi"])         # Eq. (13)


def w_fedrandom(V, state, cfg):
    """FedRandom, 2602.05693 Sec. 3.2.

    Every strategy in its ensemble {FedAvg, FedAvgM, FedAdagrad, FedAdam,
    FedYogi} weights clients uniformly; the randomisation is over the SERVER
    optimiser, applied in the update step (see sim), not over client weights.
    """
    return w_uniform(V, state, cfg)


NEW_SCORERS = {
    "FP (2026)": w_fp,
    "EE (2026)": w_ee,
    "FedTSV (2026)": w_fedtsv,
    "FedRandom (2026)": w_fedrandom,
}
ALL_SCORERS = dict(SCORERS)
ALL_SCORERS.update(NEW_SCORERS)

# The columns reported: the two TSP targets from section VI plus the four new
# ones.  The other 2017-2021 scorers are already in b1.
COLS = ["FedAvg (uniform)", "Inverse-variance",
        "FP (2026)", "EE (2026)", "FedTSV (2026)", "FedRandom (2026)"]

ATTACKS = ["none", "tsp-var", "misreport",
           "fang-trim", "shej-minmax"]

MISREPORTING = {"misreport"}
CRAFTED = {"fang-trim", "shej-minmax"}

FEDRANDOM_OPTS = ["fedavg", "fedavgm", "fedadagrad", "fedadam", "fedyogi"]
FR_BETA1, FR_BETA2, FR_TAU = 0.9, 0.99, 1e-3
# Server learning rate for the adaptive members of FedRandom's ensemble.  Not
# a free parameter we may pick to flatter ourselves: block C1.0b sweeps it and
# fixes it at FedRandom's own best no-attack MSE before any comparison runs.
FR_ETA_DEFAULT = 1.0
FR_ETA_GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]


def server_step(name, step, st, eta):
    """One server-side optimiser step (Reddi et al. 2021 forms, as cited by
    2602.05693).  step is the client-weighted pseudogradient."""
    if name == "fedavg":
        return step
    if name == "fedavgm":
        st["m"] = FR_BETA1 * st["m"] + step
        return eta * st["m"]
    st["m"] = FR_BETA1 * st["m"] + (1 - FR_BETA1) * step
    if name == "fedadagrad":
        st["v"] = st["v"] + step ** 2
    elif name == "fedadam":
        st["v"] = FR_BETA2 * st["v"] + (1 - FR_BETA2) * step ** 2
    else:                                        # fedyogi
        st["v"] = st["v"] - (1 - FR_BETA2) * step ** 2 * np.sign(
            st["v"] - step ** 2)
    return eta * st["m"] / (np.sqrt(np.abs(st["v"])) + FR_TAU)


# ---------------------------------------------------------------- simulation

def sim(attack, scorer_name, cfg, seed, v_raw=False, tsv_perm=TSV_PERM,
        fr_eta=FR_ETA_DEFAULT, score_policy="fractional"):
    """One federated run.  Identical to b1_published_baselines.sim except for
    the server validation set, the misreport channel and FedRandom's server
    optimiser."""
    rng = np.random.default_rng(seed)
    cfg = dict(cfg)
    d, N, k = cfg["d"], cfg["N"], cfg["k_adv"]
    adv, hon = np.arange(k), np.arange(k, N)

    s_adv = TSP_S_ADV if attack == "tsp-var" else None
    dl = np.zeros(d)
    if attack == "tsp-var":
        g = rng.normal(size=d)
        dl = TSP_DELTA * g / np.linalg.norm(g)

    w_star = rng.normal(size=d)
    sigma = np.empty(N)
    sigma[hon] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
    sigma[adv] = s_adv if s_adv is not None else rng.uniform(
        cfg["s_lo"], cfg["s_hi"], size=k)

    Xte = rng.normal(size=(cfg["n_test"], d))
    yte = Xte @ w_star

    w = np.zeros(d)
    scorer = ALL_SCORERS[scorer_name]
    state = {"var": np.full(N, 1.0), "hist": np.zeros((N, d)),
             "anchor_ref": np.zeros(d), "w": w,
             "tsv_phi": np.zeros(N), "tsv_rng": np.random.default_rng(seed + 7),
             "tsv_perm": tsv_perm, "v_raw": v_raw,
             "score_policy": score_policy, "fp_ee_events": {},
             "misreport": adv if attack in MISREPORTING else None}
    prev = np.zeros((N, d))
    ev = np.full(N, np.nan)
    opt_state = {"m": np.zeros(d), "v": np.zeros(d)}
    fr_rng = np.random.default_rng(seed + 11)
    hw, aw, mse, alarms, opts, cv = [], [], [], [], [], []

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + dl if i < k else w_star
            X = rng.normal(size=(cfg["n_local"], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=cfg["n_local"])
            V[i] = -cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_local"])

        if attack in CRAFTED:
            Vh = V[hon]
            if attack == "fang-trim":
                V[adv] = craft_fang_trim(Vh, k, rng)
            else:
                V[adv] = craft_shej(Vh, k, "minmax")

        # server-side validation data: the root dataset FLTrust already
        # assumes, reused here as FP/EE's common test set and FedTSV's
        # validation descent reference
        Xv = rng.normal(size=(N_VAL, d))
        yv = Xv @ w_star + rng.normal(0, cfg["s_anchor"], size=N_VAL)
        state["val_X"], state["val_y"] = Xv, yv
        state["val_step"] = -cfg["lr"] * (Xv.T @ (Xv @ w - yv) / N_VAL)
        state["w"] = w

        A = []
        for _ in range(cfg["a_anchor"]):
            X = rng.normal(size=(cfg["n_anchor"], d))
            y = X @ w_star + rng.normal(0, cfg["s_anchor"], size=cfg["n_anchor"])
            A.append(-cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_anchor"]))
        state["anchor_ref"] = np.mean(A, axis=0)

        if t > 0:
            inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
            ev = inst if t == 1 else (1 - cfg["rho"]) * ev + cfg["rho"] * inst
            state["var"] = ev
        prev = V.copy()
        state["hist"] = (1 - cfg["rho"]) * state["hist"] + cfg["rho"] * V

        wt = scorer(V, state, cfg)
        if not np.all(np.isfinite(wt)) or np.any(wt < 0) or not np.isclose(wt.sum(), 1):
            raise AssertionError("评分器没有返回有限非负单纯形权重")
        step = wt @ V
        if scorer_name == "FedRandom (2026)":
            name = FEDRANDOM_OPTS[fr_rng.integers(len(FEDRANDOM_OPTS))]
            opts.append(name)
            step = server_step(name, step, opt_state, fr_eta)
        w = w + step

        hw.append(wt[hon].mean())
        aw.append(wt[adv].sum())
        # coefficient of variation of the honest weights: how much spread the
        # scorer has left to be exploited at all (0 = indistinguishable from
        # FedAvg, so nothing to capture; large = a steep simplex)
        cv.append(float(wt[hon].std() / wt[hon].mean())
                  if wt[hon].mean() > 0 else 0.0)
        m = float(np.mean((Xte @ w - yte) ** 2))
        mse.append(m if np.isfinite(m) else np.inf)
        alarms.append(detectors2(V, adv, state, cfg))

    return dict(hw=np.array(hw), aw=np.array(aw), mse=np.array(mse),
                al=alarms, opts=opts, cv=np.array(cv),
                sigma=sigma, w_last=wt, events=state["fp_ee_events"])


def agg(attack, scorer, seeds, cfg=CFG, **kw):
    R = [sim(attack, scorer, cfg, s, **kw) for s in seeds]
    keys = R[0]["al"][0].keys()
    return dict(
        aw=float(np.mean([r["aw"][-20:].mean() for r in R])),
        hw=float(np.mean([r["hw"][-20:].mean() for r in R])),
        mse=float(np.mean([r["mse"][-20:].mean() for r in R])),
        cv=float(np.mean([r["cv"][-20:].mean() for r in R])),
        events={key: sum(r["events"].get(key, 0) for r in R)
                for key in set().union(*(r["events"] for r in R))},
        al={kk: float(np.mean([np.mean([a[kk] for a in r["al"]]) for r in R]))
            for kk in keys},
    )


# --------------------------------------------------------------------- main

if __name__ == "__main__":
    seeds = list(range(8))
    out = {}
    fair = CFG["k_adv"] / CFG["N"]

    # ---- C1.0 is the Monte Carlo check.  FedTSV's weights come from a
    # sampled Shapley value; if 32 permutations do not agree with 256, every
    # FedTSV number below is noise and nothing else in this script is worth
    # reading.
    print("=" * 78)
    print("C1.0  FedTSV Shapley estimator: 32 permutations vs 256")
    print("=" * 78)
    mc = {}
    for perm in [8, 32, 128, 256]:
        r = agg("none", "FedTSV (2026)", seeds[:4], tsv_perm=perm)
        mc[perm] = dict(aw=r["aw"], hw=r["hw"], mse=r["mse"])
        print(f"perm={perm:4d}  advW={r['aw']:.5f}  honW={r['hw']:.6f}  "
              f"mse={r['mse']:.5f}")
        sys.stdout.flush()
    out["_mc"] = {str(k): v for k, v in mc.items()}
    drift = abs(mc[TSV_PERM]["aw"] - mc[256]["aw"]) / max(mc[256]["aw"], 1e-12)
    print(f"relative advW drift {TSV_PERM} vs 256: {drift:.4f}")

    # ---- C1.0b calibrate FedRandom's server learning rate.  Its adaptive
    # members need one; comparing it at an untuned eta would be rigging the
    # comparison, so the value is chosen by its own no-attack MSE and then
    # frozen.
    print()
    print("=" * 78)
    print("C1.0b FedRandom server learning rate, chosen on its no-attack MSE")
    print("=" * 78)
    etas = {}
    for eta in FR_ETA_GRID:
        r = agg("none", "FedRandom (2026)", seeds[:4], fr_eta=eta)
        etas[eta] = r["mse"]
        print(f"eta={eta:<7} no-attack mse={r['mse']:.5f}")
        sys.stdout.flush()
    FR_ETA = min(etas, key=lambda e: etas[e])
    out["_fr_eta"] = {"grid": {str(k): v for k, v in etas.items()},
                      "chosen": FR_ETA}
    print(f"chosen eta = {FR_ETA}")
    fr_kw = {"FedRandom (2026)": {"fr_eta": FR_ETA}}

    def A(at, sc, sds=seeds, cfg=CFG, **kw):
        kw.update(fr_kw.get(sc, {}))
        return agg(at, sc, sds, cfg, **kw)

    # ---- C1.1 no-attack control per scorer
    print()
    print("=" * 78)
    print("C1.1  no-attack control per scorer (denominator of every ratio)")
    print("=" * 78)
    base = {}
    for sc in COLS:
        base[sc] = A("none", sc)
        print(f"{sc:18s} advW={base[sc]['aw']:.4f} (fair {fair:.4f}) "
              f"honW={base[sc]['hw']:.5f} mse={base[sc]['mse']:.4f} "
              f"honCV={base[sc]['cv']:.3f}")
        sys.stdout.flush()
    out["_base"] = {k: dict(aw=v["aw"], hw=v["hw"], mse=v["mse"], cv=v["cv"])
                    for k, v in base.items()}

    # ---- C1.2 the grid
    print()
    print("=" * 78)
    print("C1.2  every attack against every scorer")
    print("=" * 78)
    print(f"{'attack':14s} {'scorer':18s} {'advW':>7} {'x-fair':>7} "
          f"{'mse/base':>9} {'eps':>7} {'honCV':>6}  max-exc  fired")
    for at in ATTACKS:
        for sc in COLS:
            r = A(at, sc)
            ex = {k: r["al"][k] - base[sc]["al"].get(k, 0.0) for k in r["al"]}
            mx = max(ex.values())
            fired = sorted(k for k, v in ex.items() if v > 0.05)
            ratio = r["mse"] / base[sc]["mse"] if np.isfinite(r["mse"]) else np.inf
            out[f"{at}|{sc}"] = dict(
                aw=r["aw"], hw=r["hw"], mse=r["mse"], mse_ratio=float(ratio),
                x_fair=float(r["aw"] / fair),
                eps=float(1 - r["hw"] / base[sc]["hw"]),
                cv=r["cv"], events=r["events"],
                excess={k: float(v) for k, v in ex.items()},
                max_excess=float(mx), fired=fired)
            print(f"{at:14s} {sc:18s} {r['aw']:7.4f} {r['aw']/fair:7.2f} "
                  f"{ratio:9.2f} {1 - r['hw']/base[sc]['hw']:7.3f} "
                  f"{r['cv']:6.3f}  {mx:+.2f}    "
                  f"{','.join(fired) if fired else '-'}")
            sys.stdout.flush()
        print()

    # ---- C1.2b the misreport row is ONE attack serving two purposes, which
    # is a claim about the geometry of FP and EE, not a convenience.  Check it
    # directly on the score functions: the report that maximises FP(a) must be
    # the same report that minimises the mean EE of everyone else.  If these
    # ever came apart the two would need separate rows in the table.
    print("=" * 78)
    print("C1.2b misreport direction: does one report serve both objectives?")
    print("=" * 78)
    rng2 = np.random.default_rng(20260826)
    both = 0
    trials = 200
    for _ in range(trials):
        n = int(rng2.integers(4, 12))
        v0b, vMb = 0.5, 0.9
        pv = rng2.uniform(0.40, 0.75, n)
        qv = rng2.uniform(0.65, 0.95, n)

        def _fp(pp, qq):
            return fp_ee_weights(v0b, vMb, pp, qq, "fp")

        def _ee(pp, qq):
            return fp_ee_weights(v0b, vMb, pp, qq, "ee")

        a = 0
        oth = np.arange(1, n)
        up_p, dn_q = pv.copy(), qv.copy()
        up_p[a], dn_q[a] = pv[oth].max(), qv[oth].min()      # the report used
        dn_p, up_q = pv.copy(), qv.copy()
        dn_p[a], up_q[a] = pv[oth].min(), qv[oth].max()      # the opposite
        fp_up, fp_dn = _fp(up_p, dn_q)[a], _fp(dn_p, up_q)[a]
        ee_up = _ee(up_p, dn_q)[oth].mean()
        ee_dn = _ee(dn_p, up_q)[oth].mean()
        if fp_up > fp_dn and ee_up < ee_dn:
            both += 1
    print(f"the FP-maximising report also minimises others' EE in "
          f"{both}/{trials} random instances")
    out["_misreport_aligned"] = both / trials
    # 有限随机算例的结果，不充当全部输入的单调性定理。
    print()


    # and reports correlation degrading with N; EE's score is
    # (total - own term), so its spread shrinks like 1/(N-1) and the weights
    # flatten towards FedAvg.  Whether that flattening is what protects it
    # or what disarms it is decided here, not by argument.
    print("=" * 78)
    print("C1.3  federation size: N=9 (their default) vs N=30 (ours)")
    print("=" * 78)
    print(f"{'attack':14s} {'scorer':18s} {'N':>3} {'x-fair':>7} "
          f"{'mse/base':>9} {'spread':>8}")
    nsweep = {}
    for N in [9, 15, 30]:
        c = dict(CFG); c["N"] = N
        for sc in ["FP (2026)", "EE (2026)", "FedTSV (2026)"]:
            b = A("none", sc, seeds, c)
            for at in ["tsp-var", "misreport"]:
                r = A(at, sc, seeds, c)
                f2 = c["k_adv"] / N
                spread = r["aw"] / r["hw"] if r["hw"] > 0 else np.inf
                nsweep[f"{at}|{sc}|{N}"] = dict(
                    x_fair=float(r["aw"] / f2),
                    mse_ratio=float(r["mse"] / b["mse"]),
                    spread=float(spread))
                print(f"{at:14s} {sc:18s} {N:3d} {r['aw']/f2:7.2f} "
                      f"{r['mse']/b['mse']:9.2f} {spread:8.2f}")
                sys.stdout.flush()
    out["_nsweep"] = nsweep

    # ---- C1.4 the comparison must not rest on the one choice the papers
    # leave open in a regression harness: how their accuracy-like utility is
    # rendered.  Repeat the headline with the unbounded v = -MSE.
    print()
    print("=" * 78)
    print("C1.4  utility choice: v = 1/(1+MSE) vs v = -MSE")
    print("=" * 78)
    print(f"{'attack':14s} {'scorer':18s} {'x-fair 1/(1+m)':>15} "
          f"{'x-fair -m':>11}")
    vcheck = {}
    for sc in ["FP (2026)", "EE (2026)"]:
        for at in ["none", "tsp-var", "misreport"]:
            a = A(at, sc)["aw"] / fair
            b = A(at, sc, v_raw=True)["aw"] / fair
            vcheck[f"{at}|{sc}"] = dict(bounded=float(a), raw=float(b))
            print(f"{at:14s} {sc:18s} {a:15.2f} {b:11.2f}")
            sys.stdout.flush()
    out["_vcheck"] = vcheck

    # ---- C1.5 FedRandom sanity: the ensemble must actually be exercised
    r = sim("none", "FedRandom (2026)", CFG, 0, fr_eta=FR_ETA)
    used = sorted(set(r["opts"]))
    print()
    print(f"C1.5  FedRandom server optimisers exercised: {used}")
    out["_fedrandom_opts"] = used
    assert len(used) == len(FEDRANDOM_OPTS), used

    # C1.6 在无攻击且噪声异质的联邦中测量权重与 1/sigma_i^2 的秩相关。
    # 该诊断不等于原文对 Shapley 值的评价，也不是方法整体鲁棒性的证明。
    # 精确并列值用平均秩；常量权重保持未定义，不记为零。
    print()
    print("=" * 78)
    print("C1.6  do the weights track true client quality?  Spearman(w, 1/s^2)")
    print("      all-honest federation, no attack")
    print("=" * 78)
    disc, disc_details = {}, {}
    for sc in COLS:
        rhos = []
        for s in seeds:
            rr = sim("none", sc, CFG, s, **fr_kw.get(sc, {}))
            # sigma comes back from the run itself: reconstructing it from the
            # seed would depend on the generator's draw order (the a11_4 trap)
            rhos.append(spearman(rr["w_last"], 1.0 / rr["sigma"] ** 2))
        disc_details[sc] = correlation_summary(rhos)
        disc[sc] = disc_details[sc]["mean"]
        print(f"{sc:18s} rho={format_correlation(disc[sc])}   "
              f"defined={disc_details[sc]['defined']}/{len(seeds)}   honCV={base[sc]['cv']:.3f}")
        sys.stdout.flush()
    out["_discrim"] = disc

    out["_discrim_details"] = disc_details
    # 下游映射单独比较；不能把采用正式评分公式等同于原图像系统复现。
    out["_mapping_check"] = {}
    for sc in ("FP (2026)", "EE (2026)"):
        for at in ("none", "tsp-var", "misreport"):
            r = A(at, sc, score_policy="minshift")
            out["_mapping_check"][f"{at}|{sc}"] = dict(
                x_fair=r["aw"] / fair, mse=r["mse"], events=r["events"])
    # 旧结论作为可失败的观测项保留；不得为维持论文数字而调参。
    out["_historical_claim_checks"] = dict(
        fp_misreport_above_3=out["misreport|FP (2026)"]["x_fair"] > 3,
        ee_misreport_below_1_5=out["misreport|EE (2026)"]["x_fair"] < 1.5,
        utility_gap_below_0_05=all(abs(v["bounded"] - v["raw"]) < 0.05
                                 for v in out["_vcheck"].values()))
    here = Path(__file__).resolve().parent
    sources = ("c1_recent_scorers.py", "tools/rank_metrics.py", "e2_silence.py",
               "e2b_sweep.py", "b1_published_baselines.py")
    out["_meta"] = dict(version="iswa-positive-channels-v1", seeds=seeds, cfg=CFG,
        rank_policy="average ties; constant undefined; all-seed mean only",
        aggregation="nonnegative fractional shares; FP LOO then IOI; uniform if all vanish",
        source_sha256={n: hashlib.sha256((here / n).read_bytes()).hexdigest() for n in sources})
    (here / "results/c1_recent.json").write_text(json.dumps(out, indent=1, allow_nan=False))
    print("wrote results/c1_recent.json")
    print("C1 OK")
