"""
E2: Silence test for Trust-State Poisoning.

Instantiates the observation model  v_i = v* + b_i + eps_i  as a federated
linear regression, because it is the simplest setting in which bias and
variance are INDEPENDENTLY controllable. Classification benchmarks couple
them, which would make the TSP-Var mechanism untestable.

Honest client i:   y = X @ w_star + N(0, sigma_i^2)      sigma_i in [s_lo, s_hi]
Adversary client:  y = X @ (w_star + delta) + N(0, sigma_adv^2),  sigma_adv << s_lo

The adversary trains honestly on its own (curated) data. It fabricates
nothing. Its update is the true gradient of its true local objective.
"""

import numpy as np

# ---------------------------------------------------------------- scorers

def w_uniform(V, state, cfg):
    n = len(V)
    return np.ones(n) / n


def w_invvar(V, state, cfg):
    """Family I: inverse-variance. Needs no reference frame."""
    var = np.maximum(state["var"], cfg["var_floor"] ** 2)
    w = 1.0 / var
    return w / w.sum()


def w_krum(V, state, cfg):
    """Family II. Multi-Krum: average the m closest-to-consensus clients."""
    n = len(V)
    k = cfg["k_adv"]
    m = max(1, n - k - 2)
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2) ** 2
    scores = np.sort(D, axis=1)[:, 1:m + 1].sum(axis=1)
    sel = np.argsort(scores)[:max(1, n - k)]
    w = np.zeros(n)
    w[sel] = 1.0 / len(sel)
    return w


def w_trimmed(V, state, cfg):
    """按坐标幸存次数分配标量权重的 surrogate，并非逐坐标 trimmed mean。"""
    n, d = V.shape
    beta = cfg["trim_beta"]
    t = int(np.floor(beta * n))
    order = np.argsort(V, axis=0)
    keep = np.zeros((n, d))
    keep[order[t:n - t, np.arange(d)], np.arange(d)] = 1.0
    w = keep.sum(axis=1)
    return w / w.sum()


def w_fltrust(V, state, cfg):
    """FLTrust-inspired 余弦权重；此适配未执行原算法的更新范数重缩放。"""
    ref = state["anchor_ref"]
    rn = np.linalg.norm(ref) + 1e-12
    vn = np.linalg.norm(V, axis=1) + 1e-12
    cos = (V @ ref) / (vn * rn)
    ts = np.maximum(cos, 0.0)
    if ts.sum() < 1e-12:
        return np.ones(len(V)) / len(V)
    return ts / ts.sum()


def w_foolsgold(V, state, cfg):
    """历史余弦相似度简化评分；省略 FoolsGold 的 pardoning 和 logit 步骤。"""
    H = state["hist"]
    n = len(H)
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    CS = Hn @ Hn.T
    np.fill_diagonal(CS, 0.0)
    mx = CS.max(axis=1)
    wv = np.ones(n)
    for i in range(n):
        if mx[i] > 1e-12:
            wv[i] = 1.0 - mx[i]
    wv = np.clip(wv, 1e-6, 1.0)
    return wv / wv.sum()


SCORERS = {
    "FedAvg (uniform)": w_uniform,
    "Inverse-variance": w_invvar,
    "Krum": w_krum,
    "Trimmed mean": w_trimmed,
    "FLTrust": w_fltrust,
    "FoolsGold": w_foolsgold,
}

# ---------------------------------------------------------------- detectors

def detectors(V, adv_idx, state, cfg):
    """离线检测评价；adv_idx 是受评估集合，诚实对照使用同一集合。"""
    adv_idx = np.asarray(adv_idx, dtype=int)
    if adv_idx.ndim != 1 or not adv_idx.size:
        raise ValueError("受评估集合必须是一维非空索引")
    if np.any(adv_idx < 0) or np.any(adv_idx >= len(V)):
        raise ValueError("受评估索引越界")
    out = {}
    nrm = np.linalg.norm(V, axis=1)
    med, mad = np.median(nrm), np.median(np.abs(nrm - np.median(nrm))) + 1e-12
    out["norm"] = bool(np.any(np.abs(nrm[adv_idx] - med) / mad > cfg["z_norm"]))

    mu = V.mean(axis=0)
    dist = np.linalg.norm(V - mu, axis=1)
    dmed = np.median(dist)
    dmad = np.median(np.abs(dist - dmed)) + 1e-12
    out["dist"] = bool(np.any((dist[adv_idx] - dmed) / dmad > cfg["z_dist"]))

    krum_w = w_krum(V, state, cfg)
    out["krum_adv"] = bool(np.all(krum_w[adv_idx] == 0.0))

    ref = state["anchor_ref"]
    cos = (V @ ref) / (np.linalg.norm(V, axis=1) * np.linalg.norm(ref) + 1e-12)
    out["cos"] = bool(np.any(cos[adv_idx] < cfg["cos_thresh"]))
    return out

# ---------------------------------------------------------------- simulation

def run(scorer_name, delta_mag, cfg, seed):
    rng = np.random.default_rng(seed)
    d, N, k = cfg["d"], cfg["N"], cfg["k_adv"]
    adv_idx = np.arange(k)
    honest_idx = np.arange(k, N)

    w_star = rng.normal(size=d)
    delta = rng.normal(size=d)
    delta = delta_mag * delta / np.linalg.norm(delta)

    sigma = np.empty(N)
    sigma[honest_idx] = rng.uniform(cfg["s_lo"], cfg["s_hi"], size=N - k)
    sigma[adv_idx] = cfg["s_adv"]

    n_loc = np.full(N, cfg["n_local"])
    Xte = rng.normal(size=(cfg["n_test"], d))
    yte = Xte @ w_star

    w = np.zeros(d)
    scorer = SCORERS[scorer_name]
    state = {
        "var": np.full(N, 1.0),
        "hist": np.zeros((N, d)),
        "anchor_ref": np.zeros(d),
    }
    prev = np.zeros((N, d))
    ema_var = np.full(N, np.nan)

    rec = {"hw": [], "aw": [], "mse": [], "alarms": []}

    for t in range(cfg["rounds"]):
        V = np.empty((N, d))
        for i in range(N):
            tgt = w_star + delta if i < k else w_star
            X = rng.normal(size=(n_loc[i], d))
            y = X @ tgt + rng.normal(0, sigma[i], size=n_loc[i])
            g = X.T @ (X @ w - y) / n_loc[i]
            V[i] = -cfg["lr"] * g

        # anchor reference: a trusted clients, clean data, honest target
        A = []
        for _ in range(cfg["a_anchor"]):
            X = rng.normal(size=(cfg["n_anchor"], d))
            y = X @ w_star + rng.normal(0, cfg["s_anchor"], size=cfg["n_anchor"])
            A.append(-cfg["lr"] * (X.T @ (X @ w - y) / cfg["n_anchor"]))
        state["anchor_ref"] = np.mean(A, axis=0)

        # dispersion estimate from own across-round movement (no reference)
        if t > 0:
            inst = np.linalg.norm(V - prev, axis=1) ** 2 / d
            ema_var = inst if t == 1 else (1 - cfg["rho"]) * ema_var + cfg["rho"] * inst
            state["var"] = ema_var
        prev = V.copy()
        state["hist"] = (1 - cfg["rho"]) * state["hist"] + cfg["rho"] * V

        wt = scorer(V, state, cfg)
        w = w + wt @ V

        rec["hw"].append(wt[honest_idx].mean())
        rec["aw"].append(wt[adv_idx].sum())
        rec["mse"].append(float(np.mean((Xte @ w - yte) ** 2)))
        rec["alarms"].append(detectors(V, adv_idx, state, cfg))

    return rec


CFG = dict(
    d=20, N=30, k_adv=1, rounds=200, lr=0.30,
    n_local=50, n_test=2000,
    s_lo=0.5, s_hi=1.5, s_adv=0.02,
    a_anchor=3, n_anchor=50, s_anchor=0.3,
    rho=0.2, var_floor=1e-8, trim_beta=0.1,
    z_norm=3.5, z_dist=3.5, cos_thresh=0.0,
)

def detector_regression_checks():
    """覆盖联盟全排除、部分排除及诚实客户端被排除的区别。"""
    from unittest.mock import patch

    V = np.arange(12, dtype=float).reshape(6, 2) + 1
    state = {"anchor_ref": np.ones(2)}
    cfg = dict(CFG, N=6, k_adv=2)
    cases = [
        ([.25, .25, .25, .25, 0, 0], [0, 1], False),
        ([0, .25, .25, .25, .25, 0], [0, 1], False),
        ([0, 0, .25, .25, .25, .25], [0, 1], True),
        ([0, .25, .25, .25, .25, 0], [0], True),
        ([0, .25, .25, .25, .25, 0], [1], False),
    ]
    for weights, evaluated, expected in cases:
        with patch.dict(detectors.__globals__, {"w_krum": lambda *args: np.array(weights)}):
            out = detectors(V, evaluated, state, cfg)
        assert out["krum_adv"] is expected, (weights, evaluated, out)
        assert set(out) == {"norm", "dist", "krum_adv", "cos"}
    try:
        detectors(V, [], state, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("空集合不能被当作全部排除")
    actual = w_krum(V, state, cfg)
    assert np.count_nonzero(actual) == cfg["N"] - cfg["k_adv"]
    np.testing.assert_allclose(actual.sum(), 1.0)
    print("Krum 回归检查通过：联盟全排除语义、空集合、实际选择数量")


if __name__ == "__main__":
    import json, sys
    from pathlib import Path

    detector_regression_checks()
    seeds = [0, 1, 2, 3, 4]
    deltas = [0.0, 0.05, 0.1, 0.2, 0.5]
    results = {}

    for sname in SCORERS:
        for dm in deltas:
            runs = [run(sname, dm, CFG, s) for s in seeds]
            hw = np.array([r["hw"] for r in runs])
            aw = np.array([r["aw"] for r in runs])
            mse = np.array([r["mse"] for r in runs])
            al = {kk: float(np.mean([np.mean([a[kk] for a in r["alarms"]])
                                     for r in runs]))
                  for kk in runs[0]["alarms"][0]}
            results[f"{sname}|{dm}"] = dict(
                hw_final=float(hw[:, -20:].mean()),
                hw_init=float(hw[:, 1:6].mean()),
                aw_final=float(aw[:, -20:].mean()),
                mse_final=float(mse[:, -20:].mean()),
                alarms=al,
            )
            print(f"{sname:20s} d={dm:4.2f}  "
                  f"advW={results[f'{sname}|{dm}']['aw_final']:.3f}  "
                  f"honW={results[f'{sname}|{dm}']['hw_final']:.4f}  "
                  f"mse={results[f'{sname}|{dm}']['mse_final']:.4f}  "
                  f"alarms={ {k: round(v,2) for k,v in al.items() if v>0} }")
            sys.stdout.flush()

    output = Path(__file__).resolve().parent
    (output / "e2_results.json").write_text(json.dumps(results, indent=1))
    controls = {name: run(name, 0.0, dict(CFG, s_adv=1.0), seeds[0])
                for name in SCORERS}
    control_rates = {name: {key: float(np.mean([a[key] for a in rec["alarms"]]))
                            for key in rec["alarms"][0]}
                     for name, rec in controls.items()}
    meta = {"seeds": seeds, "config": CFG, "deltas": deltas,
            "detector_semantics": "krum_adv: all evaluated clients excluded",
            "control_seed": seeds[0], "control_s_adv": 1.0,
            "control_rates": control_rates, "python": sys.version,
            "numpy": np.__version__, "results": results}
    (output / "results").mkdir(exist_ok=True)
    (output / "results" / "e2_revision.json").write_text(json.dumps(meta, indent=2))
