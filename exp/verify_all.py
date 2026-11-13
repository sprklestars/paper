"""Run every verification script and record the output.

    ../.venv/bin/python verify_all.py            # verification only
    ../.venv/bin/python verify_all.py --figures  # and regenerate paper/fig/*.pdf

逐次运行写入 results/runs/<时间戳>/，保留输出、退出码及结构化摘要。
注册表用于声明定位，不保证每个正文数字都有断言。三张图有数字断言，
其余图只重算；negative result 与 legacy formulation check 不作为当前命题。
可用 --only 限定脚本，--output-dir 指定一个尚未使用的日志目录。
"""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"

SCRIPTS = [
    ("v1_gauge.py", "thm:unidentifiable, prop:impossible"),
    ("v2_feedback.py", "eq:feedback, prop:fixedpoint"),
    ("v3_defense_bounds.py", "lem:budget, M1 vs M2, thm:displacement"),
    ("v4_anchor_mechanism.py", "eq:firstorder, relative floor"),
    ("v5_repaired_bound.py", "thm:displacement: 1080 averaged-weight diagnostics; not a pointwise hypothesis check"),
    ("v6_capture.py", "thm:capture, prop:ceiling, prop:threshold"),
    # The linear-model track's experiment section: the silent operating point
    # sec:setup fixes, its robustness, and what M1/M2 bound there.
    ("e2_silence.py",
     "sec:setup: weighting and raw alarms; unified Krum coalition exclusion; "
     "silence/excess controls are evaluated in E2b--E2d"),
    ("e2b_sweep.py",
     "E2b: the (s_adv, delta) frontier behind the operating point, so the "
     "reported cell is a choice from a measured surface and not a lucky one"),
    ("e2c_confirm.py",
     "E2c: the operating point does not depend on the EMA memory -- rho = 1.0, "
     "a pure single-round estimate, still captures, so capture is not an "
     "artefact of the state's memory"),
    ("e2d_trustharm.py",
     "E2d: reward distortion against detectability, k = 1 to 6 -- the 4.60x "
     "uniform share at (0.35, 0.05) that sec:setup quotes, and its frontier"),
    ("e3_defense.py",
     "E3: M1's variance floor and M2's anchor mixing measured on that point, "
     "against BOTH the undefended-clean baseline and FedAvg, because a "
     "defence that removes capture by destroying the weighting is FedAvg"),
    # AI-specific track: the neural network parameter symmetry group.
    ("a1_nn_symmetry.py", "S1/S2 exact; scorers gauge-sensitive"),
    ("a2_gauge_alignment.py", "NEGATIVE: gauge descent cannot hide a backdoor"),
    ("a3_gauge_stabilise.py", "NEGATIVE: AM-GM floor on faked stability"),
    ("a4_aggregate_gauge.py", "aggregate not gauge invariant; tradeoff"),
    ("a5_honest_exclusion.py", "NULL: honest gauge drift vs linear control"),
    ("a6_audit_gap.py", "prop:auditgap: functional statistics vs aggregate loss"),
    ("mlp.py", "generic MLP: gradients vs finite diff; symmetry table"),
    ("a7_capacity.py", "capacity axis; damage is driven by gauge MAGNITUDE"),
    ("a8_normalisation.py", "LayerNorm relocates the gauge and ADDS to it"),
    ("a9_gauge_displacement.py", "NULL on displacement; cold-start round"),
    ("a10_magnitude_law.py", "the magnitude law; audit blind at every point"),
    ("a11_permutation_canon.py",
     "prop:ordering: numerical orbit checks; local function preservation, nonzero aggregate cost"),
    ("a12_policy_binding.py",
     "M3: naive equality gives false positives; digest must bind the "
     "transition map; M4-then-M3 ordering"),
    ("a13_real_data.py",
     "real-data replication: California housing by geography, UCI digits by "
     "label skew; audit gap and M4 both hold on data we did not design"),
    ("a14_auditgap_exact.py",
     "prop:auditgap / prop:gauge-damage: empirical deep-network witness; "
     "quadratic tail, IVT relative error <1e-6, float64 atol=rtol=1e-12; "
     "not a universal deep-network theorem"),
    ("a15_gauge_constrained.py",
     "what (C4) leaves when the WHOLE detector suite is on: the suite first "
     "fires at per-unit log-sd 0.05 where damage is 1.0001x, and the largest "
     "silent damage bisected per seed is a median 1.0001x -- so the gauge is "
     "an auditing result, not a stealth one. Also the seed spread behind the "
     "single draws sec:e6 quotes: the composed attack has median 7.9e5x "
     "undefended (943x to 6.3e7x) and 1.0014x under M4+"),
    ("a16_ceiling_residual.py",
     "prop:ceiling's 0.9-1.9 point residual attributed: honest-client "
     "heterogeneity through the convexity of 1/sigma^2, not estimator bias. A "
     "common sigma_H closes it to 0.03 points at every N and rho in {0.9, "
     "0.99, 0.999} changes nothing; the direction favours the defender"),
    ("a17_digest_granularity.py",
     "def:funcaudit: fixed-tolerance function values vs exact SHA-256; "
     "honest batching/reordering controls, measured error ordering, "
     "out-of-tolerance negative control"),
    ("a18_quotient_ties.py",
     "prop:ordering boundary: descriptor ties can violate the quotient; "
     "pairwise descriptor separation does not prove unique optimal assignment"),
    ("a19_dead_units.py",
     "the OTHER escape from Gamma: a dead ReLU's outgoing column is free at "
     "max|df| = 0 EXACTLY and unreachable in Gamma. M4 in isolation amplifies "
     "it (40.7x plain average vs 5.5e6 canonicalised, activation rate 0.077 -> "
     "0.486). Inverse variance contains it at 1.004x, but M2's (1-lam)/N floor "
     "admits it and the federation diverges at lam=0.99 -- the two mechanisms "
     "pull opposite ways on this move"),
    ("b1_published_baselines.py",
     "published-attack baselines (Fang USENIX'20, Shejwalkar NDSS'21, Fraboni "
     "AISTATS'21) in the identical harness: the same detector suite fires on "
     "all six and stays quiet on TSP-Var; L1 and L3 harm are orthogonal; B1.6 "
     "reads out the two statistics behind the ACE rows, where the separation "
     "is the SPREAD of the norm ratio and not its level"),
    ("b2_matcher_baselines.py",
     "sec:e6 / prop:ordering: activation and OT-inspired matching adaptations; "
     "scale-before-matching numerical checks, hard vs soft transport"),
    ("c1_recent_scorers.py",
     "2026 contribution-evaluation methods (Fair-Private and Everybody-Else, "
     "ISWA 32 (2026) 200725; Trajectory Shapley, arXiv 2605.30336; FedRandom, "
     "arXiv 2602.05693; formal chapter pending) adapted in the same "
     "harness: capture, quality correlation and detector excess under documented score mappings"),
    ("c2_deployed_family1.py",
     "E1c：FIVA 正式式 (1)--(8) 递推与 11 项回归测试；均值和方差分离、"
     "动态先验广播、无偏移诚实对照；捕获、效用、检测、初始方差、遗忘、"
     "数值稳定项及 scalar collapse；PWFL 对照正式 Eq.(5)，SGD 二阶矩适配非完整 Adam 复现"),
    ("e9_adaptive_defense.py",
     "the defences against an adversary that KNOWS them: it observes (c, "
     "lambda) and re-solves a 40-cell (s_adv, delta) grid on seeds 0-4, with "
     "the winner re-measured out of sample on seeds 5-12. Re-optimisation buys "
     "at most +0.04x against any setting in tab:defense, and M2's analytic "
     "ceiling (1-lam)/N + lam holds under it -- but silence is what binds the "
     "adversary, not the defence: dropping it reaches 24.81x undefended"),
    ("e11_hierarchical.py",
     "client-edge-cloud hierarchical aggregation (HierFAVG topology, "
     "arXiv 1905.06641): two-level averaging caps a confined adversary at 1/m "
     "of the global model, verified as a per-round assertion for m in "
     "{2,3,5,6,10}. But the cap is a topology property, not a security one: "
     "spreading the SAME k=3 adversaries one per cluster beats confining them "
     "at every m (2.90x at m=5), and cluster membership is edge attachment, "
     "which is reported rather than verified"),
    # Claims in the paper that a proof states but the harness never measured,
    # each found by reading the proofs against the code they cite.
    ("a20_impossibility_scope.py",
     "prop:impossible / eq:suppcost: relative-only identity and legacy universal-claim counterexamples"),
    ("a21_threshold_criterion.py",
     "prop:threshold: criterion and horizon sensitivity; legacy threshold interpretation check"),
    ("a22_ceiling_exact.py",
     "prop:ceiling: exact population mean-residual formula vs legacy approximation"),
    ("a23_same_harness_defense.py",
     "sec:e3: matched-harness reference vs weight mixing; legacy cross-harness comparison check"),
    ("a24_weight_cap.py",
     "sec:e3 / eq:suppcost: hard-cap utility, quality correlation, adaptive "
     "grid and coalition-size limits; favourable hard-cap control retained"),
    ("a25_orbit_scope.py",
     "sec:gauge: minimal linear regression vs redundant linear models; legacy scope check"),
    ("a26_real_confound.py",
     "sec:e10: measured dispersion-bias correlation and capture/harm separation on real partitions"),
]

# CIFAR 使用另行锁定的 torch 环境；这里不推断其他脚本的最低 NumPy 版本。
# 未找到该解释器时记录 interpreter_missing，不计为通过。
TORCH_PY = ROOT / ".venv-torch" / "bin" / "python3"
TORCH_SCRIPTS = [
    ("e10_cifar_cnn.py",
     "CIFAR-10 with a real CNN and a BatchNorm CNN, 10 Dirichlet(0.5) clients "
     "and 20 FedAvg rounds: prop:auditgap, the fig:maglaw magnitude law and M4 "
     "all replicate off the MLP harness, and BatchNorm does not close the "
     "gauge group (the a8 finding on a convnet)"),
]

# 五张图均重算；只有 maglaw、policy、scorers 带部分正文数字断言。
FIGURES = [
    ("mkfig_silence.py", "fig:silence -- E2, model layer vs trust layer"),
    ("mkfig_maglaw.py",
     "fig:maglaw -- magnitude law (asserts slope 0.940) + blind audit"),
    ("mkfig_defense.py",
     "fig:defense -- M4 on four gauge moves; honest cost 1.001x"),
    ("mkfig_policy.py",
     "fig:policy -- M3 false positive (asserts 1.236x state freedom)"),
    ("mkfig_scorers.py",
     "fig:scorers：读取已验证来源哈希的 C1/C2 结果；权重份额、平均秩与未定义相关"
     "和 FIVA-G 动态递推下的初始方差扫描；部分正文数字断言"),
]


def dependency_census():
    """The accounting sec:setup prints, computed rather than maintained.

    The paper states how many scripts need which third-party package. That
    sentence was wrong once because it was counted by reading the import lines
    of each script, which misses a transitive need: five scripts reach scipy
    only through a11_permutation_canon's linear_sum_assignment. Imports are
    followed through local modules here so the number cannot drift again.
    """
    heavy = {"scipy", "sklearn"}

    def externals(name, seen=None):
        seen = seen if seen is not None else set()
        if name in seen:
            return set()
        seen.add(name)
        txt = (HERE / name).read_text()
        mods = set(re.findall(r"^(?:import|from)\s+(\w+)", txt, re.M))
        local = {m for m in mods if (HERE / f"{m}.py").exists()}
        out = mods - local
        for m in local:
            out |= externals(f"{m}.py", seen)
        return out

    plain = [n for n, _ in SCRIPTS if not externals(n) & heavy]
    sci = [n for n, _ in SCRIPTS if externals(n) & heavy]
    total = len(SCRIPTS) + len(TORCH_SCRIPTS) + len(FIGURES)
    print(f"registry: {total} scripts -- {len(plain)} numpy only, "
          f"{len(sci)} scipy/sklearn, {len(TORCH_SCRIPTS)} torch, "
          f"{len(FIGURES)} figures")
    assert len(plain) + len(sci) == len(SCRIPTS)
    return total, len(plain), len(sci)


def artifact_state():
    """用修改时间与内容哈希识别同内容重写及新结果。"""
    paths = [HERE / "e2_results.json", *OUT.glob("*.json")]
    return {p: (p.stat().st_mtime_ns, p.stat().st_size,
                hashlib.sha256(p.read_bytes()).hexdigest())
            for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--only", nargs="+", help="指定注册表中的脚本名")
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()
    dependency_census()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.output_dir or OUT / "runs" / stamp).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (ROOT / "paper" / "fig").mkdir(parents=True, exist_ok=True)
    scripts = [(n, c, sys.executable) for n, c in SCRIPTS]
    scripts += [(n, c, str(TORCH_PY)) for n, c in TORCH_SCRIPTS]
    scripts += [(n, c, sys.executable) for n, c in FIGURES]
    known = {n for n, _, _ in scripts}
    if args.only and set(args.only) - known:
        parser.error(f"未注册脚本：{set(args.only) - known}")
    figure_names = {n for n, _ in FIGURES}
    report = dict(started_utc=stamp, python=sys.version, executable=sys.executable,
                  platform=platform.platform(), command=sys.argv,
                  thread_environment={k: os.environ.get(k) for k in
                                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                  source_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [*HERE.glob("*.py"), *(HERE / "tools").glob("*.py")]},
                  environments={}, runs=[])
    for py in sorted({py for _, _, py in scripts if pathlib.Path(py).exists()}):
        probe = subprocess.run([py, str(HERE / "tools" / "provenance.py"), "--probe"],
                               capture_output=True, text=True, cwd=HERE)
        if probe.returncode != 0:
            raise RuntimeError(f"环境记录失败：{py}\n{probe.stderr}")
        environment = json.loads(probe.stdout)
        report["environments"][py] = environment
    negative = {"a2_gauge_alignment.py", "a3_gauge_stabilise.py", "a5_honest_exclusion.py", "a9_gauge_displacement.py"}
    legacy = {"v2_feedback.py", "v3_defense_bounds.py", "v4_anchor_mechanism.py",
              "a20_impossibility_scope.py", "a21_threshold_criterion.py", "a22_ceiling_exact.py",
              "a23_same_harness_defense.py", "a25_orbit_scope.py"}
    for name, claims, py in scripts:
        kind = "negative_result" if name in negative else "legacy_and_current_check" if name in legacy else "final_verification"
        item = dict(script=name, claims=claims, interpreter=py, evidence_kind=kind,
                    environment_id=report["environments"].get(py, {}).get("environment_id"))
        if args.only and name not in args.only:
            item.update(status="SKIPPED", reason="not_selected")
        elif name in figure_names and not args.figures and not args.only:
            item.update(status="SKIPPED", reason="figures_not_requested")
        elif not pathlib.Path(py).exists():
            item.update(status="SKIPPED", reason="interpreter_missing")
        else:
            t0 = time.monotonic()
            before = artifact_state()
            try:
                result = subprocess.run([py, str(HERE / name)], capture_output=True,
                                        text=True, cwd=HERE)
                (out / f"{name}.txt").write_text(result.stdout + result.stderr)
                item.update(status="PASSED" if result.returncode == 0 else "FAILED",
                            returncode=result.returncode, seconds=time.monotonic() - t0)
                # 仅归属本次子进程实际新建或改写的文件，不混入已有历史结果。
                item["artifacts"] = []
                for artifact, state in artifact_state().items():
                    if before.get(artifact) == state:
                        continue
                    target = out / "artifacts" / name / artifact.relative_to(HERE)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    content = artifact.read_bytes()
                    target.write_bytes(content)
                    item["artifacts"].append(dict(producer=name,
                        source=str(artifact.relative_to(HERE)),
                        snapshot=str(target.relative_to(out)),
                        sha256=hashlib.sha256(content).hexdigest()))
            except OSError as error:
                item.update(status="FAILED", error=str(error), returncode=None)
            print(f"{item['status']:>7}  {name:<28} {item.get('seconds', 0):7.1f}s", flush=True)
        report["runs"].append(item)
        report["counts"] = {s: sum(r["status"] == s for r in report["runs"])
                            for s in ("PASSED", "FAILED", "SKIPPED")}
        (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"{report['counts']}; 日志：{out}", flush=True)
    return int(report["counts"]["FAILED"] > 0)


if __name__ == "__main__":
    sys.exit(main())
