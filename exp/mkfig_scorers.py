"""生成评分器比较图：份额、质量相关、动态递推下的 FIVA-G 初始方差扫描。

C1/C2 均读取已完成的结果文件并逐项检查源码哈希，不重复运行训练。
常量权重的相关显示 N/A，不在零点绘制。橙色突出 FIVA-G。
E1b/E1c 的训练步不同，分别保留其 generic inverse-variance 对照。
图为通栏三面板；先核对正文数字断言，再写图及其来源记录。
"""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import c1_recent_scorers as c1
import c2_deployed_family1 as c2
from tools.rank_metrics import format_correlation

HERE = Path(__file__).resolve().parent
c1_path = HERE / "results/c1_recent.json"
c2_path = HERE / "results/c2_deployed.json"
c1_data = json.loads(c1_path.read_text())
c2_data = json.loads(c2_path.read_text())
for data, version in ((c1_data, "iswa-positive-channels-v1"),
                      (c2_data, "fiva-equations-1-8-v3-ranks")):
    assert data["_meta"]["version"] == version
    assert data["_meta"]["seeds"] == c2.SEEDS
    for name, expected in data["_meta"]["source_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name

C_CAP = "#2a78d6"      # accent: captured
C_EXC = "#eb6834"      # 突出已修复动态递推的 FIVA-G
C_FLAT = "#898781"     # de-emphasis: resists without tracking quality
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ---------------------------------------------------------------- the data
# 份额与秩统计只取自已完成的同一结果，保留各自的种子口径。
seeds = c1_data["_meta"]["seeds"]
FR_ETA = c1_data["_fr_eta"]["chosen"]

E1B_LBL = {"FedAvg (uniform)": "FedAvg", "Inverse-variance": "inv-var (E1b)",
           "FP (2026)": "FP '26", "EE (2026)": "EE '26",
           "FedTSV (2026)": "FedTSV '26", "FedRandom (2026)": "FedRandom '26"}

print("读取来源哈希已核验的 E1b 结果...")
pts = []
for name in c1.COLS:
    cap = c1_data[f"tsp-var|{name}"]["x_fair"]
    rho = c1_data["_discrim"][name]
    pts.append([name, E1B_LBL[name], rho, cap])
    print(f"  {name:24} rho {format_correlation(rho, 4)}  reward {cap:.3f}x")

print("读取来源哈希已核验的 E1c 结果...")
for key, label in [("invvar", "inv-var (E1c)"), ("pwfl", "PWFL '25"),
                   ("fiva-p", "FIVA-P '25"), ("fiva-g", "FIVA-G '25")]:
    rho = c2_data[f"_discrim|{key}"]
    cap = c2_data[f"tsp-var|{key}"]["x_fair"]
    pts.append([label, label, rho, cap])
    print(f"  {label:24} rho {format_correlation(rho, 4)}  reward {cap:.3f}x")

pts.sort(key=lambda p: p[3])                  # ascending: barh draws bottom-up
labels = [p[1] for p in pts]
rho_v = [p[2] for p in pts]
cap_v = [p[3] for p in pts]
# Emphasis, not categorical: the color says which of the three findings a rule
# belongs to, so it must follow the rule and not its rank.
RESIST = {"FedAvg", "EE '26", "FedRandom '26", "FP '26"}
cols = [C_EXC if l == "FIVA-G '25" else C_FLAT if l in RESIST else C_CAP
        for l in labels]
y = range(len(labels))

fig = plt.figure(figsize=(7.15, 2.62))
gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.80, 1.15], wspace=0.30)
a = fig.add_subplot(gs[0])

# ------------------------------------------------------- (a) reward, sorted
a.barh(list(y), cap_v, height=0.62, color=cols, zorder=3)
a.axvline(1.0, color=AXIS, lw=0.8, zorder=2)
for i, (v, col) in enumerate(zip(cap_v, cols)):
    a.text(v + 0.16, i, f"{v:.2f}", va="center", ha="left", fontsize=7.4,
           color=INK if col != C_FLAT else INK2)
a.set_yticks(list(y))
a.set_yticklabels(labels, fontsize=7.8, color=INK2)
a.set_xlim(0, 9.4)
a.set_xticks([0, 2, 4, 6, 8])
a.set_xlabel("reward / fair share", fontsize=8, color=INK2)
a.set_title("(a) What one adversary takes", fontsize=9, color=INK,
            loc="left", pad=7)
a.grid(axis="x", lw=0.5, color=GRID, zorder=0)

# ------------------------------------- (b) does the rule rank by quality?
b = fig.add_subplot(gs[1], sharey=a)
for i, (v, col) in enumerate(zip(rho_v, cols)):
    if v is None:
        b.text(.55, i, "N/A", va="center", ha="center", fontsize=7.4, color=INK2)
    else:
        assert math.isfinite(v) and -1 <= v <= 1
        b.hlines(i, 0, v, color=col, lw=1.4, zorder=3)
        b.plot(v, i, "o", ms=5.2, color=col, mec="white", mew=1.4, zorder=4)
b.set_xlim(-0.06, 1.14)
b.set_xticks([0, 0.5, 1.0])
b.set_xlabel(r"$\rho(w_i,\,1/\sigma_i^2)$", fontsize=8, color=INK2)
b.set_title("(b) Ranks by quality?", fontsize=9, color=INK, loc="left", pad=7)
b.grid(axis="x", lw=0.5, color=GRID, zorder=0)
plt.setp(b.get_yticklabels(), visible=False)
# FIVA-G 已不再是“抵抗且排序”的例外，不保留旧机制注释。

# ---------------------------------------------- (c) the FIVA-G mechanism
V0 = [1.0, 0.1, 1e-2, 1e-4, 1e-8]
sweep = [c2_data[f"_fivag_v0|{v}"]["x_fair"] for v in V0]
ratio = [c2_data[f"_fivag_v0|{v}"]["ratio"] for v in V0]
print("\nFIVA-G 初始方差扫描（后续动态广播）：")
for v, s, r in zip(V0, sweep, ratio):
    print(f"  sigma^2_0 {v:8.0e}  reward {s:.3f}x  adv/hon est {r:.4f}")

c = fig.add_subplot(gs[2])
gen = c2_data["tsp-var|invvar"]["x_fair"]
c.axhline(gen, color=C_CAP, lw=0.9, ls="-", alpha=0.45, zorder=2)
c.text(3e-7, gen + 0.16, f"generic inverse-variance, {gen:.2f}",
       fontsize=6.8, color=INK2, ha="right")
c.axhline(1.0, color=AXIS, lw=0.8, zorder=2)
c.plot(V0, sweep, color=C_EXC, lw=2.0, solid_capstyle="round", zorder=3)
c.plot(V0, sweep, "o", ms=5.2, color=C_EXC, mec="white", mew=1.4, zorder=4)
# two labels, not five: the endpoints are the argument
c.annotate(f"initial variance $=1$\n{sweep[0]:.2f}",
           xy=(V0[0], sweep[0]), xytext=(2, -30), textcoords="offset points",
           fontsize=7.0, color=INK2, ha="left", linespacing=1.25)
c.annotate(f"{sweep[-1]:.2f}", xy=(V0[-1], sweep[-1]), xytext=(-2, -14),
           textcoords="offset points", fontsize=7.4, color=INK, ha="center")
c.set_xscale("log")
c.invert_xaxis()
c.set_xlabel(r"initial global variance $v_0$"
             "\nsubsequent rounds use Eq. (8)", fontsize=8, color=INK2)
c.set_ylabel("reward / fair share", fontsize=8, color=INK2)
c.set_ylim(0, 5.9)
c.set_title("(c) Dynamic prior, 200 rounds", fontsize=9, color=INK,
            loc="left", pad=7)
c.grid(axis="y", lw=0.5, color=GRID, zorder=0)

for ax_ in (a, b, c):
    ax_.spines[["top", "right"]].set_visible(False)
    ax_.spines[["left", "bottom"]].set_color(AXIS)
    ax_.tick_params(labelsize=7.6, colors=MUTED, length=3)
    ax_.set_axisbelow(True)
for ax_ in (a, b):
    ax_.set_ylim(-0.75, len(labels) - 0.25)

# identity never rests on color alone: three classes, named once
fig.legend(handles=[
    Line2D([], [], color=C_CAP, lw=5, label="captured by TSP-VAR"),
    Line2D([], [], color=C_FLAT, lw=5, label="FP / EE / uniform rules"),
    Line2D([], [], color=C_EXC, lw=5, label="FIVA-G, dynamic prior (panel c)")],
    loc="lower center", ncol=3, frameon=False, fontsize=7.4,
    bbox_to_anchor=(0.5, -0.02), handlelength=1.1, columnspacing=1.6,
    labelcolor=INK2)

# tight_layout cannot handle the shared-y gridspec; place the axes explicitly.
fig.subplots_adjust(left=0.115, right=0.995, top=0.885, bottom=0.30)
# 输出放在断言之后，失败不覆盖现有图。

# ---- the figure must not disagree with tab:recent or tab:deployed.
got = {p[0]: (p[2], p[3]) for p in pts}
for name, claim in [("FedAvg (uniform)", 1.00), ("Inverse-variance", 4.90),
                    ("FP (2026)", 0.71), ("EE (2026)", 1.00),
                    ("FedTSV (2026)", 7.71), ("FedRandom (2026)", 1.00),
                    ("inv-var (E1c)", 4.44), ("PWFL '25", 4.79),
                    ("FIVA-P '25", 4.92), ("FIVA-G '25", 4.09)]:
    assert abs(got[name][1] - claim) < 0.005, (
        f"{name} reward {got[name][1]:.3f} disagrees with the {claim} reported "
        "in sec:e1b/sec:e1c; figure and text must be regenerated together")
for name, claim in [("Inverse-variance", 0.95),
                    ("FP (2026)", 0.05), ("FedTSV (2026)", 0.82),
                    ("PWFL '25", 0.87), ("FIVA-P '25", 0.89),
                    ("FIVA-G '25", 0.84)]:
    assert abs(got[name][0] - claim) < 0.005, (name, got[name][0], claim)
assert got["FedAvg (uniform)"][0] is None
assert got["FedRandom (2026)"][0] is None
assert c2_data["_discrim|fedavg"] is None
assert got["EE (2026)"][0] is None
assert c1_data["_discrim_details"]["EE (2026)"]["defined"] == 2
assert c1_data["_discrim_details"]["EE (2026)"]["total"] == 8
for name, claim in (("FP (2026)", 4.37), ("EE (2026)", 1.10)):
    assert abs(c1_data[f"misreport|{name}"]["x_fair"] - claim) < 0.005
# 新评分结果与正文同时核对；未定义值不替换为零。
for actual, printed in zip(sweep, [4.09, 4.09, 4.09, 4.09, 4.83]):
    assert abs(actual - printed) < 0.005, sweep
assert all(math.isfinite(v) and v >= 0 for v in sweep + ratio)
# panel (a) must be sorted, or the form carries no information
assert cap_v == sorted(cap_v), cap_v
# and (b) must share (a)'s order for the read-across to be valid
assert [p[1] for p in pts] == labels
print("FIGURE CLAIMS: FedTSV %.2fx (rho %.2f) | PWFL %.2fx | FIVA-P %.2fx | "
      "FIVA-G %.2fx (rho %.2f)"
      % (got["FedTSV (2026)"][1], got["FedTSV (2026)"][0], got["PWFL '25"][1],
         got["FIVA-P '25"][1], got["FIVA-G '25"][1], got["FIVA-G '25"][0]))
print("               prior sweep %s -> generic %.2fx"
      % (" ".join(f"{s:.2f}" for s in sweep), gen))
target = HERE.parent / "paper/fig/scorers.pdf"
target.parent.mkdir(exist_ok=True)
fig.savefig(target, bbox_inches="tight")
fig.savefig(HERE / "results/scorers-preview.png", dpi=200, bbox_inches="tight")
evidence = dict(c1_sha256=hashlib.sha256(c1_path.read_bytes()).hexdigest(),
                c2_sha256=hashlib.sha256(c2_path.read_bytes()).hexdigest(),
                pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                source_sha256={n: hashlib.sha256((HERE / n).read_bytes()).hexdigest()
                               for n in ("mkfig_scorers.py", "c1_recent_scorers.py",
                                         "c2_deployed_family1.py", "tools/rank_metrics.py")},
                points=pts, initial_variances=V0, sweep=sweep, estimator_ratios=ratio,
                c1_seeds=seeds, c2_quality_seed=0, fedrandom_eta=FR_ETA)
(HERE / "results/scorers_figure.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
print(f"已写图与来源记录：{target}")
