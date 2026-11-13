"""
E2c: three questions E2b left open.

Q1  Is there an operating point with meaningful eps and NO detector excess?
Q2  Is Family II genuinely immune to TSP-Var, or is k=1 just too weak?
Q3  Why does rho=0 kill the attack -- amplification, or estimability?
"""

import numpy as np
from e2_silence import SCORERS, CFG
import e2b_sweep as m
from e2b_sweep import agg, sim

seeds = [0, 1, 2, 3, 4, 5, 6, 7]

print("=" * 74)
print("Q1  silent operating point   (excess = alarm rate - own false-positive)")
print("=" * 74)
fp = agg("Inverse-variance", 0.0, 1.0, seeds)
print(f"control:  honW={fp['hw']:.5f}  mse={fp['mse']:.4f}  "
      f"FP={ {k: round(v,2) for k,v in fp['al'].items()} }\n")
print(f"{'s_adv':>6} {'delta':>6} {'advW':>7} {'eps':>7} {'mse/base':>9}   max-excess")
for sa in [0.6, 0.5, 0.4, 0.35, 0.3, 0.25]:
    for dm in [0.05, 0.15, 0.3]:
        r = agg("Inverse-variance", dm, sa, seeds)
        ex = {k: r["al"][k] - fp["al"].get(k, 0.0) for k in r["al"]}
        mx = max(ex.values())
        flag = "  <-- SILENT" if mx < 0.05 else ""
        print(f"{sa:6.2f} {dm:6.2f} {r['aw']:7.3f} "
              f"{1 - r['hw']/fp['hw']:7.3f} {r['mse']/fp['mse']:9.2f}   "
              f"{mx:+.2f}{flag}")

print()
print("=" * 74)
print("Q2  Family II under a stronger adversary  (s_adv=0.1, delta=0.1)")
print("=" * 74)
print(f"{'scorer':20s} {'k=1':>8} {'k=3':>8} {'k=6':>8}   (adversary weight share)")
for nm in SCORERS:
    row = []
    for k in [1, 3, 6]:
        c = dict(CFG); c["k_adv"] = k
        old = m.CFG; m.CFG = c
        r = m.agg(nm, 0.1, 0.1, seeds)
        m.CFG = old
        row.append(r["aw"])
    fair = [k / CFG["N"] for k in [1, 3, 6]]
    print(f"{nm:20s} " + " ".join(f"{v:8.3f}" for v in row) +
          "   fair=" + ",".join(f"{f:.2f}" for f in fair))

print()
print("=" * 74)
print("Q3  rho=0 mechanism: is variance even estimable without memory?")
print("=" * 74)
for rho, lab in [(0.0, "frozen at t=1"), (0.2, "EMA"), (1.0, "latest round only")]:
    r = sim("Inverse-variance", 0.05, 0.1, CFG, 0, rho=rho)
    print(f"rho={rho:4.2f} ({lab:18s})  advW_final={r['aw'][-20:].mean():.3f}")
print()
print("If rho=1.0 (no memory at all, pure single-round estimate) still captures,")
print("then rho=0 is special only because state['var'] stays at its init value.")
