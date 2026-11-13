"""E10: the audit gap and M4 on CIFAR-10, with a real CNN.

Every network result up to E8 uses an MLP, at most [20, 128, 128, 1].  The
standing objection is that the orbit is an artefact of that toy: real vision
models use convolutions, batch normalisation and pooling, and perhaps the
gauge closes up somewhere in there.  It does not, and this measures it on the
benchmark the objection is usually phrased in terms of.

WHY THE ORBIT SURVIVES A CONVNET.  The gauge group of sec:3 needs only
positive homogeneity of the activation and linearity of the next layer in its
input.  Both hold for conv->relu->conv:

  scaling every weight and bias of output CHANNEL j of a conv layer by a_j > 0
  scales that channel's post-relu feature map by a_j everywhere (relu(a z) =
  a relu(z) for a > 0, and the scaling is spatially uniform), so dividing the
  next layer's kernel slice for INPUT channel j by a_j restores the function
  exactly.

Max-pooling commutes with a positive scalar (max(a x) = a max(x) for a > 0),
so pooling does not obstruct it either, and the group has one dimension per
CHANNEL rather than per unit -- smaller than the MLP's per-unit group at equal
parameter count, which makes this the conservative test.

BatchNorm is the interesting case and is included deliberately.  It kills the
conv layer's own scale, exactly as a8 found for LayerNorm -- but the scale
reappears in the affine weight, because BN's output is gamma*xhat + beta and
relu is still positively homogeneous in that.  So the group is intact, and
E10.2 measures it on a BN network to show the defence people expect does not
apply.

WHAT IS MEASURED.  The same claims as E8, so the numbers are comparable
across harnesses rather than freshly framed:

  E10.0  the reimplemented group is verified to BE a symmetry, before it is
         used to measure anything.
  E10.1  the audit gap (prop:auditgap) on a plain CNN.
  E10.2  the same on a BatchNorm CNN -- the case people expect to be safe.
  E10.3  the magnitude law of fig:maglaw, swept on CIFAR.
  E10.4  M4 (canonicalise, then align) as the repair, with its honest cost.

训练先运行 ROUNDS 轮 FedAvg，每轮一个本地 epoch；再执行一次本地更新，
在该批客户端模型的聚合中注入攻击。因此测量的是第 ROUNDS+1 次聚合，
并不由固定轮数推断模型已经收敛。分区固定为 seed=0，训练 seeds=0..5。

错误率的上界为 1；十类均匀随机猜测的期望错误率为 0.9，两者不同。
因此错误率之比至多为 1/clean_error，正文同时报告绝对错误率与损失比。
Test cross-entropy is unbounded and is the quantity actually optimised, so the
damage ratio is reported on the loss and the error is reported beside it in
absolute terms.  The MLP experiments report an MSE ratio, which is unbounded
in the same way, so the loss ratio is the comparable column.

该脚本使用 .venv-torch 中实测的 torch 2.2.2 / NumPy 1.26.4；主环境另行锁定。
这不是所有其他脚本在数学上都要求 NumPy 2 的声明。E10.0 验证卷积、线性层
和 BN 的数值函数保持。M4 的匹配采用 greedy 指派，不是唯一最优指派的证明。
"""

import sys
import json
from pathlib import Path

from tools.provenance import environment, file_hash
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "cpu"                      # this machine has no usable MPS/CUDA
SEEDS = [0, 1, 2, 3, 4, 5]
N_CLIENTS = 10
ROUNDS = 20                      # FedAvg rounds before the attack lands
GAUGE_SD = 1.5                   # the same log-sd as A6/A11/A13
SD_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
DATA = Path(__file__).resolve().parent / "data"
RESULTS = DATA.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ models --
class CNN(nn.Module):
    """两层 conv-relu-pool 与线性头；16/32 通道的小型 CNN，结论限于所测配置。"""

    def __init__(self, n_cls=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, n_cls)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


class CNNBN(nn.Module):
    """The same network with BatchNorm after each conv -- the case people
    expect to be safe.  Kept in eval mode for every measurement so the running
    statistics are fixed and the function is a deterministic map."""

    def __init__(self, n_cls=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.b1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.b2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, n_cls)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        return self.fc(x.flatten(1))


# ------------------------------------------------------------ gauge group ---
def _head_slice(model, ch):
    """Column indices of fc.weight belonging to input channel ch.

    The head sees c2's output flattened as (channel, row, col), so channel ch
    owns a contiguous block of H*W columns.  Getting this wrong is the obvious
    way to break the port, which is what E10.0 exists to catch.
    """
    hw = model.fc.in_features // model.c2.out_channels
    return slice(ch * hw, (ch + 1) * hw)


@torch.no_grad()
def gauge_move(model, rng, sd=GAUGE_SD):
    """A random element of the continuous gauge group, applied in place.

    Layer 1: scale conv1's (or BN1's affine) output channel j by a_j and
    divide conv2's input-channel-j kernel slice by a_j.
    Layer 2: scale conv2's (or BN2's) channel j by b_j and divide the head's
    columns for channel j by b_j.
    """
    bn = hasattr(model, "b1")
    a = np.exp(rng.normal(size=model.c1.out_channels) * sd)
    b = np.exp(rng.normal(size=model.c2.out_channels) * sd)
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)

    if bn:
        # (N2) for BatchNorm: the conv's own scale is normalised away, but the
        # affine gamma/beta carry it and relu is homogeneous in those.
        model.b1.weight.mul_(ta)
        model.b1.bias.mul_(ta)
    else:
        model.c1.weight.mul_(ta[:, None, None, None])
        model.c1.bias.mul_(ta)
    model.c2.weight.div_(ta[None, :, None, None])

    if bn:
        model.b2.weight.mul_(tb)
        model.b2.bias.mul_(tb)
    else:
        model.c2.weight.mul_(tb[:, None, None, None])
        model.c2.bias.mul_(tb)
    for ch in range(model.c2.out_channels):
        model.fc.weight[:, _head_slice(model, ch)].div_(float(b[ch]))
    return model


@torch.no_grad()
def canon(model):
    """M4 step one: the orbit representative, reached BY a gauge move.

    Each channel's scale carrier is normalised to unit norm and the scale is
    pushed forward, exactly as mlp.gauge_canon does per unit.  Sequential by
    layer, because layer 2's carrier changes once layer 1's scale lands.
    """
    bn = hasattr(model, "b1")
    if bn:
        n1 = torch.sqrt(model.b1.weight ** 2 + model.b1.bias ** 2) + 1e-30
        model.b1.weight.div_(n1)
        model.b1.bias.div_(n1)
    else:
        n1 = torch.sqrt(model.c1.weight.flatten(1).pow(2).sum(1)
                        + model.c1.bias ** 2) + 1e-30
        model.c1.weight.div_(n1[:, None, None, None])
        model.c1.bias.div_(n1)
    model.c2.weight.mul_(n1[None, :, None, None])

    if bn:
        n2 = torch.sqrt(model.b2.weight ** 2 + model.b2.bias ** 2) + 1e-30
        model.b2.weight.div_(n2)
        model.b2.bias.div_(n2)
    else:
        n2 = torch.sqrt(model.c2.weight.flatten(1).pow(2).sum(1)
                        + model.c2.bias ** 2) + 1e-30
        model.c2.weight.div_(n2[:, None, None, None])
        model.c2.bias.div_(n2)
    for ch in range(model.c2.out_channels):
        model.fc.weight[:, _head_slice(model, ch)].mul_(float(n2[ch]))
    return model


@torch.no_grad()
def align(model, ref):
    """M4 step two: quotient the channel-permutation subgroup S2.

    Greedy nearest-reference matching on each channel's incoming descriptor.
    Greedy rather than Hungarian only because this file cannot import scipy
    (numpy<2 venv); a permutation is function-preserving whichever way it is
    chosen, and E10.3 checks that in floats regardless.
    """
    def descr(w, b):
        return torch.cat([w.flatten(1), b[:, None]], 1)

    # layer 1
    d_q, d_r = descr(model.c1.weight, model.c1.bias), descr(ref.c1.weight,
                                                            ref.c1.bias)
    cost = ((d_q[:, None, :] - d_r[None, :, :]) ** 2).sum(2)
    p1 = _greedy(cost)
    model.c1.weight.copy_(model.c1.weight[p1])
    model.c1.bias.copy_(model.c1.bias[p1])
    if hasattr(model, "b1"):
        for t in (model.b1.weight, model.b1.bias, model.b1.running_mean,
                  model.b1.running_var):
            t.copy_(t[p1])
    model.c2.weight.copy_(model.c2.weight[:, p1])

    # layer 2
    d_q, d_r = descr(model.c2.weight, model.c2.bias), descr(ref.c2.weight,
                                                            ref.c2.bias)
    cost = ((d_q[:, None, :] - d_r[None, :, :]) ** 2).sum(2)
    p2 = _greedy(cost)
    model.c2.weight.copy_(model.c2.weight[p2])
    model.c2.bias.copy_(model.c2.bias[p2])
    if hasattr(model, "b2"):
        for t in (model.b2.weight, model.b2.bias, model.b2.running_mean,
                  model.b2.running_var):
            t.copy_(t[p2])
    hw = model.fc.in_features // model.c2.out_channels
    W = model.fc.weight.view(-1, model.c2.out_channels, hw)
    model.fc.weight.copy_(W[:, p2, :].reshape(model.fc.weight.shape))
    return model


def _greedy(cost):
    """Assignment by repeated global minimum.  Returns perm with q[perm] ~ ref."""
    c = cost.clone()
    n = c.shape[0]
    perm = torch.empty(n, dtype=torch.long)
    for _ in range(n):
        i = int(torch.argmin(c).item())
        r, s = divmod(i, n)
        perm[s] = r
        c[r, :] = float("inf")
        c[:, s] = float("inf")
    return perm


# -------------------------------------------------------------- federation --
def shards(n_clients, seed, alpha=0.5):
    """按 Dirichlet 标签偏斜分区，保存实际样本索引；alpha 默认 0.5。"""
    MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    SD = np.array([0.247, 0.243, 0.261], dtype=np.float32)

    def load(split):
        z = np.load(f"{DATA}/cifar10_{split}.npz")
        X = (z["X"].astype(np.float32) / 255.0 - MEAN) / SD
        return torch.from_numpy(X.transpose(0, 3, 1, 2).copy()), \
            torch.from_numpy(z["y"])

    X, y = load("train")
    Xte, yte = load("test")

    rng = np.random.default_rng(seed)
    idx = [[] for _ in range(n_clients)]
    for c in range(10):
        pool = np.where(y.numpy() == c)[0]
        rng.shuffle(pool)
        p = rng.dirichlet([alpha] * n_clients)
        for j, part in enumerate(np.split(pool, (np.cumsum(p)[:-1]
                                                 * len(pool)).astype(int))):
            idx[j].extend(part.tolist())

    flat = [i for client in idx for i in client]
    assert sorted(flat) == list(range(len(y))), "分区必须恰好覆盖训练子集"
    (RESULTS / "e10_partition.json").write_text(json.dumps(dict(
        seed=seed, alpha=alpha, clients=n_clients, indices=idx,
        index_basis="zero-based rows of the 20000-row training npz",
        train_npz_sha256=file_hash(DATA / "cifar10_train.npz"),
        numpy=np.__version__), indent=2))
    return [(X[np.array(s)], y[np.array(s)]) for s in idx], Xte, yte


def local_train(g0, Xs, ys, epochs=2, lr=0.02, batch=64, seed=0):
    m = type(g0)()
    m.load_state_dict(g0.state_dict())
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(ys), generator=gen)
        for k in range(0, len(perm) - batch + 1, batch):
            j = perm[k:k + batch]
            opt.zero_grad()
            F.cross_entropy(m(Xs[j]), ys[j]).backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def fedavg(models, replace=None):
    ms = list(models)
    if replace is not None:
        ms[0] = replace
    out = type(ms[0])()
    sd = {k: torch.stack([m.state_dict()[k].float()
                          for m in ms]).mean(0) for k in ms[0].state_dict()}
    out.load_state_dict(sd)
    out.eval()
    return out


@torch.no_grad()
def err(model, X, y):
    """错误率位于 [0,1]；0.9 仅为十类均匀随机猜测的期望错误率。"""
    model.eval()
    p = torch.cat([model(X[i:i + 500]).argmax(1) for i in range(0, len(y), 500)])
    return float((p != y).float().mean())


@torch.no_grad()
def loss(model, X, y):
    """Test cross-entropy: unbounded, and the quantity actually optimised, so
    this is what the damage ratio is computed on.  Guessing scores ln 10 =
    2.303."""
    model.eval()
    return float(F.cross_entropy(
        torch.cat([model(X[i:i + 500]) for i in range(0, len(y), 500)]), y))


@torch.no_grad()
def logits(model, X):
    model.eval()
    return torch.cat([model(X[i:i + 500]) for i in range(0, len(X), 500)])


def clone(m):
    q = type(m)()
    q.load_state_dict(m.state_dict())
    q.eval()
    return q


def federate(Cls, sh, seed, rounds=ROUNDS):
    """运行 rounds 轮 FedAvg，再返回下一次聚合使用的客户端本地模型。"""
    torch.manual_seed(seed)
    g = Cls()
    for r in range(rounds):
        g = fedavg([local_train(g, X, y, epochs=1, seed=seed * 1000 + r * 10 + j)
                    for j, (X, y) in enumerate(sh)])
    clients = [local_train(g, X, y, epochs=1, seed=seed * 1000 + 9000 + j)
               for j, (X, y) in enumerate(sh)]
    return g, clients


# ================================================================ E10.0 =====
def e10_0(Cls, Xte):
    """The port must be verified before anything is measured with it."""
    print("=" * 78)
    print(f"E10.0  is the reimplemented group actually a symmetry?  ({Cls.__name__})")
    print("=" * 78)
    torch.manual_seed(7)
    m = Cls()
    m.eval()
    f0 = logits(m, Xte[:500])
    q = gauge_move(clone(m), np.random.default_rng(0))
    d_move = float((logits(q, Xte[:500]) - f0).abs().max())
    c = canon(clone(m))
    d_canon = float((logits(c, Xte[:500]) - f0).abs().max())
    a = align(clone(m), Cls())
    d_align = float((logits(a, Xte[:500]) - f0).abs().max())
    scale = float(f0.abs().max())
    print(f"  logit scale {scale:.3f};  max |df| under a gauge move "
          f"{d_move:.2e}, under canon {d_canon:.2e}, under align {d_align:.2e}")
    print(f"  parameter L2 moves from {float(torch.cat([p.flatten() for p in m.parameters()]).norm()):.2f} "
          f"to {float(torch.cat([p.flatten() for p in q.parameters()]).norm()):.2f} "
          "-- same function, different point in parameter space\n")
    for nm, v in [("move", d_move), ("canon", d_canon), ("align", d_align)]:
        assert v < 1e-3 * max(scale, 1.0), (nm, v, scale)
    return d_move, d_canon, d_align


# ================================================================ E10.1 =====
def gap(Cls, Xte, yte, sh, seed, sd=GAUGE_SD):
    g0, clients = federate(Cls, sh, seed)
    f0 = logits(clients[0], Xte)
    own0_e, own0_l = err(clients[0], Xte, yte), loss(clients[0], Xte, yte)
    q = gauge_move(clone(clients[0]), np.random.default_rng(900 + seed), sd=sd)
    df = float((logits(q, Xte) - f0).abs().max())
    own1_e, own1_l = err(q, Xte, yte), loss(q, Xte, yte)
    gc, ga = fedavg(clients), fedavg(clients, replace=q)
    return dict(cl=clients, g0=g0, q=q, df=df,
                own_e=(own0_e, own1_e), own_l=(own0_l, own1_l),
                clean_e=err(gc, Xte, yte), after_e=err(ga, Xte, yte),
                clean_l=loss(gc, Xte, yte), after_l=loss(ga, Xte, yte))


def run_gap(Cls, Xte, yte, sh, tag):
    print("=" * 78)
    print(f"{tag}  audit gap on CIFAR-10, {N_CLIENTS} Dirichlet(0.5) clients, "
          f"{Cls.__name__}, {ROUNDS} FedAvg rounds")
    print("=" * 78)
    print(f"{'seed':>5} {'max|dlogit|':>12} {'own err':>9} {'own after':>10} "
          f"{'clean err':>10} {'err after':>10} {'clean L':>9} {'L after':>9} "
          f"{'L ratio':>8}")
    R = []
    for sd in SEEDS:
        r = gap(Cls, Xte, yte, sh, sd)
        R.append(r)
        print(f"{sd:5d} {r['df']:12.2e} {r['own_e'][0]:9.4f} "
              f"{r['own_e'][1]:10.4f} {r['clean_e']:10.4f} "
              f"{r['after_e']:10.4f} {r['clean_l']:9.4f} {r['after_l']:9.4f} "
              f"{r['after_l'] / r['clean_l']:8.2f}")
        sys.stdout.flush()
    x = np.array([r["after_l"] / r["clean_l"] for r in R])
    print(f"\n  clean federation  err {np.mean([r['clean_e'] for r in R]):.4f} "
          f"and loss {np.mean([r['clean_l'] for r in R]):.4f}, against 0.900 "
          f"and 2.303 for guessing\n  -- the baseline the ratio is measured "
          f"against has genuinely learnt CIFAR-10")
    print(f"  damage on the loss  median {np.median(x):.2f}x  "
          f"min {x.min():.2f}x  max {x.max():.2f}x   "
          f"(error rises {np.mean([r['clean_e'] for r in R]):.3f} -> "
          f"{np.mean([r['after_e'] for r in R]):.3f})")
    print(f"  functional audit  max |dlogit| {max(r['df'] for r in R):.2e}, "
          f"own error unchanged to "
          f"{max(abs(r['own_e'][1] - r['own_e'][0]) for r in R):.2e} and own "
          f"loss to {max(abs(r['own_l'][1] - r['own_l'][0]) for r in R):.2e}\n")
    return R, x


# ================================================================ E10.3 =====
def run_maglaw(Cls, Xte, yte, sh, tag):
    """fig:maglaw's sweep, on CIFAR.  The MLP harness gives damage running
    1.004 to 372x over log-sd 0.5 to 2.0; the question here is only whether
    the same monotone law appears on a convnet, not whether the exponent
    matches across two different loss functions."""
    print("=" * 78)
    print(f"{tag}  the magnitude law on CIFAR-10 ({Cls.__name__}, seed 0)")
    print("=" * 78)
    g0, clients = federate(Cls, sh, 0)
    gc = fedavg(clients)
    c_l, c_e = loss(gc, Xte, yte), err(gc, Xte, yte)
    f0 = logits(clients[0], Xte)
    base = float(torch.cat([p.flatten() for p in clients[0].parameters()]).norm())
    print(f"  clean loss {c_l:.4f}, clean err {c_e:.4f}, honest "
          f"|theta| {base:.1f}")
    print(f"\n{'log-sd':>8} {'|theta_q|':>11} {'ratio':>8} {'loss':>9} "
          f"{'damage':>9} {'err':>8} {'max|dlogit|':>12}")
    rows = []
    for s in SD_GRID:
        q = gauge_move(clone(clients[0]), np.random.default_rng(900), sd=s)
        nq = float(torch.cat([p.flatten() for p in q.parameters()]).norm())
        ga = fedavg(clients, replace=q)
        l, e = loss(ga, Xte, yte), err(ga, Xte, yte)
        d = float((logits(q, Xte) - f0).abs().max())
        rows.append((s, nq, l / c_l, e, d))
        print(f"{s:8.1f} {nq:11.1f} {nq / base:8.1f} {l:9.3f} {l / c_l:9.2f} "
              f"{e:8.4f} {d:12.2e}")
        sys.stdout.flush()
    print(f"\n  damage grows monotonically from {rows[0][2]:.2f}x to "
          f"{rows[-1][2]:.2f}x while the functional audit stays at "
          f"{max(r[4] for r in rows):.1e}:\n  the same law as fig:maglaw, on a "
          f"convnet and a different loss.\n")
    return rows


# ================================================================ E10.4 =====
def run_m4(Cls, Xte, yte, R, tag):
    print("=" * 78)
    print(f"{tag}  M4 on CIFAR-10: repair under attack, and cost without one")
    print("=" * 78)
    print(f"{'seed':>5} {'undef x':>9} {'M4 x':>8} {'honest x':>9} "
          f"{'canon |df|':>11}")
    und, rep, cost, dfs = [], [], [], []
    for r, sd in zip(R, SEEDS):
        cl, q = r["cl"], r["q"]
        f_pre = [logits(m, Xte[:500]) for m in cl]
        ref = clone(r["g0"])
        cq = [align(canon(clone(m)), ref) for m in cl]
        qq = align(canon(clone(q)), ref)
        dd = max(float((logits(z, Xte[:500]) - f).abs().max())
                 for z, f in zip(cq, f_pre))
        c_q = loss(fedavg(cq), Xte, yte)
        a_q = loss(fedavg(cq, replace=qq), Xte, yte)
        u = r["after_l"] / r["clean_l"]
        und.append(u)
        rep.append(a_q / c_q)
        cost.append(c_q / r["clean_l"])
        dfs.append(dd)
        print(f"{sd:5d} {u:9.2f} {a_q / c_q:8.4f} {c_q / r['clean_l']:9.4f} "
              f"{dd:11.2e}")
        sys.stdout.flush()
    print(f"\n  undefended median {np.median(und):.2f}x  ->  under M4 "
          f"{np.median(rep):.4f}x")
    print(f"  honest cost (M4 applied with no attacker present) median "
          f"{np.median(cost):.4f}x")
    print(f"  the canon changed no client's function: max |dlogit| "
          f"{max(dfs):.2e}\n")
    return np.array(und), np.array(rep), np.array(cost), max(dfs)


# ==================================================================== main ==
print("loading CIFAR-10 and partitioning by Dirichlet(0.5) label skew...")
sh, Xte, yte = shards(N_CLIENTS, 0)
sizes = [len(s[1]) for s in sh]
lab = [float(np.bincount(s[1].numpy(), minlength=10).max() / len(s[1]))
       for s in sh]
print(f"  {sum(sizes)} training images over {N_CLIENTS} clients, shard sizes "
      f"{min(sizes)}-{max(sizes)} ({max(sizes)/max(min(sizes),1):.1f}x "
      f"imbalance),\n  largest single class per client "
      f"{min(lab):.2f}-{max(lab):.2f} of its data -- Dirichlet's skew, not "
      f"ours\n")

d0 = e10_0(CNN, Xte)
d0bn = e10_0(CNNBN, Xte)
R_plain, x_plain = run_gap(CNN, Xte, yte, sh, "E10.1")
R_bn, x_bn = run_gap(CNNBN, Xte, yte, sh, "E10.2")
mag = run_maglaw(CNN, Xte, yte, sh, "E10.3")
u, r, c, dmax = run_m4(CNN, Xte, yte, R_plain, "E10.4")

print("=" * 78)
print("SUMMARY -- CIFAR-10 replication of prop:auditgap, fig:maglaw and M4")
print("=" * 78)
print(f"  plain CNN      damage median {np.median(x_plain):.2f}x on the loss, "
      f"functional audit max {max(z['df'] for z in R_plain):.1e}")
print(f"  BatchNorm CNN  damage median {np.median(x_bn):.2f}x on the loss, "
      f"functional audit max {max(z['df'] for z in R_bn):.1e}")
print(f"  magnitude law  {mag[0][2]:.2f}x at log-sd {mag[0][0]} rising to "
      f"{mag[-1][2]:.2f}x at {mag[-1][0]}, audit flat throughout")
print(f"  M4             {np.median(u):.2f}x -> {np.median(r):.4f}x, "
      f"honest cost {np.median(c):.4f}x")

# ---------------------------------------------------------------- assertions
# 1. The gap is the claim: the function must be preserved to numerical noise
#    while the aggregate degrades.  Both halves, or it is not the audit gap.
#    The tolerance is relative to the logit scale and to float32 epsilon --
#    torch is single precision, so the 1e-15 of the numpy experiments is not
#    available here and claiming it would be false.
for tag, R in [("plain", R_plain), ("bn", R_bn)]:
    sc = float(logits(R[0]["cl"][0], Xte).abs().max())
    assert max(z["df"] for z in R) < 1e-4 * sc, (tag, max(z["df"] for z in R), sc)
    assert max(abs(z["own_e"][1] - z["own_e"][0]) for z in R) == 0.0, tag
    assert max(abs(z["own_l"][1] - z["own_l"][0]) for z in R) < 1e-4, tag
assert np.median(x_plain) > 1.15, np.median(x_plain)
# 2. BatchNorm must NOT close the group -- the a8 finding, on a convnet.
assert np.median(x_bn) > 1.15, np.median(x_bn)
# 3. The clean federation must have learnt, or every ratio is vacuous.
assert np.mean([z["clean_e"] for z in R_plain]) < 0.60, "clean model too weak"
assert np.mean([z["clean_l"] for z in R_plain]) < 1.80, "clean loss too weak"
# 4. The magnitude law must be monotone and must span an order of magnitude,
#    or it is not the law fig:maglaw reports.
assert [m[2] for m in mag] == sorted(m[2] for m in mag), [m[2] for m in mag]
assert mag[-1][2] / mag[0][2] > 10, mag[-1][2] / mag[0][2]
# 5. M4 must repair, and must be nearly free on an honest population.
assert np.median(r) < 1.05, np.median(r)
assert np.median(c) < 1.15, np.median(c)
assert dmax < 1e-3, dmax
print("\nOK: on a real CNN and on a BatchNorm CNN the functional audit is blind "
      "while\nthe aggregate degrades, the magnitude law holds, and M4 repairs "
      "it at a cost near one.")
metrics = lambda rows: [{"seed": seed, **{k: v for k, v in row.items()
                                        if k not in ("cl", "g0", "q")}}
                        for seed, row in zip(SEEDS, rows)]
(RESULTS / "e10_revision.json").write_text(json.dumps(dict(
    environment=environment(), seeds=SEEDS, partition_seed=0, clients=N_CLIENTS,
    warmup_rounds=ROUNDS, attack_aggregation_round=ROUNDS + 1,
    local_epochs=1, lr=0.02, momentum=0.9, batch=64, drop_incomplete_batch=True,
    gauge_log_sd=GAUGE_SD, gauge_seed_rule="900 + training seed",
    partition_sha256=file_hash(RESULTS / "e10_partition.json"),
    test_npz_sha256=file_hash(DATA / "cifar10_test.npz"),
    plain=metrics(R_plain), batchnorm=metrics(R_bn), magnitude=mag,
    m4=dict(seeds=SEEDS, undefended=u.tolist(), repaired=r.tolist(),
            honest_cost=c.tolist(), max_logit_difference=dmax, matcher="greedy")), indent=2))
print("E10 CLAIMS: plain %.2fx | bn %.2fx | maglaw %.2f->%.2fx | "
      "M4 %.2f->%.4fx | cost %.4fx | audit %.1e"
      % (np.median(x_plain), np.median(x_bn), mag[0][2], mag[-1][2],
         np.median(u), np.median(r), np.median(c),
         max(z["df"] for z in R_plain)))
