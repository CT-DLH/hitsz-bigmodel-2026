# -*- coding: utf-8 -*-
"""
实验作业二：ConvLSTM 的算法应用与改进
（教材第 4 章《循环神经网络》· 时空序列预测）

在脚本自动生成的弹跳小球序列（2000 条训练 + 200 条测试，每条 10 帧 32x32）上：
  * 任务一：实现基线 ConvLSTM，用前 4 帧预测第 5 帧，记录测试 MSE / MAE / 耗时；
  * 任务二：手算 ConvLSTMCell 与输出卷积参数量，并用代码核对；
  * 任务三：完成 6 组控制变量对照实验（每组只改一个变量）；
  * 任务四：把有效改进组合成最优组合，固定随机种子复测 3 次取平均；
  * 任务五：绘制“输入帧 | 真实下一帧 | 预测帧”对比图；
  * 加分项：多步递推预测（recursive rollout）。

用法：
  python convlstm_bounce.py paramcheck   # 参数量手算核对（秒级）
  python convlstm_bounce.py probe        # 单 epoch 计时探针
  python convlstm_bounce.py stage1       # 基线 + 对照实验 1~6
  python convlstm_bounce.py stage2       # 最优组合 3 次复测 + 预测帧对比图
  python convlstm_bounce.py stage3       # 加分项：多步递推预测
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避本机 OpenMP 运行库冲突

import json
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # 无界面环境下直接出图保存
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 图中文字体（Windows）
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(16, os.cpu_count() or 8))

N_SEQ = 2200          # 2000 训练 + 200 测试
T_FRAMES = 10
SIZE = 32
BALL_R = 2

# ==================================================================
# 1. 数据合成：弹跳小球序列
# ==================================================================
def make_sequences(n_seq=N_SEQ, T=T_FRAMES, size=SIZE, r=BALL_R, seed=42):
    """生成 n_seq 条 T 帧的弹跳小球序列，返回 (N, T, size, size, 1)。

    物理规律清晰（匀速直线 + 边界反弹），便于把预测误差归因到模型本身。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    seqs = np.zeros((n_seq, T, size, size), np.float32)
    for s in range(n_seq):
        x, y = rng.uniform(4, size - 5, 2)                       # 随机初始位置
        vx, vy = rng.choice([-1, 1], 2) * rng.uniform(0.8, 1.6, 2)  # 随机初速度
        for t in range(T):
            x, y = x + vx, y + vy                                # 匀速直线运动
            if x < r or x > size - r:                            # 左右边界反弹
                vx = -vx
            if y < r or y > size - r:                            # 上下边界反弹
                vy = -vy
            seqs[s, t] = ((xx - x) ** 2 + (yy - y) ** 2 <= r * r)  # 画半径 r 的实心圆
    return seqs[..., None]                                       # (N, T, size, size, 1)


DATA = make_sequences()


def build_xy(K):
    """取前 K 帧作输入、第 K 帧（第 K+1 帧）作标签，返回 (训x, 训y, 测x, 测y)。

    注意：数据张量必须转成 (B, T, C, H, W)，这是本实验最容易踩的坑。
    """
    x = torch.tensor(DATA[:, :K]).permute(0, 1, 4, 2, 3).contiguous()   # (N,K,1,32,32)
    y = torch.tensor(DATA[:, K]).squeeze(-1).contiguous()               # (N,32,32)
    return x[:2000], y[:2000], x[2000:], y[2000:]


# ==================================================================
# 2. 模型：ConvLSTM（自写细胞）
# ==================================================================
class ConvLSTMCell(nn.Module):
    """把 LSTM 的矩阵乘法换成二维卷积的时空细胞（合并实现，4 个门共用一个卷积）。

    公式（卷积版）：
        i_t = sigmoid(W_xi * X_t + W_hi * H_{t-1} + b_i)
        f_t = sigmoid(W_xf * X_t + W_hf * H_{t-1} + b_f)
        o_t = sigmoid(W_xo * X_t + W_ho * H_{t-1} + b_o)
        g_t = tanh   (W_xg * X_t + W_hg * H_{t-1} + b_g)
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
        H_t = o_t ⊙ tanh(C_t)
    """

    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        # 4 个门共用一次卷积：输入通道 = 本层输入 + 上一时刻隐藏态
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)
        self.hid = hid_ch

    def forward(self, x, state):
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))     # (B, 4*C_h, H, W)
        i, f, g, o = z.chunk(4, dim=1)              # 切分成 4 个门
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c_new = f * c + i * torch.tanh(g)           # 细胞状态更新
        h_new = o * torch.tanh(c_new)               # 隐藏状态输出
        return h_new, c_new


class ConvLSTM(nn.Module):
    """沿时间循环调用细胞，用最后一层最后时刻的隐藏态经 1x1 位置的 3x3 卷积预测下一帧。

    层数 layers、隐藏通道 hid、卷积核 k 均可配置。
    """

    def __init__(self, in_ch=1, hid=32, k=3, layers=1):
        super().__init__()
        chs = [in_ch] + [hid] * layers
        self.cells = nn.ModuleList(
            [ConvLSTMCell(chs[i], chs[i + 1], k) for i in range(layers)])
        self.out = nn.Conv2d(hid, 1, 3, padding=1)   # 输出卷积：32 -> 1
        self.hid = hid
        self.layers = layers
        self.k = k

    def forward(self, x):                       # x: (B, T, C, H, W)
        if x.dim() != 5:                        # 形状断言：把“忘记沿时间维取帧”挡在训练之前
            raise ValueError(
                "ConvLSTM 期望 5D 输入 (B,T,C,H,W)，实际收到 %dD 张量 %s"
                % (x.dim(), tuple(x.shape)))
        b, T, _, hsz, wsz = x.shape
        states = [(torch.zeros(b, c.hid, hsz, wsz, device=x.device),
                   torch.zeros(b, c.hid, hsz, wsz, device=x.device))
                  for c in self.cells]
        for t in range(T):                      # 沿时间逐步喂入每一帧
            xt = x[:, t]
            for j, cell in enumerate(self.cells):
                states[j] = cell(xt, states[j])
                xt = states[j][0]               # 上一层输出作为下一层输入
        return self.out(states[-1][0]).squeeze(1)   # (B,32,32)


class FlattenLSTM(nn.Module):
    """对照组：普通全连接 LSTM。把 4 帧 32x32 展平成 4x1024 = 4096 维向量再送进 LSTM。

    空间结构被完全打散，且参数量远大于 ConvLSTM。
    注：指南原文写作“展平 4 帧（4×1024 维）输入 nn.LSTM(1024, 256)”，
    其中输入维度 4×1024 = 4096 与 nn.LSTM 的 input_size=1024 自相矛盾；
    按“展平 4×1024 输入”的表述，nn.LSTM 的 input_size 应取 4096（否则 shape 报错）。
    """

    def __init__(self, T=4, size=32, hid=256):
        super().__init__()
        self.T, self.size = T, size
        self.lstm = nn.LSTM(T * size * size, hid, batch_first=True)  # (4096 -> 256)
        self.fc = nn.Linear(hid, size * size)                        # (256 -> 1024)

    def forward(self, x):
        b = x.shape[0]
        v = x.reshape(b, -1).unsqueeze(1)        # (B, 1, 4096)：整个时空被展平成一个“词”
        o, _ = self.lstm(v)
        return self.fc(o[:, -1]).reshape(b, self.size, self.size)


# ==================================================================
# 3. 训练与评估
# ==================================================================
def train(model, train_x, train_y, test_x, test_y,
          loss_name="mse", lr=1e-3, epochs=5, bs=64, seed=SEED, verbose=True):
    """MSE / L1 损失 + Adam，逐 epoch 记录训练损失与测试 MSE / MAE。"""
    torch.manual_seed(seed)
    loss_fn = nn.L1Loss() if loss_name == "l1" else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.shape[0]
    g = torch.Generator().manual_seed(seed)
    hist = {"train": [], "mse": [], "mae": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):               # 小批量前向 + 损失 + 反向 + 更新
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(train_x[idx]), train_y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        hist["train"].append(tot / n)
        model.eval()
        with torch.no_grad():
            pred = model(test_x)
            hist["mse"].append((pred - test_y).pow(2).mean().item())
            hist["mae"].append((pred - test_y).abs().mean().item())
        if verbose:
            print("epoch %d train_loss %.5f test_MSE %.5f test_MAE %.5f"
                  % (ep + 1, hist["train"][-1], hist["mse"][-1], hist["mae"][-1]), flush=True)
    dt = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
    return hist, dt, pred


def plot_curve(hist, fname, title):
    """训练损失曲线 + 测试 MSE / MAE 曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = np.arange(1, len(hist["train"]) + 1)
    axes[0].plot(ep, hist["train"], "o-", ms=4, color="#d62728")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("训练损失")
    axes[0].set_title("训练损失曲线"); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, hist["mse"], "o-", ms=4, color="#1f77b4", label="测试 MSE")
    axes[1].plot(ep, hist["mae"], "s--", ms=4, color="#2ca02c", label="测试 MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("误差")
    axes[1].set_title("测试误差曲线"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(BASE, fname), dpi=150)
    plt.close(fig)


# ==================================================================
# 4. 实验调度
# ==================================================================
RESULTS = []


def do_experiment(name, fname, K=4, hid=32, k=3, layers=1, loss_name="mse",
                  lr=1e-3, epochs=5, seed=SEED, arch="convlstm", title=None,
                  verbose=True):
    """统一入口：固定种子 -> 取数据 -> 建模 -> 训练 -> 出图 -> 记录结果。"""
    torch.manual_seed(seed)
    tr_x, tr_y, te_x, te_y = build_xy(K)
    if arch == "flatten":
        model = FlattenLSTM(T=K, size=SIZE)
    else:
        model = ConvLSTM(in_ch=1, hid=hid, k=k, layers=layers)
    n_par = sum(p.numel() for p in model.parameters())
    hist, dt, pred = train(model, tr_x, tr_y, te_x, te_y, loss_name=loss_name,
                           lr=lr, epochs=epochs, seed=seed, verbose=verbose)
    cfg = ("arch=%s layers=%d hid=%d k=%d K_in=%d loss=%s lr=%g"
           % (arch, layers, hid, k, K, loss_name, lr))
    plot_curve(hist, fname,
               title or "%s（%s，参数量 %d）" % (name, cfg, n_par))
    rec = {"name": name, "config": cfg, "params": n_par,
           "mse": hist["mse"][-1], "mae": hist["mae"][-1], "time": dt,
           "seed": seed, "file": fname,
           "hist": {kk: [round(v, 6) for v in vv] for kk, vv in hist.items()}}
    RESULTS.append(rec)
    print("[%-16s] MSE=%.6f MAE=%.6f time=%.1fs params=%d  -> %s"
          % (name, rec["mse"], rec["mae"], dt, n_par, fname), flush=True)
    return rec, pred


def param_check():
    """任务二：手算参数量并用代码核对。"""
    print("=" * 72)
    print("参数量手算与代码核对")
    print("=" * 72)

    def hand(hid, k, in_ch=1, layers=1):
        """逐层手算：第 1 层输入通道为 C_in，其后各层输入通道均为 C_h。"""
        # 每个门：K*K*C_in*C_h + K*K*C_h*C_h + C_h（偏置）
        c0 = 4 * (k * k * in_ch * hid + k * k * hid * hid + hid)
        cj = 4 * (k * k * hid * hid + k * k * hid * hid + hid)
        cells = c0 + (layers - 1) * cj
        out = 3 * 3 * hid * 1 + 1          # 输出卷积固定 3x3（与 K 无关）
        return cells, out, cells + out

    for tag, hid, k, layers in [("基线 1层 C_h=32 K=3", 32, 3, 1),
                                ("实验1 2层 C_h=32 K=3", 32, 3, 2),
                                ("实验2 C_h=64 K=3", 64, 3, 1),
                                ("实验3 C_h=32 K=5", 32, 5, 1)]:
        cells, outc, tot = hand(hid, k, 1, layers)
        m = ConvLSTM(hid=hid, k=k, layers=layers)
        real = sum(p.numel() for p in m.parameters())
        print("%-24s 手算: 细胞=%d 输出卷积=%d 总计=%d | 代码: %d | %s"
              % (tag, cells, outc, tot, real, "一致" if tot == real else "不一致"))
    print("\n【基线代入过程】细胞 = 4×(K×K×C_in×C_h + K×K×C_h×C_h + C_h)")
    print("            = 4×(3×3×1×32 + 3×3×32×32 + 32) = 4×(288 + 9216 + 32) = 4×9536 = %d"
          % (4 * (3 * 3 * 1 * 32 + 3 * 3 * 32 * 32 + 32)))
    print("            输出卷积 = 3×3×32×1 + 1 = 289")
    print("            总参数量 = 38144 + 289 = 38433")
    print("\n【实验2 C_h=64 代入】4×(3×3×1×64 + 3×3×64×64 + 64) = 4×(576 + 36864 + 64) = 4×37504 = 150016")
    print("            输出卷积 = 3×3×64×1 + 1 = 577；总计 = 150593（为基线的 %.2f 倍）"
          % (150593 / 38433))
    print("\n【实验3 K=5 代入】4×(5×5×1×32 + 5×5×32×32 + 32) = 4×(800 + 25600 + 32) = 4×26432 = 105728")
    print("            输出卷积仍为 3×3 = 289；总计 = 106017")
    print("\n【实验1 2层代入】第1层 4×(3×3×1×32 + 3×3×32×32 + 32) = 38144")
    print("            第2层 4×(3×3×32×32 + 3×3×32×32 + 32) = 4×(9216 + 9216 + 32) = 4×18464 = 73856")
    print("            输出卷积 = 289；总计 = 38144 + 73856 + 289 = 112289")
    for tag, hid, k, layers in [("基线", 32, 3, 1), ("2层", 32, 3, 2), ("C_h=32 K=5", 32, 5, 1)]:
        m = ConvLSTM(hid=hid, k=k, layers=layers)
        print("\n  参数张量明细分组（%s，合计 %d）:" % (tag, sum(p.numel() for p in m.parameters())))
        for n_, p in m.named_parameters():
            print("    %-20s %-22s %d" % (n_, str(tuple(p.shape)), p.numel()))
    fm = FlattenLSTM()
    print("\n实验5 全连接 LSTM 代码参数量: %d" % sum(p.numel() for p in fm.parameters()))
    for n_, p in fm.named_parameters():
        print("    %-20s %-22s %d" % (n_, str(tuple(p.shape)), p.numel()))
    print("手算: LSTM 4×256×(4096+256) + 8×256 = %d；Linear 256×1024 + 1024 = %d；合计 %d"
          % (4 * 256 * (4096 + 256) + 8 * 256, 256 * 1024 + 1024,
             4 * 256 * (4096 + 256) + 8 * 256 + 256 * 1024 + 1024))
    print("      与基线 ConvLSTM（38433）相比为 %.1f 倍"
          % ((4 * 256 * (4096 + 256) + 8 * 256 + 256 * 1024 + 1024) / 38433))
    print("=" * 72)


def probe():
    """单 epoch 计时探针，用于估算全量实验耗时。"""
    torch.manual_seed(SEED)
    tr_x, tr_y, te_x, te_y = build_xy(4)
    m = ConvLSTM()
    h, dt, _ = train(m, tr_x, tr_y, te_x, te_y, epochs=1)
    print("baseline 单 epoch 耗时 %.1f s（2000 条训练序列，batch 64）" % dt)


def stage1():
    """基线 + 对照实验 1~6（每组只改一个变量，其余与基线一致）。"""
    print("\n########## 组 0：基线 Baseline ##########")
    do_experiment("Baseline", "result_baseline.png", K=4, hid=32, k=3, layers=1,
                  title="图1 基线 ConvLSTM（1层, C_h=32, K=3, 看4帧, MSE）")
    print("\n########## 组 1：加深层数 1 -> 2 层 ##########")
    do_experiment("Exp1-2层", "result_exp1.png", K=4, hid=32, k=3, layers=2,
                  title="图2 实验1 加深层数（2层, C_h=32, K=3, 看4帧）")
    print("\n########## 组 2：隐藏通道 C_h 32 -> 64 ##########")
    do_experiment("Exp2-Ch64", "result_exp2.png", K=4, hid=64, k=3, layers=1,
                  title="图3 实验2 隐藏通道（1层, C_h=64, K=3, 看4帧）")
    print("\n########## 组 3：卷积核 K=3 -> 5 ##########")
    do_experiment("Exp3-K5", "result_exp3.png", K=4, hid=32, k=5, layers=1,
                  title="图4 实验3 卷积核（1层, C_h=32, K=5, 看4帧）")
    print("\n########## 组 4：输入帧数 4 -> 8 ##########")
    do_experiment("Exp4-Kin8", "result_exp4.png", K=8, hid=32, k=3, layers=1,
                  title="图5 实验4 输入帧数（1层, C_h=32, K=3, 看8帧）")
    print("\n########## 组 5：结构对照 全连接 LSTM ##########")
    do_experiment("Exp5-LSTM", "result_exp5.png", K=4, arch="flatten",
                  title="图6 实验5 结构对照（全连接 LSTM, 展平4帧->1024, 隐藏256）")
    print("\n########## 组 6：损失函数 MSE -> L1 ##########")
    do_experiment("Exp6-L1", "result_exp6.png", K=4, hid=32, k=3, layers=1,
                  loss_name="l1",
                  title="图7 实验6 损失函数（1层, C_h=32, K=3, 看4帧, L1损失）")
    with open(os.path.join(BASE, "results_stage1.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    print("\nstage1 完成，结果写入 results_stage1.json")


def stage2(best_cfg=None, seeds=(42, 43, 44)):
    """最优组合：固定种子复测 3 次取平均，并绘制预测帧对比图（复用第 1 次的模型输出）。"""
    cfg = best_cfg or dict(K=4, hid=64, k=5, layers=2, loss_name="mse")
    print("\n########## 组 7：最优组合 %s ##########" % cfg, flush=True)
    recs, preds = [], []
    for i, sd in enumerate(seeds, start=1):
        r, p = do_experiment("Best-run%d" % i, "result_best_run%d.png" % i, seed=sd,
                             title=("图9 最优组合（种子%d）：看%d帧, C_h=%d, K=%d, %d层, %s损失"
                                    % (sd, cfg["K"], cfg["hid"], cfg["k"],
                                       cfg["layers"], cfg["loss_name"].upper())),
                             **cfg)
        recs.append(r)
        preds.append(p)

    # ---- 平凡参考线：直接预测“全零图”的 MSE / MAE（用于判断模型是否真的学到了东西）----
    tr_x, tr_y, te_x, te_y = build_xy(cfg["K"])
    zero_mse = float((te_y ** 2).mean())
    zero_mae = float(te_y.abs().mean())
    print("平凡参考（全零预测）: MSE=%.6f MAE=%.6f" % (zero_mse, zero_mae), flush=True)

    avg = {"mse": float(np.mean([r["mse"] for r in recs])),
           "mae": float(np.mean([r["mae"] for r in recs])),
           "time": float(np.mean([r["time"] for r in recs]))}
    print("最优组合 3 次平均: MSE=%.6f MAE=%.6f time=%.1fs"
          % (avg["mse"], avg["mae"], avg["time"]), flush=True)

    # ---- 任务五：预测帧对比图（3 个样本 x 输入末帧|真实下一帧|预测帧，复用种子 42 的模型输出）----
    pred = preds[0]
    sel = [0, 7, 23]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 7.6))
    notes = []
    for r, si in enumerate(sel):
        inp = te_x[si, -1, 0].numpy()
        gt = te_y[si].numpy()
        pr = pred[si].numpy()
        pmse = float(((pred[si] - te_y[si]) ** 2).mean())
        pmae = float((pred[si] - te_y[si]).abs().mean())
        # 真实球心与预测球心（灰度质心），用于讨论位移误差
        gy, gx = np.nonzero(gt > 0.5)
        py, px = np.nonzero(pr > 0.5)
        iy, ix = np.nonzero(inp > 0.5)
        cx, cy = ix.mean(), iy.mean()
        gcx, gcy = gx.mean(), gy.mean()
        if len(px):
            pcx, pcy = px.mean(), py.mean()
        else:
            pcx = pcy = float("nan")
        notes.append({"sample": int(si), "mse": pmse, "mae": pmae,
                      "center_in": [round(float(cx), 2), round(float(cy), 2)],
                      "center_gt": [round(float(gcx), 2), round(float(gcy), 2)],
                      "center_pred": [round(float(pcx), 2), round(float(pcy), 2)],
                      "shift": [round(float(vx - v0), 2) for vx, v0 in
                                zip((gcx, gcy), (cx, cy))]})
        for c, (img, ttl) in enumerate([(inp, "输入末帧"), (gt, "真实下一帧"), (pr, "预测帧")]):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(ttl, fontsize=10)
            if c == 0:
                ax.set_ylabel("样本 %d" % (r + 1), fontsize=10)
            if c == 2:
                ax.set_xlabel("MSE=%.4f" % pmse, fontsize=9)
    fig.suptitle("图8 输入末帧 | 真实下一帧 | 预测帧 对比（最优组合，看%d帧）" % cfg["K"], fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(BASE, "result_pred.png"), dpi=150)
    plt.close(fig)
    print("预测对比图已保存:", notes, flush=True)

    # ---- 附加：MSE 训练 vs L1 训练的预测锐度对比（实验 6 的视觉证据，均为基线结构、看 4 帧）----
    x4, y4, tx4, ty4 = build_xy(4)
    _, _, p_mse = do_experiment_direct(ConvLSTM(hid=32, k=3, layers=1), x4, y4, tx4, ty4,
                                       "mse", 42)
    _, _, p_l1 = do_experiment_direct(ConvLSTM(hid=32, k=3, layers=1), x4, y4, tx4, ty4,
                                      "l1", 42)
    si = sel[0]
    m_mse = float(((p_mse[si] - ty4[si]) ** 2).mean())
    m_l1 = float(((p_l1[si] - ty4[si]) ** 2).mean())
    gray_mse = float(p_mse[si].std())
    gray_l1 = float(p_l1[si].std())
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    panels = [(x4[si, -1, 0].numpy(), "输入末帧"),
              (ty4[si].numpy(), "真实下一帧"),
              (p_mse[si].numpy(), "MSE 训练预测"),
              (p_l1[si].numpy(), "L1 训练预测")]
    for ax, (img, ttl) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MSE 与 L1 损失的预测帧锐度对比（基线结构，看 4 帧，同种子）", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(BASE, "result_mse_vs_l1.png"), dpi=150)
    plt.close(fig)
    print("MSE 预测 MSE=%.6f 灰度std=%.4f | L1 预测 MSE=%.6f 灰度std=%.4f"
          % (m_mse, gray_mse, m_l1, gray_l1), flush=True)

    out = {"best_config": cfg, "seeds": list(seeds),
           "runs": [{"seed": r["seed"], "mse": r["mse"], "mae": r["mae"], "time": r["time"],
                     "params": r["params"], "hist": r["hist"]} for r in recs],
           "avg": avg, "trivial_zero": {"mse": zero_mse, "mae": zero_mae},
           "pred_samples": notes,
           "sharpness": {"mse_train_gray_std": gray_mse, "l1_train_gray_std": gray_l1,
                         "mse_train_mse": m_mse, "l1_train_mse": m_l1}}
    with open(os.path.join(BASE, "results_stage2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("stage2 完成，结果写入 results_stage2.json")


def do_experiment_direct(model, tr_x, tr_y, te_x, te_y, loss_name, seed, epochs=5):
    """不落盘图片的轻量训练入口（用于附加对比）。"""
    torch.manual_seed(seed)
    hist, dt, pred = train(model, tr_x, tr_y, te_x, te_y, loss_name=loss_name,
                           epochs=epochs, seed=seed, verbose=False)
    return hist, dt, pred


def stage3(best_cfg=None, steps=4):
    """加分项：多步递推预测（把预测帧回灌作为输入，逐步外推）。"""
    cfg = best_cfg or dict(K=8, hid=64, k=5, layers=1, loss_name="l1")
    K = cfg["K"]
    print("\n########## 加分项：多步递推预测（%d 步）##########" % steps)
    torch.manual_seed(SEED)
    tr_x, tr_y, te_x, te_y = build_xy(K)
    model = ConvLSTM(in_ch=1, hid=cfg["hid"], k=cfg["k"], layers=cfg["layers"])
    h, dt, _ = train(model, tr_x, tr_y, te_x, te_y, loss_name=cfg["loss_name"],
                     epochs=5, seed=SEED, verbose=False)
    # 用完整 10 帧数据做递推：取前 K 帧，逐步外推
    full = torch.tensor(DATA[2000:]).permute(0, 1, 4, 2, 3).contiguous()[:, :, 0]  # (200,10,32,32)
    seq = full[:, :K].clone()
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for s in range(steps):
            p = model(seq.unsqueeze(2))          # (B,32,32) 预测第 K+s 帧
            preds.append(p)
            gts.append(full[:, K + s])
            seq = torch.cat([seq[:, 1:], p.unsqueeze(1)], dim=1)   # 回灌
    per_step = []
    for s in range(steps):
        mse = float((preds[s] - gts[s]).pow(2).mean())
        mae = float((preds[s] - gts[s]).abs().mean())
        per_step.append({"step": s + 1, "mse": mse, "mae": mae})
        print("  第 %d 步预测: MSE=%.6f MAE=%.6f" % (s + 1, mse, mae), flush=True)
    fig, axes = plt.subplots(2, steps, figsize=(2.3 * steps, 4.8))
    si = 0
    for s in range(steps):
        axes[0, s].imshow(gts[s][si].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, s].set_title("真实 t+%d" % (s + 1), fontsize=9)
        axes[1, s].imshow(preds[s][si].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, s].set_title("预测 t+%d (MSE %.4f)" % (s + 1, per_step[s]["mse"]), fontsize=9)
        for r in range(2):
            axes[r, s].set_xticks([]); axes[r, s].set_yticks([])
    fig.suptitle("加分项 多步递推预测（最优组合，样本 1，看%d帧起步）" % K, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(BASE, "result_multistep.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(BASE, "results_stage3.json"), "w", encoding="utf-8") as f:
        json.dump({"steps": per_step, "config": cfg}, f, ensure_ascii=False, indent=2)
    print("stage3 完成，结果写入 results_stage3.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    t0 = time.time()
    if cmd == "paramcheck":
        param_check()
    elif cmd == "probe":
        probe()
    elif cmd == "stage1":
        stage1()
    elif cmd == "stage2":
        best = json.load(open(os.path.join(BASE, "best_config.json"), encoding="utf-8")) \
            if os.path.exists(os.path.join(BASE, "best_config.json")) else None
        stage2(best_cfg=best)
    elif cmd == "stage3":
        best = json.load(open(os.path.join(BASE, "best_config.json"), encoding="utf-8")) \
            if os.path.exists(os.path.join(BASE, "best_config.json")) else None
        stage3(best_cfg=best)
    else:
        print("未知命令:", cmd)
    print("总耗时 %.1f s" % (time.time() - t0))
