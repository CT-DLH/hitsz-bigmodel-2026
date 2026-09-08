# -*- coding: utf-8 -*-
"""
实验作业一：设计更好的神经网络（load_digits 手写数字分类）
在 sklearn load_digits 数据集上从零构建 PyTorch MLP 分类器：
  - 先运行朴素基线（Baseline）
  - 再围绕激活函数 / 网络深度 / 优化器 / 学习率 / L2 / Dropout / BatchNorm
    开展控制变量对照实验（每组只改一个变量）
  - 附加：学习率过大的失败实验 + 学习率扫描（寻找发散临界点）
  - 最优组合固定 3 个不同种子各复测一次，避免偶然性
所有实验共用同一份数据划分（8:2 分层抽样）与随机种子，保证对比公平。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 本机 Windows 下 OpenMP 运行库冲突的规避措施
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境下直接出图保存
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 图中文字体（Windows）
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
torch.manual_seed(SEED)          # 全局随机种子，保证可复现
np.random.seed(SEED)

# ---------- 1. 数据：所有实验共用同一划分 ----------
X, y = load_digits(return_X_y=True)
X = X.astype(np.float32) / 16.0                     # 像素归一化到 [0, 1]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)  # 8:2 分层抽样
# Windows 上 numpy 标签是 int32，CrossEntropyLoss 要求 Long 类型
X_tr = torch.tensor(X_tr)
y_tr = torch.tensor(y_tr, dtype=torch.long)
X_te = torch.tensor(X_te)
y_te = torch.tensor(y_te, dtype=torch.long)

# ---------- 2. 模型：结构各维度均可配置 ----------
class MLP(nn.Module):
    """可配置 MLP：隐藏层数/宽度、激活函数、Dropout、BatchNorm 均为参数。

    改进点插入位置：
      改进点① BatchNorm —— 每个隐藏层 Linear 之后插入 nn.BatchNorm1d
      改进点② Dropout  —— 激活函数之后按 p 随机失活
      改进点③ 优化器   —— 在 run() 中按 opt_name 切换 SGD / Adam
    """
    def __init__(self, hidden=(32,), act="sigmoid", dropout=0.0, use_bn=False):
        super().__init__()
        act_layer = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
                     "relu": nn.ReLU, "gelu": nn.GELU}[act]
        layers, prev = [], 64                      # 输入为 8x8=64 维
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:                             # 改进点①：BatchNorm
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())             # 激活函数引入非线性
            if dropout > 0:                        # 改进点②：Dropout
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 10))         # 输出层：10 类 logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- 3. 训练与评估：全批量，完整三步循环 ----------
def run(model, opt_name="sgd", lr=0.1, epochs=30, wd=0.0):
    """训练并逐 epoch 记录训练损失与测试准确率，返回历史与耗时。"""
    if opt_name == "sgd":                          # 改进点③：优化器可切换
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()                # 多分类用交叉熵损失
    hist = {"loss": [], "acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()                            # 梯度清零
        loss = loss_fn(model(X_tr), y_tr)          # 前向传播 + 损失计算
        loss.backward()                            # 反向传播求梯度
        opt.step()                                 # 优化器沿负梯度更新参数
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(1) == y_te).float().mean().item()
        hist["loss"].append(loss.item())
        hist["acc"].append(acc)
    return hist, time.time() - t0

def plot_curve(hist, fname, title):
    """绘制损失曲线与测试准确率曲线，保存为图片。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = np.arange(1, len(hist["loss"]) + 1)
    axes[0].plot(ep, hist["loss"], "o-", ms=3, color="#d62728")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("训练损失（交叉熵）")
    axes[0].set_title("训练损失曲线"); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, np.array(hist["acc"]) * 100, "o-", ms=3, color="#1f77b4")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("测试准确率（%）")
    axes[1].set_title("测试准确率曲线"); axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fname, dpi=150)
    plt.close(fig)

def describe(hidden, act, opt_name, lr, wd=0.0, dropout=0.0, use_bn=False):
    return (f"hidden={'x'.join(map(str, hidden))}, act={act}, opt={opt_name}, "
            f"lr={lr}, wd={wd}, dropout={dropout}, bn={use_bn}")

RESULTS = []

def do_experiment(name, fname, hidden=(32,), act="sigmoid", opt_name="sgd",
                  lr=0.1, wd=0.0, dropout=0.0, use_bn=False, seed=SEED,
                  title_prefix=None):
    """统一入口：固定种子 -> 建模 -> 训练 -> 出图 -> 记录结果。"""
    torch.manual_seed(seed)                        # 每组实验均固定种子
    model = MLP(hidden=hidden, act=act, dropout=dropout, use_bn=use_bn)
    hist, dt = run(model, opt_name=opt_name, lr=lr, wd=wd)
    cfg = describe(hidden, act, opt_name, lr, wd, dropout, use_bn)
    title = title_prefix or f"{name}（{cfg}）"
    plot_curve(hist, fname, title)
    acc = hist["acc"][-1] * 100
    RESULTS.append({"name": name, "config": cfg, "acc": acc, "time": dt,
                    "seed": seed, "file": fname})
    print(f"[{name:<14s}] acc={acc:6.2f}%  time={dt:5.2f}s  {cfg}  -> {fname}")
    return hist, dt

if __name__ == "__main__":
    t_start = time.time()

    # ---- 组 0：朴素基线：1x32 Sigmoid + SGD(0.1)，无任何正则化 ----
    do_experiment("Baseline", "result_baseline.png",
                  title_prefix="图1 基线模型 Baseline（1×32 Sigmoid + SGD, lr=0.1）")

    # ---- 组 1：换激活函数 Sigmoid -> ReLU（其余与基线一致）----
    do_experiment("Exp1-ReLU", "result_exp1.png", act="relu",
                  title_prefix="图2 实验1 换激活函数（1×32 ReLU + SGD, lr=0.1）")

    # ---- 组 2：加深网络 1层 -> 2层×128（仍为 Sigmoid）----
    do_experiment("Exp2-Deep", "result_exp2.png", hidden=(128, 128),
                  title_prefix="图3 实验2 加深网络（2×128 Sigmoid + SGD, lr=0.1）")

    # ---- 组 3：换优化器 SGD -> Adam(0.01)（结构与基线一致）----
    do_experiment("Exp3-Adam", "result_exp3.png", opt_name="adam", lr=0.01,
                  title_prefix="图4 实验3 换优化器（1×32 Sigmoid + Adam, lr=0.01）")

    # ---- 组 4：调学习率 0.1 -> 0.01（SGD，其余与基线一致）----
    do_experiment("Exp4-LR0.01", "result_exp4.png", lr=0.01,
                  title_prefix="图5 实验4 调学习率（1×32 Sigmoid + SGD, lr=0.01）")

    # ---- 组 5：加 L2 正则化 weight_decay=1e-4 ----
    do_experiment("Exp5-L2", "result_exp5.png", wd=1e-4,
                  title_prefix="图6 实验5 L2正则化（1×32 Sigmoid + SGD, lr=0.1, wd=1e-4）")

    # ---- 组 6：加 Dropout p=0.2 ----
    do_experiment("Exp6-Dropout", "result_exp6.png", dropout=0.2,
                  title_prefix="图7 实验6 Dropout（1×32 Sigmoid + SGD, lr=0.1, p=0.2）")

    # ---- 组 7：加 BatchNorm（每个隐藏层后插入 BatchNorm1d）----
    do_experiment("Exp7-BN", "result_exp7.png", use_bn=True,
                  title_prefix="图8 实验7 BatchNorm（1×32 Sigmoid+BN + SGD, lr=0.1）")

    # ---- 组 8：最优组合：ReLU + 2×128 + BatchNorm + Adam(0.001) + Dropout(0.2) ----
    # 3 个不同种子各复测一次（表 5），主图取第 1 次
    best_hists = []
    for i, sd in enumerate([42, 43, 44], start=1):
        h, d = do_experiment(
            f"Exp8-run{i}", f"result_exp8_run{i}.png",
            hidden=(128, 128), act="relu", opt_name="adam", lr=0.001,
            wd=1e-4, dropout=0.2, use_bn=True, seed=sd,
            title_prefix=(f"图9 最优组合 ReLU+2×128+BN+Dropout+Adam(0.001)（种子{sd}）"
                          if i == 1 else f"最优组合复测（种子{sd}）"))
        best_hists.append((h, d))

    # ---- 失败实验：学习率过大 lr=10.0（不计入改进组，必须写入报告）----
    do_experiment("LR=10.0", "result_lr_too_big.png", lr=10.0,
                  title_prefix="图10 失败实验：学习率过大（1×32 Sigmoid + SGD, lr=10.0）")

    # ---- 附加：学习率扫描，寻找发散临界点（加分项）----
    print("\n学习率扫描（基线其余配置不变）:")
    scan = []
    for lr_s in [0.1, 1.0, 5.0, 10.0, 20.0]:
        h, d = do_experiment(f"Scan-lr={lr_s}", f"result_scan_lr{lr_s}.png", lr=lr_s,
                             title_prefix=f"学习率扫描 lr={lr_s}（1×32 Sigmoid + SGD）")
        scan.append({"lr": lr_s, "acc": h["acc"][-1] * 100,
                     "final_loss": h["loss"][-1],
                     "loss_std": float(np.std(h["loss"][-5:]))})

    # ---- 汇总保存 ----
    summary = {"results": RESULTS, "lr_scan": scan}
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n全部实验完成，总耗时 {time.time() - t_start:.1f}s，结果已写入 results.json")
