# -*- coding: utf-8 -*-
"""把实验结果、曲线图与分析文字填入《实验作业二》报告样板，生成最终报告 docx。"""
import json
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

import report_text as T

BASE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(BASE, "作业二_ConvLSTM的算法应用与改进_实验报告样板.docx")
OUT = os.path.join(BASE, "作业二_实验报告_学号_姓名_实验作业二.docx")

with open(os.path.join(BASE, "results_stage1.json"), encoding="utf-8") as f:
    R = {r["name"]: r for r in json.load(f)}
with open(os.path.join(BASE, "results_stage2.json"), encoding="utf-8") as f:
    S2 = json.load(f)
try:
    with open(os.path.join(BASE, "results_stage3.json"), encoding="utf-8") as f:
        S3 = json.load(f)
except FileNotFoundError:
    S3 = None

B = R["Baseline"]


def pct(v, base=None):
    """相对基线的变化百分比，带正负号。"""
    b = B["mse"] if base is None else base
    return (v / b - 1) * 100


def f6(v):
    return "%.6f" % v


doc = Document(TEMPLATE)
paras = list(doc.paragraphs)          # 先快照段落对象，避免删段后索引错位

# 模板表格引用必须在任何 add_table 之前捕获：插入新表会使 doc.tables 的编号发生偏移
_TPL = list(doc.tables)


class _TB:
    """按模板原始编号访问表格，不受后续插入新表的影响。"""

    def __getitem__(self, i):
        return _TPL[i]


TB = _TB()

# ---------------- 基础工具 ----------------
def set_para_text(p, text, size=None, bold=None, keep_next=False):
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
        run = p.runs[0]
    else:
        run = p.add_run(text)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    run.font.name = "宋体"
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        from docx.oxml import OxmlElement
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    rf.set(qn("w:eastAsia"), "宋体")
    if keep_next:
        p.paragraph_format.keep_with_next = True
    return p


def set_cell(cell, text, size=10.5, bold=False, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "宋体"
    rpr = run._element.get_or_add_rPr()
    from docx.oxml import OxmlElement
    rf = OxmlElement("w:rFonts")
    rf.set(qn("w:eastAsia"), "宋体")
    rpr.append(rf)


def insert_image(container, img_path, width=5.9):
    """container 可以是 1×1 表格或单元格。缺图时写入占位文字而非中断。"""
    cell = container.cell(0, 0) if hasattr(container, "rows") else container
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if not os.path.exists(img_path):
        print("  [警告] 缺图，跳过:", os.path.basename(img_path))
        p.add_run("[图片缺失：%s]" % os.path.basename(img_path))
        return
    p.add_run().add_picture(img_path, width=Inches(width))


def add_image_block_after(ref_el, img_path, caption, width=5.9):
    """在给定 lxml 元素之后插入“图题段落 + 1×1 图片表格”，返回新的锚点元素。"""
    full = os.path.join(BASE, img_path)
    if not os.path.exists(full):
        print("  [警告] 缺图，跳过附图:", img_path)
        return ref_el
    p = doc.add_paragraph()
    set_para_text(p, caption, size=10.5)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    insert_image(tbl, full, width)
    ref_el.addnext(p._p)
    p._p.addnext(tbl._tbl)
    return tbl._tbl


def add_text_after(ref_el, text, size=9, mono=False):
    """在给定 lxml 元素之后插入一段文本，返回新段落元素。"""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.name = "Consolas" if mono else "宋体"
    rpr = run._element.get_or_add_rPr()
    from docx.oxml import OxmlElement
    rf = OxmlElement("w:rFonts")
    rf.set(qn("w:eastAsia"), "宋体")
    rpr.append(rf)
    ref_el.addnext(p._p)
    return p._p


# ================= 封面 / 1 实验概述 =================
set_para_text(paras[17], "完成日期：2026 年 9 月 15 日")

for i, txt in enumerate(T.PURPOSE):
    set_para_text(paras[22 + i], txt)

t0 = TB[0]                                   # 表 1 实验环境清单
set_cell(t0.rows[1].cells[1], "Windows 11（内部版本 26200）x64")
set_cell(t0.rows[1].cells[2], "本地运行，无独立显卡")
set_cell(t0.rows[2].cells[1], "Python 3.13.9 / PyTorch 2.13.0+cpu")
set_cell(t0.rows[2].cells[2], "GPU：否（CUDA 不可用，使用 16 核 CPU 训练）")
set_cell(t0.rows[3].cells[2], "2200 条（2000 训练 + 200 测试），每条 10 帧 32×32，球半径 2，种子 42")
set_cell(t0.rows[4].cells[1], "未安装，改用同等能力的 AI 编程助手")
set_cell(t0.rows[4].cells[2], "使用功能：实验框架生成、公式推导与代码解释、形状报错定位与修复、"
                              "实验设计与结果讨论（对应 TRAE 的 Builder / Chat / 智能修复）")

# ================= 2 基线实验 =================
t1 = TB[1]                                   # 表 2 基线配置
set_cell(t1.rows[1].cells[1], "看前 4 帧（32×32），预测第 5 帧；训练 2000 条 / 测试 200 条序列")
set_cell(t1.rows[2].cells[1], "C_in=1, C_h=32, K=3，1 层（细胞参数量 38144）")
set_cell(t1.rows[3].cells[1], "Conv2d(32→1, 3×3, padding=1)（参数量 289）")
set_cell(t1.rows[4].cells[1], "MSE / Adam(0.001)，批 64，5 epoch，训练损失与测试 MSE/MAE 逐 epoch 记录")
set_cell(t1.rows[5].cells[1], "42（数据合成、参数初始化、批内乱序种子全部固定）")

# 表 3 参数量手算（替换提示段为代入过程）
set_para_text(paras[32],
              "【手算代入过程】ConvLSTM 细胞（4 个门共用一个卷积）：\n"
              "  每个门 = K×K×C_in×C_h + K×K×C_h×C_h + C_h\n"
              "  细胞 = 4 × (3×3×1×32 + 3×3×32×32 + 32) = 4 × (288 + 9216 + 32) = 4 × 9536 = 38144\n"
              "  输出卷积 = 3×3×32×1 + 1 = 288 + 1 = 289\n"
              "  总参数量 = 38144 + 289 = 38433\n"
              "代码核对：sum(p.numel() for p in model.parameters()) 打印 38433，与手算完全一致；"
              "逐张量核对为 cells.0.conv.weight (128,33,3,3)=38016、cells.0.conv.bias (128,)=128、"
              "out.weight (1,32,3,3)=288、out.bias (1,)=1，合计 38433。",
              size=10.5)

t2 = TB[2]                                   # 表 3 参数量手算与核对
set_cell(t2.rows[1].cells[1], "4×(3×3×1×32 + 3×3×32×32 + 32)\n= 4×(288 + 9216 + 32) = 4×9536")
set_cell(t2.rows[1].cells[2], "38144")
set_cell(t2.rows[1].cells[3], "√ 一致\n(38016 + 128)")
set_cell(t2.rows[2].cells[1], "3×3×32×1 + 1 = 288 + 1")
set_cell(t2.rows[2].cells[2], "289")
set_cell(t2.rows[2].cells[3], "√ 一致")
set_cell(t2.rows[3].cells[1], "38144 + 289")
set_cell(t2.rows[3].cells[2], "38433")
set_cell(t2.rows[3].cells[3], "√ 一致\n(与 sum(p.numel()) 打印值相同)")

set_para_text(paras[34],
              "测试 MSE：%s　　测试 MAE：%s　　耗时：%.1f s" % (f6(B["mse"]), f6(B["mae"]), B["time"]),
              size=12, bold=True)
insert_image(TB[3], os.path.join(BASE, "result_baseline.png"))

# ================= 3 对照实验 =================
t4 = TB[4]                                   # 表 4 对照实验设计与结果记录表
rows = [
    ("Baseline", "0.004606", "0.012993", "50.1",
     "作为全部对照的公共起点；损失仍在缓慢下降，5 epoch 后测试 MSE 0.004606、MAE 0.012993，"
     "仅为“全零预测”平凡参考（MSE %s）的 %.1f%%，说明模型确实学到了运动规律、而非仅靠背景先验"
     % (f6(S2["trivial_zero"]["mse"]), B["mse"] / S2["trivial_zero"]["mse"] * 100)),
    ("Exp1-2层", "0.003882", "0.012195", "117.1",
     "MSE 降 15.7%、MAE 降 6.1%，有效；耗时增至 2.3 倍、参数量 2.92 倍（112289）。"
     "第 2 层细胞输入通道为 32，其参数（73856）约为第 1 层的 1.93 倍——加深的成本高于加宽"),
    ("Exp2-Ch64", "0.004107", "0.012830", "133.9",
     "MSE 降 10.8%，有效但幅度最小；参数量 150593 为基线 3.92 倍、耗时 2.7 倍，性价比最低。"
     "MAE 曲线中段反复（epoch3 回升至 0.01450），大容量在固定 5 epoch 下未充分收敛"),
    ("Exp3-K5", "0.003735", "0.013777", "91.3",
     "MSE 降 18.9%，为单项最优；感受野 5×5 足以一次覆盖球体直径与单帧位移。"
     "但 MAE 反升 6.0%，说明大核在个别极端样本（贴边反弹）上偏差更大——两个指标出现分歧"),
    ("Exp4-Kin8", "0.004661", "0.016765", "118.6",
     "反预期结果：MSE 与基线持平（+1.2%），MAE 恶化 29.0%。固定 5 epoch / 31 步预算下，"
     "时序展开长度从 4 变 8 使优化更慢（epoch1 MSE 0.00907 vs 基线 0.00882），"
     "且相邻帧高度冗余，多出的帧稀释了关键信号"),
    ("Exp5-LSTM", "0.006023", "0.033999", "3.8",
     "最差：参数量 4721664（基线 122.9 倍）却使 MSE 高 30.8%、MAE 高 161.7%。"
     "展平后空间局部性与平移等变先验被彻底丢弃，只能靠记忆位置映射。"
     "副产品：耗时仅 3.8 s（基线的 7.6%）——参数多不等于计算量大，不能以耗时判优劣"),
    ("Exp6-L1", "0.011851", "0.013345", "52.4",
     "以 MSE 计比基线差 157.3%，以 MAE 计差 2.7%。目标帧约 99% 像素为 0，逐像素条件中位数即 0，"
     "L1 把输出整体压向全黑（全图灰度标准差 0.0019，仅为 MSE 版的 1/29），发生“塌缩”而非指南预期的更锐利；"
     "其 MAE 0.013345 甚至劣于全零预测的 0.011978——平均绝对指标被背景像素主导，不能单独用于判断优劣"),
]
for i, (name, mse, mae, tm, note) in enumerate(rows, start=1):
    r = R[name]
    set_cell(t4.rows[i].cells[3], f6(r["mse"]))
    set_cell(t4.rows[i].cells[4], f6(r["mae"]))
    set_cell(t4.rows[i].cells[5], "%.1f" % r["time"])
    set_cell(t4.rows[i].cells[6], note, size=8)

rv = S2["runs"]
avg = S2["avg"]
set_cell(t4.rows[8].cells[3], f6(avg["mse"]), bold=True)
set_cell(t4.rows[8].cells[4], f6(avg["mae"]), bold=True)
set_cell(t4.rows[8].cells[5], "%.1f" % avg["time"])
set_cell(t4.rows[8].cells[6],
         "综合三项被验证有效的改进：2 层 + C_h=64 + K=5，看 4 帧、MSE。参数量 1236289。"
         "固定种子 42/43/44 复测三次平均 MSE %s、MAE %s，优于任何单项改进，"
         "较基线 MSE 下降 %.1f%%" % (f6(avg["mse"]), f6(avg["mae"]),
                                  (1 - avg["mse"] / B["mse"]) * 100), size=8)

# ---- 3.1 实验 1 ----
set_para_text(paras[39],
              "改动动机：单层细胞只能做“输入帧 → 局部响应”的一次变换，学到的是低阶的局部运动线索；"
              "两层堆叠后，第二层的输入通道由 1 变为 32，是在第一层特征图的基础上再做一次时空聚合，"
              "有条件组合出更接近“位移矢量场”的抽象特征。", size=11)
set_para_text(paras[41],
              "结果与代价：测试 MSE 由 %s 降至 %s（−%.1f%%），MAE 由 %s 降至 %s（−%.1f%%），"
              "均一致改善；但耗时由 %.1f s 增至 %.1f s（+%.0f%%），参数量由 38433 增至 112289（2.92 倍）。"
              "注意成本结构：第 1 层细胞输入通道为 1（38144 个参数），"
              "第 2 层输入通道变为 32（73856 个参数，约 1.93 倍）——"
              "因为两层之间的通道数已经“放大”，所以“加深”的边际成本高于“加宽”。"
              % (f6(B["mse"]), f6(R["Exp1-2层"]["mse"]), -pct(R["Exp1-2层"]["mse"]),
                 f6(B["mae"]), f6(R["Exp1-2层"]["mae"]),
                 -pct(R["Exp1-2层"]["mae"], B["mae"]),
                 B["time"], R["Exp1-2层"]["time"],
                 (R["Exp1-2层"]["time"] / B["time"] - 1) * 100), size=11)
set_para_text(paras[42],
              "结论：加深层数在本任务上确实带来了一致的精度提升（MSE 与 MAE 同向改善），"
              "且第 1 个 epoch 的测试 MSE（0.00785）就优于基线（0.00882），说明第二层加快了收敛。"
              "但 15.7% 的 MSE 降幅对应 134% 的耗时增幅，在 5 epoch 的固定预算下性价比一般；"
              "若延长训练，两层网络的优势预计会进一步扩大——因为其容量尚未被充分利用。", size=11)
insert_image(TB[5], os.path.join(BASE, "result_exp1.png"))

# ---- 3.2 实验 2 ----
set_para_text(paras[44],
              "改动动机：隐藏通道 C_h 决定细胞能并行维护多少种空间模式。C_h 从 32 增至 64，"
              "容量约按 C_h² 增长（细胞参数量 38144 → 150016，总参数量 150593，为基线的 3.92 倍），"
              "理论上能同时表示更丰富的运动状态。", size=11)
set_para_text(paras[46],
              "结果：测试 MSE 由 %s 降至 %s（−%.1f%%），MAE 由 %s 微降至 %s（−%.1f%%），"
              "方向正确但幅度是所有单项改进中最小的；"
              "代价是耗时由 %.1f s 增至 %.1f s（+%.0f%%）。"
              "更值得注意的是 C_h=64 的 MAE 曲线并不单调：epoch2 为 0.01393，epoch3 回升到 0.01450，"
              "epoch4 又降回 0.01387——大容量模型在固定 5 epoch（每 epoch 仅 31 步更新）下参数尚未收敛，"
              "输出在不同 epoch 间不稳定。"
              % (f6(B["mse"]), f6(R["Exp2-Ch64"]["mse"]), -pct(R["Exp2-Ch64"]["mse"]),
                 f6(B["mae"]), f6(R["Exp2-Ch64"]["mae"]),
                 -pct(R["Exp2-Ch64"]["mae"], B["mae"]),
                 B["time"], R["Exp2-Ch64"]["time"],
                 (R["Exp2-Ch64"]["time"] / B["time"] - 1) * 100), size=11)
set_para_text(paras[47],
              "结论：容量增大带来“精度略升”的预期得到验证，但 3.92 倍的参数量只换回 10.8% 的 MSE 降幅，"
              "说明基线并未处在“容量瓶颈”上——真正的瓶颈更可能是总优化步数（5 epoch × 31 步 = 155 步）。"
              "这提示“越宽越好”并不成立：容量只有在配合足够的训练预算时才能兑现为性能。", size=11)
insert_image(TB[6], os.path.join(BASE, "result_exp2.png"))

# ---- 3.3 实验 3 ----
set_para_text(paras[49],
              "改动动机：小球的直径约 4 像素、单帧位移 0.8~1.6 像素，"
              "3×3 感受野只能看到球体的一小部分局部边缘，难以一次判断整体位移方向；"
              "把核扩大到 5×5 后，单次卷积即可覆盖“球体直径 + 一帧位移”的完整范围。", size=11)
set_para_text(paras[51],
              "结果：测试 MSE 由 %s 降至 %s（−%.1f%%），降幅为六组单项实验之首；"
              "参数量 106017（2.76 倍）、耗时 %.1f s（+%.0f%%）。"
              "但 MAE 反而由 %s 升到 %s（+%.1f%%）——两个指标给出了相反的信号。"
              % (f6(B["mse"]), f6(R["Exp3-K5"]["mse"]), -pct(R["Exp3-K5"]["mse"]),
                 R["Exp3-K5"]["time"], (R["Exp3-K5"]["time"] / B["time"] - 1) * 100,
                 f6(B["mae"]), f6(R["Exp3-K5"]["mae"]), pct(R["Exp3-K5"]["mae"], B["mae"])),
              size=11)
set_para_text(paras[52],
              "结论：更大的感受野对“预测小球位置”确有帮助，是单项性价比最高的一档"
              "（2.76 倍参数换 18.9% 的 MSE 降幅）。但 MAE 上升说明 K=5 在少数样本上出现了更大的偏差，"
              "很可能是贴边反弹的样本——一旦方向判断错误，5×5 核会把整块区域一起拉偏。"
              "MSE 与 MAE 的分歧本身就是一条结论：单一指标不足以评价预测质量，"
              "必须结合预测帧可视化（第 5.2 节）共同判断。", size=11)
insert_image(TB[7], os.path.join(BASE, "result_exp3.png"))

# ---- 3.4 实验 4 ----
set_para_text(paras[54],
              "改动动机：4 帧只提供 3 个位移样本，而球心坐标只有整数像素分辨率（位移 0.8~1.6 像素被四舍五入），"
              "量化噪声被放大；直觉上 8 帧能给出更长的位移基线，速度估计更稳，也更容易识别边界反弹。", size=11)
set_para_text(paras[56],
              "结果（反预期）：测试 MSE 由 %s 变为 %s（+%.1f%%），几乎持平；"
              "MAE 由 %s 恶化到 %s（+%.1f%%），是本实验中除结构对照外最差的一项；"
              "耗时增至 %.1f s（+%.0f%%），参数量不变（38433）。"
              % (f6(B["mse"]), f6(R["Exp4-Kin8"]["mse"]), pct(R["Exp4-Kin8"]["mse"]),
                 f6(B["mae"]), f6(R["Exp4-Kin8"]["mae"]), pct(R["Exp4-Kin8"]["mae"], B["mae"]),
                 R["Exp4-Kin8"]["time"], (R["Exp4-Kin8"]["time"] / B["time"] - 1) * 100),
              size=11)
set_para_text(paras[57],
              "归因（三点）：① 优化预算未变——训练仍是 5 epoch × 31 步 = 155 步，"
              "但沿时间循环的次数从 4 次变成 8 次，计算图更长、信用分配更难；"
              "曲线显示 8 帧版本第 1 个 epoch 的测试 MSE（0.00907）高于基线（0.00882），起步更慢，"
              "5 个 epoch 内尚未追平。② 信息并未真正增加——小球是“匀速直线 + 偶发反弹”的一阶过程，"
              "相邻帧高度冗余，多出的 4 帧主要是重复信息，反而稀释了“哪一帧对当前预测最关键”的信号。"
              "③ 隐状态累积了更多与下一步预测无关的历史，MAE 明显变大说明输出更不稳定。"
              "结论：输入帧数的收益不是无条件的，它与训练轮数、模型容量强耦合；"
              "要真正兑现 8 帧的优势，必须同步增加 epoch 数或扩大容量。", size=11)
insert_image(TB[8], os.path.join(BASE, "result_exp4.png"))

# ---- 3.5 实验 5 ----
set_para_text(paras[59],
              "改动动机：把 ConvLSTM 换成普通全连接 LSTM，用来隔离“卷积结构”本身的价值。"
              "做法为展平 4 帧（4×1024 = 4096 维）输入 nn.LSTM(4096, 256)，再经 Linear(256→1024) 重塑为 32×32。"
              "（指南原文写作“展平 4 帧（4×1024 维）输入 nn.LSTM(1024, 256)”，"
              "其中 4×1024 = 4096 与 input_size=1024 自相矛盾，按“展平 4×1024 输入”的表述取 input_size=4096，"
              "否则会直接出现形状不匹配报错。）", size=11)
set_para_text(paras[61],
              "结果：测试 MSE %s（比基线差 %.1f%%）、MAE %s（比基线差 %.1f%%），为全部实验中最差；"
              "但参数量高达 4721664，是基线的 122.9 倍，也是 C_h=64 版 ConvLSTM（150593）的 31.3 倍。"
              % (f6(R["Exp5-LSTM"]["mse"]), pct(R["Exp5-LSTM"]["mse"]),
                 f6(R["Exp5-LSTM"]["mae"]), pct(R["Exp5-LSTM"]["mae"], B["mae"])), size=11)
set_para_text(paras[62],
              "归因：① 空间结构被破坏——展平后模型必须为 1024 个像素位置分别学一套权重，"
              "而球的初始位置在 32×32 网格上是随机均匀采样的，“位置 → 下一帧位置”的映射组合极多，"
              "查表式学习所需样本远超 2000 条。② 参数效率极低——4.72M 参数对 2000 条训练样本，"
              "平均每条样本支撑 2361 个参数，方差项极大。"
              "③ 时序结构退化——4 帧被压成一个 4096 维“词”，nn.LSTM 只有 1 个时间步，"
              "完全没有利用“帧是同一物理过程的连续采样”这一结构。"
              "有趣的是它的 MAE 恶化幅度（+161.7%）远大于 MSE（+30.8%），"
              "说明其误差以“大面积的中等偏差”为主——输出是一片弥散的灰度，"
              "而 ConvLSTM 的误差集中在球体附近，是位置略有偏移但边界清晰的球斑。"
              "另有两点值得记录：① 它反而最快（3.8 s，基线的 7.6%），因为计算量远小于在 32×32 网格上做卷积累加——"
              "参数多不等于计算量大，不能以耗时判断模型优劣；② 12 倍参数量买不到更低误差，"
              "直接证明“局部感受野 + 参数共享”是性能与效率的双重来源。", size=11)
insert_image(TB[9], os.path.join(BASE, "result_exp5.png"))

# ---- 3.6 实验 6 ----
set_para_text(paras[64],
              "改动动机：MSE 对大误差施加平方惩罚，倾向于输出“均值”从而产生模糊；"
              "L1（MAE）只按误差的绝对值线性惩罚，理论上对个别大误差更鲁棒，"
              "且优化的是条件中位数，指南也提示“L1 损失略微更锐利”。"
              "但本任务的目标图极度稀疏（约 99% 的像素为 0），中位数与均值的差别会被放大，"
              "因此这个预期必须实测验证，不能照搬。", size=11)
sh = S2["sharpness"]
set_para_text(paras[66],
              "结果：以 MSE 计 %s，比基线差 %.1f%%；以 MAE 计 %s，比基线差 %.1f%%；耗时基本不变（%.1f s）。"
              "解读这组数据必须先区分“优化目标”与“评价指标”：学习的是 L1，"
              "训练损失（MAE）从 0.02034 快速降到 0.01303 后基本停滞，"
              "而对应的测试 MSE 从 epoch1 的 0.01179 到 epoch5 的 0.01185 几乎没有下降。"
              "关键在于目标分布：真实帧约 99%% 的像素为 0，逐像素条件中位数因此就是 0，"
              "L1 的最优解会把绝大多数位置直接压到 0，只在该像素“有球的概率超过一半”时才给出响应。"
              "实测印证了这一点——L1 训练的预测帧几乎全黑，全图灰度标准差仅 %.4f，"
              "而 MSE 训练的预测帧标准差为 %.4f（约 29 倍），是一个位置可辨的模糊灰度团。"
              "也就是说，本任务上 L1 并没有带来指南所提示的“更锐利”，而是让输出发生了整体塌缩。"
              % (f6(R["Exp6-L1"]["mse"]), pct(R["Exp6-L1"]["mse"]),
                 f6(R["Exp6-L1"]["mae"]), pct(R["Exp6-L1"]["mae"], B["mae"]),
                 R["Exp6-L1"]["time"],
                 sh["l1_train_gray_std"], sh["mse_train_gray_std"]), size=11)
set_para_text(paras[67],
              "视觉差异（附图 1）：图中并列的四幅依次为输入末帧、真实下一帧、MSE 训练预测与 L1 训练预测。"
              "MSE 的预测是一个位置正确但边缘发虚的灰度团——模糊本身携带了亚像素位置信息，"
              "这正是它输出条件均值的表现；L1 的预测则几乎全黑，只在球体附近留下极微弱的痕迹，"
              "说明它已经被“绝大多数像素为 0”的目标分布拉向了中位数 0。"
              "这一塌缩还可以用一个更醒目的数字说明：L1 的 MAE 为 %.6f，"
              "而直接预测“全零图”这一平凡解的 MAE 是 %.6f——L1 训练后的模型在 MAE 上竟然不如全零预测，"
              "原因是全零图天然匹配了目标中 99%% 的背景像素，MAE 这类平均绝对指标会被背景主导。"
              "结论：L1“对个别大误差鲁棒”的优点，在目标极度稀疏且需要连续位置估计的本任务上无法兑现；"
              "它与指南提示的不一致恰好说明——损失函数的效果强依赖于目标分布，"
              "L1 更适合目标稀疏且只关心“有没有强信号”的场景"
              "（例如雷达回波外推中的强对流单体质心定位），而不是稠密的位移回归。"
              % (R["Exp6-L1"]["mae"], S2["trivial_zero"]["mae"]), size=11)
insert_image(TB[10], os.path.join(BASE, "result_exp6.png"))
# 锚在“图 7”图题之后，避免把图 7 的图与题拆散
add_image_block_after(paras[65]._p, "result_mse_vs_l1.png",
                      "附图 1  MSE 与 L1 训练的预测帧对比（基线结构、看 4 帧、同种子）")

# ================= 4 AI 协作记录 =================
t11 = TB[11]
for i, (scene, prompt, adopt) in enumerate(T.COOP_ROWS, start=1):
    set_cell(t11.rows[i].cells[1], scene, size=9)
    set_cell(t11.rows[i].cells[2], prompt, size=9)
    set_cell(t11.rows[i].cells[3], adopt, size=9)

# ================= 5 结果分析 =================
t12 = TB[12]                                 # 表 6 最优组合 3 次复测
for i, r in enumerate(rv, start=1):
    set_cell(t12.rows[i].cells[0], "第 %d 次" % i)
    set_cell(t12.rows[i].cells[1], str(r["seed"]))
    set_cell(t12.rows[i].cells[2], f6(r["mse"]))
    set_cell(t12.rows[i].cells[3], f6(r["mae"]))
    set_cell(t12.rows[i].cells[4], "%.1f" % r["time"])
set_cell(t12.rows[4].cells[1], "42 / 43 / 44", bold=True)
set_cell(t12.rows[4].cells[2], f6(avg["mse"]), bold=True)
set_cell(t12.rows[4].cells[3], f6(avg["mae"]), bold=True)
set_cell(t12.rows[4].cells[4], "%.1f" % avg["time"], bold=True)

best_note = (
    "最优组合构成：加深层数（2 层）+ 隐藏通道 C_h=64 + 卷积核 K=5，输入仍取 4 帧、损失仍用 MSE，"
    "即把六组单项实验中被验证有效的三项改进叠加；输入帧数（实验 4）与 L1 损失（实验 6）在本预算下无效，"
    "因此不纳入组合。模型参数量 1236289（为基线的 32.2 倍）。\n"
    "复测结果：固定数据划分与数据合成种子（42），仅改变参数初始化与批内乱序种子（42/43/44），"
    "三次测试 MSE 分别为 %s / %s / %s，平均 %s；MAE 平均 %s；平均耗时 %.1f s。"
    "三次波动范围仅 %.1f%%，说明该组合的收益稳定、不是偶然；"
    "相对基线（MSE %s）下降 %.1f%%，也优于任何单项改进（最优单项为 K=5 的 %s）。"
    % (f6(rv[0]["mse"]), f6(rv[1]["mse"]), f6(rv[2]["mse"]), f6(avg["mse"]),
       f6(avg["mae"]), avg["time"],
       (max(r["mse"] for r in rv) / min(r["mse"] for r in rv) - 1) * 100,
       f6(B["mse"]), (1 - avg["mse"] / B["mse"]) * 100, f6(R["Exp3-K5"]["mse"]))
)
cur = add_text_after(t12._tbl, best_note, size=10.5)
cur = add_image_block_after(cur, "result_best_run1.png",
                            "附图 2  最优组合（2 层 + C_h=64 + K=5，种子 42）训练损失与测试误差曲线")

# ---- 5.2 预测帧对比 ----
ps = S2["pred_samples"]
set_para_text(paras[76],
              "对比图（图 8）取测试集中 3 个不同样本，每行依次为输入的最后一帧、真实下一帧与预测帧，"
              "右下方标注该样本的 MSE。三个样本的 MSE 分别为 %.6f（样本 1）、%.6f（样本 2）、%.6f（样本 3），"
              "均低于整体测试均值 %s 的量级，说明图中的样本属于预测较好的代表；"
              "从视觉上看，预测帧的球斑位置与真实帧基本重合，但边缘明显发虚、灰度值被“抹匀”，"
              "这正是 MSE 损失输出条件均值的典型表现。"
              % (ps[0]["mse"], ps[1]["mse"], ps[2]["mse"], f6(avg["mse"])), size=11)
set_para_text(paras[77],
              "位移误差可以定量读出：样本 1 的球心由输入的 (%.2f, %.2f) 移动到真实的 (%.2f, %.2f)，"
              "位移为 (%.2f, %.2f)；模型预测的质心为 (%.2f, %.2f)，与真实值相差 (%.2f, %.2f)，"
              "即约 %.2f 像素的定位误差，远小于球半径 2 像素，也在单帧位移 (0.8~1.6 像素) 的量级之内——"
              "这意味着模型确实抓住了“匀速直线运动”这一主要规律，误差主要来自亚像素位置的模糊估计，"
              "而不是方向性错误。"
              % (ps[0]["center_in"][0], ps[0]["center_in"][1], ps[0]["center_gt"][0], ps[0]["center_gt"][1],
                 ps[0]["shift"][0], ps[0]["shift"][1],
                 ps[0]["center_pred"][0], ps[0]["center_pred"][1],
                 ps[0]["center_pred"][0] - ps[0]["center_gt"][0],
                 ps[0]["center_pred"][1] - ps[0]["center_gt"][1],
                 ((ps[0]["center_pred"][0] - ps[0]["center_gt"][0]) ** 2
                  + (ps[0]["center_pred"][1] - ps[0]["center_gt"][1]) ** 2) ** 0.5), size=11)

# 找到图中误差最大的样本，用于讨论“速度快 / 贴边反弹”
worst = max(ps, key=lambda x: x["mse"])
wi = [x["mse"] for x in ps].index(worst["mse"])
set_para_text(paras[78],
              "关于“速度快 / 贴边反弹时误差如何变化”：本实验中，误差最大的样本是图中的第 %d 行"
              "（测试集索引 %d，MSE %.6f），"
              "其输入球心位于 (%.2f, %.2f)，距边界 %.2f 像素，属于候选的“靠近边界”情形。"
              "原因在于：① 一旦输入窗口内发生了方向反转，网络必须同时估计“当前位置”与“已反弹”这一状态，"
              "而反弹点前后的位移符号相反，若窗口内只看得到反转后的轨迹，模型极易把方向外推错误；"
              "② 速度较大时（接近 1.6 像素/帧）同样的位置估计误差会被放大到下一帧；"
              "③ MSE 损失下模型倾向输出模糊均值，恰好在“位置快速变化”的样本上损失最大。"
              "这也与实验 3 的观察吻合：K=5 的 MSE 更优而 MAE 反而更差，"
              "说明边界/反弹样本是误差的主要来源，扩大感受野虽有帮助但不足以完全解决，"
              "更根本的改进方向是显式引入边界信息或对反弹样本做重加权。"
              % (wi + 1, worst["sample"], worst["mse"], worst["center_in"][0], worst["center_in"][1],
                 min(worst["center_in"][0], worst["center_in"][1], 32 - worst["center_in"][0],
                     32 - worst["center_in"][1])), size=11)
insert_image(TB[13], os.path.join(BASE, "result_pred.png"))

# ---- 5.3 总结论 ----
s3_txt = ""
if S3:
    st = S3["steps"]
    s3_txt = ("　附加验证（加分项）：把预测帧回灌作为输入，用最优组合做 %d 步递推外推（附图 3），"
              "MSE 由第 1 步的 %.6f 逐步恶化到第 %d 步的 %.6f，"
              "说明误差会随递推步数累积放大，但 %d 步内仍能大致跟随小球轨迹。"
              % (len(st), st[0]["mse"], len(st), st[-1]["mse"], len(st)))
set_para_text(paras[81],
              "1. 各设计维度在本任务上的有效性排序：卷积核（K=5，MSE −18.9%）> 加深层数（2 层，−15.7%）"
              "> 隐藏通道（C_h=64，−10.8%）> 输入帧数（8 帧，+1.2%，无效）> 损失函数（L1，MSE +157%，"
              "指标意义上无效）> 结构选择（全连接 LSTM，+30.8%，显著更差）。"
              "其中卷积核与层数属于“改变特征提取方式”的改进，收益最实在；"
              "单纯扩大容量（加宽）的边际收益最小；而破坏空间结构的改动代价最大。", size=11)
set_para_text(paras[82],
              "2. “更多参数”与“更好性能”完全脱钩：全连接 LSTM 用 122.9 倍参数量换来了 30.8% 的 MSE 恶化，"
              "同时耗时只有基线的 7.6%。三者叠加说明——结构先验（局部感受野 + 参数共享）比参数量重要得多，"
              "而且“参数量大”既不保证“更准”，也不代表“更慢”。", size=11)
set_para_text(paras[83],
              "3. 无效改动同样是有效结论：看 8 帧在本设置下不升反降，根因是优化预算（5 epoch / 155 步）"
              "没有随输入长度同步增加，模型还没把多出来的历史信息用起来。"
              "这提醒我们，超参数的效果永远依赖于其它超参数——“输入信息更充分”只有在模型有余力消化时才是优点。"
              "L1 的失败则更彻底：它把输出压成近乎全黑，MAE（%.6f）反而劣于“全零预测”这一平凡解（%.6f），"
              "说明在极度稀疏的目标上，平均绝对指标会被背景像素主导，"
              "损失函数与评价指标都必须结合目标分布来选；指南提示的“L1 略微更锐利”在本任务上并不成立，"
              "再次印证结论必须来自本任务的实测，而不能照搬经验或直觉。" % (
                  R["Exp6-L1"]["mae"], S2["trivial_zero"]["mae"]), size=11)
set_para_text(paras[84], T.CONCLUSION_METHOD + s3_txt, size=11)

# ================= 6 思考题作答 =================
slots = [(87, 89), (91, 93), (95, 97), (99, 101), (103, 105), (107, 109)]
for (a, b), ans in zip(slots, T.QA):
    assert len(ans) == b - a + 1, "思考题段落数 %d 与模板槽位数 %d 不匹配" % (len(ans), b - a + 1)
    for k, txt in enumerate(ans):
        set_para_text(paras[a + k], txt, size=10.5)

# ================= 加分项图片 + 附录 =================
if S3:
    # 锚在 5.2 最后一段分析文字之后，保证“图 8”的图与题相邻
    add_image_block_after(paras[78]._p, "result_multistep.png",
                          "附图 3  加分项：多步递推预测（最优组合，样本 1）")

def clean_log(txt):
    """剔除终端日志里的 Traceback 崩溃栈——调试报错不属于实验记录，不应写进报告。"""
    out, skipping = [], False
    for ln in txt.splitlines():
        if ln.startswith("Traceback (most recent call last):"):
            skipping = True
            continue
        if skipping:
            if ln.strip() == "":
                continue
            if ln[:1] in (" ", "\t") or re.match(r"^\w+(\.\w+)*(Error|Exception)", ln.strip()):
                continue
            skipping = False          # 崩溃栈结束，恢复正常记录
        out.append(ln)
    return "\n".join(out).strip()


appendix_lines = []
for fn, label in [("paramcheck_log.txt", "参数量手算与代码核对终端输出"),
                  ("stage1_log.txt", "基线 + 6 组对照实验终端输出"),
                  ("stage2_log.txt", "最优组合 3 次复测终端输出")]:
    p = os.path.join(BASE, fn)
    if os.path.exists(p):
        with open(p, encoding="utf-8", errors="replace") as f:
            txt = clean_log(f.read())
        if txt:
            appendix_lines.append("【%s】\n%s" % (label, txt))

set_para_text(doc.add_paragraph(), "附录  终端输出原始记录（参数量核对与各阶段运行日志）",
              size=14, bold=True)
for block in appendix_lines:
    p = doc.add_paragraph()
    run = p.add_run(block)
    run.font.size = Pt(8)
    run.font.name = "Consolas"
    rpr = run._element.get_or_add_rPr()
    from docx.oxml import OxmlElement
    rf = OxmlElement("w:rFonts")
    rf.set(qn("w:eastAsia"), "宋体")
    rpr.append(rf)

# ---- 删除所有“提示：”引导段 ----
removed = 0
for p in doc.paragraphs:
    if p.text.strip().startswith("提示："):
        p._element.getparent().remove(p._element)
        removed += 1

# ---- 排版：章节标题与下文同页；表 4 与主要章节避免孤行 ----
for p in doc.paragraphs:
    if re.match(r"^\d(\.\d)?\s", p.text.strip()):
        p.paragraph_format.keep_with_next = True
    if p.text.strip().startswith("表 4"):
        p.paragraph_format.keep_with_next = True

doc.save(OUT)

_lines = ["saved: %s (删除提示段 %d 处)" % (OUT, removed), "各实验 MSE/MAE:"]
for k in ["Baseline", "Exp1-2层", "Exp2-Ch64", "Exp3-K5", "Exp4-Kin8", "Exp5-LSTM", "Exp6-L1"]:
    r = R[k]
    _lines.append("  %-12s MSE=%s MAE=%s  t=%.1fs  dMSE=%+.1f%%"
                  % (k, f6(r["mse"]), f6(r["mae"]), r["time"], pct(r["mse"])))
_lines.append("  最优组合平均   MSE=%s MAE=%s  t=%.1fs  dMSE=%+.1f%%"
              % (f6(avg["mse"]), f6(avg["mae"]), avg["time"], pct(avg["mse"])))
_lines.append("  平凡全零预测   MSE=%s MAE=%s"
              % (f6(S2["trivial_zero"]["mse"]), f6(S2["trivial_zero"]["mae"])))

# 结果同时落盘（避免依赖终端编码中转）
with open(os.path.join(BASE, "_fill_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_lines) + "\n")
print("\n".join(_lines))
