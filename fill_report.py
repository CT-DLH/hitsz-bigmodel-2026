# -*- coding: utf-8 -*-
"""把实验结果、曲线图与分析文字填入实验报告样板，生成最终报告 docx。"""
import copy
import json
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
TEMPLATE = os.path.join(ROOT, "作业一_实验报告样板.docx")
OUT = os.path.join(ROOT, "作业一_实验报告_学号_姓名_实验作业一.docx")

with open(os.path.join(BASE, "results.json"), encoding="utf-8") as f:
    S = json.load(f)
R = {r["name"]: r for r in S["results"]}
SCAN = {s["lr"]: s for s in S["lr_scan"]}

doc = Document(TEMPLATE)
paras = list(doc.paragraphs)  # 先快照段落对象，避免删除提示段后索引错位


def set_para_text(p, text):
    """替换段落文本，保留原首个 run 的格式。"""
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)


def set_cell(cell, text, size=10.5, bold=False):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def insert_image(container, img_path, width=5.9):
    """container 可以是 1×1 表格或单元格。"""
    cell = container.cell(0, 0) if hasattr(container, "rows") else container
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(img_path, width=Inches(width))


def insert_para_after_table(table, text):
    """在表格后插入一个新段落（用于表 5 下的复测分析）。"""
    new_p = OxmlElement("w:p")
    table._tbl.addnext(new_p)
    from docx.text.paragraph import Paragraph
    para = Paragraph(new_p, table._parent)
    run = para.add_run(text)
    run.font.size = Pt(12)
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    return para


fmt = "6.2f"
# ---------- 封面副标题：如实反映使用 ZCode 完成 ----------
set_para_text(paras[7], "使用 AI 编程助手（ZCode）设计更好的神经网络")

# ---------- 1.1 实验目的 ----------
set_para_text(paras[21], "1. 掌握人工神经元与多层感知器（MLP）的基本结构，理解加权和、偏置与激活函数在模型中的作用，"
                         "明确只有引入非线性激活，网络才能表达超越线性模型的函数；")
set_para_text(paras[22], "2. 用 PyTorch 从零实现神经网络的完整训练流程——前向传播、交叉熵损失计算、反向传播与参数更新，"
                         "理解“训练 = 反复执行三步循环”的本质；")
set_para_text(paras[23], "3. 通过控制变量对照实验，定量理解激活函数、优化器、学习率、L2 正则化、Dropout 与 BatchNorm "
                         "对模型性能的影响；培养固定随机种子、逐项归因的严谨实验记录习惯与基于数据的模型改进思维。")

# ---------- 1.2 表1 实验环境 ----------
t0 = doc.tables[0]
set_cell(t0.rows[1].cells[1], "Windows 11（内部版本 26200）x64")
set_cell(t0.rows[1].cells[2], "本地运行，无独立显卡")
set_cell(t0.rows[2].cells[1], "Python 3.13.9")
set_cell(t0.rows[3].cells[1], "PyTorch 2.13.0+cpu")
set_cell(t0.rows[3].cells[2], "CUDA 可用：否（CPU 训练即可完成本实验）")
set_cell(t0.rows[4].cells[1], "scikit-learn 1.7.2 / NumPy 2.3.5 / Matplotlib 3.10.6")
set_cell(t0.rows[5].cells[1], "未安装，改用 ZCode AI 编程助手（等价能力）")
set_cell(t0.rows[5].cells[2], "使用的核心功能：AI 对话生成与解释代码、按提示词生成实验框架、辅助分析实验结果（对应 TRAE 的 Chat / Builder / 智能调试）")

# ---------- 2.1 表2 基线配置 ----------
t1 = doc.tables[1]
set_cell(t1.rows[2].cells[1], "64 → 隐藏层 1×32（Sigmoid 激活） → 输出 10")
set_cell(t1.rows[3].cells[1], "SGD / 学习率 0.1")

# ---------- 2.2 基线结果 ----------
b = R["Baseline"]
set_para_text(paras[32], f"最终测试准确率：{b['acc']:.2f} %　　　训练耗时：{b['time']:.2f} s"
                         "（脚本终端实际打印；本机首次运行含环境预热开销，各组耗时均为真实单次记录）")
insert_image(doc.tables[2], os.path.join(BASE, "result_baseline.png"))

# ---------- 2.3 基线现象观察 ----------
set_para_text(paras[36], "损失下降极慢：30 个 epoch 内训练损失仅从 2.36 缓慢降到约 2.29；测试准确率前 20 个 epoch 一直停在 "
                         "10%（相当于 10 类随机猜测水平），直到第 21 个 epoch 之后才开始缓慢爬升，最终只有 22.22%，"
                         "远低于 50% 的及格预期。")
set_para_text(paras[37], "原因一（激活函数）：Sigmoid 的导数最大值仅 0.25，反向传播时梯度逐层连乘迅速衰减，网络几乎得不到有效的学习信号；"
                         "原因二（优化器与学习率）：全批量 SGD 每个 epoch 只更新一次参数，学习率 0.1 相对本任务的梯度尺度偏小，"
                         "30 步更新不足以让参数走出初始平坦区。")
set_para_text(paras[38], "结论：基线“能学但学不好”，损失确实在下降（说明梯度通路畅通），但收敛速度与最终精度都远不理想，"
                         "这为后续各单项改进提供了明确的提升空间。")

# ---------- 表3 对照实验汇总 ----------
t3 = doc.tables[3]
rows = [
    ("22.22", "0.80", "收敛极慢，准确率约为随机水平（10%）的 2 倍，作为全部对照的公共起点"),
    ("47.50", "0.27", "ReLU 正区间导数恒为 1，缓解梯度消失，收敛明显加快，准确率翻倍"),
    ("7.22", "0.34", "深层 Sigmoid 梯度消失更严重，容量增大反而难以训练，准确率跌回随机水平"),
    ("82.22", "0.41", "Adam 自适应学习率显著加快收敛，是单项收益最大的改进"),
    ("10.00", "0.30", "lr=0.01 时 SGD 在 30 epoch 内几乎未学习，准确率仍停留在随机水平"),
    ("22.22", "0.14", "与基线几乎完全相同：基线本身欠拟合，正则化收益约等于零"),
    ("21.39", "0.28", "训练信号被随机失活削弱，欠拟合的基线上 Dropout 略有负面作用"),
    ("79.17", "0.18", "BatchNorm 稳定各层输入分布，使 Sigmoid 处于梯度健康的区间，单项提升 57 个百分点"),
    ("96.67", "3.60", "ReLU+2×128+BN+Dropout(0.2)+Adam(0.001)+wd=1e-4；3 种子复测平均 94.82%（见表 5）"),
    ("36.94", "0.44", "lr=10.0 时损失剧烈震荡、无法稳定下降，准确率大幅波动（详见 5.2 节）"),
]
for i, (acc, tm, note) in enumerate(rows, start=1):
    set_cell(t3.rows[i].cells[3], acc)
    set_cell(t3.rows[i].cells[4], tm)
    set_cell(t3.rows[i].cells[5], note, size=9)
set_cell(t3.rows[10].cells[2], "学习率过大（lr = 10.0）")

# ---------- 3.1–3.8 各组图片与分析 ----------
img_map = [
    (4, "result_exp1.png"), (5, "result_exp2.png"), (6, "result_exp3.png"),
    (7, "result_exp4.png"), (8, "result_exp5.png"), (9, "result_exp6.png"),
    (10, "result_exp7.png"), (11, "result_exp8_run1.png"),
]
for tbl_idx, fname in img_map:
    insert_image(doc.tables[tbl_idx], os.path.join(BASE, fname))

set_para_text(paras[45], "改动动机：Sigmoid 的导数 σ(1−σ) 最大仅 0.25，且输入绝对值较大时进入饱和区导数趋近 0，"
                         "反向传播梯度逐层衰减；ReLU 在正区间导数恒为 1，梯度几乎无衰减地回传。")
set_para_text(paras[46], "结果：测试准确率从 22.22% 提升到 47.50%（+25.3 个百分点），且准确率曲线从第 1 个 epoch 起"
                         "就持续爬升，不再有基线前 20 个 epoch 的“停滞期”，验证了缓解梯度消失对收敛速度的直接帮助。")

set_para_text(paras[50], "结果：准确率不升反降，从基线的 22.22% 跌到 7.22%（约等于随机猜测）。这是本实验中最重要的"
                         "“负结果”之一。")
set_para_text(paras[51], "原因：① 梯度消失被放大——每多过一层 Sigmoid，梯度至少再乘 0.25，两层隐藏层使到达第一层的"
                         "梯度不足原来的 1/16；② 容量过剩——参数量约为基线的 30 倍，而训练样本只有 1437 个，在梯度信号本就微弱的"
                         "SGD(0.1) 下，大网络的参数几乎留在初始化附近无法被有效训练。容量增大只有在梯度通路健康的前提下才有意义。")

set_para_text(paras[55], "结果：同样的 1×32 Sigmoid 结构，Adam(lr=0.01) 在 30 个 epoch 内达到 82.22%，比基线高出 60 个百分点，"
                         "是所有单项改进中收益最大的一项。")
set_para_text(paras[56], "原因：Adam 为每个参数维护梯度的一阶矩（动量）与二阶矩（梯度幅值）估计，"
                         "用一阶矩与二阶矩平方根之比把更新步长归一化到与 lr 同量级，相当于自动补偿了本任务中 Sigmoid 导致的微小梯度，"
                         "粗调不敏感、收敛快；而 SGD 的有效步长完全等于 lr×梯度，梯度小时几乎不更新。")

set_para_text(paras[60], "结果：SGD 学习率从 0.1 降到 0.01 后，30 个 epoch 内准确率始终停留在 10%（随机水平），"
                         "训练损失从 2.36 只降到约 2.35，网络基本没有学习。")
set_para_text(paras[61], "对比实验 3 可以看出学习率的影响与优化器强相关：Adam 在 lr=0.01 下仍能达到 82.22%，"
                         "而 SGD 已完全失效——SGD 依赖“足够大的步长 × 梯度”才能更新，学习率缩小 10 倍即被冻结；"
                         "Adam 的步长经过自适应归一化，对小学习率的容忍度明显更高。")

set_para_text(paras[65], "结果：测试准确率 22.22%，与基线几乎完全相同，损失曲线也基本重合。")
set_para_text(paras[66], "原因：L2 正则化通过惩罚大权重来抑制过拟合、缩小训练-测试差距，但本实验的基线是“欠拟合”"
                         "而非“过拟合”——测试准确率低是因为网络根本没学进去，此时进一步约束权重只会限制拟合能力，"
                         "收益约等于零。正则化只有在模型容量足够、开始记忆训练集时才能发挥作用（见实验 8）。")

set_para_text(paras[70], "结果：21.39%，比基线的 22.22% 还略低一点，同样属于“无收益甚至略有害”。")
set_para_text(paras[71], "原因：Dropout 训练时随机丢弃 20% 的隐藏神经元，等价于在每次更新中使用一个更小的子网络，"
                         "它是一种牺牲部分拟合能力换取泛化的正则化手段；对本来就没拟合上的基线来说，"
                         "有效神经元进一步减少，训练信号更弱，因此准确率不升反降。Dropout 应与大容量网络配合使用。")

set_para_text(paras[75], "结果：79.17%，比基线提升 57 个百分点，是仅次于换优化器的单项改进，而且训练损失下降速度"
                         "明显快于基线。")
set_para_text(paras[76], "原因：BatchNorm 把每个隐藏层的输入标准化为均值 0、方差 1 的分布，缓解了内部协变量偏移"
                         "（前面层参数一更新，后面层的输入分布就漂移的问题）；标准化后的取值落在 Sigmoid 的"
                         "非饱和区域，梯度不再因饱和而消失，等效地“放大”了学习信号，使同样的 SGD(0.1) 也能稳定快速收敛。")

set_para_text(paras[80], "最优组合配置：ReLU + 2 层×128 神经元 + BatchNorm + Dropout(0.2) + Adam(lr=0.001) + L2(wd=1e-4)。"
                         "单项实验中被验证有效的改进全部组合：ReLU 解决梯度消失，BN 稳定训练，Adam 提供快速稳定的优化，"
                         "Dropout 与 L2 则在大容量网络上抑制过拟合。")
set_para_text(paras[81], "结果：固定数据划分，用 3 个不同随机种子（42/43/44）复测分别得到 96.67%、92.78%、95.00%，"
                         "平均 94.82%，显著优于任何单项改进（最高 82.22%），且优于基线 72.6 个百分点。"
                         "3 次结果波动约 4 个百分点，说明小数据集上复测取平均确有必要。")

# ---------- 4 AI 协作记录 ----------
set_para_text(paras[83], "说明：本实验因环境限制未安装 TRAE IDE，全部 AI 协作改用 ZCode AI 编程助手完成，"
                         "其对话生成、代码解释与调试能力与 TRAE 的 Chat / Builder / 智能修复一一对应，"
                         "协作模式（提示词 → AI 输出 → 人工理解、修改与验证）保持一致。下表记录 3 处典型协作。")
t12 = doc.tables[12]
coop = [
    ("实验框架生成（对应 Builder）",
     "“请在当前目录创建 mlp_digits.py，用 PyTorch 实现手写数字分类实验框架：加载 load_digits，8:2 分层划分，"
     "随机种子 42；MLP 的隐藏层/激活/Dropout/BatchNorm 可配置；训练循环含前向、交叉熵、反向传播、参数更新；"
     "逐 epoch 记录损失与准确率并绘图保存……”（实验指南步骤 1 原文）",
     "AI 一次性生成完整框架，与指南参考代码结构一致。我逐行通读后重点核查了三处：数据划分确实 stratify=y；"
     "标签已转 torch.long（Windows int32 会报错）；损失函数为 CrossEntropyLoss。采纳后自行补充了 3 处修改："
     "① 设 KMP_DUPLICATE_LIB_OK 环境变量规避本机 OpenMP 冲突；② matplotlib 改 Agg 后端便于无界面出图；"
     "③ 增加统一的 do_experiment 入口与学习率扫描。"),
    ("单项改动与原理解释（对应 Chat）",
     "“请把模型中的 Sigmoid 激活函数改为 ReLU，其他保持不变，并告诉我这一改动为什么通常能加快收敛。”",
     "AI 给出最小改动方案（仅 act='relu'）并解释：Sigmoid 导数最大 0.25、逐层连乘导致梯度消失，"
     "ReLU 正区间导数恒 1。采纳该解释框架，并自己补充了数学推导（两层网络梯度衰减到 1/16 以下）"
     "与实验 2 的负结果对照，确认解释与数据一致后才写入报告。"),
    ("失败实验分析与学习率扫描设计（对应 Chat + 智能调试）",
     "“学习率设为 10.0 后损失曲线剧烈震荡无法稳定下降，请解释原因；再帮我在 0.1、1.0、5.0、10.0、20.0 之间"
     "做学习率扫描，寻找本设置下的发散临界点。”",
     "AI 解释：步长过大时单次更新跨过损失面谷底，参数在最优值两侧来回跳动甚至被甩向更高处；"
     "并指出本实验全批量 30 步更新极少，大学习率反而可能更好。采纳该提示后自行编写扫描代码，"
     "实测 lr=1.0 达 78.89% 为扫描最优、发散临界点位于 10～20 之间，把这一反直觉现象写入 5.2 节。"),
]
for i, (scene, prompt, adopt) in enumerate(coop, start=1):
    set_cell(t12.rows[i].cells[1], scene, size=9)
    set_cell(t12.rows[i].cells[2], prompt, size=9)
    set_cell(t12.rows[i].cells[3], adopt, size=9)
# 删除样板自带的“4（可选）”空行，避免孤行跨页
opt_row = t12.rows[4]._tr
opt_row.getparent().remove(opt_row)

# ---------- 5.1 表5 最优组合复测 ----------
t13 = doc.tables[13]
runs_data = [("第 1 次", "42", R["Exp8-run1"]), ("第 2 次", "43", R["Exp8-run2"]), ("第 3 次", "44", R["Exp8-run3"])]
for i, (label, seed, r) in enumerate(runs_data, start=1):
    set_cell(t13.rows[i].cells[0], label)
    set_cell(t13.rows[i].cells[1], seed)
    set_cell(t13.rows[i].cells[2], f"{r['acc']:.2f}")
    set_cell(t13.rows[i].cells[3], f"{r['time']:.2f}")
set_cell(t13.rows[4].cells[1], "42/43/44")
set_cell(t13.rows[4].cells[2], "94.82", bold=True)
set_cell(t13.rows[4].cells[3], f"{sum(r['time'] for _, _, r in runs_data)/3:.2f}")
insert_para_after_table(
    t13, "3 次复测平均准确率 94.82%（波动范围 92.78%～96.67%），显著优于所有单项改进组（最高 82.22%），"
         "也远高于基线 22.22%。波动主要来自参数初始化与 Dropout 随机掩码的差异；训练耗时的差异（1.5～3.6 s）"
         "主要来自首次运行的环境预热，与模型本身无关。")

# ---------- 5.2 失败实验分析 ----------
insert_image(doc.tables[14], os.path.join(BASE, "result_lr_too_big.png"))
set_para_text(paras[92], "现象：学习率设为 10.0 后，训练损失在前 2 个 epoch 冲高到约 3.75，随后一直在 2.0～2.7 之间"
                         "剧烈震荡、无法稳定下降；测试准确率在 10%～37% 之间大幅跳动，相邻两个 epoch 之间可以相差"
                         "20 个百分点以上，最终停在 36.94%。")
set_para_text(paras[93], "原因：学习率过大时，单步参数更新的步长超过了损失面谷底的宽度，参数每一步都直接跨过谷底"
                         "落到对面坡上，下一步又被大幅拉回，于是参数在最优值两侧“来回甩”，损失呈现锯齿状震荡；"
                         "更严重时参数甚至被甩到损失更高的区域，训练完全发散。")
set_para_text(paras[94], "学习率扫描（基线其余配置不变）：lr=0.1 → 22.22%，lr=1.0 → 78.89%，lr=5.0 → 64.72%，"
                         "lr=10.0 → 36.94%，lr=20.0 → 10.28%。可见发散临界点位于 10 与 20 之间。"
                         "反直觉的是 lr=1.0 反而是扫描中最优：这是因为本实验数据集极小且全批量训练每个 epoch 只更新一次参数，"
                         "30 个 epoch 总共只有 30 步更新，需要较大的步长才能走出初始平坦区——“学习率多大方才合适”"
                         "与总更新步数、优化器类型强耦合，并非一个孤立的常数。")

# ---------- 5.3 总结论 ----------
set_para_text(paras[97], "1. 各设计维度在本任务上的有效性排序：优化器（Adam，+60.0 pp）≈ BatchNorm（+57.0 pp）"
                         "> 激活函数（ReLU，+25.3 pp）> 调学习率（lr=0.01 时 SGD 反而失效）；L2 与 Dropout 在欠拟合的"
                         "基线上收益为零，但在大容量的最优组合中是防止过拟合、稳定复测结果的必要部件——正则化的价值取决于模型是否已经“学得动”。")
set_para_text(paras[98], "2. “更大更深”不等于更好：2 层×128 的 Sigmoid 网络准确率跌到 7.22%，说明容量只有在梯度通路健康"
                         "（ReLU/BN）且配合足够优化能力的优化器时才能转化为性能；最优组合正是“ReLU+BN+Adam+正则化”"
                         "环环相扣的结果，任何单一改进都无法达到 94.82% 的水平。")
set_para_text(paras[99], "3. 学习率没有普适最优值：本设置下 SGD 的最佳学习率是 1.0（非常规的 0.01/0.1），"
                         "因为它全批量 30 步的更新次数极少；学习率过大的发散临界点在 10～20 之间。"
                         "评价超参数必须结合更新步数与优化器机制，而非死记“太大发散、太小太慢”。")
set_para_text(paras[100], "4. 方法论体会：固定随机种子 + 每组只改一个变量的控制变量法，是把 82.22%、79.17% 这样的数字"
                          "准确归因到单一设计点的前提；负结果（深层 Sigmoid 崩溃、正则化无效）与正结果同样有价值，"
                          "3 次复测取平均避免了把偶然波动当成结论。")

# ---------- 6 思考题 ----------
set_para_text(paras[103], "收敛速度显著加快：基线（Sigmoid）前 20 个 epoch 准确率停留在 10%，30 个 epoch 仅达 22.22%；"
                          "换成 ReLU 后准确率从第 1 个 epoch 起就持续上升，30 个 epoch 达到 47.50%。")
set_para_text(paras[104], "数学原因：Sigmoid 的导数 σ′(z)=σ(z)(1−σ(z))，在 z=0 处取得最大值 0.25，且 |z| 增大时迅速趋于 0。"
                          "反向传播中梯度要逐层乘以激活函数导数，每过一层梯度至少缩小为原来的 1/4（饱和区接近 0），"
                          "两层以上网络中早期层的梯度呈指数衰减，参数几乎得不到更新信号。")
set_para_text(paras[105], "ReLU 的导数在正区间恒为 1、负区间为 0，不存在连乘衰减因子，梯度可以几乎无衰减地回传到早期层；"
                          "同时负区间输出 0 带来稀疏性。因此网络在前几个 epoch 就能获得有效梯度，收敛速度大幅提升。")

set_para_text(paras[107], "不相同，SGD 受到的影响远大于 Adam。SGD 的有效步长 = lr × 梯度：本实验基线梯度本就微小（Sigmoid + 输入归一化），"
                          "lr 从 0.1 降到 0.01 后准确率从 22.22% 跌回 10%，网络基本冻结。")
set_para_text(paras[108], "Adam 为每个参数维护梯度的一阶矩 m̂（动量）与二阶矩 v̂（幅值），更新量为 lr 乘以"
                          "（一阶矩 ÷ 二阶矩的平方根）。该比值的量级约为 ±1，相当于对梯度做了逐参数的幅值归一化，"
                          "有效步长主要由 lr 决定，对梯度的绝对大小不敏感。")
set_para_text(paras[109], "因此 lr 缩小 10 倍时 Adam 只是整体放慢但仍在稳定学习（lr=0.01 仍达 82.22%），而 SGD 已完全失效。"
                          "这也解释了实践中的经验：SGD 必须精心调 lr，Adam 用默认 lr=0.001 就能工作。")

set_para_text(paras[111], "内部协变量偏移指训练中前面层参数不断更新，导致后面层看到的输入分布持续漂移，"
                          "后层不得不不断适应新分布，损失面随之“移动”，训练不稳定。")
set_para_text(paras[112], "BatchNorm 把每个隐藏层的输入强制标准化为均值 0、方差 1（再用 γ、β 恢复表达能力），"
                          "无论前层参数怎么更新，后层的输入分布保持稳定，相当于损失面被“固定”得更平滑。")
set_para_text(paras[113], "标准化把激活值限制在 Sigmoid 的非饱和区（|z| 较小处，梯度 ≥ 0.2），即使大步长把参数推偏，"
                          "下一层的输入仍被重新拉回标准区间，不会一步进入梯度消失区或跨过谷底。因此可以放心使用更大的学习率——"
                          "本实验中 BN 使 SGD(0.1) 从 22.22% 提升到 79.17%，损失曲线也明显更陡、更平稳。")

set_para_text(paras[115], "PyTorch 采用 inverted dropout：训练时以概率 p 把激活置零，并对保留下来的激活乘以 1/(1−p) 放大。"
                          "例如 p=0.2 时，保留的 80% 神经元各放大 1/0.8=1.25 倍，使激活的期望值与不丢弃时一致。")
set_para_text(paras[116], "推理时调用 model.eval() 后，Dropout 层变为恒等映射：不丢弃任何神经元、不缩放。")
set_para_text(paras[117], "由于训练时的期望已经被 1/(1−p) 预先补偿，推理时的输出规模与训练期望自动一致，"
                          "无需在推理阶段做任何缩放。这与早期论文的“训练不缩放、推理按 (1−p) 缩放”做法等价，"
                          "但把缩放放到训练侧使推理更简单。")

set_para_text(paras[119], "偏差—方差权衡：模型总误差 ≈ 偏差（拟合能力不足）+ 方差（对训练采样过度敏感）。"
                          "本实验训练集只有 1437 个样本，2 层×128 的网络参数量远超样本量，方差项急剧上升——"
                          "网络有能力逐个“记住”训练样本，表现为训练误差低而测试误差高，即过拟合。")
set_para_text(paras[120], "本实验还叠加了第二个因素：Sigmoid 深层网络的梯度消失使大容量根本无法被有效训练，"
                          "7.22% 的结果说明“容量大”与“能训练”是两个独立的前提，缺一不可。")
set_para_text(paras[121], "因此小数据集上正确的做法是：用 ReLU/BN 保证可训练性，用适度容量控制方差，"
                          "再用 Dropout/L2 压制过拟合（实验 8 的组合），而不是一味加深加宽。")

set_para_text(paras[123], "AI 最有价值的环节：① 按提示词一次性生成实验框架，省去搭建训练循环、绘图等重复劳动；"
                          "② 解释报错与现象（如 Long 类型转换、学习率震荡的原因），把排查时间从小时级压到分钟级；"
                          "③ 补全重复性代码。这些是“体力密集型”工作，AI 的产出质量高且可控。")
set_para_text(paras[124], "必须由人做的决策：实验设计（控制变量、固定种子、分组顺序）、对 AI 生成代码的逐行核查"
                          "（本实验中我验证了分层划分、标签类型、损失函数三处关键点）、对每个实验数字真实性的确认，"
                          "以及判断负结果是否成立、结论如何归因——AI 可以给出解释，但只有结合数据才能确认解释正确。")
set_para_text(paras[125], "总体理解：AI 生成代码是“起点”而非“终点”。人机协作的理想分工是 AI 负责生成与解释，"
                          "人负责设计、验证与判断；只有当每一个数字都来自自己运行、每一条结论都能被曲线佐证时，"
                          "AI 带来的效率提升才真正转化为可靠的实验结果。")

# ---------- 排版修正：标题与下文同页，避免孤行 ----------
import re
for p in doc.paragraphs:
    if re.match(r"^\d(\.\d)?\s", p.text.strip()):  # “1 实验概述” / “3.7 实验 7”等章节标题
        p.paragraph_format.keep_with_next = True
    if p.text.strip().startswith("表 3"):          # 表 3 整体推到下一页，避免表头孤悬页底
        p.paragraph_format.page_break_before = True

# ---------- 删除所有“提示：”引导段 ----------
removed = 0
for p in doc.paragraphs:
    if p.text.strip().startswith("提示："):
        p._element.getparent().remove(p._element)
        removed += 1

doc.save(OUT)
print(f"saved: {OUT}  (删除提示段 {removed} 处)")
