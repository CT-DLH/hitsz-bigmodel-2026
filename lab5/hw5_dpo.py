# -*- coding: utf-8 -*-
"""
实验作业五：SmolVLM-256M 的 QLoRA 高效微调与 DPO 偏好对齐
------------------------------------------------------------------
以《实验指南》第七节参考框架为基础实现，为适配「无 GPU 的 CPU 环境」以及
修正参考代码中会影响结论的口径问题，做了以下改动（报告中逐条说明）：

  [C1] 判分口径修正：参考代码 judge() 对 gold="有" 的题目会把答「没有」误判为正确
       （因为字符串「没有」包含「有」）。本脚本改用 _is_negative() 显式识别否定词，
       同时保留参考口径 judge_guide() 以便量化它高估了多少。
  [C2] 训练步数修正：参考代码 epochs 固定（SFT 2 / DPO 1），当 --steps 大于
       epochs×N 时循环提前结束、实际步数被静默截断。本脚本按 --steps 反推所需 epochs。
  [C3] 参考模型前向加 no_grad，并缓存 π_ref 的序列对数概率：π_ref 在整轮 DPO 中冻结，
       其取值与 policy 无关，因此只需在训练前算一次。DPO 单步前向次数由 4 次降为 2 次。
  [C4] labels 掩码的 prompt_len 加了断言校验：逐元素比对「prompt 单独编码」与
       「prompt+answer 合并编码」的前 prompt_len 个 token 是否一致，防止图像 token
       展开导致掩码错位（指南「智能补全」一节点名要求自查此处）。
  [C5] CPU 无 bitsandbytes 时自动回退为 fp32 基座 + LoRA，仅训适配器的思想不变；
       量化路径在有 NVIDIA GPU 且装了 bitsandbytes 时自动启用。
  [C6] 参考模型可选 --ref base（指南口径）或 --ref sft（DPO 论文口径），用于对照。
  [C7] 新增 5 组自拟偏好对 MY_PAIRS（任务三），并新增 --mode sweep_beta / sweep_rank
       完成「任务五」超参敏感性扫描。
  [C8] 修正参考代码 dpo_check() 的符号错误：其 margin>=0 分支写成
       −log(1+e^{−margin})，margin<0 分支写成 margin−log(1+e^{margin})，
       两者都是正确答案 log(1+e^{−margin}) 的相反数。指南 4.3 节手算例给出
       loss≈+0.693，而按其代码会打印 −0.688 —— 代码与自己的手算例矛盾。
       本脚本改用数值稳定式 max(−m,0)+log1p(e^{−|m|})。
  [C10] DPO 阶段改为在「裸基座」上单层包装 PeftModel。参考代码的 eval 路径把
        PeftModel 套在已经包装过的模型上（嵌套），保存出的适配器键名会多一层
        「base_model.model.」前缀，导致评测时 load_adapter 静默匹配失败、
        DPO 适配器等效于全零、三阶段评测退化成基线。本脚本统一用单层包装，
        并在评测侧改为直接 PeftModel.from_pretrained(base, DPO_DIR)。
  [C9] 图像预处理可切 --img_mode：split=指南默认（256×256 图被放大到 2048 后切成
       4×4 拼块 + 1 张全图，共 17 个图像槽 = 1088 个视觉 token，占序列长度 94%）；
       native=不切块、最长边 512（= 64 个视觉 token）。CPU 训练默认 native，
       并用 split 跑基线作为对照，量化该预处理选择对正确率的影响。

常用命令：
  python hw5_dpo.py --mode make_data
  python hw5_dpo.py --mode infer
  python hw5_dpo.py --mode sft   --steps 10
  python hw5_dpo.py --mode dpo   --steps 4 --ref base
  python hw5_dpo.py --mode eval
  python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1
  python hw5_dpo.py --mode sweep_beta
  python hw5_dpo.py --mode sweep_rank
"""

import argparse
import copy
import json
import math
import os
import platform
import random
import re
import sys
import time

# ---- 环境变量必须在 import transformers 之前设置 ------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows OpenMP 重复加载保护
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
LOCAL_MODEL = os.path.join(BASE, "_models", "SmolVLM-256M-Instruct")
IMG_DIR = os.path.join(BASE, "hw5_imgs")
SFT_DIR = os.path.join(BASE, "sft_adapter")
DPO_DIR = os.path.join(BASE, "dpo_adapter")
RES_DIR = os.path.join(BASE, "results")
ADAPTER_LABEL = {"sft": SFT_DIR, "dpo": DPO_DIR, "r4": os.path.join(BASE, "sft_adapter_r4"),
                 "r16": os.path.join(BASE, "sft_adapter_r16")}
SEED = 42


# ============================================================================
# 0. 日志：所有日志用 Python 以 UTF-8 自己写，避免 PowerShell 重定向的 GBK 乱码
# ============================================================================
class Tee:
    def __init__(self, path=None):
        self.f = open(path, "w", encoding="utf-8") if path else None
        self.buf = []

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.buf.append(str(msg))
        if self.f:
            self.f.write(str(msg) + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def results_path(name):
    os.makedirs(RES_DIR, exist_ok=True)
    return os.path.join(RES_DIR, name)


def load_json(name):
    p = results_path(name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump_json(name, obj):
    p = results_path(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return p


# ============================================================================
# 1. 合成图像库「几何图形」：15 张图，三类问题的答案均已埋入
# ============================================================================
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],                    # 判断题素材：只有一个黄方
    "img12": [("红", "方", 1), ("蓝", "方", 1)],    # 没有圆形 → 「有没有红圆」答「没有」
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],
}


def make_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    rnd = random.Random(SEED)
    boxes = {}
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(im)
        for color, shape, n in spec:
            for _ in range(n):
                x, y = rnd.randint(10, 200), rnd.randint(10, 200)
                s = rnd.randint(30, 50)
                box = [x, y, x + s, y + s]
                if shape == "圆":
                    d.ellipse(box, outline=COLORS[color], width=4)
                else:
                    d.rectangle(box, outline=COLORS[color], width=4)
        im.save(f"{IMG_DIR}/{img_id}.png")
    print(f"[data] 已生成 {len(IMG_SPEC)} 张图像到 {IMG_DIR}/ (seed={SEED})")
    return boxes


def load_image(img_id):
    p = f"{IMG_DIR}/{img_id}.png"
    if not os.path.exists(p):
        make_images()
    return Image.open(p).convert("RGB")


# ============================================================================
# 2. 问题集：3 类 × 5 题
# ============================================================================
QUESTIONS = [
    {"id": "P01", "type": "count", "img": "img01", "q": "图中有几个红色圆形？", "gold": "2"},
    {"id": "P02", "type": "count", "img": "img02", "q": "图中有几个绿色圆形？", "gold": "3"},
    {"id": "P03", "type": "count", "img": "img03", "q": "图中有几个黄色方形？", "gold": "2"},
    {"id": "P04", "type": "count", "img": "img06", "q": "图中有几个黄色圆形？", "gold": "4"},
    {"id": "P05", "type": "count", "img": "img10", "q": "图中有几个蓝色圆形？", "gold": "3"},
    {"id": "P06", "type": "attr", "img": "img01", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P07", "type": "attr", "img": "img05", "q": "图中红色的图形是什么形状？", "gold": "圆形"},
    {"id": "P08", "type": "attr", "img": "img09", "q": "图中绿色的图形是什么形状？", "gold": "方形"},
    {"id": "P09", "type": "attr", "img": "img13", "q": "图中黄色的图形是什么形状？", "gold": "圆形"},
    {"id": "P10", "type": "attr", "img": "img14", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P11", "type": "exist", "img": "img11", "q": "图里有没有黄色方形？", "gold": "有"},
    {"id": "P12", "type": "exist", "img": "img12", "q": "图里有没有红色圆形？", "gold": "没有"},
    {"id": "P13", "type": "exist", "img": "img07", "q": "图里有没有绿色圆形？", "gold": "有"},
    {"id": "P14", "type": "exist", "img": "img08", "q": "图里有没有绿色方形？", "gold": "没有"},
    {"id": "P15", "type": "exist", "img": "img15", "q": "图里有没有蓝色图形？", "gold": "没有"},
]
Q_BY_ID = {q["id"]: q for q in QUESTIONS}
TYPES = ("count", "attr", "exist")


def gold_answer(q):
    """标准答案的书写形式（SFT 目标 / DPO chosen）"""
    return f"{q['gold']}个。" if q["type"] == "count" else f"{q['gold']}。"


SFT_DATA = [{"qid": q["id"], "img": q["img"], "prompt": q["q"], "answer": gold_answer(q)}
            for q in QUESTIONS]


# ============================================================================
# 2b. 偏好对：12 组内置（指南）+ 5 组自拟（任务三，共 17 组）
# ============================================================================
PRE_PAIRS = [
    {"qid": "P01", "why": "数错（off-by-one）", "rejected": "3 个。图中有三个红色圆形整齐排列。"},
    {"qid": "P02", "why": "数错（off-by-one）", "rejected": "2 个。"},
    {"qid": "P03", "why": "视而不见（否认图中确有之物）", "rejected": "没有看到黄色图形。"},
    {"qid": "P04", "why": "数错（off-by-one）", "rejected": "3 个。"},
    {"qid": "P05", "why": "形状幻觉（圆说成方）", "rejected": "图中有三个蓝色方形。"},
    {"qid": "P06", "why": "形状幻觉（方说成圆）", "rejected": "圆形。"},
    {"qid": "P07", "why": "形状幻觉 + 虚构数量", "rejected": "方形，而且有两个红色方形。"},
    {"qid": "P08", "why": "视而不见", "rejected": "图中没有绿色图形。"},
    {"qid": "P09", "why": "颜色/形状张冠李戴", "rejected": "方形，图中黄色的都是方形。"},
    {"qid": "P10", "why": "形状幻觉（方说成圆）", "rejected": "圆形。"},
    {"qid": "P11", "why": "硬答「没有」（漏报）", "rejected": "没有。"},
    {"qid": "P12", "why": "硬答「有」+ 空间位置编造", "rejected": "有，图左上角有一个红色圆形。"},
    {"qid": "P13", "why": "硬答「没有」（漏报）", "rejected": "图里没有绿色圆形。"},
    {"qid": "P14", "why": "硬答「有」+ 虚构数量", "rejected": "有，一共两个绿色方形。"},
    {"qid": "P15", "why": "硬答「有」+ 虚构对象", "rejected": "有，两个蓝色圆形。"},
]
# 自拟 5 组：补齐 4 类新的失败模式，并显式标注对应哪种失败模式（任务三要求）
MY_PAIRS = [
    {"qid": "P02", "why": "冗长废话型（信息稀释，不答数目却堆砌形容词）",
     "rejected": "图中一共有若干个绿色的圆形，它们分布在画面的不同位置，看起来数量不算多也不算少，整体比较稀疏。"},
    {"qid": "P08", "why": "冗长废话型 + 不给出结论",
     "rejected": "这张图里包含绿色的几何图形，如果要判断它的形状，需要仔细观察边框的走向，这个问题比较复杂，我建议再看一看。"},
    {"qid": "P13", "why": "否定式幻觉（把「有」答成「没有」并附编造理由）",
     "rejected": "没有。图中只有蓝色和黄色的图形，没有绿色。"},
    {"qid": "P04", "why": "颜色混淆（数了别的颜色）",
     "rejected": "图中有 4 个绿色圆形，没有黄色圆形。"},
    {"qid": "P14", "why": "形状幻觉（方说成圆）",
     "rejected": "有，图中有两个绿色的圆形。"},
]


def build_pairs(specs):
    out = []
    for s in specs:
        q = Q_BY_ID[s["qid"]]
        out.append({"qid": q["id"], "type": q["type"], "img": q["img"], "prompt": q["q"],
                    "chosen": gold_answer(q), "rejected": s["rejected"], "why": s["why"]})
    return out


ALL_PAIRS = build_pairs(PRE_PAIRS) + build_pairs(MY_PAIRS)


# ============================================================================
# 3. 模型加载：QLoRA（NF4 + LoRA），CPU 自动回退 fp32 + LoRA
# ============================================================================
def pick_dtype():
    return torch.float32


def make_processor(src, img_mode="native", log=None):
    """[C9] 图像预处理：split=指南默认 / native=不切块、最长边 512"""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(src)
    ip = proc.image_processor
    if img_mode == "native":
        ip.do_image_splitting = False
        ip.size = {"longest_edge": 512}
    n_img = {"native": 64, "split": 1088}[img_mode]
    if log:
        log(f"[img] 预处理模式={img_mode}  do_image_splitting={ip.do_image_splitting} "
            f"size={ip.size} max_image_size={ip.max_image_size} → 视觉 token ≈ {n_img}")
    return proc


def load_model(mode="infer", lora_r=8, lora_alpha=None, log=None, quant="auto",
               img_mode="native"):
    """mode: infer=推理 / train=量化(若可用)+LoRA
    返回 (processor, model, info)
    """
    from transformers import AutoModelForVision2Seq
    info = {}
    src = LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else MODEL_ID
    info["model_path"] = src
    info["img_mode"] = img_mode
    processor = make_processor(src, img_mode, log)

    kwargs = {"torch_dtype": pick_dtype()}
    use_4bit = False
    if mode == "train" and quant in ("auto", "4bit"):
        try:
            import bitsandbytes  # noqa: F401
            if torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float32)
                use_4bit = True
        except Exception:
            use_4bit = False
    info["quant_4bit"] = use_4bit
    if mode == "train" and not use_4bit:
        info["fallback"] = ("无 NVIDIA GPU / 未安装 bitsandbytes，已回退为 fp32 基座 + LoRA；"
                            "只训适配器的思想不变，但未走 4-bit NF4 量化路径")
    model = AutoModelForVision2Seq.from_pretrained(src, **kwargs)
    info["base_params"] = sum(p.numel() for p in model.parameters())

    if mode == "train":
        from peft import LoraConfig, get_peft_model
        for p in model.parameters():
            p.requires_grad = False
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r,
                          lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        if log:
            model.print_trainable_parameters()
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        info["trainable_params"] = tr
        info["total_params"] = tot
        # 两种占比口径：分母含 / 不含 LoRA 适配器本身
        info["pct_of_total"] = 100.0 * tr / tot
        info["pct_of_base"] = 100.0 * tr / info["base_params"]
        info["lora_r"] = lora_r
        info["lora_alpha"] = lora_alpha or 2 * lora_r
        info["lora_targets"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
        info["lora_dropout"] = 0.05
        names = list(getattr(model, "targeted_module_names", []))
        info["n_target_modules"] = len(names)
        # 统计命中的模块按后缀归类（说明 LoRA 挂在哪些子模块上）
        suffix = {}
        for n in names:
            s = n.split(".")[-1]
            suffix[s] = suffix.get(s, 0) + 1
        info["target_suffix_count"] = suffix
        info["target_examples"] = names[:3]
        model.config.use_cache = False
    else:
        info["total_params"] = info["base_params"]
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return processor, model, info


def build_prompt(processor, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)


def encode(processor, prompt, img, answer=None):
    text = prompt if answer is None else prompt + answer
    enc = processor(text=[text], images=[img], return_tensors="pt")
    return enc


@torch.no_grad()
def answer(processor, model, img, q, max_new=24):
    inputs = encode(processor, build_prompt(processor, q), img).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()


_NEG = ("没有", "没", "未", "无", "不")


def _is_negative(pred):
    """预测里是否出现明确的否定词（用于 exist 类判分）"""
    p = pred.replace(" ", "")
    return any(w in p for w in ("没有", "没看", "未见", "未发现", "不存在", "不含", "无")) or \
        (p.startswith("没") or p.startswith("不"))


def judge(q, pred):
    """[C1] 修正后的宽松判分：标准答案出现即算对，但 gold='有' 必须排除否定式回答。"""
    p = pred.replace(" ", "")
    g = q["gold"]
    if g == "没有":
        return _is_negative(p)
    if g == "有":
        return ("有" in p) and (not _is_negative(p))
    return g in p


def judge_guide(q, pred):
    """参考代码原口径（保留，用于量化它的高估幅度）"""
    p, g = pred.replace(" ", ""), q["gold"]
    if g == "没有":
        return "没有" in p and "有" in p
    return g in p


_NUM_EN = {"1": ("one", "1"), "2": ("two", "2"), "3": ("three", "3"),
           "4": ("four", "4"), "0": ("zero", "0")}
_SHAPE_EN = {"方形": ("square", "rectangle"), "圆形": ("circle", "round", "circular")}


def _has_word(p, words):
    """英文侧按词边界匹配，避免 know 里的 no、network 里的 two 这类误命中"""
    for w in words:
        if w.isdigit():
            if w in p:
                return True
        elif re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", p):
            return True
    return False


def judge_loose(q, pred):
    """中英双语容错判分：把「语言没对上」与「知识没对上」分开。
    SmolVLM-256M 的基线几乎全用英文回答，只按中文 gold 匹配会把语义正确也判错。"""
    p = pred.lower()
    g = q["gold"]
    if q["type"] == "count":
        return _has_word(p, _NUM_EN.get(g, (g,))) or (g in p)
    if q["type"] == "attr":
        return _has_word(p, _SHAPE_EN.get(g, ())) or (g in p)
    # exist：先判否定、再判肯定，避免「没有」被当成「有」而误判
    has_neg = ("没有" in p) or ("未" in p) or ("不存在" in p) or ("无" in p) or \
        _has_word(p, ("no", "not", "none", "never", "without")) or ("n't" in p)
    has_pos = (("有" in p) and ("没有" not in p)) or _has_word(p, ("yes", "there is", "there are"))
    if g == "没有":
        return has_neg and not has_pos
    return has_pos and not has_neg


def score_rows(rows):
    by = {}
    for tp in TYPES:
        sel = [r for r in rows if r["type"] == tp]
        by[tp] = {"n": len(sel), "ok": sum(1 for r in sel if r["ok"]),
                  "ok_guide": sum(1 for r in sel if r["ok_guide"]),
                  "ok_loose": sum(1 for r in sel if r["ok_loose"])}
    by["all"] = {"n": len(rows),
                 "ok": sum(1 for r in rows if r["ok"]),
                 "ok_guide": sum(1 for r in rows if r["ok_guide"]),
                 "ok_loose": sum(1 for r in rows if r["ok_loose"])}
    return by


def run_questions(processor, model, tag, log, max_new=24):
    rows = []
    t0 = time.time()
    for q in QUESTIONS:
        pred = answer(processor, model, load_image(q["img"]), q["q"], max_new=max_new)
        rows.append({"id": q["id"], "type": q["type"], "img": q["img"], "q": q["q"],
                     "gold": q["gold"], "pred": pred,
                     "ok": judge(q, pred), "ok_guide": judge_guide(q, pred),
                     "ok_loose": judge_loose(q, pred)})
        log(f"[{tag}] {q['id']}·{q['type']:<5} {'√' if rows[-1]['ok'] else '×'} "
            f"Q:{q['q']} A:{pred!r} (gold:{q['gold']})")
    dt = time.time() - t0
    by = score_rows(rows)
    for tp in TYPES:
        log(f"[{tag}] {tp:<6} 正确 {by[tp]['ok']}/{by[tp]['n']}"
            f"  (指南口径 {by[tp]['ok_guide']}/{by[tp]['n']}, 双语容错 {by[tp]['ok_loose']}/{by[tp]['n']})")
    log(f"[{tag}] 总计 正确 {by['all']['ok']}/15"
        f"  (指南口径 {by['all']['ok_guide']}/15, 双语容错 {by['all']['ok_loose']}/15)"
        f"  用时 {dt:.1f}s (平均 {dt/15:.1f}s/题)")
    return {"rows": rows, "by_type": by, "total": by["all"]["ok"],
            "total_guide": by["all"]["ok_guide"], "total_loose": by["all"]["ok_loose"],
            "seconds": dt, "sec_per_q": dt / 15}


# ============================================================================
# 4. SFT：只对答案 token 计损失
# ============================================================================
def masked_batch(processor, img, prompt, answer):
    """返回 full 输入 + labels + n_ans，并断言掩码位置正确（[C4]）"""
    full = encode(processor, build_prompt(processor, prompt), img, answer)
    with torch.no_grad():
        only = encode(processor, build_prompt(processor, prompt), img)
    plen = only["input_ids"].shape[1]
    ids = full["input_ids"]
    assert ids.shape[1] > plen, "答案被截断了：full=%d plen=%d" % (ids.shape[1], plen)
    # 逐元素比对 prompt 前缀，防图像 token 展开导致错位
    assert torch.equal(ids[0, :plen], only["input_ids"][0]), \
        "prompt_len 掩码错位：合并编码的前 %d 个 token 与单独编码不一致" % plen
    labels = ids.clone()
    labels[:, :plen] = -100
    full["labels"] = labels
    return full, plen, ids.shape[1] - plen


def sft_step(model, processor, opt, steps=60, log_every=1, log=print, lr=1e-4):
    """[C2] 按 steps 反推 epochs，保证实际执行步数等于 steps（参考代码会静默截断）"""
    n = len(SFT_DATA)
    epochs = max(1, math.ceil(steps / n))
    hist, done, t0 = [], 0, time.time()
    model.train()
    for ep in range(epochs):
        rnd = random.Random(SEED + ep)
        order = list(range(n))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            if done >= steps:
                break
            s = SFT_DATA[idx]
            full, plen, n_ans = masked_batch(processor, load_image(s["img"]), s["prompt"], s["answer"])
            full = full.to(next(model.parameters()).device)
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            done += 1
            hist.append({"step": done, "loss": round(float(loss.item()), 4), "qid": s["qid"]})
            if done % log_every == 0 or done == steps:
                log(f"[sft] ep{ep+1} step{done}/{steps} qid={s['qid']} loss={loss.item():.4f}")
        if done >= steps:
            break
    model.eval()
    log(f"[sft] 完成 {done} 步，用时 {time.time()-t0:.1f}s；尾部 loss={hist[-1]['loss']:.4f}")
    return {"hist": hist, "steps": done, "n_samples": n, "epochs_run": epochs,
            "lr": lr, "seconds": time.time() - t0, "final_loss": hist[-1]["loss"],
            "loss_head": hist[0]["loss"], "loss_tail_mean": round(
                sum(h["loss"] for h in hist[-min(3, len(hist)):]) / min(3, len(hist)), 4),
            "epochs_note": f"{n} 条样本 × {epochs} epoch ≥ {steps} 步（参考代码固定 2 epoch，"
                           f"当 steps>{2*n} 时实际步数被静默截断）"}


# ============================================================================
# 5. DPO
# ============================================================================
def seq_logprob(model, inputs, plen, need_grad=False):
    """答案 token 的序列总对数概率（Σ log π）。[C3] 默认 no_grad。"""
    n_ans = inputs["input_ids"].shape[1] - plen
    assert n_ans > 0
    with torch.set_grad_enabled(need_grad):
        try:
            logits = model(**inputs, num_logits_to_keep=n_ans).logits
        except TypeError:
            logits = model(**inputs).logits[:, -n_ans:]
        tgt = inputs["input_ids"][:, plen:]
        lp = torch.log_softmax(logits.float(), dim=-1)
        return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()


def encode_pair(processor, pair):
    img = load_image(pair["img"])
    prompt = build_prompt(processor, pair["prompt"])
    out = {}
    for side in ("chosen", "rejected"):
        enc = encode(processor, prompt, img, pair[side])
        with torch.no_grad():
            only = encode(processor, prompt, img)
        plen = only["input_ids"].shape[1]
        assert torch.equal(enc["input_ids"][0, :plen], only["input_ids"][0]), \
            "DPO 掩码错位：%s/%s" % (pair["qid"], side)
        out[side] = (enc, plen)
    return out


def compute_ref_logps(model, processor, pairs, log, tag):
    """[C3] 冻结参考模型的序列对数概率只算一次，全流程复用"""
    cache, t0 = [], time.time()
    for i, p in enumerate(pairs):
        if (i + 1) % 4 == 0 or i == 0:
            log("      [ref] %d/%d %s … %.1fs" % (i + 1, len(pairs), p["qid"],
                                                  time.time() - t0))
        enc = encode_pair(processor, p)
        dev = next(model.parameters()).device
        r = {}
        for side in ("chosen", "rejected"):
            e, plen = enc[side]
            r[side] = float(seq_logprob(model, e.to(dev), plen, need_grad=False).item())
        cache.append(r)
    log(f"[dpo] 参考模型（{tag}）对数概率已缓存 {len(cache)} 对，用时 {time.time()-t0:.1f}s")
    return cache


def dpo_step(model, processor, opt, ref_cache, beta=0.1, steps=40, log_every=1,
             log=print, lr=5e-5, pairs=None):
    pairs = pairs if pairs is not None else ALL_PAIRS
    n = len(pairs)
    epochs = max(1, math.ceil(steps / n))
    hist, done, t0 = [], 0, time.time()
    dev = next(model.parameters()).device
    model.train()
    for ep in range(epochs):
        rnd = random.Random(SEED + ep)
        order = list(range(n))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            if done >= steps:
                break
            p = pairs[idx]
            enc = encode_pair(processor, p)
            pol_w = seq_logprob(model, enc["chosen"][0].to(dev), enc["chosen"][1], need_grad=True)
            pol_l = seq_logprob(model, enc["rejected"][0].to(dev), enc["rejected"][1], need_grad=True)
            ref_w, ref_l = ref_cache[idx]["chosen"], ref_cache[idx]["rejected"]
            margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
            loss = -F.logsigmoid(margin)
            loss.backward()
            opt.step()
            opt.zero_grad()
            done += 1
            hist.append({"step": done, "loss": round(float(loss.item()), 4),
                         "margin": round(float(margin.item()), 4), "qid": p["qid"]})
            if done % log_every == 0 or done == steps:
                log(f"[dpo] ep{ep+1} step{done}/{steps} qid={p['qid']} "
                    f"loss={loss.item():.4f} margin={margin.item():.4f} "
                    f"[πθw={float(pol_w):.3f} πθl={float(pol_l):.3f} "
                    f"πrefw={ref_w:.3f} πrefl={ref_l:.3f}]")
        if done >= steps:
            break
    model.eval()
    log(f"[dpo] 完成 {done} 步 β={beta}，用时 {time.time()-t0:.1f}s；"
        f"loss {hist[0]['loss']:.4f}→{hist[-1]['loss']:.4f}，"
        f"margin {hist[0]['margin']:.4f}→{hist[-1]['margin']:.4f}")
    return {"hist": hist, "steps": done, "beta": beta, "lr": lr, "n_pairs": n,
            "seconds": time.time() - t0, "margin_head": hist[0]["margin"],
            "margin_tail": hist[-1]["margin"], "loss_head": hist[0]["loss"],
            "loss_tail": hist[-1]["loss"]}


def dpo_check(arg, log=print):
    """[C8] DPO loss 手算复核。

    数值稳定式：−log σ(m) = log(1 + e^{−m}) = max(−m, 0) + log1p(e^{−|m|})。
    注意 m→0 时该式 → log2 ≈ 0.6931，且恒为正；这可以作为快速自检。
    """
    pw, rw, pl, rl, beta = [float(x) for x in arg.split(",")]
    diff_w, diff_l = pw - rw, pl - rl
    margin = beta * (diff_w - diff_l)
    loss = max(-margin, 0.0) + math.log1p(math.exp(-abs(margin)))
    sigma = 1.0 / (1.0 + math.exp(-margin))
    log(f"--dpo_check 输入: πθ(yw)={pw}, πref(yw)={rw}, πθ(yl)={pl}, πref(yl)={rl}, β={beta}")
    log(f"  (πθ−πref)(yw) = {pw} − {rw} = {diff_w:.4f}")
    log(f"  (πθ−πref)(yl) = {pl} − {rl} = {diff_l:.4f}")
    log(f"  margin = β·[(πθ−πref)(yw) − (πθ−πref)(yl)] = {beta}×({diff_w:.4f} − {diff_l:.4f})"
        f" = {margin:.4f}")
    log(f"  σ(margin) = {sigma:.4f}")
    log(f"  DPO loss = −log σ(margin) = {loss:.4f}")
    if margin > 0:
        log(f"  margin>0：策略把 chosen 相对 rejected 的概率差拉得比参考模型更大，"
            f"loss={loss:.4f} < log2=0.6931，优化方向正确。")
    else:
        log(f"  margin<0：策略反而更偏好 rejected（或相对参考模型退步），"
            f"loss={loss:.4f} > log2=0.6931，需要继续训练。")
    return {"inputs": [pw, rw, pl, rl, beta], "diff_w": diff_w, "diff_l": diff_l,
            "margin": margin, "loss": loss, "sigma": sigma}


# ============================================================================
# 6. 评测 / 各模式
# ============================================================================
def env_info():
    def cpu_name():
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            return platform.processor() or "unknown"
    import transformers, peft
    ram = None
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        ram = st.ullTotalPhys / 1e9
    except Exception:
        pass
    return {"cpu": cpu_name(), "cores": os.cpu_count(), "ram_gb": round(ram, 1) if ram else None,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无（CPU 全程）",
            "os": f"{platform.system()} {platform.release()}",
            "bitsandbytes": _bnb_version()}


def _bnb_version():
    try:
        import bitsandbytes
        return getattr(bitsandbytes, "__version__", "已安装")
    except Exception:
        return "未安装（CPU 环境不需要）"


def main():
    ap = argparse.ArgumentParser(description="实验作业五：SmolVLM-256M QLoRA + DPO")
    ap.add_argument("--mode", default="infer",
                    choices=["make_data", "infer", "sft", "dpo", "eval",
                             "sweep_beta", "sweep_rank", "verify_ref", "all"])
    ap.add_argument("--steps", type=int, default=None, help="覆盖默认训练步数")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO 温度系数 β")
    ap.add_argument("--ref", default="base", choices=["base", "sft"],
                    help="π_ref 取「关适配器的基座」(base, 指南口径) 或「SFT 模型」(sft, DPO 论文口径)")
    ap.add_argument("--rank", type=int, default=8, help="LoRA r")
    ap.add_argument("--img_mode", default="native", choices=["native", "split"],
                    help="图像预处理：native=不切块/最长边 512（64 视觉 token）；"
                         "split=指南默认（放大到 2048 后切 4×4 拼块，1088 视觉 token）")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--out_dir", default=None, help="覆盖 DPO 适配器输出目录")
    ap.add_argument("--dpo_check", default=None, help="如 1.2,0.8,0.9,0.6,0.1")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 8)
    tag = {"make_data": "data", "infer": "infer", "sft": "sft", "dpo": "dpo",
           "eval": "eval", "sweep_beta": "sweep_beta", "sweep_rank": "sweep_rank",
           "verify_ref": "verify_ref", "all": "all"}[args.mode]
    t_start = time.time()

    if args.dpo_check:
        log = Tee(results_path("log_dpo_check.txt"))
        log("===== DPO loss 手算复核 =====")
        res = dpo_check(args.dpo_check, log)
        res["env"] = env_info()
        dump_json("res_dpo_check.json", res)
        log.close()
        return

    extra = ("_" + args.ref) if args.mode == "dpo" else ""
    log = Tee(results_path(f"log_{tag}_{args.img_mode}{extra}.txt"))
    log("=" * 78)
    log(f"实验作业五 | mode={args.mode} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 78)
    E = env_info()
    for k, v in E.items():
        log(f"  {k:<14}= {v}")

    if args.mode == "make_data":
        make_images()
        log.close()
        return

    if args.mode == "infer":
        make_images()
        processor, model, info = load_model("infer", log=log, img_mode=args.img_mode)
        log(json.dumps({k: v for k, v in info.items()}, ensure_ascii=False))
        r = run_questions(processor, model, "基线", log, args.max_new)
        r.update({"env": E, "model_info": info, "seconds": round(r["seconds"], 1)})
        dump_json(f"res_infer_{args.img_mode}.json", r)
        log(f"已写出 {results_path('res_infer_' + args.img_mode + '.json')}")
        log.close()
        return

    if args.mode == "sft":
        make_images()
        steps = args.steps or 60
        processor, model, info = load_model("train", lora_r=args.rank, log=log,
                                           img_mode=args.img_mode)
        log(f"可训练参数 {info['trainable_params']:,} / 基座参数 {info['base_params']:,} "
            f"= {info['pct_of_base']:.4f}%；"
            f"/ 含适配器总参数 {info['total_params']:,} = {info['pct_of_total']:.4f}%")
        log(f"LoRA 命中 {info['n_target_modules']} 个模块，按后缀统计 {info['target_suffix_count']}；"
            f"示例 {info['target_examples']}")
        if info.get("fallback"):
            log("[env] " + info["fallback"])
        from torch.optim import AdamW
        lr = args.lr or 1e-4
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        r = sft_step(model, processor, opt, steps=steps, log=log, lr=lr)
        model.save_pretrained(SFT_DIR)
        log(f"[sft] 适配器已保存到 {SFT_DIR}")
        r.update({"env": E, "model_info": info, "adapter_dir": SFT_DIR})
        dump_json(f"res_sft_{args.img_mode}.json", r)
        log(f"已写出 {results_path('res_sft_' + args.img_mode + '.json')}")
        log.close()
        return

    if args.mode == "verify_ref":
        # 思考题(6)要求的可证伪验证：disable_adapter 是否真的把适配器关掉了？
        # 判据：同一 (图, 问题, 回答) 下
        #   纯基座 seq logprob  vs  「挂适配器后在 disable_adapter 内」的 seq logprob
        #   两者应几乎相等（只差数值路径噪声）；而
        #   纯基座 vs 「适配器生效」的 seq logprob 应出现更大差异。
        # 若 disable_adapter 失效，第一组差值会与第二组同量级 → 判据可证伪。
        make_images()
        from peft import PeftModel
        proc_b, base, info = load_model("infer", log=log, img_mode=args.img_mode)
        dev = next(base.parameters()).device
        pick = ["P14", "P06", "P15", "P05"]
        pairs = [p for p in ALL_PAIRS if p["qid"] in pick]
        log(f"[verify] 选取偏好对 {[p['qid'] for p in pairs]}")
        rows = []
        for p in pairs:
            enc = encode_pair(proc_b, p)
            r0 = {}
            for side in ("chosen", "rejected"):
                e, plen = enc[side]
                r0[side] = float(seq_logprob(base, e.to(dev), plen, need_grad=False).item())
            rows.append({"qid": p["qid"], "base": r0, "enc": enc})
        base.eval()
        wraps = []
        for r in rows:
            for side in ("chosen", "rejected"):
                e, plen = r["enc"][side]
                wraps.append((r["qid"], side, e, plen))
        pm = PeftModel.from_pretrained(base, SFT_DIR, is_trainable=False)
        pm.to(dev)
        pm.eval()
        for r in rows:
            r["active"] = {}
            for side in ("chosen", "rejected"):
                e, plen = r["enc"][side]
                r["active"][side] = float(seq_logprob(pm, e.to(dev), plen, need_grad=False).item())
        with pm.disable_adapter():
            for r in rows:
                r["disabled"] = {}
                for side in ("chosen", "rejected"):
                    e, plen = r["enc"][side]
                    r["disabled"][side] = float(
                        seq_logprob(pm, e.to(dev), plen, need_grad=False).item())
        log("")
        log("  题号  侧       纯基座          disable_adapter  适配器生效      |关闭−基座|  |生效−基座|")
        d_dis, d_act = [], []
        for r in rows:
            for side in ("chosen", "rejected"):
                b0, dd, ac = r["base"][side], r["disabled"][side], r["active"][side]
                d_dis.append(abs(dd - b0))
                d_act.append(abs(ac - b0))
                log("  %-5s %-8s %-15.4f %-16.4f %-13.4f %-11.5f %.5f" % (
                    r["qid"], side, b0, dd, ac, abs(dd - b0), abs(ac - b0)))
            r.pop("enc", None)
        out = {"pairs": rows, "max_abs_diff_disabled_vs_base": max(d_dis),
               "max_abs_diff_active_vs_base": max(d_act),
               "median_abs_diff_disabled": sorted(d_dis)[len(d_dis) // 2],
               "ok": max(d_dis) < 0.05 < max(d_act),
               "env": E, "ref_cache_in_dpo": (load_json("res_dpo_native.json") or {}).get("ref_cache")}
        log("")
        log(f"  disable_adapter 关闭后与纯基座的最大差异 = {max(d_dis):.5f}")
        log(f"  适配器生效时与纯基座的最大差异         = {max(d_act):.5f}")
        log(f"  判据（关闭差异 << 生效差异，且关闭差异 < 0.05）：{'通过' if out['ok'] else '不通过'}")
        dump_json("res_verify_ref.json", out)
        log(f"已写出 {results_path('res_verify_ref.json')}")
        log.close()
        return

    if args.mode == "dpo":
        make_images()
        steps = args.steps or 40
        if not os.path.isdir(SFT_DIR):
            raise SystemExit("先运行 --mode sft 生成 SFT 适配器")
        from peft import PeftModel
        from torch.optim import AdamW
        processor, base, info = load_model("infer", log=log, img_mode=args.img_mode)
        # [C10] 裸基座 + 单层 PeftModel。若写成 PeftModel.from_pretrained(
        #       load_model("train"), SFT_DIR) 会嵌套两层，save_pretrained 出的键名
        #       多一层 base_model.model. 前缀，评测侧静默加载失败、DPO 等效于全零。
        model = PeftModel.from_pretrained(base, SFT_DIR, is_trainable=True)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.config.use_cache = False
        info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log("[dpo] 单层包装后的适配器参数键名样例：%s"
            % [k for k in model.state_dict() if "lora_A" in k][0])
        # 训练前快照 π_ref
        if args.ref == "base":
            with model.disable_adapter():
                ref = compute_ref_logps(model, processor, ALL_PAIRS, log, "base/关适配器")
        else:
            model.eval()
            ref = compute_ref_logps(model, processor, ALL_PAIRS, log, "sft/SFT 适配器")
            model.train()
        lr = args.lr or 5e-5
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        r = dpo_step(model, processor, opt, ref, beta=args.beta, steps=steps,
                     log=log, lr=lr)
        out_dir = args.out_dir or DPO_DIR
        model.save_pretrained(out_dir)
        log(f"[dpo] 适配器已保存到 {out_dir}")
        r.update({"env": E, "model_info": info, "ref": args.ref,
                  "ref_cache": ref, "adapter_dir": args.out_dir or DPO_DIR,
                  "ref_desc": "关闭 LoRA 适配器后的基座（指南口径）" if args.ref == "base"
                  else "DPO 训练前的 SFT 模型（DPO 论文口径）"})
        dump_json(f"res_dpo_{args.img_mode}_{args.ref}.json", r)
        log(f"已写出 {results_path('res_dpo_' + args.img_mode + '_' + args.ref + '.json')}")
        log.close()
        return

    if args.mode == "eval":
        make_images()
        out = {"env": E}
        processor, model, info = load_model("infer", log=log, img_mode=args.img_mode)
        out["baseline"] = run_questions(processor, model, "基线", log, args.max_new)
        out["baseline"]["seconds"] = round(out["baseline"]["seconds"], 1)
        from peft import PeftModel
        if os.path.isdir(SFT_DIR):
            sft_model = PeftModel.from_pretrained(model, SFT_DIR)
            sft_model.to(model.device)
            out["sft"] = run_questions(processor, sft_model, "SFT", log, args.max_new)
            out["sft"]["seconds"] = round(out["sft"]["seconds"], 1)
            if os.path.isdir(DPO_DIR):
                # [C10] DPO_DIR 里保存的已是「SFT 初始化 + DPO 更新」的完整适配器，
                # 因此在裸基座上单层加载即可，不可再叠一层 SFT、也不可用
                # load_adapter+set_adapter（嵌套保存的键名会多一层前缀而静默失效）
                del sft_model
                _, base2, _ = load_model("infer", log=log, img_mode=args.img_mode)
                dpo_model = PeftModel.from_pretrained(base2, DPO_DIR)
                dpo_model.to(base2.device)
                log("[eval] DPO 适配器键名样例：%s" % list(dpo_model.state_dict())[0])
                out["dpo"] = run_questions(processor, dpo_model, "SFT+DPO", log, args.max_new)
                out["dpo"]["seconds"] = round(out["dpo"]["seconds"], 1)
        out["model_info"] = info
        out["total_seconds"] = round(time.time() - t_start, 1)
        dump_json(f"res_eval_{args.img_mode}.json", out)
        log(f"已写出 {results_path('res_eval_' + args.img_mode + '.json')}")
        log.close()
        return

    if args.mode == "sweep_beta":
        make_images()
        if not os.path.isdir(SFT_DIR):
            raise SystemExit("先运行 --mode sft")
        from peft import PeftModel
        from torch.optim import AdamW
        steps = args.steps or 6
        out = {"env": E, "steps": steps, "ref": args.ref, "runs": []}
        for beta in (0.05, 0.1, 0.5):
            processor, base, info = load_model("infer", log=log, img_mode=args.img_mode)
            model = PeftModel.from_pretrained(base, SFT_DIR, is_trainable=True)
            model.to("cpu")
            model.config.use_cache = False
            if args.ref == "base":
                with model.disable_adapter():
                    ref = compute_ref_logps(model, processor, ALL_PAIRS, log, "base")
            else:
                ref = compute_ref_logps(model, processor, ALL_PAIRS, log, "sft")
            opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
            tr = dpo_step(model, processor, opt, ref, beta=beta, steps=steps, log=log)
            d = os.path.join(BASE, f"dpo_adapter_beta{beta}")
            model.save_pretrained(d)
            ev = None
            try:
                proc2, base2, _ = load_model("infer", log=log, img_mode=args.img_mode)
                m2 = PeftModel.from_pretrained(base2, d)
                m2.to(base2.device)
                ev = run_questions(proc2, m2, f"β={beta}", log, args.max_new)
                del m2, base2
            except Exception as e:
                log(f"[sweep_beta] β={beta} 评测失败：{e}")
            out["runs"].append({"beta": beta, "train": tr,
                                "eval": {"by_type": ev["by_type"], "total": ev["total"],
                                         "total_guide": ev["total_guide"]} if ev else None})
        dump_json(f"res_sweep_beta_{args.img_mode}.json", out)
        log(f"已写出 {results_path('res_sweep_beta_' + args.img_mode + '.json')}")
        log.close()
        return

    if args.mode == "sweep_rank":
        make_images()
        from torch.optim import AdamW
        steps = args.steps or 10
        out = {"env": E, "steps": steps, "runs": []}
        for r_ in (4, 8, 16):
            processor, model, info = load_model("train", lora_r=r_, log=log,
                                               img_mode=args.img_mode)
            opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            tr = sft_step(model, processor, opt, steps=steps, log=log, lr=1e-4)
            d = os.path.join(BASE, f"sft_adapter_r{r_}")
            model.save_pretrained(d)
            ev = None
            try:
                proc2, base2, _ = load_model("infer", log=log, img_mode=args.img_mode)
                m2 = PeftModel = None  # noqa
                from peft import PeftModel as PM
                m2 = PM.from_pretrained(base2, d)
                m2.to(base2.device)
                ev = run_questions(proc2, m2, f"r={r_}", log, args.max_new)
                del m2, base2
            except Exception as e:
                log(f"[sweep_rank] r={r_} 评测失败：{e}")
            out["runs"].append({"rank": r_, "trainable_params": info["trainable_params"],
                                "pct_of_total": info["pct_of_total"],
                                "pct_of_base": info["pct_of_base"],
                                "train": tr,
                                "eval": {"by_type": ev["by_type"], "total": ev["total"],
                                         "total_guide": ev["total_guide"]} if ev else None})
        dump_json(f"res_sweep_rank_{args.img_mode}.json", out)
        log(f"已写出 {results_path('res_sweep_rank_' + args.img_mode + '.json')}")
        log.close()
        return


if __name__ == "__main__":
    main()
