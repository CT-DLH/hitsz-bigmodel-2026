# -*- coding: utf-8 -*-
"""
SmolVLM-256M 精准定位的 RSI 递归自进化实验 —— 实验作业六主脚本（纯 CPU）

设计要点（承自教材第 11 章 DeepSeek Harness / Hermes 自进化闭环）：
① 不训练任何检测器：定位「真值」由合成数据的 GT 框直接给出，零训练开销；
② 验证器是解析式的颜色掩码连通块分析（PIL 像素统计，毫秒级），不是学习模型；
③ 智能体侧用 RSI（Recursive Self-Improvement）：
   预测 → 渲染高亮格回灌自查 → 验证器纠错 → 通过验证的样本入池 → 再训练 → 下一轮；
④ 输出空间用 4×4 网格（16 类）而非像素坐标，便于 256M 小模型学习。

常用命令：
  python hw6_rsi_localize.py --mode make_data                 # 生成 55 张合成图 + GT 框
  python hw6_rsi_localize.py --mode make_sft                  # GT 框 → 网格坐标种子数据
  python hw6_rsi_localize.py --mode infer                     # 基线逐题推理 15 题
  python hw6_rsi_localize.py --mode sft --steps 10            # QLoRA 种子 SFT → sft_adapter/
  python hw6_rsi_localize.py --mode eval                      # 三阶段评测 → eval_log.json
  python hw6_rsi_localize.py --mode rsi --rounds 2            # RSI 2 轮 → rsi_memory.json / rsi_summary.json
  python hw6_rsi_localize.py --mode sweep_grid                # 拓展 A：4×4 vs 8×8 对比 → grid_sweep.json

相对指南参考实现的改良（均在报告中 3.4 说明）：
[改良1] 模型本地加载：优先 ./_models/SmolVLM-256M-Instruct，否则经 HF_ENDPOINT=hf-mirror.com 下载到该目录。
[改良2] --img_mode native：关闭 do_image_splitting。lab5 实测 CPU 提速约 6.6×、且理解/定位不掉点。
[改良3] PeftModel 一律「裸基座 + 单层 from_pretrained」加载，防 lab5 踩过的「嵌套 PeftModel 静默失效」。
[改良4] 全流程 JSON 落盘 + UTF-8 日志（results/log_*.txt），数字可追溯、便于填表。
[改良5] parse_cell 泛化到 [1-8]，配合 --grid 8 支撑拓展 A。
[改良6] sft_step 按 save_dir 保存；RSI 各轮 adapter 分开存（rsi_adapter_r{r}），便于分轮分析。
[改良7] 计时：每阶段墙钟耗时写入 JSON，供报告表 1「产出/耗时」填写。
"""
import argparse, json, os, random, re, time, sys
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
LOCAL_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_models", "SmolVLM-256M-Instruct")
IMG_DIR, SFT_DIR, MEM_JSON = "hw6_imgs", "sft_adapter", "rsi_memory.json"
RESULTS_DIR = "results"
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
TRAIN_N, EVAL_N = 40, 15
GRID = 4            # 可被 --grid 覆盖（拓展 A）
IMG = 256
CELL = IMG // GRID
IMG_MODE = "native" # native / split

# ---------- 轻量日志：同时写 stdout 与 UTF-8 文件 ----------
_LOG = None
def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n"); _LOG.flush()

def open_log(tag):
    global _LOG
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _LOG = open(os.path.join(RESULTS_DIR, f"log_{tag}.txt"), "w", encoding="utf-8")

def dump_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ============================================================
# 1. 合成数据集：白底 4 色 × 圆/方，1~4 个物体，绘制时记录 GT 框
# ============================================================
def draw_scene(rnd):
    items = []
    for _ in range(rnd.randint(1, 4)):
        color = rnd.choice(list(COLORS))
        shape = rnd.choice(["圆", "方"])
        s = rnd.randint(34, 54)
        x, y = rnd.randint(8, IMG - s - 8), rnd.randint(8, IMG - s - 8)
        items.append((f"{color}{shape}", [x, y, x + s, y + s]))
    return items

def render(items, path):
    im = Image.new("RGB", (IMG, IMG), "white")
    d = ImageDraw.Draw(im)
    for cls, box in items:
        color, shape = cls[0], cls[1:]
        if shape == "圆":
            d.ellipse(box, outline=COLORS[color], width=4)
        else:
            d.rectangle(box, outline=COLORS[color], width=4)
    im.save(path)

def make_dataset():
    os.makedirs(f"{IMG_DIR}/train", exist_ok=True)
    os.makedirs(f"{IMG_DIR}/eval", exist_ok=True)
    gt = {}
    rnd = random.Random(2026)
    for i in range(TRAIN_N):
        items = draw_scene(rnd)
        render(items, f"{IMG_DIR}/train/t{i:02d}.png")
        gt[f"t{i:02d}"] = items
    rnd = random.Random(42)  # 评测集种子固定，人人可复现
    for i in range(EVAL_N):
        items = draw_scene(rnd)
        render(items, f"{IMG_DIR}/eval/e{i:02d}.png")
        gt[f"e{i:02d}"] = items
    dump_json(gt, f"{IMG_DIR}/gt.json")
    log(f"[data] 已生成 {TRAIN_N} 张训练图 + {EVAL_N} 张评测图 → {IMG_DIR}/")

def load_gt():
    if not os.path.exists(f"{IMG_DIR}/gt.json"):
        make_dataset()
    return json.load(open(f"{IMG_DIR}/gt.json", encoding="utf-8"))

# ============================================================
# 2. 定位编码：GRID×GRID 网格（行-列），把 IMG×IMG 像素压缩成 GRID^2 个类别
# ============================================================
def cell_of(box):
    """GT 框 → 中心所在网格 (row, col)"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(GRID - 1, int(cy // CELL)), min(GRID - 1, int(cx // CELL))

def cell_box(r, c):
    return [c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL]

def cell_text(r, c):
    return f"第{r+1}行第{c+1}列"

def parse_cell(text):
    """从模型输出解析 (row, col)；兼容「第 X 行第 Y 列」「rXcY」「X,Y」等写法（[改良5] 泛化到 [1-GRID]）"""
    t = text.strip()
    hi = GRID
    m = re.search(rf"第\s*([1-{hi}])\s*行\s*第\s*([1-{hi}])\s*列", t)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    m = re.search(rf"r\s*([1-{hi}])\s*c\s*([1-{hi}])", t, re.I)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    m = re.search(rf"([1-{hi}])\s*[,\-，、]\s*([1-{hi}])", t)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    return None

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# ============================================================
# 3. 解析式验证器：颜色掩码连通块分析（零训练，毫秒级）
#    回答两件事：目标颜色像素是否落在指定网格内（验证定位）、共有几个连通块（数数）
# ============================================================
def color_components(img, color):
    im = img.convert("RGB")
    px = im.load()
    ref = COLORS[color]
    W, H = im.size
    def near(p):
        return sum((p[i] - ref[i]) ** 2 for i in range(3)) < 9000
    s = {(x, y) for y in range(H) for x in range(W) if near(px[x, y])}
    comps = []
    while s:
        seed = s.pop()
        stack, pts = [seed], [seed]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q in s:
                        s.discard(q); stack.append(q); pts.append(q)
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        comps.append(([min(xs), min(ys), max(xs) + 1, max(ys) + 1], len(pts)))
    return comps

def color_mask_stats(img, color, cell=None):
    im = img.convert("RGB")
    px = im.load()
    ref = COLORS[color]
    W, H = im.size
    def near(p):
        return sum((p[i] - ref[i]) ** 2 for i in range(3)) < 9000
    pts = [(x, y) for y in range(H) for x in range(W) if near(px[x, y])]
    total = len(pts)
    inside = 0
    if cell is not None:
        cb = cell_box(*cell)
        inside = sum(1 for (x, y) in pts if cb[0] <= x < cb[2] and cb[1] <= y < cb[3])
    return {"total": total, "inside": inside, "components": len(color_components(img, color))}

# ============================================================
# 4. 问题集：loc=网格定位（主攻）/ count=数数 / exist=存在性
# ============================================================
def name_of(cls):
    return f"{cls[0]}色{cls[1:]}形"

def build_questions(gt):
    qs = []
    for i in range(EVAL_N):
        eid = f"e{i:02d}"
        items = gt[eid]
        present = sorted({c for c, _ in items})
        cls = present[i % len(present)]
        box = next(b for c, b in items if c == cls)
        r, c = cell_of(box)
        qs.append(dict(id=f"L{i+1:02d}", type="loc", img=eid, target=cls,
                       q=f"图中的{name_of(cls)}位于第几行第几列？请用「第 X 行第 Y 列」回答（共{GRID}行{GRID}列）。",
                       gold=cell_text(r, c), cell=[r, c], box=box))
        same = sum(1 for cc, _ in items if cc == cls)
        qs.append(dict(id=f"C{i+1:02d}", type="count", img=eid, target=cls,
                       q=f"图中有几个{name_of(cls)}？", gold=f"{same}个", n=same))
        qs.append(dict(id=f"E{i+1:02d}", type="exist", img=eid, target=cls,
                       q=f"图里有没有{name_of(cls)}？", gold="有"))
    locs = [q for q in qs if q["type"] == "loc"][:8]
    cnts = [q for q in qs if q["type"] == "count"][:4]
    exs = [q for q in qs if q["type"] == "exist"][:3]
    return locs + cnts + exs

def judge(q, pred):
    """宽松判分：定位看网格命中（并记录 IoU），数数看数字，存在性看有无（中英双语容错）"""
    p = pred.strip().lower()
    if q["type"] == "loc":
        c = parse_cell(p)
        return c is not None and c == tuple(q["cell"])
    if q["type"] == "count":
        nums = re.findall(r"\d+", p)
        return bool(nums) and int(nums[0]) == q["n"]
    # exist：需明确肯定信号（修 judge 漏洞：纯数字/无否定词不再误判为「有」）
    body = p.replace("有没有", "")
    yes = ("有" in body) or ("yes" in body)
    no = ("没有" in body) or ("无" in body) or ("no" in body) or ("not" in body)
    return (yes and not no) if q["gold"] == "有" else no

def loc_iou(q, pred):
    c = parse_cell(pred)
    return iou(cell_box(*c), q["box"]) if c else 0.0

# ============================================================
# 5. 模型封装：本地加载 / 提示 / 生成 / QLoRA SFT
# ============================================================
def _resolve_model_path():
    if os.path.isdir(LOCAL_MODEL) and os.path.exists(os.path.join(LOCAL_MODEL, "model.safetensors")):
        return LOCAL_MODEL
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    return MODEL_ID

def load_model(attach_lora=False):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    path = _resolve_model_path()
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForVision2Seq.from_pretrained(path, torch_dtype=torch.float32)
    # [改良2] 图像切分策略：native 关闭 splitting（lab5 实测 CPU 提速 ~6.6×）
    try:
        processor.image_processor.do_image_splitting = (IMG_MODE == "split")
    except Exception:
        pass
    if attach_lora:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"[lora] 可训练参数 {trainable:,} / {total:,} = {trainable/total*100:.4f}%")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return processor, model

def build_prompt(processor, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)

@torch.no_grad()
def answer(processor, model, img, q, hint=None, max_new=24):
    """hint 非空时把验证器反馈拼进问题——Harness/RSI 纠错的关键动作"""
    text = q if hint is None else f"{q}\n（校验提示：{hint}）"
    inputs = processor(text=[build_prompt(processor, text)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

def make_sft(gt, out="sft_data.json"):
    """种子 SFT 数据：定位答案直接由 GT 框换算成网格坐标（零人工标注、零外部模型）"""
    data = []
    for q in build_questions(gt):
        ans = q["gold"] if q["type"] in ("loc", "count") else "有"
        data.append({"img": q["img"], "prompt": q["q"], "answer": ans})
    dump_json(data, out)
    log(f"[sft] 种子数据 {len(data)} 条（定位答案由 GT 框换算）→ {out}")
    return data

def sft_step(model, processor, opt, steps=10, log_every=5, data=None, save_dir=SFT_DIR):
    data = data or json.load(open("sft_data.json", encoding="utf-8"))
    dev = next(model.parameters()).device
    model.train()
    losses = []
    for i in range(steps):
        s = data[i % len(data)]
        img = Image.open(f"{IMG_DIR}/eval/{s['img']}.png").convert("RGB")
        full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                         images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            plen = processor(text=[build_prompt(processor, s["prompt"])],
                             images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :plen] = -100  # 只学答案 token
        full["labels"] = labels
        loss = model(**full).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # [改良8] 梯度裁剪，防单样本高 lr 发散
        opt.step(); opt.zero_grad()
        losses.append(round(loss.item(), 4))
        if (i + 1) % log_every == 0:
            log(f"[sft] step{i+1}/{steps} loss={loss.item():.4f}")
    model.save_pretrained(save_dir)
    model.eval()
    log(f"[sft] 适配器已保存到 ./{save_dir}（本轮训练样本 {len(data)} 条）")
    return losses

# ============================================================
# 6. RSI 递归自进化闭环
# ============================================================
def overlay(img, cell, path):
    """把模型预测的网格渲染成高亮方框，作为「可视化工具输出」回灌给模型"""
    im = img.copy()
    d = ImageDraw.Draw(im)
    cb = cell_box(*cell)
    d.rectangle(cb, outline=(255, 0, 255), width=5)
    d.rectangle([cb[0] + 6, cb[1] + 6, cb[2] - 6, cb[3] - 6], outline=(255, 0, 255), width=2)
    im.save(path)
    return path

def verify_cell(img, q, cell):
    """解析式验证器：判断预测网格是否真的定位到目标。
    判据：目标连通块中心落在该格内（跨格物体也成立），或网格覆盖物体 ≥50% 面积。"""
    comps = color_components(img, q["target"][0])
    if not comps:
        return False, "该颜色在图中不存在"
    cb = cell_box(*cell)
    best_overlap = 0.0
    for box, _ in comps:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if cb[0] <= cx < cb[2] and cb[1] <= cy < cb[3]:
            return True, "高亮格确实定位到了目标（目标中心在该格内）"
        x1, y1 = max(cb[0], box[0]), max(cb[1], box[1])
        x2, y2 = min(cb[2], box[2]), min(cb[3], box[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        best_overlap = max(best_overlap, inter / area if area else 0.0)
    if best_overlap >= 0.5:
        return True, f"高亮格覆盖了目标 {best_overlap:.0%} 的面积"
    return False, f"高亮格没有定位到目标（重叠仅 {best_overlap:.0%}），请重新给出位置"

def rsi_round(processor, model, gt, QUESTIONS, round_idx, mem):
    """一轮 RSI：模型作答 → 渲染回灌自查 → 验证器判定 → 收集可用样本（含分项统计，供表 7）"""
    log(f"\n===== RSI 第 {round_idx} 轮 =====")
    new_data = []
    n_ok = n_self = n_tool = 0
    n_self_fixed = n_tool_fixed = 0
    ious = []
    for q in QUESTIONS:
        img = Image.open(f"{IMG_DIR}/eval/{q['img']}.png").convert("RGB")
        pred1 = answer(processor, model, img, q["q"])
        ok1 = judge(q, pred1)
        if q["type"] == "loc":
            ious.append(loc_iou(q, pred1))
        n_ok += ok1
        rec = {"id": q["id"], "type": q["type"], "pred1": pred1, "ok1": ok1}
        if q["type"] == "loc" and not ok1:
            # ① 智能体自查：把高亮格渲染回图像，让模型自己检查
            c = parse_cell(pred1)
            if c:
                ov = overlay(img, c, f"{IMG_DIR}/ov_{q['id']}.png")
                ov_img = Image.open(ov).convert("RGB")
                pred2 = answer(processor, model, ov_img,
                               f"图中紫色高亮框是上一次给出的位置。{q['q']}\n"
                               "若高亮框没有圈住目标，请直接给出正确位置。", max_new=16)
                rec["pred2_self"] = pred2
                if parse_cell(pred2) != c:
                    n_self += 1
                    if judge(q, pred2):
                        n_self_fixed += 1
            # ② 验证器纠错：解析式颜色掩码判定高亮格是否成立
            c = parse_cell(pred1) or (0, 0)
            passed, msg = verify_cell(img, q, c)
            if not passed:
                n_tool += 1
                pred3 = answer(processor, model, img, q["q"], hint=msg)
                rec["hint"] = msg
                rec["pred2_tool"] = pred3
                if judge(q, pred3):
                    n_tool_fixed += 1
                    new_data.append({"img": q["img"], "prompt": q["q"], "answer": pred3})
        mem.append({**rec, "round": round_idx})
    stats = dict(round=round_idx, n_ok=n_ok, n_total=len(QUESTIONS),
                 n_self=n_self, n_tool=n_tool, n_self_fixed=n_self_fixed,
                 n_tool_fixed=n_tool_fixed, new_samples=len(new_data),
                 mean_iou=round(sum(ious) / len(ious), 4) if ious else None)
    log(f"[rsi] 轮 {round_idx}：VLM 答对 {n_ok}/{len(QUESTIONS)}，"
        f"自查改写 {n_self} 次（改对 {n_self_fixed}），验证器纠错 {n_tool} 次（改对 {n_tool_fixed}），"
        f"新增可用样本 {len(new_data)} 条")
    if ious:
        log(f"[rsi] 轮 {round_idx}：定位平均 IoU = {sum(ious)/len(ious):.3f}")
    return new_data, stats

def rsi_loop(rounds=2, steps=10, lr=2e-5):
    """完整 RSI 递归：每轮 = 纠错采样 → 数据池扩充 → 再训练 → 评测"""
    gt = load_gt()
    QUESTIONS = build_questions(gt)
    processor, model = load_model(attach_lora=True)
    from torch.optim import AdamW
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    pool = make_sft(gt)
    mem, summary = [], []
    log(f"[rsi] 初始数据池 {len(pool)} 条，开始 {rounds} 轮递归自进化")
    summary.append(dict(round="初始", pool=len(pool), n_ok=None, n_self=None,
                        n_tool=None, fixed=None, acc=None))
    for r in range(1, rounds + 1):
        t0 = time.time()
        new_data, st = rsi_round(processor, model, gt, QUESTIONS, r, mem)
        if new_data:
            pool = pool + new_data
            dump_json(pool, "sft_data.json")
        save_dir = f"rsi_adapter_r{r}"
        sft_step(model, processor, opt, steps=steps, data=pool, save_dir=save_dir)
        acc = evaluate(processor, model, f"RSI第{r}轮")
        st.update(pool=len(pool), acc=acc, secs=round(time.time() - t0, 1))
        summary.append(st)
        log(f"[rsi] 轮 {r} 结束：数据池 {len(pool)} 条，轮末正确率 {acc}/15，耗时 {st['secs']}s")
    dump_json(mem, MEM_JSON)
    dump_json(summary, "rsi_summary.json")
    log(f"[rsi] 记忆写入 {MEM_JSON}（{len(mem)} 条）；数据池最终 {len(pool)} 条")
    return summary

# ============================================================
# 7. 评测：分阶段 / 纠错后
# ============================================================
def _score_type(processor, model, QUESTIONS, tp):
    qs = [q for q in QUESTIONS if q["type"] == tp]
    ok, ious, detail = 0, [], []
    for q in qs:
        img = Image.open(f"{IMG_DIR}/eval/{q['img']}.png").convert("RGB")
        pred = answer(processor, model, img, q["q"])
        hit = judge(q, pred)
        ok += hit
        if tp == "loc":
            iv = loc_iou(q, pred)
            ious.append(iv)
            log(f"  [{q['id']}] {'√' if hit else '×'} A: {pred[:40]} gold: {q['gold']} IoU={iv:.2f}")
        detail.append(dict(id=q["id"], pred=pred, hit=bool(hit)))
    return ok, len(qs), (sum(ious) / len(ious) if ious else None), detail

def evaluate(processor, model, tag):
    """三阶段评测的单个阶段：返回总正确数，并落盘分题型正确率"""
    gt = load_gt()
    QUESTIONS = build_questions(gt)
    log(f"\n===== {tag} × {len(QUESTIONS)} 题 =====")
    by_type, all_detail = {}, []
    for tp in ("loc", "count", "exist"):
        ok, n, miou, detail = _score_type(processor, model, QUESTIONS, tp)
        by_type[tp] = dict(ok=ok, n=n, mean_iou=(round(miou, 4) if miou is not None else None))
        all_detail += detail
        log(f"  {tag:<12}{tp:<7}正确 {ok}/{n}")
    total = sum(v["ok"] for v in by_type.values())
    n_all = sum(v["n"] for v in by_type.values())
    log(f"  {tag:<12}总计 正确 {total}/{n_all}")
    rec = dict(tag=tag, by_type=by_type, total=total, n=n_all,
               mean_iou=by_type["loc"]["mean_iou"], detail=all_detail)
    fn = os.path.join(RESULTS_DIR, f"eval_{re.sub(r'[^0-9A-Za-z]+', '_', tag)}.json")
    dump_json(rec, fn)
    return total

def eval_with_correction(processor, model, tag):
    """SFT + 验证器纠错：错题把验证器反馈注入提示词重答一次；统计由错变对 / 由对变错"""
    gt = load_gt()
    QUESTIONS = build_questions(gt)
    log(f"\n===== {tag} + 验证器纠错 × {len(QUESTIONS)} 题 =====")
    ok1 = ok2 = 0
    wrong_to_right = right_to_wrong = 0
    detail = []
    for q in QUESTIONS:
        img = Image.open(f"{IMG_DIR}/eval/{q['img']}.png").convert("RGB")
        p1 = answer(processor, model, img, q["q"])
        hit1 = judge(q, p1)
        ok1 += hit1
        if hit1:
            p2, hit2 = p1, True
        else:
            if q["type"] == "loc":
                c = parse_cell(p1) or (0, 0)
                _, msg = verify_cell(img, q, c)
            else:
                st = color_mask_stats(img, q["target"][0])
                msg = (f"图中该颜色共 {st['components']} 处" if q["type"] == "count"
                       else f"图中该颜色像素数 {st['total']}")
            p2 = answer(processor, model, img, q["q"], hint=msg)
            hit2 = judge(q, p2)
        ok2 += hit2
        if (not hit1) and hit2:
            wrong_to_right += 1
        if hit1 and (not hit2):
            right_to_wrong += 1
        log(f"  [{q['id']}·{q['type']}] {'√' if hit2 else '×'} A1: {p1[:30]} → A2: {p2[:30]} gold: {q['gold']}")
        detail.append(dict(id=q["id"], type=q["type"], p1=p1, p2=p2,
                           hit1=bool(hit1), hit2=bool(hit2)))
    log(f"  {tag:<12}纯 VLM 正确 {ok1}/{len(QUESTIONS)} 纠错后正确 {ok2}/{len(QUESTIONS)}"
        f"（由错变对 {wrong_to_right}，由对变错 {right_to_wrong}）")
    dump_json(dict(tag=tag, ok1=ok1, ok2=ok2, wrong_to_right=wrong_to_right,
                   right_to_wrong=right_to_wrong, detail=detail),
              os.path.join(RESULTS_DIR, "eval_correction.json"))
    return ok1, ok2

# ============================================================
# 8. 主入口
# ============================================================
def _load_with_adapter(adapter):
    """[改良3] 裸基座 + 单层 PeftModel 加载，杜绝嵌套 PeftModel 静默失效"""
    from peft import PeftModel
    processor, base = load_model(attach_lora=False)
    model = PeftModel.from_pretrained(base, adapter).to(base.device)
    model.eval()
    return processor, model

def main():
    global GRID, CELL, IMG_MODE
    ap = argparse.ArgumentParser(description="实验作业六：SmolVLM-256M 精准定位的 RSI 递归自进化")
    ap.add_argument("--mode", choices=["make_data", "make_sft", "infer", "sft", "eval", "rsi", "sweep_grid"],
                    default="infer")
    ap.add_argument("--steps", type=int, default=10, help="每轮 SFT 步数（CPU 建议 5~10）")
    ap.add_argument("--rounds", type=int, default=2, help="RSI 递归轮数")
    ap.add_argument("--grid", type=int, default=4, help="网格粒度（拓展 A 设 8）")
    ap.add_argument("--img_mode", choices=["native", "split"], default="native")
    ap.add_argument("--lr", type=float, default=1e-4, help="SFT 学习率（对齐实验5 有效配置：lr=1e-4 + steps=60/4epoch）")
    args = ap.parse_args()
    GRID = args.grid
    CELL = IMG // GRID
    IMG_MODE = args.img_mode
    open_log(f"{args.mode}_g{GRID}")
    log(f"[env] torch={torch.__version__} device={'cuda' if torch.cuda.is_available() else 'cpu'} "
        f"grid={GRID}x{GRID} img_mode={IMG_MODE}")

    gt = load_gt()
    QUESTIONS = build_questions(gt)

    if args.mode == "make_data":
        return
    if args.mode == "make_sft":
        make_sft(gt); return
    if args.mode == "infer":
        t0 = time.time()
        processor, model = load_model()
        rec = []
        for q in QUESTIONS:
            img = Image.open(f"{IMG_DIR}/eval/{q['img']}.png").convert("RGB")
            pred = answer(processor, model, img, q["q"])
            hit = judge(q, pred)
            extra = f" IoU={loc_iou(q, pred):.2f}" if q["type"] == "loc" else ""
            log(f"[{q['id']}·{q['type']}] {'√' if hit else '×'} A: {pred[:50]}（gold: {q['gold']}）{extra}")
            rec.append(dict(id=q["id"], type=q["type"], pred=pred, hit=bool(hit)))
        dump_json(dict(secs=round(time.time() - t0, 1), detail=rec),
                  os.path.join(RESULTS_DIR, "baseline_infer.json"))
        return
    if args.mode == "sft":
        if not os.path.exists("sft_data.json"):
            make_sft(gt)
        processor, model = load_model(attach_lora=True)
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        sft_step(model, processor, opt, steps=args.steps)
        return
    if args.mode == "eval":
        # 三阶段：基线 / SFT / SFT+验证器纠错；若存在 RSI 各轮 adapter 则一并评测
        processor, base = load_model(attach_lora=False)
        evaluate(processor, base, "基线")
        if os.path.exists(SFT_DIR):
            from peft import PeftModel
            m_sft = PeftModel.from_pretrained(base, SFT_DIR).to(base.device); m_sft.eval()
            evaluate(processor, m_sft, "SFT")
            eval_with_correction(processor, m_sft, "SFT")
        for r in (1, 2):
            ad = f"rsi_adapter_r{r}"
            if os.path.exists(ad):
                _, m_r = _load_with_adapter(ad)
                evaluate(processor, m_r, f"RSI第{r}轮")
        return
    if args.mode == "rsi":
        rsi_loop(rounds=args.rounds, steps=args.steps, lr=args.lr); return
    if args.mode == "sweep_grid":
        # 拓展 A：GRID×GRID 网格粒度对比（本进程 GRID 已由 --grid 设置）
        from torch.optim import AdamW
        make_sft(gt, out=f"sft_data_g{GRID}.json")
        processor, model = load_model(attach_lora=True)
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        data = json.load(open(f"sft_data_g{GRID}.json", encoding="utf-8"))
        t0 = time.time()
        losses = sft_step(model, processor, opt, steps=args.steps, data=data, save_dir=f"sft_adapter_g{GRID}")
        ok, n, miou, _ = _score_type(processor, model, QUESTIONS, "loc")
        res = dict(grid=GRID, loc_ok=ok, loc_n=n,
                   loc_acc=round(ok / n, 4), mean_iou=round(miou, 4) if miou else None,
                   losses=losses, secs=round(time.time() - t0, 1))
        dump_json(res, os.path.join(RESULTS_DIR, f"grid_sweep_g{GRID}.json"))
        log(f"[sweep] GRID={GRID} 定位正确 {ok}/{n} 平均IoU={res['mean_iou']}")
        return

if __name__ == "__main__":
    main()
