# -*- coding: utf-8 -*-
"""
手搓最小 LLM —— 使用 CPU 训练（实验作业三）
对应教材第 5 章《Transformer 模型》

单文件、零依赖外部数据：内置唐诗语料，字符级 GPT，CPU 数分钟~二十几分钟训完。

用法示例：
    python min_llm.py                              # 基线：训练 2000 步并生成
    python min_llm.py --iters 800                  # 快速跑通
    python min_llm.py --n_layer 1 --n_embd 64      # 对照实验：改模型
    python min_llm.py --no_pos                     # 对照实验：去掉位置编码
    python min_llm.py --temperature 1.5            # 采样温度
    python min_llm.py --sample_only --load model.pt  # 只采样（复用已训练模型）

本文件在指南参考框架基础上做了以下工程化扩展（不改变模型与训练语义）：
    1. 每个 run 把 loss 曲线（JSON）与结果摘要（JSON）落盘，便于后续合成图表与核查；
    2. 生成结果按配置各自命名（generated_{tag}.txt），不再互相覆盖；
    3. --save_model / --load / --sample_only，使采样实验可复用基线模型；
    4. --gen_prompts / --gen_segs 支持「春」「月」多提示词多段落；
    5. generate() 支持 no_repeat_ngram 重复惩罚（进阶任务 b）。
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))

# ============ 1. 内置语料：精选唐诗（可自行替换/追加） ============
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
    "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
    "远芳侵古道，晴翠接荒城。又送王孙去，萋萋满别情。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "鹅，鹅，鹅，曲项向天歌。白毛浮绿水，红掌拨清波。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
    "李白乘舟将欲行，忽闻岸上踏歌声。桃花潭水深千尺，不及汪伦送我情。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
    "折戟沉沙铁未销，自将磨洗认前朝。东风不与周郎便，铜雀春深锁二乔。",
    "巴山楚水凄凉地，二十三年弃置身。怀旧空吟闻笛赋，到乡翻似烂柯人。",
    "沉舟侧畔千帆过，病树前头万木春。今日听君歌一曲，暂凭杯酒长精神。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
    "湖光秋月两相和，潭面无风镜未磨。遥望洞庭山水翠，白银盘里一青螺。",
    "杨柳青青江水平，闻郎江上踏歌声。东边日出西边雨，道是无晴却有晴。",
    "自古逢秋悲寂寥，我言秋日胜春朝。晴空一鹤排云上，便引诗情到碧霄。",
    "前不见古人，后不见来者。念天地之悠悠，独怆然而涕下。",
    "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。",
    "黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。",
    "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。",
    "月黑雁飞高，单于夜遁逃。欲将轻骑逐，大雪满弓刀。",
    "好雨知时节，当春乃发生。随风潜入夜，润物细无声。",
    "晓看红湿处，花重锦官城。野径云俱黑，江船火独明。",
    "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。",
    "烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
    "细草微风岸，危樯独夜舟。星垂平野阔，月涌大江流。",
    "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。",
    "花近高楼伤客心，万方多难此登临。锦江春色来天地，玉垒浮云变古今。",
    "独怜幽草涧边生，上有黄鹂深树鸣。春潮带雨晚来急，野渡无人舟自横。",
    "天街小雨润如酥，草色遥看近却无。最是一年春好处，绝胜烟柳满皇都。",
    "昔人已乘黄鹤去，此地空余黄鹤楼。黄鹤一去不复返，白云千载空悠悠。",
    "晴川历历汉阳树，芳草萋萋鹦鹉洲。日暮乡关何处是，烟波江上使人愁。",
    "客舍青青柳色新，渭城朝雨浥轻尘。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "山光忽西落，池月渐东上。散发乘夕凉，开轩卧闲敞。",
    "荷笠带斜阳，青山独归远。苍苍竹林寺，杳杳钟声晚。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "人闲桂花落，夜静春山空。月出惊山鸟，时鸣春涧中。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "君自故乡来，应知故乡事。来日绮窗前，寒梅著花未。",
    "山中相送罢，日暮掩柴扉。春草明年绿，王孙归不归。",
    "花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。",
    "小时不识月，呼作白玉盘。又疑瑶台镜，飞在青云端。",
    "长安一片月，万户捣衣声。秋风吹不尽，总是玉关情。",
    "弃我去者，昨日之日不可留。乱我心者，今日之日多烦忧。",
    "抽刀断水水更流，举杯消愁愁更愁。人生在世不称意，明朝散发弄扁舟。",
    "千里黄云白日曛，北风吹雁雪纷纷。莫愁前路无知己，天下谁人不识君。",
    "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。",
    "山重水复疑无路，柳暗花明又一村。莫笑农家腊酒浑，丰年留客足鸡豚。",
    "纸上得来终觉浅，绝知此事要躬行。古人学问无遗力，少壮功夫老始成。",
    "死去元知万事空，但悲不见九州同。王师北定中原日，家祭无忘告乃翁。",
    "小荷才露尖尖角，早有蜻蜓立上头。泉眼无声惜细流，树阴照水爱晴柔。",
    "接天莲叶无穷碧，映日荷花别样红。毕竟西湖六月中，风光不与四时同。",
    "胜日寻芳泗水滨，无边光景一时新。等闲识得东风面，万紫千红总是春。",
    "半亩方塘一鉴开，天光云影共徘徊。问渠那得清如许，为有源头活水来。",
    "郁孤台下清江水，中间多少行人泪。西北望长安，可怜无数山。",
    "人生自古谁无死，留取丹心照汗青。辛苦遭逢起一经，干戈寥落四周星。",
    "咬定青山不放松，立根原在破岩中。千磨万击还坚劲，任尔东西南北风。",
    "千锤万凿出深山，烈火焚烧若等闲。粉骨碎身浑不怕，要留清白在人间。",
    "浩荡离愁白日斜，吟鞭东指即天涯。落红不是无情物，化作春泥更护花。",
    "九州生气恃风雷，万马齐喑究可哀。我劝天公重抖擞，不拘一格降人才。",
    "力微任重久神疲，再竭衰庸定不支。苟利国家生死以，岂因祸福避趋之。",
]


def build_corpus(extra_file=None):
    """拼接语料；若同目录下有 corpus_extra.txt（学生自备），自动追加"""
    text = "\n".join(POEMS)
    if extra_file and os.path.exists(extra_file):
        with open(extra_file, "r", encoding="utf-8") as f:
            text = text + "\n" + f.read()
    return text


# ============ 2. 字符级分词器 ============
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ============ 3. 最小 GPT 模型 ============
class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力（教材 5.4 节）"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)     # Q/K/V 一次算出
        self.proj = nn.Linear(n_embd, n_embd)        # 输出投影 W_O
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # 拆多头: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))       # 缩放点积
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # 因果掩码
        att = F.softmax(att, dim=-1)                                  # 归一化
        y = att @ v                                                   # 加权求和
        y = y.transpose(1, 2).contiguous().view(B, T, C)              # 拼接多头
        return self.proj(y)


class Block(nn.Module):
    """Pre-Norm Transformer 块：LN -> 注意力 -> 残差，LN -> FFN -> 残差"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))      # 残差连接
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2,
                 block_size=128, use_pos=True):
        super().__init__()
        self.block_size = block_size
        self.use_pos = use_pos
        self.tok_emb = nn.Embedding(vocab_size, n_embd)               # 词嵌入
        self.pos_emb = nn.Embedding(block_size, n_embd) if use_pos else None  # 可学习位置编码
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                        # 权重共享
        self.apply(self._init_weights)                                # 小方差初始化

    def _init_weights(self, m):
        """GPT 标准初始化：权重 N(0, 0.02)，偏置清零 → 初始 loss ≈ ln V"""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(pos)        # 词向量 + 位置向量
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=1.0, top_k=None,
                 no_repeat_ngram=0):
        """自回归生成：每步预测下一个字符并拼接回输入（教材代码 5-6 思想）

        no_repeat_ngram>0 时启用重复惩罚（进阶任务 b）：
        对已经出现过的 n-gram，强制其后继字符的 logits 置为 -inf。
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]        # 截取上下文窗口
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if no_repeat_ngram and no_repeat_ngram > 0 and idx.size(1) >= no_repeat_ngram:
                seq = idx[0].tolist()
                n = no_repeat_ngram
                prefix = tuple(seq[-n:])
                banned = set()
                for i in range(len(seq) - n):
                    if tuple(seq[i:i + n]) == prefix:
                        banned.add(seq[i + n])
                for b in banned:
                    logits[:, b] = float("-inf")
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits[logits < kth] = float("-inf")    # 只保留 top-k
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)       # 按概率采样
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ============ 4. 数据批生成：随机截取 (block_size+1) 的片段 ============
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])   # 错开一位作标签
    return x.to(device), y.to(device)


# ============ 工具：输出文件名 tag ============
def make_tag(a):
    """按配置生成 tag，与指南约定的曲线文件名一致。"""
    if a.no_pos:
        return "no_pos"
    s = "L%d_E%d" % (a.n_layer, a.n_embd)
    if a.block_size != 128:
        s += "_B%d" % a.block_size
    return s + "_lr%g" % a.lr


def param_breakdown(vocab_size, n_embd, n_layer, block_size, use_pos):
    """参数量分项统计（用于与手算公式逐项核对）"""
    tok = vocab_size * n_embd
    pos = block_size * n_embd if use_pos else 0
    per_block = 0
    ln = 2 * n_embd
    qkv = n_embd * 3 * n_embd + 3 * n_embd
    proj = n_embd * n_embd + n_embd
    mlp = (n_embd * 4 * n_embd + 4 * n_embd) + (4 * n_embd * n_embd + n_embd)
    per_block = ln + qkv + proj + ln + mlp
    return {
        "tok_emb": tok, "pos_emb": pos,
        "per_block": per_block, "blocks_total": per_block * n_layer,
        "ln_f": ln,
        "approx_formula": tok + pos + n_layer * 12 * n_embd ** 2,
        "total_exact": tok + pos + per_block * n_layer + ln,
    }


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--n_embd", type=int, default=128)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--no_pos", action="store_true", help="去掉位置编码（对照实验 5）")
    ap.add_argument("--extra_corpus", type=str, default="corpus_extra.txt",
                    help="自备补充语料文件（可选）")
    # ---- 工程化扩展 ----
    ap.add_argument("--tag", type=str, default=None, help="自定义输出文件名 tag")
    ap.add_argument("--gen_prompts", type=str, default="春,月",
                    help="生成用的提示词，逗号分隔")
    ap.add_argument("--gen_segs", type=int, default=2, help="每个提示词生成几段")
    ap.add_argument("--save_model", action="store_true", help="保存模型权重")
    ap.add_argument("--load", type=str, default=None, help="加载模型权重")
    ap.add_argument("--sample_only", action="store_true",
                    help="只做采样，不训练（需配合 --load）")
    ap.add_argument("--no_repeat_ngram", type=int, default=0,
                    help="重复惩罚：禁止重复的 n-gram（进阶任务 b）")
    ap.add_argument("--no_curve", action="store_true", help="不绘制曲线（加速）")
    return ap


def do_generate(model, tok, prompts, segs, max_new_tokens, temperature, top_k,
                no_repeat_ngram=0):
    """对每个提示词生成 segs 段，返回 [(prompt, text), ...]"""
    outs = []
    for p in prompts:
        for s in range(segs):
            # 每段用不同但可复现的采样种子，避免同一提示词两段完全一样。
            # 注意不能用内置 hash()：它受 PYTHONHASHSEED 影响，跨进程不可复现。
            seed_p = sum(ord(c) for c in p)
            torch.manual_seed(1000 * (seed_p % 97) + s)
            idx = torch.tensor([tok.encode(p)])
            out = model.generate(idx, max_new_tokens, temperature, top_k,
                                 no_repeat_ngram)
            outs.append((p, tok.decode(out[0].tolist())))
    return outs


def main():
    ap = build_parser()
    args = ap.parse_args()
    tag = args.tag or make_tag(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cpu"
    print("PyTorch %s | 设备: %s | 线程数: %d"
          % (torch.__version__, device, torch.get_num_threads()))

    # 数据
    text = build_corpus(os.path.join(BASE, args.extra_corpus))
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print("语料 %d 字符 | 词表大小 V = %d" % (len(text), tok.vocab_size))

    # 模型
    model = MiniGPT(tok.vocab_size, args.n_embd, args.n_head, args.n_layer,
                    args.block_size, use_pos=not args.no_pos)
    n_params = sum(p.numel() for p in model.parameters())
    print("模型参数量: %s (%.3fM) | 配置: n_layer=%d n_head=%d n_embd=%d "
          "block=%d pos=%s"
          % (format(n_params, ","), n_params / 1e6, args.n_layer, args.n_head,
             args.n_embd, args.block_size, "有" if not args.no_pos else "无"))

    pb = param_breakdown(tok.vocab_size, args.n_embd, args.n_layer,
                         args.block_size, not args.no_pos)
    print("参数量分项: 词嵌入=%d 位置编码=%d 每层=%d 全部层=%d ln_f=%d | "
          "公式近似=%d 精确=%d"
          % (pb["tok_emb"], pb["pos_emb"], pb["per_block"], pb["blocks_total"],
             pb["ln_f"], pb["approx_formula"], pb["total_exact"]))
    if pb["total_exact"] != n_params:
        print("!! 分项统计与 sum(p.numel()) 不一致，请核对")
    else:
        print("✓ 分项统计与 sum(p.numel()) 一致")

    if args.load:
        sd = torch.load(os.path.join(BASE, args.load), map_location="cpu")
        model.load_state_dict(sd)
        print("已加载权重: %s" % args.load)

    summary = {
        "tag": tag, "config": vars(args), "vocab_size": tok.vocab_size,
        "corpus_chars": len(text), "n_params": n_params, "param_breakdown": pb,
    }

    # ---------------- 只采样模式 ----------------
    if args.sample_only:
        prompts = [p for p in args.gen_prompts.split(",") if p]
        t0 = time.time()
        outs = do_generate(model, tok, prompts, args.gen_segs,
                           args.max_new_tokens, args.temperature, args.top_k,
                           args.no_repeat_ngram)
        dt = time.time() - t0
        lines = ["=" * 60,
                 "配置 tag=%s | temperature=%s top_k=%s no_repeat_ngram=%s"
                 % (tag, args.temperature, args.top_k, args.no_repeat_ngram),
                 "=" * 60]
        for p, s in outs:
            lines += ["【提示词「%s」】" % p, s, ""]
        txt = "\n".join(lines)
        print(txt)
        fname = "generated_sample_%s.txt" % tag
        with open(os.path.join(BASE, fname), "w", encoding="utf-8") as f:
            f.write(txt)
        summary["generated_file"] = fname
        summary["segments"] = [{"prompt": p, "text": s} for p, s in outs]
        summary["gen_time"] = dt
        with open(os.path.join(BASE, "results_%s.json" % tag), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("已保存 %s 与 results_%s.json（采样耗时 %.1fs）" % (fname, tag, dt))
        return

    # ---------------- 训练 ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    t0 = time.time()
    for step in range(1, args.iters + 1):
        xb, yb = get_batch(data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == 1:
            speed = step / (time.time() - t0)
            eta = (args.iters - step) / speed
            print("step %5d/%d | loss %.4f | %.2f it/s | 预计剩余 %.0fs"
                  % (step, args.iters, loss.item(), speed, eta), flush=True)
    train_time = time.time() - t0
    print("训练完成，耗时 %.2f 分钟 | 最终 loss %.4f (前 100 步均值 %.4f)"
          % (train_time / 60, losses[-1], sum(losses[:100]) / 100))

    # loss 曲线（英文标签避免中文乱码）
    if not args.no_curve:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4))
            plt.plot(losses, linewidth=0.8)
            plt.xlabel("Iteration")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("Training Loss (n_layer=%d, n_embd=%d, block=%d, lr=%g, "
                      "pos=%s)" % (args.n_layer, args.n_embd, args.block_size,
                                   args.lr, "off" if args.no_pos else "on"))
            plt.tight_layout()
            plt.savefig(os.path.join(BASE, "loss_curve_%s.png" % tag), dpi=150)
            plt.close()
            print("已保存 loss_curve_%s.png" % tag)
        except ImportError:
            print("未安装 matplotlib，跳过曲线绘制")

    # 生成
    prompts = [p for p in args.gen_prompts.split(",") if p]
    outs = do_generate(model, tok, prompts, args.gen_segs, args.max_new_tokens,
                       args.temperature, args.top_k, args.no_repeat_ngram)
    lines = ["=" * 60,
             "配置 tag=%s | temperature=%s top_k=%s"
             % (tag, args.temperature, args.top_k),
             "=" * 60]
    for p, s in outs:
        lines += ["【提示词「%s」】" % p, s, ""]
    txt = "\n".join(lines)
    print(txt)

    fname = "generated_%s.txt" % tag
    with open(os.path.join(BASE, fname), "w", encoding="utf-8") as f:
        f.write(txt)
    # 基线配置额外按指南要求输出 generated_base.txt
    if tag == "L2_E128_lr0.001":
        with open(os.path.join(BASE, "generated_base.txt"), "w",
                  encoding="utf-8") as f:
            f.write(txt)
        print("已保存 generated_base.txt（基线）")

    if args.save_model:
        mname = "model_%s.pt" % tag
        torch.save(model.state_dict(), os.path.join(BASE, mname))
        print("已保存模型 %s" % mname)

    # 结果落盘
    summary.update({
        "final_loss": losses[-1],
        "init_loss": losses[0],
        "first100_mean": sum(losses[:100]) / 100,
        "min_loss": min(losses),
        "train_time_sec": train_time,
        "train_time_min": train_time / 60,
        "iters_per_sec": args.iters / train_time,
        "generated_file": fname,
        "segments": [{"prompt": p, "text": s} for p, s in outs],
    })
    with open(os.path.join(BASE, "results_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(BASE, "loss_hist_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(losses, f)
    print("已保存 generated_%s.txt / results_%s.json / loss_hist_%s.json"
          % (tag, tag, tag))


if __name__ == "__main__":
    main()
