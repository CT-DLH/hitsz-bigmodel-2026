# -*- coding: utf-8 -*-
"""
实验作业八 · 自制语言模型的知识蒸馏（纯 CPU，单文件、零外部数据）

教师 0.74 M 参数 -> 学生 0.15 M 参数，压缩约 5x；语料由代码现场生成，断网可跑。

与指南参考实现的差异（均在报告中说明）：
  1. teacher.pt 旁另存 teacher_meta.json，复用教师时能恢复训练曲线与耗时，
     否则图 1 只剩两条学生曲线（指南实现的 res["teacher"]["curve"] 为空）。
  2. evaluate() 结束后恢复模型原有 train/eval 状态，不再无条件 model.train()。
  3. 生成样例对教师 / 从零 / 蒸馏三个模型各出一段，用于表 8 对比。
  4. 增加可证伪的对照与探针：标签平滑、乱序教师、α=1、T=100、漏乘 T^2、
     步数扫描、教师规模扫描、自蒸馏、动态量化、真实（唐诗）语料。

常用命令：
  python distill_lm.py --mode quick                     # 30~60 s 自检
  python distill_lm.py --mode teacher --retrain         # 训练教师并缓存
  python distill_lm.py --mode all                       # 从零 + 蒸馏 + 样例
  python distill_lm.py --mode ablate                    # T x alpha 4x4 网格
  python distill_lm.py --mode ext --ext A,B,C,D,E,F      # 拓展加分项
  python distill_lm.py --mode probe                      # 思考题用探针
  python distill_lm.py --mode sweep                      # 步数扫描
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "runs_distill")
RES_DIR = os.path.join(BASE, "results")

# ============================================================
# 1. 语料
# ============================================================
SUBJ = ["小明", "小红", "阿远", "书生", "渔夫", "牧童", "旅人", "诗人",
        "少年", "故人", "归客", "山僧", "孤客", "游子"]
PLACE = ["江边", "山寺", "长亭", "渡口", "竹林", "荒村", "城楼", "野渡",
         "溪桥", "松间", "篱下", "窗前", "陌上", "舟中"]
VERB = ["望见", "听见", "想起", "送去", "寻到", "梦见", "独对", "遥看",
        "静听", "又逢", "恰遇", "空忆"]
OBJ = ["明月", "落花", "孤舟", "归雁", "残灯", "细雨", "青山", "晚钟",
       "白雪", "寒江", "疏星", "暮云", "秋草", "远帆"]
MOOD = ["心中惆怅", "不觉天明", "欣然归去", "泪湿衣襟", "长叹一声",
        "默然无语", "忽忆旧游", "欲语还休", "此意难平", "一笑而别"]
POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "渭城朝雨浥轻尘，客舍青青柳色新。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
]
# 作业三用过的唐诗语料（拓展 F），运行时若找不到则以 POEMS 兜底
TANG_POEMS = [
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


def build_corpus(n_lines=3000, seed=0):
    """合成「仿古诗句」：四个槽位 + 两个收尾模板，下一个字的分布有真实多样性。"""
    rng = random.Random(seed)
    out = []
    for i in range(n_lines):
        s, p, v, o, m = (rng.choice(x) for x in (SUBJ, PLACE, VERB, OBJ, MOOD))
        if i % 3 == 0:
            out.append("%s%s%s%s，%s。" % (s, p, v, o, m))
        elif i % 3 == 1:
            out.append("%s 在%s%s 了一夜%s，%s。" % (s, p, o, rng.choice(["看", "听", "守", "忆"]), m))
        else:
            out.append("%s%s 道：“%s%s。”%s。" % (s, v, o, rng.choice(["正好", "可怜", "可惜", "依旧"]), m))
        out.append("\n")
    out.extend(POEMS)
    return "".join(out)


def build_tang_corpus(n_chars=50000, seed=0):
    """拓展 F：把作业三的唐诗按句重排拼成同量级语料。

    原始 143 句仅约 1.8 k 字，直接训练会让 0.74 M 的教师整篇背下来；
    这里按句随机拼接（1~4 句一行）扩到 5 万字，字符分布与搭配仍是真实唐诗，
    跨句衔接是合成的 —— 这一点在报告里必须写明。
    """
    pool = TANG_POEMS or POEMS
    rng = random.Random(seed)
    out, n = [], 0
    while n < n_chars:
        k = rng.choice([1, 1, 1, 2, 2, 4])
        line = "".join(rng.choice(pool) for _ in range(k))
        out.append(line + "\n")
        n += len(line) + 1
    return "".join(out)


# ============================================================
# 2. 字符级分词
# ============================================================
class Tokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# ============================================================
# 3. 最小 GPT（Pre-Norm）
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        assert C % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.c_attn = nn.Linear(C, 3 * C)
        self.c_proj = nn.Linear(C, C)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg["n_embd"]
        self.fc = nn.Linear(C, 4 * C)
        self.proj = nn.Linear(4 * C, C)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg["n_embd"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["n_embd"])
        self.head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.apply(self._init)
        self.n_param = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, need_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        if need_hidden:
            return logits, loss, x
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=60, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ============================================================
# 4. 批采样 / 评估 / 一致率
# ============================================================
def get_batch(data, bs, block, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, bs, block, iters=20, seed=1234):
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, bs, block, gen)
        _, loss = model(x, y)
        tot += loss.item()
    if was_training:
        model.train()
    return tot / iters


@torch.no_grad()
def agreement(student, teacher, data, bs, block, iters=10, seed=4321):
    """学生与教师在留出集上的 top-1 一致率。"""
    s_was, t_was = student.training, teacher.training
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    hit = tot = 0
    for _ in range(iters):
        x, _ = get_batch(data, bs, block, gen)
        a = student(x)[0].argmax(-1)
        b = teacher(x)[0].argmax(-1)
        hit += (a == b).sum().item()
        tot += b.numel()
    if s_was:
        student.train()
    if t_was:
        teacher.train()
    return hit / tot


@torch.no_grad()
def soft_entropy(teacher, data, bs, block, T_list=(1, 2, 4, 8, 100), iters=8, seed=777):
    """教师软标签在温度 T 下的平均熵（nats）——用来解释 T 过大/过小的失效。"""
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    batches = [get_batch(data, bs, block, gen)[0] for _ in range(iters)]
    out = {}
    for T in T_list:
        tot, n = 0.0, 0
        for x in batches:
            logits = teacher(x)[0].reshape(-1, teacher.head.out_features)
            logp = F.log_softmax(logits / T, dim=-1)
            p = logp.exp()
            ent = -(p * logp).sum(-1)          # (N,)
            tot += ent.sum().item()
            n += ent.numel()
        out[str(T)] = round(tot / n, 4)
    teacher.train()
    return out


# ============================================================
# 5. 训练：硬标签 / 蒸馏 / 特征蒸馏 / 各种对照
# ============================================================
def train(model, data, iters, bs, block, lr=3e-3, seed=0, teacher=None,
          T=4.0, alpha=0.5, feat_weight=0.0, apply_t2=True,
          label_smoothing=0.0, shuffle_teacher=None,
          log_every=50, tag=""):
    """alpha = 0 -> 纯交叉熵；alpha > 0 -> (1-a)*CE + a*T^2*KL(t || s)。

    shuffle_teacher: "fixed" 用同一个词表置换打乱教师输出（有结构但无信息），
                     "random" 每个样本换一次（纯噪声、熵不变）——用于区分
                     「更多信息」与「更强正则」两种解释。
    apply_t2: 置 False 可复现「漏乘 T^2」的错误，用于思考题 2。
    """
    proj = None
    if teacher is not None and feat_weight > 0:
        proj = nn.Linear(model.cfg["n_embd"], teacher.cfg["n_embd"])
    params = list(model.parameters()) + (list(proj.parameters()) if proj is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)
    hist, ce_hist, kd_hist = [], [], []
    V = model.head.out_features

    perm = None
    if teacher is not None and shuffle_teacher == "fixed":
        perm = torch.randperm(V, generator=torch.Generator().manual_seed(99991))

    if teacher is not None:
        teacher.eval()
    model.train()
    t0 = time.time()
    kd = torch.zeros(())
    for it in range(iters):
        x, y = get_batch(data, bs, block, gen)
        logits = model(x)[0]
        ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1),
                             label_smoothing=label_smoothing)
        loss = ce
        if teacher is not None and alpha > 0:
            with torch.no_grad():
                t_logits = teacher(x)[0]
                if shuffle_teacher == "fixed":
                    t_logits = t_logits[..., perm]
                elif shuffle_teacher == "random":
                    N = t_logits.shape[0] * t_logits.shape[1]
                    flat = t_logits.reshape(N, V)
                    idx = torch.rand(N, V, generator=torch.Generator().manual_seed(
                        1000 + it)).argsort(dim=1)
                    t_logits = torch.gather(flat, 1, idx).reshape(t_logits.shape)
            # 关键：先 reshape 成 (B*T, V) 再算 KL。
            # 若直接对 (B,T,V) 调 kl_div(reduction="batchmean")，只除以 B，
            # 序列维被重复累加，KL 会被放大 T 倍（T=96 时数值大得离谱）。
            s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
            t_p = F.softmax(t_logits.reshape(-1, V) / T, dim=-1)
            kd = F.kl_div(s_lp, t_p, reduction="batchmean")
            if apply_t2:
                kd = kd * (T * T)
            loss = (1 - alpha) * ce + alpha * kd
        if proj is not None:
            _, _, hs = model(x, need_hidden=True)
            with torch.no_grad():
                _, _, ht = teacher(x, need_hidden=True)
            loss = loss + feat_weight * F.mse_loss(proj(hs), ht)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        ce_hist.append(ce.item())
        kd_hist.append(float(kd.detach()) if torch.is_tensor(kd) else 0.0)
        if (it + 1) % log_every == 0 or it == 0:
            extra = ""
            if teacher is not None and alpha > 0:
                extra = " ce %.4f kd(T^2) %.4f" % (ce.item(), float(kd))
            print("  [%s] iter %4d/%d loss %.4f%s (%.1fs)"
                  % (tag or "train", it + 1, iters, loss.item(), extra, time.time() - t0))
    tail = max(1, min(10, iters))
    stats = {"ce_last": sum(ce_hist[-tail:]) / tail,
             "kd_last": sum(kd_hist[-tail:]) / tail,
             # KD 的总损失含 alpha*T^2*KL，量纲与纯 CE 不同，不能与从零组同图画；
             # 单独留一条 CE 曲线，供图 1 做同口径对比。
             "ce_curve": ce_hist}
    return hist, time.time() - t0, stats


# ============================================================
# 6. 主流程
# ============================================================
TEACHER_CFG = dict(n_layer=2, n_head=4, n_embd=160, block_size=96, dropout=0.0)
STUDENT_CFG = dict(n_layer=2, n_head=4, n_embd=64, block_size=96, dropout=0.0)


def make_data(block, corpus="synth"):
    if corpus == "tang":
        train_text = build_tang_corpus(50000, seed=0)
        val_text = build_tang_corpus(6000, seed=999)
    else:
        train_text = build_corpus(3000, seed=0)
        val_text = build_corpus(300, seed=999)
    tok = Tokenizer(train_text)
    train_ids = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tok.encode(val_text), dtype=torch.long)
    return tok, train_ids, val_ids


def load_meta(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def dump(obj, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("[save] %s" % p)
    return p


def get_teacher(args, tok, train_ids, val_ids, bs, block, C=None, corpus="synth", tag=None):
    """取教师：有缓存就复用（并从 meta 恢复曲线/耗时），否则训练。"""
    cfg = dict(TEACHER_CFG)
    if C:
        cfg["n_embd"] = C
    if tag is None:
        tag = "teacher" if not C else "teacher_C%d" % C
    path = os.path.join(OUT_DIR, "%s.pt" % tag)
    meta_path = os.path.join(OUT_DIR, "%s_meta.json" % tag)
    torch.manual_seed(args.seed)
    teacher = GPT(tok.vocab_size, cfg)
    meta = load_meta(meta_path)
    reuse = (corpus == "synth" and os.path.exists(path) and not args.retrain
             and args.mode != "teacher" and "val_loss" in meta)
    if reuse:
        teacher.load_state_dict(torch.load(path, map_location="cpu"))
        tv = evaluate(teacher, val_ids, bs, block)
        print("[teacher] 复用 %s val_loss=%.4f ppl=%.2f（训练耗时 %.1fs，来自 meta）"
              % (os.path.basename(path), tv, math.exp(tv), meta.get("time_s", 0.0)))
        return teacher, {"params": teacher.n_param, "val_loss": tv, "ppl": math.exp(tv),
                         "time_s": meta.get("time_s", 0.0), "iters": meta.get("iters", 0),
                         "curve": meta.get("curve", []), "cached": True}
    th, ttime, st = train(teacher, train_ids, args.teacher_iters, bs, block,
                          lr=args.lr, seed=args.seed, log_every=args.log_every, tag=tag)
    tv = evaluate(teacher, val_ids, bs, block)
    info = {"params": teacher.n_param, "val_loss": tv, "ppl": math.exp(tv),
            "time_s": ttime, "iters": args.teacher_iters, "curve": th,
            "ce_last": st["ce_last"], "cfg": cfg, "cached": False}
    torch.save(teacher.state_dict(), path)
    if corpus == "synth":
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    print("[teacher] params %.3f M val_loss=%.4f ppl=%.2f %.1fs"
          % (teacher.n_param / 1e6, tv, math.exp(tv), ttime))
    return teacher, info


def new_student(tok, seed):
    torch.manual_seed(seed + 7)
    return GPT(tok.vocab_size, STUDENT_CFG)


def gen_sample(model, tok, prompt="月", n=60, seed=2026):
    torch.manual_seed(seed)
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        g = model.generate(ids, n, temperature=0.9, top_k=20)
    return tok.decode(g[0].tolist())


@torch.no_grad()
def sample_ce(model, tok, text):
    """把一段文本在某个模型下的 teacher-forced 平均交叉熵算出来。

    用于 7.4：谁的文本更像谁 —— 若蒸馏学生生成的文本在教师下 CE 更低，
    说明它的用词/句式确实更接近教师，而不是只在留出集数字上好看。
    """
    ids = tok.encode(text)
    if len(ids) < 3:
        return float("nan")
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    was = model.training
    model.eval()
    logits = model(x)[0]
    v = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    if was:
        model.train()
    return v


def teach_row(m, hist, tcost, tag, stats, extra):
    row = {"curve": hist, "time_s": tcost, "ce_last": stats["ce_last"]}
    row.update(extra)
    print("[%s] val_loss=%.4f ppl=%.2f agree=%.3f %.1fs"
          % (tag, row["val_loss"], row["ppl"], row["agree"], tcost))
    return row


def run_all(args, tok, train_ids, val_ids, teacher, tinfo, bs, block, tag="all"):
    res = {"data": {"vocab": tok.vocab_size, "train_chars": int(len(train_ids)),
                    "val_chars": int(len(val_ids)), "block": block, "batch": bs,
                    "corpus": "synth"},
           "teacher": tinfo}
    # (1) 从零训练
    s0 = new_student(tok, args.seed)
    h0, t0, st0 = train(s0, train_ids, args.student_iters, bs, block, lr=args.lr,
                        seed=args.seed, log_every=args.log_every, tag="scratch")
    v0 = evaluate(s0, val_ids, bs, block)
    res["scratch"] = teach_row(s0, h0, t0, "scratch", st0, {
        "params": s0.n_param, "val_loss": v0, "ppl": math.exp(v0),
        "agree": agreement(s0, teacher, val_ids, bs, block), "T": None, "alpha": 0.0,
        "ce_curve": st0["ce_curve"]})
    # (2) 蒸馏训练
    s1 = new_student(tok, args.seed)
    h1, t1, st1 = train(s1, train_ids, args.student_iters, bs, block, lr=args.lr,
                        seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
                        feat_weight=args.feat, log_every=args.log_every, tag="kd")
    v1 = evaluate(s1, val_ids, bs, block)
    res["kd"] = teach_row(s1, h1, t1, "kd", st1, {
        "params": s1.n_param, "val_loss": v1, "ppl": math.exp(v1),
        "agree": agreement(s1, teacher, val_ids, bs, block), "T": args.T, "alpha": args.alpha,
        "feat": args.feat, "kd_last": st1["kd_last"], "ce_curve": st1["ce_curve"]})
    res["compare"] = {"delta": v0 - v1, "rel": 100 * (v0 - v1) / v0}
    print("\n[对比] 从零 %.4f -> 蒸馏 %.4f 相对下降 %.1f%%"
          % (v0, v1, res["compare"]["rel"]))
    # (3) 三个模型的生成样例（表 8）
    res["sample"] = {"teacher": gen_sample(teacher, tok),
                     "scratch": gen_sample(s0, tok),
                     "kd": gen_sample(s1, tok)}
    for k, v in res["sample"].items():
        print("[generate:%s] %s" % (k, v.replace("\n", " ")))
    # 3x3：每段文本在三个模型下的平均 CE，判断文风更像谁
    res["sample_score"] = {}
    for name, txt in res["sample"].items():
        res["sample_score"][name] = {
            m: round(sample_ce(model, tok, txt), 4)
            for m, model in (("teacher", teacher), ("scratch", s0), ("kd", s1))}
    for name, row in res["sample_score"].items():
        print("[sample_score:%s] %s" % (name, row))
    dump(res, "results_%s.json" % tag)
    return res


def run_ablate(args, tok, train_ids, teacher, val_ids, bs, block):
    grid_T = [1, 2, 4, 8]
    grid_a = [0.3, 0.5, 0.7, 0.9]
    V = tok.vocab_size
    gen = torch.Generator().manual_seed(2024)
    cache = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(train_ids, bs, block, gen)
            cache.append((x, y, teacher(x)[0]))
    rows = []
    for T in grid_T:
        for a in grid_a:
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, STUDENT_CFG)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
            m.train()
            for it in range(args.abl_iters):
                x, y, tl = cache[it % len(cache)]
                logits = m(x)[0]
                ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
                s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
                t_p = F.softmax(tl.reshape(-1, V) / T, dim=-1)
                kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
                loss = (1 - a) * ce + a * kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            v = evaluate(m, val_ids, bs, block)
            print("  T=%d alpha=%.1f -> val %.4f" % (T, a, v))
            rows.append({"T": T, "alpha": a, "val_loss": v, "ppl": math.exp(v)})
    best = min(rows, key=lambda r: r["val_loss"])
    worst = max(rows, key=lambda r: r["val_loss"])
    print("[ablate] 最优 T=%d alpha=%.1f val=%.4f ｜ 最差 T=%d alpha=%.1f val=%.4f"
          % (best["T"], best["alpha"], best["val_loss"],
             worst["T"], worst["alpha"], worst["val_loss"]))
    out = {"rows": rows, "best": best, "worst": worst, "iters": args.abl_iters}
    dump(out, "results_ablate.json")
    return out


def run_fine(args, tok, train_ids, val_ids, teacher, bs, block):
    """α 加密扫描：4x4 网格太粗，T=8 的 α=0.3 与 0.5 几乎等高，
    需要更密的点才能说清「最优点在哪、是尖峰还是平台」。"""
    grid_T = [4, 8]
    grid_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    V = tok.vocab_size
    gen = torch.Generator().manual_seed(2024)
    cache = []
    teacher.eval()
    with torch.no_grad():
        for _ in range(32):
            x, y = get_batch(train_ids, bs, block, gen)
            cache.append((x, y, teacher(x)[0]))
    rows = []
    for T in grid_T:
        for a in grid_a:
            torch.manual_seed(args.seed + 7)
            m = GPT(tok.vocab_size, STUDENT_CFG)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01)
            m.train()
            for it in range(args.abl_iters):
                x, y, tl = cache[it % len(cache)]
                logits = m(x)[0]
                ce = F.cross_entropy(logits.view(-1, V), y.reshape(-1))
                s_lp = F.log_softmax(logits.reshape(-1, V) / T, dim=-1)
                t_p = F.softmax(tl.reshape(-1, V) / T, dim=-1)
                kd = F.kl_div(s_lp, t_p, reduction="batchmean") * (T * T)
                loss = (1 - a) * ce + a * kd
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            v = evaluate(m, val_ids, bs, block)
            print("  T=%d alpha=%.2f -> val %.4f" % (T, a, v))
            rows.append({"T": T, "alpha": a, "val_loss": v, "ppl": math.exp(v)})
    best = min(rows, key=lambda r: r["val_loss"])
    print("[fine] 最优 T=%d alpha=%.2f val=%.4f" % (best["T"], best["alpha"], best["val_loss"]))
    out = {"rows": rows, "best": best, "iters": args.abl_iters}
    dump(out, "results_fine.json")
    return out


def run_probe(args, tok, train_ids, val_ids, teacher, bs, block, base):
    """思考题 1/2/5 的实测探针。"""
    out = {"base": {"scratch": base["scratch"]["val_loss"], "kd": base["kd"]["val_loss"]}}
    out["entropy"] = soft_entropy(teacher, val_ids, bs, block)
    print("[probe] 教师软标签平均熵 (nats):", out["entropy"],
          "上界 log V =", round(math.log(tok.vocab_size), 4))

    cases = [
        ("alpha1_T4", dict(T=4.0, alpha=1.0), "α=1：完全丢弃硬标签"),
        ("T100", dict(T=100.0, alpha=0.5), "T=100：软标签趋于均匀"),
        ("T8_noT2", dict(T=8.0, alpha=0.5, apply_t2=False), "T=8 漏乘 T²"),
        ("T8_noT2_lr4", dict(T=8.0, alpha=0.5, apply_t2=False, lr=args.lr * 4),
         "T=8 漏乘 T² 但把学习率乘 4"),
        ("T8_t2_lr4", dict(T=8.0, alpha=0.5, lr=args.lr * 4),
         "T=8 正常乘 T²，学习率也乘 4（对照上一行）"),
    ]
    out["cases"] = {}
    for name, kw, note in cases:
        lr = kw.pop("lr", args.lr)
        m = new_student(tok, args.seed)
        h, t, st = train(m, train_ids, args.student_iters, bs, block, lr=lr,
                         seed=args.seed, teacher=teacher, log_every=args.log_every,
                         tag=name, **kw)
        v = evaluate(m, val_ids, bs, block)
        out["cases"][name] = {"val_loss": v, "ppl": math.exp(v), "note": note,
                              "T": kw.get("T"), "alpha": kw.get("alpha"),
                              "apply_t2": kw.get("apply_t2", True), "lr": lr,
                              "time_s": t, "curve": h,
                              "agree": agreement(m, teacher, val_ids, bs, block)}
        print("[probe:%s] val=%.4f ppl=%.2f (%s)" % (name, v, math.exp(v), note))
    dump(out, "results_probe.json")
    return out


def run_sweep(args, tok, train_ids, val_ids, teacher, bs, block, base):
    """4.4 节「收益只在欠拟合区间」的验证：同一初始化、同数据顺序，只改步数。"""
    steps = [60, 120, 200, 400]
    out = {"steps": [], "teacher_val": base["teacher"]["val_loss"]}
    for s in steps:
        if s == args.student_iters:
            out["steps"].append({"steps": s,
                                 "scratch": base["scratch"]["val_loss"],
                                 "kd": base["kd"]["val_loss"], "reused": True})
            continue
        a = new_student(tok, args.seed)
        train(a, train_ids, s, bs, block, lr=args.lr, seed=args.seed,
              log_every=max(50, s // 2), tag="sw-scratch-%d" % s)
        va = evaluate(a, val_ids, bs, block)
        b = new_student(tok, args.seed)
        train(b, train_ids, s, bs, block, lr=args.lr, seed=args.seed,
              teacher=teacher, T=args.T, alpha=args.alpha,
              log_every=max(50, s // 2), tag="sw-kd-%d" % s)
        vb = evaluate(b, val_ids, bs, block)
        out["steps"].append({"steps": s, "scratch": va, "kd": vb, "reused": False})
        print("[sweep] %d 步 scratch %.4f kd %.4f (%.1f%%)"
              % (s, va, vb, 100 * (va - vb) / va))
    for r in out["steps"]:
        r["gain_rel"] = 100 * (r["scratch"] - r["kd"]) / r["scratch"]
    dump(out, "results_sweep.json")
    return out


def run_control(args, tok, train_ids, val_ids, teacher, bs, block, base):
    """思考题 4：把「更多信息」与「更强正则」拆开。

    三个对照都不含教师的真实语义信息，但都提供了「非 one-hot 的软目标」：
      * 标签平滑 eps=0.1：纯正则，无任何教师参与；
      * 固定置换教师：输出仍有结构（同一映射），但与真实下一个字无关；
      * 逐样本置换教师：输出熵不变，但对学生完全不可预测。
    若三者都追不上真教师蒸馏，说明收益主要来自「信息」而非「平滑」。
    """
    ent = soft_entropy(teacher, val_ids, bs, block, T_list=(1, 2, 4, 8))
    out = {"scratch": base["scratch"]["val_loss"], "kd": base["kd"]["val_loss"],
           "agree_base": base["kd"]["agree"], "entropy": ent, "cases": {}}
    cases = [
        ("label_smooth_0.1", dict(label_smoothing=0.1), None,
         "标签平滑 eps=0.1（熵约 0.9 nats，远弱于软标签）"),
        ("label_smooth_0.5", dict(label_smoothing=0.5), None,
         "标签平滑 eps=0.5（熵约 3.6 nats）"),
        ("label_smooth_0.9", dict(label_smoothing=0.9), None,
         "标签平滑 eps=0.9（熵约 5.5 nats，与 T=4 软标签相当）"),
        ("shuffle_fixed", dict(T=args.T, alpha=args.alpha, shuffle_teacher="fixed"), teacher,
         "教师输出做固定词表置换（有结构、无信息）"),
        ("shuffle_random", dict(T=args.T, alpha=args.alpha, shuffle_teacher="random"), teacher,
         "教师输出逐样本置换（熵不变、纯噪声）"),
    ]
    for name, kw, tch, note in cases:
        m = new_student(tok, args.seed)
        h, t, st = train(m, train_ids, args.student_iters, bs, block, lr=args.lr,
                         seed=args.seed, teacher=tch, log_every=args.log_every,
                         tag=name, **kw)
        v = evaluate(m, val_ids, bs, block)
        out["cases"][name] = {"val_loss": v, "ppl": math.exp(v), "note": note,
                              "time_s": t, "curve": h, "ce_curve": st["ce_curve"],
                              "agree": agreement(m, teacher, val_ids, bs, block)}
        print("[control:%s] val=%.4f ppl=%.2f (%s)" % (name, v, math.exp(v), note))
    dump(out, "results_control.json")
    return out


def run_ext(args, tok, train_ids, val_ids, teacher, tinfo, bs, block, base, which):
    """拓展加分项 A/B/C/D/E/F。"""
    ext = {}
    want = set(which)
    V = tok.vocab_size

    # ---- A 特征蒸馏 ----
    if "A" in want:
        m = new_student(tok, args.seed)
        h, t, st = train(m, train_ids, args.student_iters, bs, block, lr=args.lr,
                         seed=args.seed, teacher=teacher, T=args.T, alpha=args.alpha,
                         feat_weight=0.1, log_every=args.log_every, tag="feat0.1")
        v = evaluate(m, val_ids, bs, block)
        ext["A"] = {"feat": 0.1, "val_loss": v, "ppl": math.exp(v), "time_s": t,
                    "agree": agreement(m, teacher, val_ids, bs, block),
                    "base_val": base["kd"]["val_loss"], "curve": h}
        print("[extA] feat=0.1 val=%.4f (baseline kd %.4f)" % (v, base["kd"]["val_loss"]))

    # ---- C 自蒸馏：教师 = 训练更久的同结构学生 ----
    if "C" in want:
        big = new_student(tok, args.seed)
        hb, tb, _ = train(big, train_ids, 400, bs, block, lr=args.lr, seed=args.seed,
                          log_every=100, tag="self-teacher")
        vb = evaluate(big, val_ids, bs, block)
        print("[extC] 学生教师(400 步) val=%.4f params=%.3f M" % (vb, big.n_param / 1e6))
        m = new_student(tok, args.seed)
        h, t, _ = train(m, train_ids, args.student_iters, bs, block, lr=args.lr,
                        seed=args.seed, teacher=big, T=args.T, alpha=args.alpha,
                        log_every=args.log_every, tag="self-kd")
        v = evaluate(m, val_ids, bs, block)
        ext["C"] = {"self_teacher": {"val_loss": vb, "ppl": math.exp(vb),
                                     "iters": 400, "params": big.n_param,
                                     "time_s": tb, "curve": hb},
                    "self_kd": {"val_loss": v, "ppl": math.exp(v), "time_s": t,
                                "agree": agreement(m, big, val_ids, bs, block),
                                "curve": h},
                    "scratch_val": base["scratch"]["val_loss"],
                    "real_kd_val": base["kd"]["val_loss"]}
        print("[extC] 自蒸馏 val=%.4f（从零 %.4f ／ 真教师蒸馏 %.4f）"
              % (v, base["scratch"]["val_loss"], base["kd"]["val_loss"]))

    # ---- D 动态量化 ----
    if "D" in want:
        m = new_student(tok, args.seed)
        train(m, train_ids, args.student_iters, bs, block, lr=args.lr, seed=args.seed,
              teacher=teacher, T=args.T, alpha=args.alpha,
              log_every=args.log_every, tag="quant-base")
        v_fp32 = evaluate(m, val_ids, bs, block)

        def nbytes(mod):
            p = os.path.join(OUT_DIR, "_tmp_state.pt")
            torch.save(mod.state_dict(), p)
            n = os.path.getsize(p)
            os.remove(p)
            return n

        def bench(mod, n=60, batch=1):
            mod.eval()
            x = torch.randint(0, V, (batch, block))
            with torch.no_grad():
                for _ in range(5):
                    mod(x)
                t0 = time.perf_counter()
                for _ in range(n):
                    mod(x)
            return (time.perf_counter() - t0) / n

        size_fp32 = nbytes(m)
        lat_fp32 = bench(m)
        try:
            try:
                from torch.ao.quantization import quantize_dynamic
            except Exception:
                from torch.quantization import quantize_dynamic
            q = quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
            v_int8 = evaluate(q, val_ids, bs, block)
            size_int8 = nbytes(q)
            lat_int8 = bench(q)
            ext["D"] = {"fp32": {"val_loss": v_fp32, "ppl": math.exp(v_fp32),
                                 "bytes": size_fp32, "ms": lat_fp32 * 1e3},
                        "int8": {"val_loss": v_int8, "ppl": math.exp(v_int8),
                                 "bytes": size_int8, "ms": lat_int8 * 1e3},
                        "ratio_bytes": size_fp32 / size_int8,
                        "ratio_latency": lat_fp32 / lat_int8}
            print("[extD] fp32 %.1f KB %.2f ms -> int8 %.1f KB %.2f ms（%.2fx, %.2fx）"
                  % (size_fp32 / 1024, lat_fp32 * 1e3, size_int8 / 1024, lat_int8 * 1e3,
                     size_fp32 / size_int8, lat_fp32 / lat_int8))
        except Exception as e:                      # 量化不是所有 torch 构建都支持
            ext["D"] = {"error": repr(e), "fp32": {"val_loss": v_fp32, "bytes": size_fp32,
                                                   "ms": lat_fp32 * 1e3}}
            print("[extD] 动态量化失败：%r" % e)

    # ---- E 教师规模扫描 ----
    if "E" in want:
        scan = []
        for C in (96, 160, 256):
            if C == 160:
                th, ti = teacher, tinfo
            else:
                th, ti = get_teacher(args, tok, train_ids, val_ids, bs, block, C=C)
            m = new_student(tok, args.seed)
            train(m, train_ids, args.student_iters, bs, block, lr=args.lr, seed=args.seed,
                  teacher=th, T=args.T, alpha=args.alpha,
                  log_every=args.log_every, tag="scan-C%d" % C)
            v = evaluate(m, val_ids, bs, block)
            row = {"C": C, "teacher_params": th.n_param, "teacher_val": ti["val_loss"],
                   "teacher_ppl": math.exp(ti["val_loss"]), "kd_val": v,
                   "kd_ppl": math.exp(v),
                   "gain_rel": 100 * (base["scratch"]["val_loss"] - v) / base["scratch"]["val_loss"],
                   "agree": agreement(m, th, val_ids, bs, block)}
            scan.append(row)
            print("[extE] C=%d 教师 %.3f M val %.4f -> 学生 %.4f（收益 %.1f%%）"
                  % (C, th.n_param / 1e6, ti["val_loss"], v, row["gain_rel"]))
        ext["E"] = {"scan": scan, "scratch_val": base["scratch"]["val_loss"]}

    # ---- F 真实语料 ----
    if "F" in want:
        tok2, tr2, va2 = make_data(block, corpus="tang")
        print("[extF] 唐诗重排语料 vocab=%d train=%d val=%d"
              % (tok2.vocab_size, len(tr2), len(va2)))
        th2, ti2 = get_teacher(args, tok2, tr2, va2, bs, block, corpus="tang",
                               tag="teacher_tang")
        torch.manual_seed(args.seed + 7)
        a = GPT(tok2.vocab_size, STUDENT_CFG)
        train(a, tr2, args.student_iters, bs, block, lr=args.lr, seed=args.seed,
              log_every=args.log_every, tag="tang-scratch")
        va = evaluate(a, va2, bs, block)
        torch.manual_seed(args.seed + 7)
        b = GPT(tok2.vocab_size, STUDENT_CFG)
        train(b, tr2, args.student_iters, bs, block, lr=args.lr, seed=args.seed,
              teacher=th2, T=args.T, alpha=args.alpha,
              log_every=args.log_every, tag="tang-kd")
        vb = evaluate(b, va2, bs, block)
        ext["F"] = {"vocab": tok2.vocab_size, "train_chars": int(len(tr2)),
                    "val_chars": int(len(va2)),
                    "teacher": {"params": th2.n_param, "val_loss": ti2["val_loss"],
                                "ppl": math.exp(ti2["val_loss"]), "time_s": ti2["time_s"]},
                    "scratch": {"val_loss": va, "ppl": math.exp(va)},
                    "kd": {"val_loss": vb, "ppl": math.exp(vb),
                           "agree": agreement(b, th2, va2, bs, block)},
                    "gain_rel": 100 * (va - vb) / va}
        print("[extF] teacher %.4f scratch %.4f kd %.4f (%.1f%%)"
              % (ti2["val_loss"], va, vb, ext["F"]["gain_rel"]))

    dump(ext, "results_ext.json")
    return ext


def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    tok, train_ids, val_ids = make_data(args.block, corpus="synth")
    bs, block = args.batch_size, args.block
    print("[data] vocab=%d train_chars=%d val_chars=%d block=%d batch=%d threads=%d"
          % (tok.vocab_size, len(train_ids), len(val_ids), block, bs,
             torch.get_num_threads()))

    teacher, tinfo = get_teacher(args, tok, train_ids, val_ids, bs, block)
    if args.mode == "teacher":
        dump({"teacher": tinfo}, "results_teacher.json")
        return
    if args.mode == "ablate":
        run_ablate(args, tok, train_ids, teacher, val_ids, bs, block)
        return

    base = run_all(args, tok, train_ids, val_ids, teacher, tinfo, bs, block)
    if args.mode == "all":
        return
    if args.mode == "probe":
        run_probe(args, tok, train_ids, val_ids, teacher, bs, block, base)
    elif args.mode == "sweep":
        run_sweep(args, tok, train_ids, val_ids, teacher, bs, block, base)
    elif args.mode == "control":
        run_control(args, tok, train_ids, val_ids, teacher, bs, block, base)
    elif args.mode == "fine":
        run_fine(args, tok, train_ids, val_ids, teacher, bs, block)
    elif args.mode == "ext":
        run_ext(args, tok, train_ids, val_ids, teacher, tinfo, bs, block, base,
                [x.strip().upper() for x in args.ext.split(",") if x.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["quick", "teacher", "all", "ablate", "ext", "probe",
                             "sweep", "control", "fine"])
    ap.add_argument("--ext", default="A,B,C,D,E,F", help="拓展项编号，逗号分隔")
    ap.add_argument("--teacher-iters", type=int, default=200)
    ap.add_argument("--student-iters", type=int, default=120)
    ap.add_argument("--abl-iters", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--feat", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--retrain", action="store_true", help="忽略已缓存的 teacher.pt")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.mode == "quick":
        global OUT_DIR
        OUT_DIR = os.path.join(OUT_DIR, "quick")
        args.teacher_iters, args.student_iters, args.abl_iters = 30, 20, 10
        args.batch_size, args.log_every = 8, 10
        args.mode = "all"
    run(args)


if __name__ == "__main__":
    main()
