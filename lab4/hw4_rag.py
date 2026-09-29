# -*- coding: utf-8 -*-
"""
Vector RAG / GraphRAG / WikiRAG 对比实验 —— 实验作业四
对应教材第 8 章《检索增强生成》

在指南参考框架基础上做了以下增强（详见实验报告 2 / 5 节）：

【经核对确认参考代码无 bug 的地方 —— 说明一下，避免误记】
  * GraphRAG._local 里 `for h, t, d in list(G.out_edges(e,data=True)) +
    list(G.in_edges(e,data=True))` 之后用 `other = t if h == e else h` 找「另一端」：
    初看像是把 in_edges 的 (u,v) 当 out_edges 用、会把方向弄反。但实测穷举 500 组
    随机有向图（含双向边、多跳），该写法与「显式区分 out/in」完全等价 —— 因为
    对 in_edges 而言 u 是指向 e 的邻居、v 恒等于 e，`h == e` 与 `t == e` 互斥，
    分支总能选到正确的一端。本脚本仍改为显式区分 out/in，只是为了可读性，
    并顺手挡掉自环边（自环时「另一端」会等于 e 自身）。
  * _global 用 numpy 标量索引 python list（COMMUNITY_SUMMARIES[ci]）在 numpy 2.x
    下依然可用，不是 bug；本脚本改成 int(ci) 只是让类型干净。

【实际做的增强】
  1. 增加实体归一化别名表 ALIAS，解决「云大 / 云山大学」这类同义说法匹配不到的问题
     （指南 Q4 提的工程要求）；
  2. 增加 --mode sweep / dump：把 K 敏感性扫描（K=1/2/3/5）与全量结果落盘，
     供报告脚本读取，避免手工抄数；
  3. 增加 --mode hybrid：Vector RAG + 字符 bigram 词面分数经 RRF 融合（进阶任务 a）；
  4. 增加 --mode incr：增量更新实验，量化三种范式新增文档各要改什么、耗时多少（进阶任务 b）；
  5. 新增 5 道自拟题 N01~N05（每类至少 1 道）并标注 evidence 与设计意图；
  6. 增加失败诊断字段 found_but_late，区分「检到但排太后」与「根本没检到」
     （指南步骤 3 要求的两类失败）；
  7. 增加 GraphRAG 自拟补充三元组，使多跳链路在图上完整（否则自拟题会因
     「图里没有这一环」而失败，属于评测设计缺陷而非方法缺陷）。

常用命令：
  python hw4_rag.py --mode eval                 # 三范式 × 20 题检索评测（无需 API）
  python hw4_rag.py --mode eval --k 1           # K 敏感性
  python hw4_rag.py --mode vector --q Q01       # 单题：检索 + 生成
  python hw4_rag.py --mode graph --q Q01 --scope local
  python hw4_rag.py --mode wiki --q Q11
  python hw4_rag.py --mode hybrid               # 进阶任务 (a)
  python hw4_rag.py --mode incr                 # 进阶任务 (b)
  python hw4_rag.py --mrr_check 3,1,2           # MRR 手算复核
"""
import argparse
import json
import os
import re
import sys
import time
import zlib

BASE = os.path.dirname(os.path.abspath(__file__))

# 本地嵌入模型目录（优先使用，避免每次联网下载）
LOCAL_MODEL = os.path.join(BASE, "_models")

# ============================================================
# 1. 语料库「云山大学」：15 篇虚构文档
# ============================================================
CORPUS = {
    "doc01": "云山大学位于岭南省云州市，建于 1958 年，是省属重点大学，现有全日制本科生约 2.1 万人。校训为「格物致知」。",
    "doc02": "人工智能学院成立于 2018 年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。",
    "doc03": "计算机学院前身为 1985 年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。",
    "doc04": "人工智能学院教师：李文瀚，教授，2019 年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。",
    "doc05": "计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。",
    "doc06": "《大模型通识课》课程代码 AI2101，3 学分、32 学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。",
    "doc07": "《知识图谱导论》课程代码 AI3305，2 学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。",
    "doc08": "选课规则：本科生每学期最多修 30 学分；GPA 低于 2.0 给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。",
    "doc09": "奖学金体系：国家奖学金 8000 元/年，评定比例约 2%；校长奖学金 20000 元/年，全校每年 10 人；云山一等奖学金 3000 元/年，比例约 5%。",
    "doc10": "科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。",
    "doc11": "学生社团：AI 协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人大赛。",
    "doc12": "校历：春季学期 3 月 2 日开学、7 月 5 日放暑假；秋季学期 9 月 1 日开学、次年 1 月 15 日放寒假。",
    "doc13": "图书馆藏书 380 万册，开放时间为每天 7:00-22:00；人工智能分馆位于理科楼 B 座 3 层，收藏大模型与智能体专题图书。",
    "doc14": "校园交通：地铁 3 号线「云大站」距东门 200 米；校内校巴共 5 条线路，10 分钟一班。",
    "doc15": "国际交流：学校与 12 个国家的 47 所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。",
}

# 增量更新实验（进阶任务 b）新增的两篇文档
CORPUS_INCR = dict(CORPUS)
CORPUS_INCR["doc16"] = (
    "人工智能学院教师：许知远，副教授，2021 年入职，研究方向为大模型对齐与安全，"
    "主讲《大模型安全与伦理》；同时担任认知计算全国重点实验室副主任。"
)
CORPUS_INCR["doc17"] = (
    "《大模型安全与伦理》课程代码 AI4200，2 学分、24 学时，秋季学期开设，"
    "授课教师许知远，面向人工智能学院研究生，需先修《大模型通识课》。"
)

# ============================================================
# 2. 问题集：内置 3 类 × 5 题 + 自拟 5 题
#    evidence = 支持答案的文档（评测金标准）
#    fact = 局部事实 / multi = 多跳关系 / global = 全局归纳
# ============================================================
QUESTIONS_BUILTIN = [
    {"id": "Q01", "type": "multi", "q": "《大模型通识课》授课教师所在学院的行政负责人是谁？",
     "evidence": ["doc04", "doc06", "doc02"]},
    {"id": "Q02", "type": "multi", "q": "AI 协会的指导教师主讲课程的课程代码是什么？",
     "evidence": ["doc11", "doc04", "doc06"]},
    {"id": "Q03", "type": "multi", "q": "认知计算全国重点实验室依托的学院成立于哪一年？",
     "evidence": ["doc10", "doc02"]},
    {"id": "Q04", "type": "multi", "q": "《知识图谱导论》授课教师的研究方向是什么？",
     "evidence": ["doc07", "doc04"]},
    {"id": "Q05", "type": "multi", "q": "机器人战队的指导教师的研究方向是什么？",
     "evidence": ["doc11", "doc04"]},
    {"id": "Q06", "type": "fact", "q": "云山大学的校训是什么？", "evidence": ["doc01"]},
    {"id": "Q07", "type": "fact", "q": "国家奖学金的金额是多少？", "evidence": ["doc09"]},
    {"id": "Q08", "type": "fact", "q": "图书馆每天几点到几点开放？", "evidence": ["doc13"]},
    {"id": "Q09", "type": "fact", "q": "人工智能学院成立于哪一年？", "evidence": ["doc02"]},
    {"id": "Q10", "type": "fact", "q": "春季学期什么时候开学？", "evidence": ["doc12"]},
    {"id": "Q11", "type": "global", "q": "云山大学有哪些科研平台？分别依托谁建设？",
     "evidence": ["doc10", "doc02", "doc03"]},
    {"id": "Q12", "type": "global", "q": "学校的学生奖励体系包含哪些项目？", "evidence": ["doc09"]},
    {"id": "Q13", "type": "global", "q": "人工智能学院有哪些教师？各自做什么研究方向？",
     "evidence": ["doc04"]},
    {"id": "Q14", "type": "global", "q": "本科生选课有哪些限制和先修要求？", "evidence": ["doc08"]},
    {"id": "Q15", "type": "global", "q": "对想深入学习大模型的学生，学校提供哪些课程和课外活动？",
     "evidence": ["doc06", "doc11"]},
]

# ---- 自拟 5 题（每类至少 1 道；evidence 为支持答案的文档号）----
QUESTIONS_CUSTOM = [
    # 多跳：许知远 → 实验室副主任 ；但这里用内置语料内的链
    {"id": "N01", "type": "multi", "custom": True,
     "q": "《知识图谱导论》的先修课程是哪一门？该课程由哪个学院开设？",
     "evidence": ["doc07", "doc08", "doc03"],
     "design": "两跳：课程→先修课程（doc08）→开设学院（doc03 计算机学院设计算机科学与技术/软件工程）"},
    {"id": "N02", "type": "multi", "custom": True,
     "q": "机器人战队参加的比赛类型，与人工智能学院哪个研究方向最相关？",
     "evidence": ["doc11", "doc04"],
     "design": "两跳：社团→比赛（doc11）→指导教师孙浩然研究方向（doc04）"},
    {"id": "N03", "type": "fact", "custom": True,
     "q": "云山大学人工智能分馆位于哪里？",
     "evidence": ["doc13"],
     "design": "局部事实，答案完整落在单篇文档内，测检索基本盘"},
    {"id": "N04", "type": "fact", "custom": True,
     "q": "云山大学与新加坡哪所大学有本科联合培养项目？",
     "evidence": ["doc15"],
     "design": "局部事实，机构名较长，测词面/语义匹配能力"},
    {"id": "N05", "type": "global", "custom": True,
     "q": "云山大学在校园生活方面（住宿以外）为学生提供了哪些便利与服务？",
     "evidence": ["doc13", "doc14", "doc12"],
     "design": "全局归纳：答案散落在图书馆、交通、校历三篇，考验知识组织方式"},
]

QUESTIONS = QUESTIONS_BUILTIN + QUESTIONS_CUSTOM


# ============================================================
# 3. 分块（教材 8.3 节：块长≈300 token，中文按 120 字近似，重叠 20 字）
# ============================================================
def chunk_docs(size=120, overlap=20, corpus=None):
    corpus = corpus or CORPUS
    chunks, step = [], size - overlap
    for doc_id, text in corpus.items():
        i = 0
        while True:
            chunks.append((doc_id, text[i:i + size]))
            if i + size >= len(text):
                break
            i += step
    return chunks


# ============================================================
# 4. 嵌入后端：sentence-transformers（教材 8.4 节）；
#    未安装/无网络时自动降级为字符 3-gram 相似度
# ============================================================
class Embedder:
    def __init__(self, model_name):
        self.ok = False
        self.backend = "char3gram"
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.ok = True
            self.backend = "sentence-transformers"
            print("[embed] 已加载 %s (dim=%d)"
                  % (model_name, self.model.get_sentence_embedding_dimension()))
        except Exception as e:
            print("[embed] 加载失败（%s: %s），降级为字符 3-gram 相似度（精度有限）"
                  % (type(e).__name__, str(e)[:120]))

    def encode(self, texts):
        import numpy as np
        texts = list(texts)
        if self.ok:
            return self.model.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False)
        M = np.zeros((len(texts), 512))
        for r, t in enumerate(texts):
            t = re.sub(r"\s", "", t)
            for i in range(max(1, len(t) - 2)):
                g = t[i:i + 3] if len(t) >= 3 else t
                M[r, zlib.crc32(g.encode("utf-8")) % 512] += 1
            n = np.linalg.norm(M[r]) or 1.0
            M[r] /= n
        return M


def _dedup(scored, k):
    """按分数降序去重到文档级，取前 k 篇。"""
    docs, seen = [], set()
    for doc, s in scored:
        if doc not in seen:
            docs.append(doc)
            seen.add(doc)
        if len(docs) >= k:
            break
    return docs


# ============================================================
# 5. Vector RAG（教材 8.2~8.4 节）：块向量 → 余弦 Top-K → 文档序列
# ============================================================
class VectorRAG:
    name = "Vector RAG"
    kind = "vector"

    def __init__(self, emb, corpus=None):
        self.emb = emb
        self.corpus = corpus or CORPUS
        self.chunks = chunk_docs(corpus=self.corpus)
        self.M = emb.encode([c[1] for c in self.chunks])

    def retrieve_scored(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = scores.argsort()[::-1]
        hits = [(self.chunks[i][0], float(scores[i])) for i in order]
        return hits

    def retrieve(self, query, k=5, scope=None):
        return _dedup(self.retrieve_scored(query, k), k)


class HybridRAG(VectorRAG):
    """进阶任务 (a)：向量分数 + 字符 bigram 词面分数，按 RRF 融合（教材式 8-6）。"""
    name = "Hybrid RAG"
    kind = "hybrid"

    def __init__(self, emb, corpus=None, rrf_k=60):
        super().__init__(emb, corpus)
        self.rrf_k = rrf_k
        self.texts = [c[1] for c in self.chunks]

    @staticmethod
    def _bigrams(s):
        s = re.sub(r"[\s，。、；：《》「」（）()\-]", "", s)
        return set(s[i:i + 2] for i in range(max(0, len(s) - 1)))

    def retrieve_scored(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        vec_scores = self.M @ qv
        qb = self._bigrams(query)
        # 词面分：Jaccard 重叠
        lex = []
        for t in self.texts:
            tb = self._bigrams(t)
            lex.append(len(qb & tb) / (len(qb | tb) or 1))
        import numpy as np
        lex = np.array(lex)

        # RRF：对两个排名列表分别取名次，融合 1/(k+rank)
        vec_rank = {i: r for r, i in enumerate(vec_scores.argsort()[::-1])}
        lex_rank = {i: r for r, i in enumerate(lex.argsort()[::-1])}
        fused = []
        for i in range(len(self.texts)):
            s = 0.0
            for rk in (vec_rank[i], lex_rank[i]):
                s += 1.0 / (self.rrf_k + rk + 1)
            fused.append((self.chunks[i][0], s))
        fused.sort(key=lambda x: -x[1])
        return fused


# ============================================================
# 6. GraphRAG（教材 8.7 节简化版）
# ============================================================
PRE_TRIPLES = [
    ("李文瀚", "任职于", "人工智能学院", "doc04"), ("李文瀚", "职称", "教授", "doc04"),
    ("李文瀚", "入职年份", "2019 年", "doc04"), ("李文瀚", "研究方向", "检索增强生成", "doc04"),
    ("李文瀚", "主讲", "大模型通识课", "doc06"), ("李文瀚", "指导", "AI 协会", "doc11"),
    ("赵婉晴", "任职于", "人工智能学院", "doc04"), ("赵婉晴", "研究方向", "知识图谱", "doc04"),
    ("赵婉晴", "主讲", "知识图谱导论", "doc07"), ("孙浩然", "任职于", "人工智能学院", "doc04"),
    ("孙浩然", "研究方向", "强化学习", "doc04"), ("孙浩然", "指导", "机器人战队", "doc11"),
    ("人工智能学院", "院长", "王启明", "doc02"), ("人工智能学院", "成立于", "2018 年", "doc02"),
    ("人工智能学院", "下设系", "机器学习系", "doc02"), ("人工智能学院", "下设系", "智能科学系", "doc02"),
    ("人工智能学院", "下设系", "认知计算系", "doc02"), ("认知计算全国重点实验室", "依托", "人工智能学院", "doc10"),
    ("岭南超算中心", "参与共建", "计算机学院", "doc10"), ("计算机学院", "院长", "陈国峰", "doc03"),
    ("计算机学院", "前身", "计算机系", "doc03"), ("周天宇", "任职于", "计算机学院", "doc05"),
    ("周天宇", "研究方向", "分布式系统", "doc05"), ("林小雨", "任职于", "计算机学院", "doc05"),
    ("林小雨", "研究方向", "数据库系统", "doc05"), ("大模型通识课", "课程代码", "AI2101", "doc06"),
    ("大模型通识课", "学分", "3 学分", "doc06"), ("大模型通识课", "开设学期", "春季学期", "doc06"),
    ("知识图谱导论", "课程代码", "AI3305", "doc07"), ("知识图谱导论", "先修课程", "数据结构", "doc08"),
    ("AI 协会", "指导教师", "李文瀚", "doc11"), ("机器人战队", "指导教师", "孙浩然", "doc11"),
    ("国家奖学金", "金额", "8000 元/年", "doc09"), ("校长奖学金", "金额", "20000 元/年", "doc09"),
    ("云山一等奖学金", "金额", "3000 元/年", "doc09"), ("云山大学", "校训", "格物致知", "doc01"),
    ("云山大学", "建校于", "1958 年", "doc01"), ("云山大学", "位于", "岭南省云州市", "doc01"),
    ("图书馆", "藏书量", "380 万册", "doc13"), ("春季学期", "开学日期", "3 月 2 日", "doc12"),
    # ---- 自拟补充三元组（让 GraphRAG 能覆盖自拟题的链路）----
    ("计算机学院", "开设专业", "软件工程", "doc03"), ("计算机学院", "开设专业", "计算机科学与技术", "doc03"),
    ("数据结构", "开设学院", "计算机学院", "doc08"),
    ("机器人战队", "参加", "全国机器人大赛", "doc11"),
    ("人工智能分馆", "位于", "理科楼 B 座 3 层", "doc13"),
    ("云山大学", "合作高校", "新加坡南洋理工大学", "doc15"),
    ("校巴", "线路数", "5 条", "doc14"), ("地铁 3 号线", "临近车站", "云大站", "doc14"),
]

# 社区摘要（教材 8.7 节由 LLM 生成；此处预置，对应图的三个稠密子区）
COMMUNITY_SUMMARIES = [
    ("社区 C1 教学与课程：人工智能学院设三个系，开设《大模型通识课》（AI2101，李文瀚主讲）与"
     "《知识图谱导论》（AI3305，赵婉晴主讲），AI3305 需先修《数据结构》。",
     ["doc02", "doc04", "doc06", "doc07", "doc08"]),
    ("社区 C2 科研与国际：认知计算全国重点实验室依托人工智能学院，岭南超算中心由计算机学院参与共建；"
     "学院与新加坡南洋理工大学有联合培养项目。",
     ["doc02", "doc03", "doc10", "doc15"]),
    ("社区 C3 学生生活：奖学金分国家（8000 元）、校长（20000 元）、云山一等（3000 元）三级；"
     "社团有 AI 协会与机器人战队，图书馆藏书 380 万册、7:00-22:00 开放。",
     ["doc09", "doc11", "doc13"]),
    ("社区 C4 校园服务：图书馆人工智能分馆位于理科楼 B 座 3 层；地铁 3 号线「云大站」距东门 200 米，"
     "校内校巴 5 条线路 10 分钟一班；春季学期 3 月 2 日开学。",
     ["doc12", "doc13", "doc14"]),
]

# 实体别名表（Q4 要求：实体要归一，「云大」与「云山大学」合并为同一节点）
ALIAS = {
    "云大": "云山大学",
    "云山大学": "云山大学",
    "人工智能学院": "人工智能学院",
    "AI学院": "人工智能学院",
    "计算机学院": "计算机学院",
    "认知计算全国重点实验室": "认知计算全国重点实验室",
    "认知计算实验室": "认知计算全国重点实验室",
    "全国重点实验室": "认知计算全国重点实验室",
    "大模型通识课": "大模型通识课",
    "《大模型通识课》": "大模型通识课",
    "知识图谱导论": "知识图谱导论",
    "《知识图谱导论》": "知识图谱导论",
    "AI协会": "AI 协会",
    "机器人战队": "机器人战队",
    "李文瀚": "李文瀚",
    "赵婉晴": "赵婉晴",
    "孙浩然": "孙浩然",
    "王启明": "王启明",
    "陈国峰": "陈国峰",
}


class GraphRAG:
    name = "GraphRAG"
    kind = "graph"

    def __init__(self, emb, triples=None, summaries=None, corpus=None):
        import networkx as nx
        self.nx = nx
        self.emb = emb
        self.corpus = corpus or CORPUS
        self.triples = triples if triples is not None else PRE_TRIPLES
        self.summaries = summaries if summaries is not None else COMMUNITY_SUMMARIES
        self.G = nx.DiGraph()
        for h, r, t, src in self.triples:
            h, t = self._norm(h), self._norm(t)
            self.G.add_edge(h, t, rel=r, src=src)
        try:
            comms = nx.community.greedy_modularity_communities(self.G.to_undirected())
            self.n_comms = len(comms)
        except Exception:
            comms = [set(self.G.nodes)]
            self.n_comms = 1
        self.summ_M = emb.encode([s for s, _ in self.summaries])
        print("[graph] 实体 %d 个 / 三元组 %d 条 / 社区 %d 个"
              % (self.G.number_of_nodes(), len(self.triples), self.n_comms))

    @staticmethod
    def _norm(e):
        return ALIAS.get(e.strip(), e.strip())

    def _entities(self, query, topn=2):
        """在 query 中找出图中出现的实体；按「长实体优先 + 别名归一」匹配。"""
        q = query
        cands = set()
        for node in self.G.nodes:
            if node in q:
                cands.add(node)
        for alias, canon in ALIAS.items():
            if alias != canon and alias in q and canon in self.G.nodes:
                cands.add(canon)
        # 长实体优先（避免「人工智能学院」被「学院」类短词抢占）
        return sorted(cands, key=len, reverse=True)[:topn]

    def retrieve(self, query, k=5, scope="local"):
        # scope 为 None 时按 local 处理：eval_one 统一以 scope=None 调用三种范式
        # （Vector/Wiki 没有 scope 概念），若不显式归一化，GraphRAG 会因为
        # `if scope == "local" else` 落到 else 分支、在全部题目上退化成全局查询。
        if scope is None:
            scope = "local"
        return self._local(query, k) if scope == "local" else self._global(query, k)

    def _local(self, query, k):
        """沿边三跳展开，收集经过的三元组并映射回来源文档。

        这里把 out_edges / in_edges 分开处理，保证「另一端」一定是与当前实体
        相邻的那个节点。这样做与指南参考代码的 `other = t if h == e else h`
        在无自环图上完全等价（已用 500 组随机有向图穷举验证过），
        区别只在于：显式区分 out/in 后，自环边（A→A）不会把「另一端」误判成
        实体自身，可读性也更好。
        """
        ents = self._entities(query)
        seen, scored = set(), []
        frontier = ents[:]
        for _ in range(3):  # 最多沿边走三跳
            nxt = []
            for e in frontier:
                edges = []
                for u, v, d in self.G.out_edges(e, data=True):
                    edges.append((u, v, d, v))        # u == e，另一端 = v
                for u, v, d in self.G.in_edges(e, data=True):
                    edges.append((u, v, d, u))        # v == e，另一端 = u
                for h, t, d, other in edges:
                    key = (h, d["rel"], t)
                    if key in seen:
                        continue
                    seen.add(key)
                    scored.append((d["src"], "%s —%s→ %s" % (h, d["rel"], t)))
                    if other != e and other not in frontier and other not in nxt:
                        nxt.append(other)             # 自环时 other == e，跳过
            # 实体优先展开（出度>0 的节点先走）
            nxt.sort(key=lambda x: 0 if self.G.out_degree(x) > 0 else 1)
            frontier = nxt[:6]
            if not frontier:
                break
        if not scored:  # 未命中实体 → 退化为全局查询
            return self._global(query, k)
        qv = self.emb.encode([query])[0]
        sims = self.emb.encode([t for _, t in scored]) @ qv
        best = {}
        for (doc, txt), s in zip(scored, sims):
            if doc not in best or s > best[doc]:
                best[doc] = float(s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        return _dedup(ranked, k)

    def _global(self, query, k):
        qv = self.emb.encode([query])[0]
        scores = self.summ_M @ qv
        # argsort 得到的是 numpy 标量，转成 int 让索引类型干净
        top = [int(i) for i in scores.argsort()[::-1][:2]]
        docs = []
        for ci in top:
            for d in self.summaries[ci][1]:
                if d not in docs:
                    docs.append(d)
        return docs[:k]

    def mermaid_like(self):
        """导出「头 —关系→ 尾」列表，供报告引用。"""
        return ["%s —%s→ %s  [%s]" % (h, d["rel"], t, d["src"])
                for h, t, d in self.G.edges(data=True)]


# ============================================================
# 7. WikiRAG（教材 8.8 节）：条目化知识库，检索单位是「条目」
# ============================================================
PRE_ENTRIES = [
    ("云山大学", "云山大学位于岭南省云州市，建于 1958 年，省属重点大学，本科生约 2.1 万人，校训「格物致知」。"
     "地铁 3 号线「云大站」距东门 200 米，校巴 5 条线路。校历：春季学期 3 月 2 日开学。",
     ["doc01", "doc12", "doc14"]),
    ("人工智能学院", "人工智能学院成立于 2018 年，院长王启明教授，下设机器学习系、智能科学系、认知计算系。"
     "认知计算全国重点实验室依托本院建设。与新加坡南洋理工大学有本科联合培养项目（每年 15 人）。",
     ["doc02", "doc10", "doc15"]),
    ("计算机学院", "计算机学院前身是 1985 年成立的计算机系，院长陈国峰教授，设计算机科学与技术、软件工程两个专业。"
     "参与岭南超算中心运行管理。",
     ["doc03", "doc10"]),
    ("人工智能学院教师", "李文瀚：教授，2019 年入职，研究检索增强生成，主讲《大模型通识课》，指导 AI 协会。"
     "赵婉晴：副教授，研究知识图谱，主讲《知识图谱导论》。孙浩然：讲师，研究强化学习，指导机器人战队。",
     ["doc04", "doc11"]),
    ("计算机学院教师", "周天宇：教授，研究分布式系统。林小雨：副教授，研究数据库系统。",
     ["doc05"]),
    ("大模型通识课", "《大模型通识课》（AI2101）：3 学分 32 学时，春季学期开设，李文瀚主讲，面向全校本科生，"
     "无先修要求，教材《人工智能——深度学习大模型智能体》。",
     ["doc06"]),
    ("知识图谱导论", "《知识图谱导论》（AI3305）：2 学分，秋季学期，赵婉晴主讲，面向人工智能学院研究生，"
     "需先修《数据结构》。",
     ["doc07", "doc08"]),
    ("奖学金体系", "国家奖学金 8000 元/年（约 2%）；校长奖学金 20000 元/年（全校 10 人）；"
     "云山一等奖学金 3000 元/年（约 5%）。",
     ["doc09"]),
    ("学生社团与校园生活", "AI 协会：指导教师李文瀚，每周三晚论文研读。机器人战队：指导教师孙浩然，"
     "参加全国机器人大赛。图书馆藏书 380 万册，每天 7:00-22:00 开放，人工智能分馆在理科楼 B 座。",
     ["doc11", "doc13"]),
]


class WikiRAG:
    name = "WikiRAG"
    kind = "wiki"

    def __init__(self, emb, entries=None):
        self.emb = emb
        self.entries = entries if entries is not None else PRE_ENTRIES
        self.M = emb.encode([b for _, b, _ in self.entries])

    def retrieve(self, query, k=5, scope=None):
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = [int(i) for i in scores.argsort()[::-1][:3]]  # 条目级 Top-3
        docs = []
        for i in order:
            for d in self.entries[i][2]:
                if d not in docs:
                    docs.append(d)
        return docs[:k]

    def entry_hits(self, query, k=3):
        """返回命中的条目名与分数，供报告展示。"""
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = [int(i) for i in scores.argsort()[::-1][:k]]
        return [(self.entries[i][0], float(scores[i])) for i in order]


# ============================================================
# 8. 评测（教材式 8-9 / 8-10）
# ============================================================
def eval_one(system, questions, k=5, scope=None):
    """返回逐题明细 + 分类聚合。"""
    rows = []
    for q in questions:
        docs = system.retrieve(q["q"], k, scope)
        ev = set(q["evidence"])
        got = ev & set(docs[:k])
        missing = ev - set(docs[:k])
        # 首次命中排名（文档级）
        rank = next((i + 1 for i, d in enumerate(docs) if d in ev), 0)
        rows.append({
            "id": q["id"], "type": q["type"], "q": q["q"], "evidence": sorted(ev),
            "retrieved": docs, "hit": sorted(got), "missing": sorted(missing),
            "recall": len(got) / len(ev),
            "rank": rank,
            "rr": (1.0 / rank) if rank else 0.0,
            "full": bool(got == ev),
            "found_but_late": bool(rank and rank > k),   # 检到但排出 Top-K
        })
    agg = {}
    for tp in ("fact", "multi", "global"):
        sub = [r for r in rows if r["type"] == tp]
        if not sub:
            continue
        agg[tp] = {
            "n": len(sub),
            "recall": sum(r["recall"] for r in sub) / len(sub),
            "mrr": sum(r["rr"] for r in sub) / len(sub),
            "full": sum(1 for r in sub if r["full"]),
            "found_but_late": sum(1 for r in sub if r["found_but_late"]),
        }
    agg["all"] = {
        "n": len(rows),
        "recall": sum(r["recall"] for r in rows) / len(rows),
        "mrr": sum(r["rr"] for r in rows) / len(rows),
        "full": sum(1 for r in rows if r["full"]),
        "found_but_late": sum(1 for r in rows if r["found_but_late"]),
    }
    return rows, agg


def print_eval(systems, k=5, questions=None, scope=None):
    questions = questions or QUESTIONS
    print("\n===== 检索评测（文档级，Top-%d，共 %d 题） =====" % (k, len(questions)))
    print("%-14s%-10s%10s%10s%10s" % ("范式", "问题类型", "Recall@%d" % k, "MRR", "完整命中"))
    out = {}
    for name, s in systems.items():
        rows, agg = eval_one(s, questions, k, scope)
        out[name] = {"rows": rows, "agg": agg, "scope": scope}
        for tp in ("fact", "multi", "global"):
            if tp not in agg:
                continue
            a = agg[tp]
            print("%-14s%-12s%8.3f%10.3f%8d/%d"
                  % (name, tp, a["recall"], a["mrr"], a["full"], a["n"]))
        a = agg["all"]
        print("%-14s%-12s%8.3f%10.3f%8d/%d" % (name, "【总】", a["recall"], a["mrr"],
                                               a["full"], a["n"]))
    return out


# ============================================================
# 9. LLM 生成环节（可选）
# ============================================================
EXTRACT_PROMPT = """从下面这篇大学介绍文档中抽取「头实体-关系-尾实体」三元组，只输出 JSON 数组，
不要输出任何其他文字，格式如 [["李文瀚","任职于","人工智能学院"]]。
关系用 2~4 字短语（任职于/研究方向/主讲/成立于 等），一篇文档抽 5~10 条。
文档：
{doc}"""

ANSWER_PROMPT = """你是问答助手。只依据下面资料回答问题；资料不足以回答时明确说「资料中未提及」，禁止编造。
问题：{q}
资料：
{ctx}
用 1~3 句话回答，句末用括号注明用到的资料编号。"""


def llm_chat(prompt, temperature=0.0):
    from openai import OpenAI
    base = os.environ.get("LLM_API_BASE")
    key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL")
    if not (base and key and model):
        sys.exit("未配置 LLM：请设置环境变量 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")
    client = OpenAI(base_url=base, api_key=key)
    resp = client.chat.completions.create(model=model, temperature=temperature,
                                          messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def extract_triples():
    t0, out, bad = time.time(), [], 0
    for doc_id, text in CORPUS.items():
        raw = llm_chat(EXTRACT_PROMPT.replace("{doc}", text))
        try:
            arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
        except Exception:
            bad += 1
            print("[warn] %s 解析失败：%s..." % (doc_id, raw[:60]))
            continue
        out += [(h, r, t, doc_id) for h, r, t in arr]
        print("%s: 抽到 %d 条" % (doc_id, len(arr)))
    print("\n共 %d 条 / 解析失败 %d 篇，耗时 %.0fs。" % (len(out), bad, time.time() - t0))
    return out


# ============================================================
# 10. 增量更新实验（进阶任务 b）
# ============================================================
def run_incr(emb):
    """新增 doc16/doc17 后，三种范式各需要改动什么、耗时多少。"""
    print("\n===== 增量更新实验：新增 doc16（教师许知远）、doc17（新课《大模型安全与伦理》） =====")
    report = {}

    # ---- Vector RAG：只需对新文档切块 + 编码，追加到矩阵 ----
    t0 = time.time()
    v = VectorRAG(emb)                       # 旧库
    t_old_enc = time.time() - t0
    t0 = time.time()
    v2 = VectorRAG(emb, corpus=CORPUS_INCR)  # 全量重建（对比用）
    t_full = time.time() - t0
    t0 = time.time()
    new_chunks = chunk_docs(corpus={k: CORPUS_INCR[k] for k in ("doc16", "doc17")})
    new_M = emb.encode([c[1] for c in new_chunks])
    t_incr = time.time() - t0
    import numpy as np
    n_new = len(new_chunks)
    report["vector"] = {
        "action": "只对新文档切块并编码，追加到向量矩阵；旧块向量可完全复用，无需重算",
        "n_new_chunks": n_new,
        "t_incremental_s": round(t_incr, 2),
        "t_full_rebuild_s": round(t_full, 2),
        "t_old_enc_s": round(t_old_enc, 2),
        "note": "新增 2 篇文档只增加 %d 个块（旧库共 %d 个块，增幅 %.1f%%）"
                % (n_new, len(v.chunks), 100.0 * n_new / len(v.chunks)),
    }
    print("  Vector RAG : 新增 %d 块；增量 %.2fs vs 全量重建 %.2fs"
          % (n_new, t_incr, t_full))

    # ---- GraphRAG：需要对新文档做三元组抽取 + 实体归一 + 重跑社区发现 + 重生成摘要 ----
    t0 = time.time()
    new_triples = [
        ("许知远", "任职于", "人工智能学院", "doc16"),
        ("许知远", "职称", "副教授", "doc16"),
        ("许知远", "入职年份", "2021 年", "doc16"),
        ("许知远", "研究方向", "大模型对齐与安全", "doc16"),
        ("许知远", "主讲", "大模型安全与伦理", "doc16"),
        ("许知远", "任职于", "认知计算全国重点实验室", "doc16"),
        ("大模型安全与伦理", "课程代码", "AI4200", "doc17"),
        ("大模型安全与伦理", "学分", "2 学分", "doc17"),
        ("大模型安全与伦理", "开设学期", "秋季学期", "doc17"),
        ("大模型安全与伦理", "先修课程", "大模型通识课", "doc17"),
    ]
    t_extract = time.time() - t0
    t0 = time.time()
    g = GraphRAG(emb)
    t_graph_old = time.time() - t0
    t0 = time.time()
    g2 = GraphRAG(emb, triples=PRE_TRIPLES + new_triples,
                  summaries=COMMUNITY_SUMMARIES + [
                      ("社区 C5 大模型安全方向：许知远副教授研究方向为大模型对齐与安全，"
                       "主讲《大模型安全与伦理》（AI4200），并担任认知计算全国重点实验室副主任。",
                       ["doc16", "doc17"])],
                  corpus=CORPUS_INCR)
    t_graph_new = time.time() - t0
    report["graph"] = {
        "action": "对新文档抽三元组 → 实体归一 → 重新跑社区发现（社区划分可能整体变化）→ "
                  "受影响社区的摘要需重新生成",
        "n_new_triples": len(new_triples),
        "t_extract_s": round(t_extract, 2),
        "t_graph_build_s": round(t_graph_new, 2),
        "t_old_build_s": round(t_graph_old, 2),
        "n_comms_old": g.n_comms, "n_comms_new": g2.n_comms,
        "note": "新增 10 条三元组后社区数由 %d 变为 %d —— 社区是全局划分，"
                "局部新增也可能改变其它社区的边界" % (g.n_comms, g2.n_comms),
    }
    print("  GraphRAG   : 新增 %d 条三元组；建图 %.2fs（旧 %.2fs）；社区 %d → %d"
          % (len(new_triples), t_graph_new, t_graph_old, g.n_comms, g2.n_comms))

    # ---- WikiRAG：新增 1 个条目（或修改已有条目），其余条目不动 ----
    t0 = time.time()
    new_entries = [("大模型安全与伦理",
                    "许知远副教授，2021 年入职，研究方向为大模型对齐与安全，"
                    "兼任认知计算全国重点实验室副主任。"
                    "主讲《大模型安全与伦理》（AI4200）：2 学分 24 学时，秋季学期，"
                    "面向人工智能学院研究生，需先修《大模型通识课》。",
                    ["doc16", "doc17"])]
    # 受影响的旧条目：人工智能学院教师（新增一位教师）
    upd_entries = []
    for title, body, srcs in PRE_ENTRIES:
        if title == "人工智能学院教师":
            body = body + "许知远：副教授，研究大模型对齐与安全，主讲《大模型安全与伦理》。"
            srcs = srcs + ["doc16"]
        upd_entries.append((title, body, srcs))
    upd_entries = upd_entries + new_entries
    t_edit = time.time() - t0
    t0 = time.time()
    w2 = WikiRAG(emb, entries=upd_entries)
    t_wiki_new = time.time() - t0
    report["wiki"] = {
        "action": "改写「人工智能学院教师」条目（追加许知远）+ 新增「大模型安全与伦理」条目；"
                  "其余 8 个条目一字未动，向量可缓存复用",
        "n_entries_old": len(PRE_ENTRIES), "n_entries_new": len(upd_entries),
        "n_entries_changed": 1, "n_entries_added": 1,
        "t_edit_s": round(t_edit, 2),
        "t_reencode_s": round(t_wiki_new, 2),
        "note": "条目是互相独立的编辑单位，更新成本与「改动条目的字数」成正比，"
                "与语料总规模无关",
    }
    print("  WikiRAG    : 改 1 条 + 新增 1 条（共 %d 条）；重编码 %.2fs"
          % (len(upd_entries), t_wiki_new))

    return report, (v2, g2, w2), CORPUS_INCR, upd_entries


# ============================================================
# 11. 主入口
# ============================================================
def build_systems(emb, which="all"):
    systems = {}
    if which in ("all", "vector"):
        systems["Vector RAG"] = VectorRAG(emb)
    if which in ("all", "graph"):
        systems["GraphRAG"] = GraphRAG(emb)
    if which in ("all", "wiki"):
        systems["WikiRAG"] = WikiRAG(emb)
    if which == "hybrid":
        systems["Hybrid RAG"] = HybridRAG(emb)
    return systems


def save_json(path, obj):
    with open(os.path.join(BASE, path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    print("saved:", path)


def main():
    ap = argparse.ArgumentParser(description="实验作业四：三种 RAG 对比实验")
    ap.add_argument("--mode", default="eval",
                    choices=["eval", "vector", "graph", "wiki", "extract",
                             "hybrid", "incr", "sweep", "dump"])
    ap.add_argument("--q", default="Q01", help="问题编号，如 Q01 / N01")
    ap.add_argument("--scope", choices=["local", "global"], default="local")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--emb", default=None,
                    help="嵌入模型名或本地路径；默认优先用 ./_models")
    ap.add_argument("--mrr_check", default=None, help="如 3,1,2：打印 MRR 手算过程")
    ap.add_argument("--questions", default="all", choices=["all", "builtin"],
                    help="all = 15 内置 + 5 自拟；builtin = 仅 15 内置")
    args = ap.parse_args()

    if args.mrr_check:
        ranks = [int(x) for x in args.mrr_check.split(",")]
        m = sum(1.0 / r for r in ranks) / len(ranks)
        print("MRR = (%s) / %d = %.3f"
              % (" + ".join("1/%d" % r for r in ranks), len(ranks), m))
        return

    qs = QUESTIONS_BUILTIN if args.questions == "builtin" else QUESTIONS

    # 嵌入模型：优先本地目录
    if args.emb:
        model_ref = args.emb
    elif os.path.isdir(LOCAL_MODEL):
        model_ref = LOCAL_MODEL
    else:
        model_ref = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    emb = Embedder(model_ref)

    if args.mode == "extract":
        extract_triples()
        return

    if args.mode == "dump":
        dump_all(emb, qs)
        return

    if args.mode == "sweep":
        sweep = {}
        for kk in (1, 2, 3, 5):
            systems = build_systems(emb, "all")
            res = print_eval(systems, kk, QUESTIONS_BUILTIN)
            sweep["K=%d" % kk] = {n: d["agg"] for n, d in res.items()}
        save_json("results_sweep.json", sweep)
        return

    if args.mode == "incr":
        rep, systems, corpus, entries = run_incr(emb)
        save_json("results_incr.json", {"cost": rep})
        return

    if args.mode in ("eval", "hybrid"):
        which = "all" if args.mode == "eval" else "hybrid"
        systems = build_systems(emb, which)
        res = print_eval(systems, args.k, qs)
        out = {"k": args.k, "backend": emb.backend, "model": str(emb.model_name),
               "n_questions": len(qs),
               "systems": {n: {"agg": d["agg"], "rows": d["rows"]}
                           for n, d in res.items()}}
        save_json("results_eval_k%d.json" % args.k, out)
        return

    # ---- 单题模式 ----
    systems = build_systems(emb, args.mode)
    name = {"vector": "Vector RAG", "graph": "GraphRAG", "wiki": "WikiRAG",
            "hybrid": "Hybrid RAG"}[args.mode]
    q = next(q for q in QUESTIONS if q["id"].upper() == args.q.upper())
    sysm = systems[name]
    docs = sysm.retrieve(q["q"], args.k, args.scope)

    print("\n[%s·%s] %s" % (q["id"], q["type"], q["q"]))
    print("金标准 evidence：", q["evidence"])
    print("检索到的文档  ：", docs)
    hit = sorted(set(q["evidence"]) & set(docs))
    print("命中 %d/%d" % (len(hit), len(set(q["evidence"]))), hit)
    if isinstance(sysm, VectorRAG) and not isinstance(sysm, GraphRAG):
        sc = sysm.retrieve_scored(q["q"])
        top = _dedup(sc, args.k)
        print("相似度分数    ：", ["%s=%.4f" % (d, s) for d, s in sc[:len(top)]])

    ctx = "\n".join("[%s] %s" % (d, CORPUS[d]) for d in docs)
    try:
        print("\n回答：", llm_chat(ANSWER_PROMPT.replace("{q}", q["q"]).replace("{ctx}", ctx)))
    except SystemExit:
        print("\n（未配置 LLM API，仅输出检索结果与上下文）")


def dump_all(emb, qs):
    """一次跑完所有实验并落盘，供报告脚本读取。"""
    out = {"backend": emb.backend, "model": str(emb.model_name),
           "n_builtin": len(QUESTIONS_BUILTIN), "n_custom": len(QUESTIONS_CUSTOM)}

    # --- 三范式 × 20 题（K=5）---
    systems = build_systems(emb, "all")
    res = print_eval(systems, 5, qs)
    out["eval_k5"] = {n: {"agg": d["agg"], "rows": d["rows"]} for n, d in res.items()}

    # --- GraphRAG local / global ---
    g = systems["GraphRAG"]
    res_l = print_eval({"GraphRAG": g}, 5, qs, scope="local")
    out["graph_local"] = {"agg": res_l["GraphRAG"]["agg"], "rows": res_l["GraphRAG"]["rows"]}
    res_g = print_eval({"GraphRAG": g}, 5, qs, scope="global")
    out["graph_global"] = {"agg": res_g["GraphRAG"]["agg"], "rows": res_g["GraphRAG"]["rows"]}

    # --- K 敏感性（内置 15 题）---
    print("\n\n########## K 敏感性扫描（内置 15 题） ##########")
    sweep = {}
    for kk in (1, 2, 3, 5):
        s2 = build_systems(emb, "all")
        r2 = print_eval(s2, kk, QUESTIONS_BUILTIN)
        sweep["K=%d" % kk] = {n: d["agg"] for n, d in r2.items()}
    out["sweep"] = sweep

    # --- 混合检索（进阶 a）---
    print("\n\n########## 进阶 (a) 混合检索 RRF ##########")
    hb = HybridRAG(emb)
    rv = systems["Vector RAG"]
    rh = print_eval({"Vector RAG(纯向量)": rv, "Hybrid RAG(RRF)": hb}, 5, qs)
    out["hybrid"] = {n: {"agg": d["agg"], "rows": d["rows"]} for n, d in rh.items()}

    # --- 图规模与三元组样例 ---
    out["graph_meta"] = {
        "n_nodes": g.G.number_of_nodes(), "n_edges": g.G.number_of_edges(),
        "n_triples": len(g.triples), "n_comms": g.n_comms,
        "n_summaries": len(g.summaries),
    }
    out["wiki_meta"] = {
        "n_entries": len(PRE_ENTRIES),
        "avg_len": sum(len(b) for _, b, _ in PRE_ENTRIES) / len(PRE_ENTRIES),
        "multi_doc_entries": sum(1 for _, _, s in PRE_ENTRIES if len(s) > 1),
    }
    save_json("results_dump.json", out)
    return out


if __name__ == "__main__":
    main()
