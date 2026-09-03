# 重定位方案:从"负面结果"到"机制 + 协议"

状态:改投/答辩准备材料。**不修改任何冻结的实验设定、数据或已报告数值。**
本文档提出的全部改动都属于**叙事与呈现层面**,新增分析均为冻结审计产物的确定性重新聚合。

日期:2026-09-03

---

## 1. 问题诊断:现在的框架为什么脆弱

当前稿件把自己表述为"我们冻结了一批简单诊断指标,发现它们不work"。这个框架有两个可被攻击的软肋:

1. **贡献读起来是空的。** 编辑在桌拒阶段扫标题和摘要,看到的是 "no stable incremental value"。没有新方法、没有 SOTA、没有机制解释——只有一个否定观察。
2. **规则像稻草人。** 审稿人会问:你的图组合里有三个异配专用架构,却用一个同配阈值去决定要不要用图,这不是注定失败吗?

**关键在于:第 2 条质疑是对的——但它恰恰是这篇论文最有价值的发现,只是现在没被写出来。**
把它从"审稿人的反驳"变成"论文的主论点",整篇文章的性质就变了。

---

## 2. 新的贡献叙事(三支柱)

### 支柱一(新的主论点):结构性错配,而非阈值调优失败

不是"这条规则没调好",而是**任何单一同配统计量在面对异配感知组合时都必然失败**,因为二者在结构上反相关。

支撑证据(全部为冻结审计的确定性重聚合,见 §3):

- Combined 规则 **95.1%** 的 regret 集中在 4 个低同配数据集
- 这 4 个数据集的 $h_1$ 全部 < 0.55,故 Combined 在这 40 个单元上 **100% 不选图**
- 而这 40 个单元中 **88%(35/40)** 由异配专用架构(H2GCN/LINKX/GPR-GNN)获胜
- 全基准 110 个单元中 **86%(95/110)** 由异配专用架构获胜

**核心论断:** 同配统计量度量的是"GCN 式单一模型是否受益于图",而决策目标是"一个含异配专用架构的组合是否受益于图"。这两个量在低同配区间**方向相反**。因此这不是可以通过调阈值修复的失败,而是**被测量对象与决策目标之间的错配**。

> 这条论断的价值在于:它对所有"用单一结构统计量做图模型选择"的工作都成立,而不只是对本文测的这几条规则成立。

### 支柱二(工程/协议):可复用的决策评估基础设施

这是本仓库最扎实的部分,目前在稿件里被降级成了"可复现性附录"。它应当成为**一等贡献**:

- 决策时信息边界的**机器可执行**定义(标签作用域、禁止输入,evaluator 拒绝违规记录)
- 冻结的九策略评估套件:含 trivial 基线(always-graph/always-MLP/random)、abstention + coverage、oracle regret、以及 validation-selection 上界参照
- 不可变的逐单元记录:source commit、配置摘要、数据 checksum、split id、四次 trial、选中 checkpoint、环境 provenance
- test-once 会计 + 数据集级(非 seed 级)推断
- 确定性审计:`audit_route_a_claims.py` 使正文数值与冻结 JSON 绑定;`git hash-object` 可逐字节复现两个汇总产物
- 路径脱敏的公开 artifact 发布流程,双 SHA-256(原始 + 脱敏)清单

**表述方式:** "我们交付的不只是一个负面观察,而是一套让'图诊断指标能否用于部署决策'这一问题**可被证伪**的评估协议与审计基础设施。任何后续诊断指标或学习型选择器都可以直接在这套协议上被检验。"

### 支柱三(既有):机制—决策分离

保留现有的保度随机化干预。它证明诊断失败 ≠ 图无用。位置不变,但从"主结果之一"降为"支撑论证",给支柱一让出篇幅。

---

## 3. 新增分析(确定性重聚合,无需重跑实验)

### 表 A:Regret 集中度

| 数据集 | $h_1$ | Combined regret (pp) | 占总 regret |
|---|---:|---:|---:|
| Roman-empire | 0.047 | 23.66 | 28.8% |
| Squirrel | 0.221 | 22.06 | 26.9% |
| Amazon-ratings | 0.380 | 16.45 | 20.1% |
| Chameleon | 0.236 | 15.88 | 19.4% |
| **小计** | | | **95.1%** |
| Wisconsin | 0.172 | 1.89 | 2.3% |
| Cornell | 0.114 | 1.75 | 2.1% |
| Actor | 0.219 | 0.34 | 0.4% |
| CiteSeer / Coauthor-CS / Cora / PubMed | ≥0.73 | 0.00 | 0.0% |

### 表 B:获胜图架构分布(错配的直接证据)

| 数据集 | $h_1$ | 获胜图架构 | 异配专用 | Combined 选图? |
|---|---:|---|---:|---|
| Roman-empire | 0.047 | LINKX 9, H2GCN 1 | 10/10 | 否 |
| Cornell | 0.114 | GPR-GNN 4, H2GCN 4, SAGE 2 | 8/10 | 否 |
| Wisconsin | 0.172 | GPR-GNN 7, LINKX 1, SAGE 1, H2GCN 1 | 9/10 | 否 |
| Actor | 0.219 | H2GCN 9, SAGE 1 | 9/10 | 否 |
| Squirrel | 0.221 | LINKX 10 | 10/10 | 否 |
| Chameleon | 0.236 | LINKX 5, GAT 4, SAGE 1 | 5/10 | 否 |
| Amazon-ratings | 0.380 | LINKX 10 | 10/10 | 否 |
| CiteSeer | 0.732 | H2GCN 5, GCN 3, SAGE 2 | 5/10 | 是 |
| PubMed | 0.804 | H2GCN 10 | 10/10 | 是 |
| Cora | 0.808 | GPR-GNN 8, H2GCN 1, GAT 1 | 9/10 | 是 |
| Coauthor-CS | 0.810 | GPR-GNN 10 | 10/10 | 是 |
| **合计** | | | **95/110 = 86%** | |

**读法:** 异配专用架构在**整个基准**上占优(86%),包括全部四个高同配数据集。
同配阈值把"选图"限制在高同配区间,而组合的优势**并不集中在**那里——这正是错配。

生成命令见 §7。

---

## 4. 零功率披露(必须加)

### 事实

符号翻转检验的双侧 p 值下限为 $2/2^{k}$,$k$ = 配对差异非零的数据集数。

- 4 个数据集(Cora/CiteSeer/PubMed/Coauthor-CS)两策略 regret 均恒为 0 → $k \le 7$
- always-graph 实测 $k=7$,7 个差异全部同号(最极端模式)→ raw $p = 2/2^7 = 0.015625$,**恰为理论下限**
- Holm 首步 $\times 8$ → 0.125
- 若要在 $\alpha=0.05$ 下拒绝,需 $2^{1-k}\cdot 8 \le 0.05$,即 $k \ge 9$

**结论:预注册的确认性检验在设计上功率为零——即使效应完美也无法拒绝。**

### 为什么主结论不受影响

预注册的论文级决策规则是:"combined 只有在**降低** regret 时才能主张增量价值"。
实测 7.46 pp vs always-graph 0.26 pp,方向明确失败。**该结论是描述性的,不依赖任何显著性检验。**
检验只影响次级论断("always-graph 显著更优"),而论文已正确地拒绝做此论断。

### 待插入段落(§4.3 末尾或 §7.3)

```latex
\paragraph{Attainable resolution of the predeclared test.}
The dataset-level sign-flip test has a two-sided $p$-value floor of $2^{1-k}$,
where $k$ counts datasets with a nonzero paired difference. Four datasets
(Cora, CiteSeer, PubMed, Coauthor-CS) yield identically zero regret under both
policies, so $k\leq 7$ here; the always-graph comparison attains $k=7$ with all
seven differences sharing one sign, which is the most extreme attainable
pattern and gives the floor value $p=0.015625$. Rejection at $\alpha=0.05$
after Holm correction across eight comparisons would require $k\geq 9$. The
predeclared confirmatory test therefore could not have rejected under any
realization of the data, and we report it as a preregistered limitation rather
than as evidence of a small effect. This does not affect the primary
conclusion: the predeclared decision rule grants the combined diagnostic an
incremental-value claim only if it \emph{reduces} regret, and its 7.46-point
full-set regret against 0.26 points for always-graph is a descriptive
comparison that requires no significance test. Future protocols at this scale
should either preregister fewer comparisons or treat dataset-level regret
differences with an estimation-only analysis.
```

**为什么主动写:** 审稿人自己算出来 → "作者没意识到检验没功率";作者主动给出下限推导 → 严谨性加分。

---

## 5. 具体改写草案

### 5.1 标题

现:`A Prospective Evaluation of Simple Graph Diagnostics for Graph-vs-MLP Model Selection`
问题:承诺的是"一次评估",不是一个发现。

候选:
- **A**(推荐)`Why Homophily Statistics Cannot Select Graph Models: A Frozen Decision Benchmark`
- **B** `Structural Mismatch Between Graph Diagnostics and Heterophily-Aware Model Portfolios`
- **C** `Measuring the Wrong Quantity: Graph Diagnostics as Deployment Decisions`

A 的优点:`Why ... Cannot` 明确承诺机制解释;`Frozen Decision Benchmark` 保留工程贡献。

### 5.2 摘要(改写要点)

结构从「测试 → 失败」改为「失败 → **为什么** → 交付了什么」:

1. 第 1–2 句不变(操作性决策的动机)
2. **提前给出机制**:诊断失败并非阈值问题,而是被测量量与决策目标的结构性错配——同配统计量刻画单一 GCN 式模型的收益,而组合中 86% 的获胜模型是异配专用架构
3. 再给数字(55.5 / 68.2 / 7.46 vs 80.9 / 0.26 / 91.8 / 0.22)以及 95.1% 的集中度
4. 保留保度随机化(45.5 / 29.9 / 17.3)
5. **末句改为交付物**,而非否定式收尾:一套冻结的、可审计的决策评估协议,后续诊断指标与学习型选择器可直接在其上被检验

避免以 "do not reliably choose" 收尾。收尾应是贡献,不是否定。

### 5.3 引言贡献列表

```latex
\begin{enumerate}
    \item \textbf{A structural explanation for diagnostic failure.} We show that
    the failure of homophily-based selection is not a threshold-tuning problem
    but a mismatch between the measured quantity and the decision target: a
    homophily statistic characterizes when a GCN-style model benefits from the
    graph, whereas the decision concerns a portfolio in which
    heterophily-aware architectures win 95 of 110 units. The two quantities are
    anti-correlated over the low-homophily regime, and 95.1\% of the combined
    rule's regret concentrates on the four datasets where this mismatch is
    sharpest.

    \item \textbf{An auditable decision-evaluation protocol.} We deliver a
    frozen, machine-executable specification of decision-time information
    boundaries, a nine-policy suite with matched trivial baselines, explicit
    abstention and coverage, oracle-portfolio regret, test-once accounting,
    dataset-level inference, and immutable per-unit provenance. Manuscript
    values are bound to the frozen artifacts by a deterministic claim audit,
    and both published summaries are byte-reproducible. Any subsequent
    diagnostic or learned selector can be tested against the same protocol.

    \item \textbf{A mechanism--decision separation.} A degree-preserving paired
    intervention shows that diagnostic failure cannot be reduced to graph
    irrelevance, while explicitly not isolating homophily as a single causal
    mechanism.
\end{enumerate}
```

### 5.4 新增 §7.1(讨论,置于最前)

```latex
\subsection{The Diagnostic and the Portfolio Measure Different Quantities}

The negative result admits a structural explanation rather than a
threshold-tuning one. Table~\ref{tab:winning_architectures} reports the
validation-selected graph architecture on every unit. Heterophily-aware
architectures (H2GCN, LINKX, GPR-GNN) win 95 of 110 units, including all ten
units on PubMed and Coauthor-CS, whose training-label homophily exceeds 0.80.
The graph portfolio is therefore not a proxy for GCN-style message passing; it
is a portfolio whose realized advantage is largely supplied by architectures
designed to remain effective when one-hop homophily is low.

An edge-homophily threshold selects the graph action exactly on the
high-homophily datasets. The portfolio's advantage, however, is largest
elsewhere: the four datasets carrying 95.1\% of the combined rule's regret
(Roman-empire, Squirrel, Amazon-ratings, Chameleon) all have $h_1<0.55$, so the
rule declines the graph action on all 40 of those units, and 35 of those 40
units are won by a heterophily-aware architecture with mean gaps between 15.9
and 23.7 percentage points.

The diagnostic and the decision target are thus anti-correlated over precisely
the regime that dominates the loss. No threshold on $h_1$ can repair this,
because the statistic characterizes a property of one-hop label mixing, while
the action concerns whether \emph{any} member of a heterophily-aware portfolio
beats a tuned MLP. This generalizes beyond the rules evaluated here: any
selection rule built on a scalar summary of one-hop homophily inherits the same
mismatch whenever the candidate portfolio contains heterophily-aware models.
```

### 5.5 其余章节

- §2 相关工作:保持不变(已经写得很好且定位诚实)
- §5 干预:保持不变,但在 §7 中降为支撑论证
- §6 固定度分析:**考虑移入附录**。它是全文最弱的新颖性环节(自己的 novelty audit 已指出与 Wang et al. ICML 2024 重叠),且占约 4 页。移入附录可为支柱一与支柱二腾出篇幅。数学本身正确(已逐行验证),不必删除。

---

## 6. 投稿策略

| 目标 | 需要的改动 | 说明 |
|---|---|---|
| **TMLR**(首选) | §5 全套 | novelty 明确**不是**接收标准;两条标准为"claims 是否被证据支持"与"是否有读者感兴趣"。本文最大弱点在此不扣分,最大优点(证据链)正中第一条 |
| LoG | §5 全套 + 压缩至会议长度 | 图方向专门会议,对负面结果/可复现性友好 |
| Neural Processing Letters | §5 全套 | 已发表 Katsman 2026《When is the Graph Actually Necessary?》,同一问题空间,fit 已被证明 |
| Neurocomputing 重投 | **不可行**,除非拒稿信明确邀请 | 桌拒不受理申诉 |

**注意事项**

- 本仓库公开(含 `CITATION.cff` 与 GitHub 用户名),TMLR 审稿期间匿名性会被削弱。TMLR 允许 preprint,不违规,但需知情。
- 若走 Elsevier Article Transfer Service,先看清推荐的目标期刊档次再决定。

---

## 7. 复现本文档中的新增数字

```bash
python3 - <<'PY'
import json, collections
d = json.load(open('results/diagnostic/route_a_prospective_v2/analysis/diagnostic_audit.json'))
HET = {"H2GCN", "LINKX", "GPR-GNN"}
by = collections.defaultdict(list)
for u in d['units']:
    by[u['dataset']].append(u)
het_all = sum(1 for u in d['units'] if u['selected_graph'] in HET)
print(f"heterophily-aware wins: {het_all}/{len(d['units'])}")
for ds in sorted(by, key=lambda x: sum(u['homophily'] for u in by[x]) / len(by[x])):
    us = by[ds]
    h = sum(u['homophily'] for u in us) / len(us)
    c = collections.Counter(u['selected_graph'] for u in us)
    het = sum(v for k, v in c.items() if k in HET)
    print(f"{ds:16s} h1={h:.3f} het={het}/{len(us)} {dict(c)}")
PY
```

Regret 集中度取自 `sections/09_reproducibility_appendix.tex` 表 A.2(其本身由
`scripts/summarize_reviewer_appendix.py` 生成)。

零功率下限:$2/2^{k}$,$k=7$ → 0.015625;Holm 首步 $\times 8$ → 0.125;
需 $k\ge 9$ 方可在 $\alpha=0.05$ 下拒绝。
