# 本轮修订与交付记录 — 2026-09-28

## 完成范围与版本

已完成当前 Applied Intelligence 稿件的证据组织修订、期刊补充包清理、
本机构建与解压复核。标题、原 frozen benchmark、实验条件和原始结果保留。
未启动训练，未提交期刊，未提交或推送 Git，未把新稳健性结果混入旧均值。

- 分支：`publication/applied-intelligence-2026-09-28`。
- 基础 HEAD：`ef48699cdfdce8137b3fc560ff838977751e6e52`。
- 实际修订基线是上一轮 **41 页工作区版本**，不是 HEAD 本身。
  其完整保留包 SHA-256 为
  `1c7573f3285536e7e5738b2a5ca19cd32bc2c8f4067e12f52f4b9a28beb62f39`。
- 当前源码含上一轮未提交修复与本轮修订。包内 source manifest 标识实际字节；
  不把基础提交当成本轮已提交源码。

## 改动、原因与 claim 边界

下面给出代表性原文和改后表述；逐行完整差异见
[本轮 diff](evidence-focused-revision-2026-09-28.diff)。此前修复详见
[上一轮报告](review-revision-report-2026-09-28.md)，其文件与证据均保留。

| 文件 | 原文或原组织 | 修改后 | 原因与 claim 影响 |
|---|---|---|---|
| `main_applied_intelligence.tex` 摘要 | “The combined diagnostic and its declared fallback incur 7.46 …”；摘要未交代损失集中度 | “Under this configuration …”；增加 “Four datasets account for 95.1% …” 及两类基线/数据版本问题 | 将 headline 限定到既有配置，说明量级依赖；数值不变，收窄解释 |
| 同上摘要 | “Decisions that abstain … contribute 75.9% …” | “Abstaining units … account for 75.9% of recorded loss.” | 说明记录分组贡献，不将占比写成已识别的 fallback 因果效应 |
| `sections_applied/01_introduction.tex` | 主要按总体 regret → portfolio reversal 组织 | 将四数据集集中度、输入敏感性与完整策略损失一并介绍 | 核心贡献为既有政策决策损失的量化与审计，不增加新方法贡献 |
| `sections_applied/04_prospective_results.tex` | fallback 分解与基线问题分散 | 增加 95.15% 集中度及明确的两数据集敏感性回引；保留完整分解表 | 连接已有证据；不声称四个数据集的损失全部由缺陷造成 |
| 同上及 `sections_applied/12_execution_details.tex` | 正文详述 sign-flip/Holm 可达下界 | 正文保留观察值、区间与限制；完整推导移至既有附录 D.4 | 正文更聚焦；公式、结构上限、观察 p 值与修正值不变 |
| `sections_applied/07_discussion.tex` | “The results connect diagnostic utility to the complete decision policy.” 等较抽象解释 | 具体说明 Roman/Amazon/Squirrel 的 LINKX/H2GCN 验证选择、Chameleon 的混合选择、GCN/GAT 限制与 Cornell/Wisconsin 影响 | 把机制描述限于记录能显示的结构，不独立归因于架构；保留 GPR 反例 |
| 同上 | fallback 与预处理证据分别讨论 | “The 75.9% loss share after abstention identifies a group of records.”；20.876→13.299 明确是另一次两数据集配对敏感性 | 不拼成 11 数据集的“修正效应量”，不声称 stronger MLP 预算问题已解决 |
| `sections_applied/08_conclusion.tex` | 首句强调 portfolio dependence，重复 7.46 | 优先说明配置、损失集中与条件化的 portfolio ordering | 保留描述性结论，避免把高损失写成普遍诊断失败 |
| availability、`README.md`、新 journal index/provenance note | 指向含内部管理材料的 author companion | 指向独立期刊包，说明三份原档案与一份路径脱敏派生档案 | 提高交付可用性；原档案与历史证据不删除、不覆盖 |
| `scripts/verify_journal_supplement.py`、`tests/test_journal_supplement.py` | 无对应派生包校验路径 | 增加档案、内部完成绑定、模型工件、来源、汇总及脱敏约束校验 | 校验发布工件，不改变科学代码或训练结果；篡改/重复/危险路径必须拒绝 |

摘要从 220 词变为 213 词。PDF 仍为 41 页，不能称整篇已经压缩页数。
本轮未改 `references.bib`、历史 TMLR 源码、实验代码、训练配置和研究记录。

## 交付物

均位于 `output/evidence-focused-revision-2026-09-28/`：

| 工件 | 字节数 | SHA-256 |
|---|---:|---|
| `pdf/main_applied_intelligence.pdf` | 375407 | `1007165e817ac64b8cc8e6e3f76d28351fcacf201895b6be429d0db311ac52b3` |
| `applied-intelligence-journal-supplement-2026-09-28.zip` | 321291208 | `6d0925620097e803fa181ce64556d0b6b4b8c814e0e276c2de5c21a2290aab11` |
| `applied-intelligence-manuscript-source-2026-09-28.zip` | 156517 | `136244422a38fb84eba69384dac31c327075137e21ac18302fedd64fbc963e78` |

期刊包包含 113 个源码文件、对应 PDF、三份原始字节不变的证据 ZIP，
以及一份明确标识的 repeatability 派生 ZIP。排除了当前源码树中的旧稿、
投稿信、内部审稿/编辑/管理材料；历史档案内部说明维持其历史身份。

派生仅涉及 24 个 JSON 的本机 executable/library 路径及受影响的完成/manifest
哈希。模型张量、logits、历史、日志、来源快照与科学字段不变。
`JOURNAL_DERIVATION.json` 明确列出原始到交付字节的映射。
用保留的原档案直接比较已通过；读者没有原档案时，能检查派生说明、科学字段
哈希与交付内部绑定，不能独立恢复已脱敏的原路径。此边界在包内说明。

## 实际验收

- 89 项已有本机测试与既有 claim audit 通过。
- 对最终 ZIP **实际解压**后，13 项定向检查通过，其中 5 项为既有附录检查，
  8 项为新档案负例/正例检查；不把两批简单相加当作独立测试总数。
- 附录摘要从包内记录重算后逐字节相同。
- 910 条原档案记录的字节哈希、其他两档案 500/652 项、派生档案 125 项、
  113 个完成绑定文件、模型状态/logits 和来源绑定均通过；22 个 repeatability
  worker 重新汇总匹配。
- **没有重新聚合全部 770 条模型记录，没有加载数据集、重训或新测试评价。**
- 原 22 张表数值、14 个 display equations、43 个引用键、标题和 60 个受保护文件
  与本轮实际基线一致；从 110 个决策单元复核 7.457189 pp、95.149953% 和 75.876204%。
- 模块源码与平铺源码均构建成功，逐页提取文本相同，41 页；无未定义引用或溢出。
  41 页总览及结果/推导放大页已检查。模板 underfull 警告保留，不影响生成。
- 本轮没有远端 CI；此前其他 SHA 的 CI 不冒充本轮验证。当前修改仍在工作区。

逐项凭据：`evidence-focused-verification-2026-09-28.json`，以及交付目录内
`journal-supplement-verification.json`、`journal-content-audit.json`、
`journal-derivation-verification.json`、`journal-verification/` 和构建/测试日志。

## 用户后续提供的新稳健性结果：独立状态更新

此前 17:24 北京时间的 1538/1540 状态是当时运行快照，现已被后续实际工件更新。
新的只读增量报告单独保存在相邻 `applied-intelligence-review-2026-09-28/`，
文件名 `extension-evidence-completion-audit-2026-09-28.md` / `.json`。

已核实 1540/1540 正式记录，无缺失或重复，6160 trials、1540 次选择后测试，
`complete.json` 与记录 digest 及保存的严格校验报告匹配。实际执行源码
SHA 为 `eb43adfe7c366154170526f30d9ffe2477a70436`，区别于本稿基础提交。
本轮检查 checkpoint 文件存在与大小，**没有再次计算全部大型 checkpoint 哈希**。
两条件、九策略、63 portfolios、LODO 与后续统计结果确实存在。

但还不能称整个实验闭环验收完成：

1. 七个历史 open attempt 未完成终端证据审查；历史 journal 字节未变。
2. 最新 segment 007 的 89041 个事件回放与外部快照匹配、62 个尝试已关闭，
   但最后计账仍对应 17:24:42 的最后模型单位，最终校验及外部分析没有对应计账。
3. 新分析 JSON 尚无 source/config/record-digest 绑定；本地分析包装脚本未入 Git。
   跨条件 bootstrap 的种子派生属于本地分析选择，需明确审查与归档。

科学上，这些新数字支持预处理改善后仍存在完整 portfolio 的描述性策略差距。
不能写“全部结论完全复现”：Combined regret 的跨条件变化区间仍跨零，
GCN/GAT 有利点估计的区间也跨零。截图中的“4 到 5 个非零差，p 下界 0.125”
需改正：关键比较两条件均有 7 个非零差，原始可达下界 0.015625，观察值为 0.125。
具体数值审查另存 `robustness-inference-audit-2026-09-28.md`。

本次交付 PDF 和补充包 **尚未加入这些新结果**，避免把待收尾分析冒充已冻结证据。
MLP 24-trial 与已发表诊断研究扩展没有因这 6160 次四-trial 训练而自动完成。

## 仍待处理及投稿建议

尚未消除的问题包括诊断代表性、MLP 总调参预算不对称、过滤版图的验证、
11 个数据集的推断分辨率、历史来源摘要哈希差异的原始原因，以及期刊匹配度。
已完成的文字和交付修复使这些问题更清楚，不等于给出新的接收保证。

建议保留本轮稿件为可审阅快照；将新稳健性分析按实际执行来源归档、完成已有
计账审查后，再作为独立 sensitivity 结果并入下一版。不要用新数据替换原 frozen
表格，也不要为了文档更新重训。此报告与计划属于内部管理文件，不进入期刊包。
