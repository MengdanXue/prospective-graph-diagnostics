# Applied Intelligence 改写报告

本轮交付为 **Applied Intelligence 格式的现有证据改写版**，尚未投稿。原题保留；科学设计、结果、负结果和既有研究记录均未改动。

## 基准与保存位置

- 基准分支：`publication/tmlr-submission`。
- 基准 SHA：`6495a64fdbb45670b0ce6f66f33a89a7984d4175`。较旧的本地 main 未用作稿件基准。
- 改写分支：`publication/applied-intelligence-2026-09-28`。
- 工作入口：`main_applied_intelligence.tex`；章节：`sections_applied/`。
- 阅读版：`output/pdf/main_applied_intelligence.pdf`。
- 平铺源包：`output/applied-intelligence-submission-source/`；压缩包：`output/applied-intelligence-manuscript-source.zip`。这是论文源文件包，不是研究记录补充包。
- 原 TMLR 检出的未提交脚本、测试和历史报告保持原样；新检出没有修改 TMLR 源稿、参考文献、实验代码、配置或结果文件。

## 使用的技能

- Nature Skills：`84880815fb37317b3766bff2c2abba395b8993c3`，19项技能及共享目录，749个文件校验一致。
- Kiterlin Anti-Defensive Writing：`224d182e908c3e6409f60bb488682142e75e87c1`，完整技能目录2文件校验一致。
- 本轮采用 nature-writing 的 research / manuscript / English / generic 路径，结合 Anti-Defensive 的重复措辞清理规则。Applied Intelligence 不套用 Nature 期刊的篇幅或投稿政策。
- 不安装自动更新 hooks、MCP 服务或可选运行依赖。安装凭据保存在用户技能目录外的 `.codex/skill-install-records/`。

## 主要修改及理由

完整逐行对照见 `manuscript.diff`；术语表、结果分配及段落功能见 `editorial-plan.md`。

| 位置 | 原文／原组织 | 修改后／新组织 | 原因与 claim 影响 |
|---|---|---|---|
| 摘要 | “We evaluate the measured decision utility…”；结果和限制逐项堆叠 | “Graph diagnostics can guide model selection only when their decisions improve on a relevant reference policy.”；问题→设计→损失与反转→关键边界→用途 | 220词，突出选模问题；保留24对4预算、75.9% fallback、3/63及数据集影响、Holm限制；不扩大科学claim |
| 引言 | 多段逐项列举结果和解释限制 | 从模型选择的决策成本进入，集中预览portfolio/fallback结果，保留原三项贡献 | 约652→456英文词；减少重复，不新增算法或方法优越性 |
| Related Work | “not a new…”, “not the first…”, “neither the first nor the largest…”分散出现 | 图统计→候选架构→模型选择与结构控制，3小节 | 约910→613词；所有原引用键保留，诊断代表性限制仍明确 |
| 方法 | 长段落解释阈值历史及多个否定句 | 阈值的exploratory origin与冻结时点留正文，详细来源放附录D | 保留非whole-project preregistration、决策规则含糊处、信息边界、所有公式和训练设置 |
| 主结果 | 执行历史、曲线解释与核心结果并列 | 损失、fallback分解、统计分辨率构成主线；执行修订和曲线移附录D | 约1415→1029词；主文仍保留完整Holm推导、观察值和fallback分解 |
| 敏感性 | 多次说明“不能证明”“不替代” | 每种分析在对应位置清楚给出其用途和必要边界，删除重复句 | 约2149→1621词（含表题和表内词）；六表数字不变，Cornell/Wisconsin、LODO、预算、预处理边界均保留 |
| Discussion | 逐项重述数值并反复申明不作何种claim | 决策场景的含义→fallback与训练→集中说明实际证据限制 | 约1008→686词；预算、代表性、结构零差、重复训练、数据版本限制仍在正文 |
| Conclusion | 回顾多个表和多个否定式声明 | 一项有边界的观察、一个可执行的评价建议 | 约230→143词；保留观察范围和反转依赖，不作普适诊断失败结论 |
| 排版 | TMLR article/style | 官方Springer Nature `sn-jnl`、`sn-basic`数字引文、6关键词、作者页和Declarations | 没有修改出版社class/bst；宽表通过表头换行和列距适配；附录链接采用独立标识 |
| 作者信息 | 匿名稿 | Mengdan Xue、MSU Faculty of Computational Mathematics and Cybernetics、用户提供通讯邮箱 | 邮箱与“无资助”由用户本轮确认；无利益冲突沿用已有明确答复 |
| AI说明 | “ChatGPT was used to suggest specific text and table edits.” | Methods/Research Tools中如实列明Codex的代码、测试、编排、核验、分析及稿件编辑辅助 | 依据当前期刊指南，实质用途超出纯copyediting；不把实际贡献缩写为仅文字润色 |

按一致的英文词形统计（非TeXCount，排除引用/label，含表题和表内文字），八个正文section合计 **8,321→6,444词，约缩短22.6%**。主稿所含附录、表、公式均保留。原TMLR参考PDF31页；Springer单栏版本40页，其中结论位于第18–19页，余下包括声明、完整附录与参考文献。不同版式页数不能直接代表文字长度变化。

## 核验

- 22张表的表头及数据中的数值序列逐表与基准一致；排版列宽参数不纳入科学数值比较。
- 14个展示公式逐项一致；43个实际引用键集合一致。
- `references.bib`与基准提交字节一致，没有新增或猜测DOI/arXiv。
- 原稿对应的60项结果、评分、fallback、portfolio、阈值及训练诊断检查通过；不是训练执行验收。
- 新稿claim审计通过。
- 模块化稿与平铺源包分别成功编译；逐页提取文本一致。
- 两份编译日志均无未定义引用、重复label、溢出框；控制台无重复PDF链接目标。保留少量非致命underfull排版提示。
- 检查全部页面缩略图，并放大检查首页、预处理表和架构选择表；未发现裁切或重叠。
- 详细文件哈希和验证数据：`revision-verification.json`。构建与检查日志位于 `output/`。

## 保留的真实缺口

本版仍依据原冻结基准和已报告的两数据集／训练诊断结果。完整11数据集输入稳健性运行、已发表诊断补充和24-trial MLP补充没有在本轮被认定为已完成或写入论文。软件测试通过不能替代其科学结果和运行完成凭据。既有预算、账本及运行保持不变。

投稿前应完成并核验已经约定的补充结果，再决定与本版如何合并；核对上传平台实际匿名要求、研究记录补充包与稿件的对应关系、作者最终判断及其他在投状态。当前不是“全部投稿门槛通过”的声明。

Applied Intelligence 的方法与技术创新取向仍是范围风险。改写将实证评价贡献说清楚，没有把它包装成新GNN或假定转刊会提高录用概率。

## 官方要求来源

- https://link.springer.com/journal/10489/submission-guidelines
- https://link.springer.com/journal/10489/aims-and-scope
- https://www.springernature.com/gp/authors/campaigns/latex-author-support

核验日期：2026-09-28。期刊指南同时出现旧smallcondensed和新模板指引；旧下载失效，采用当前官方可下载模板。未从其通用匿名条款推断该刊实行双盲。
