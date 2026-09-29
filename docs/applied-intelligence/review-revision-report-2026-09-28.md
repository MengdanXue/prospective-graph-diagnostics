# Applied Intelligence 定点修订记录（2026-09-28）

本轮处理模拟评审确认的两项主要论证问题和六项局部问题。标题、摘要、实验结果、训练配置和原始证据保持不变；没有新增训练、研究测试评价或投稿操作。

## 版本与基线

- 当前分支：`publication/applied-intelligence-2026-09-28`。
- 修改基线：`ef48699cdfdce8137b3fc560ff838977751e6e52`，父提交 `6495a64fdbb45670b0ce6f66f33a89a7984d4175`。
- 外部截图提到的 `9d6d23f77672c9cec44e70ddbae5998713cc334c` 是同一父提交的另一子提交，修改三个 CI/浮点重构相关文件，没有修改稿件。它不是本轮 Applied 稿件基线，也没有被宣称合入当前工作树。
- 本轮交付用“基线提交 + companion 源文件哈希清单”标识；没有冒称新的提交 SHA，未推送或核验新的远端 CI。
- 旧 40 页 PDF 保留在 `output/pdf/main_applied_intelligence.pdf`，SHA-256 为 `a12277e8038293b74ba11495971ccc4e2f1fe1e34395fb3ca86bebd1bd00991e`。
- 新 41 页 PDF：`output/review-revision-2026-09-28/pdf/main_applied_intelligence.pdf`，SHA-256 为 `60c5e2246058a027de4f0d96a409a0b69249d5d63cd82cd9784dae09a63ea39c`。

## 逐项修订

### M1：贡献与已发表方法的关系

**文件：** `sections_applied/01_introduction.tex`、`02_related_work.tex`、`07_discussion.tex`、`08_conclusion.tex`。

**原文要点：** “For transparent graph-versus-MLP rules, a complementary question is how much accuracy their decisions cost ...”；协议笼统连接信息边界、验证选择、regret、coverage。

**修改后：** “We use simple threshold rules as transparent case studies ...”；“We combine established principles ... in a record-based audit of regret, coverage, and available improvement over reference policies.” 相关工作明确对照 CPM 的配对模型优势检验、Tri-Hom 的信息/相关性目标、GLEMOS 的跨图模型选择，以及本文的完整策略损失、fallback 与组合比较。

**理由：** 把经验发现、既有评价原则的组合实施和未验证推广分开；简单规则用于解释受控案例，不代表所有现代诊断。

**Claim 影响：** 收窄创新性和代表性表达；三项贡献及现有结果不变，未加入新方法或对发表方法的胜负判断。

### M2：实际决策阶段

**文件：** 上述四章及 `sections_applied/03_decision_protocol.tex`。

**原文要点：** “Validation accuracy is available because model tuning precedes the decision.” 表 2 原有 Policy / Decision rule 两列。

**修改后：** 明确这是共同训练结果上的离线策略审计；表 2 增加 Information / stage，区分训练前可用统计、MLP 调参后的 Combined 弃权/覆盖信息、双方调参后的 validation selection。表注说明阶段对应 fallback 前规则；选定家族后仍需训练/选择预测器。

**理由：** 避免把离线预测精度损失当作已证实的训练节省或部署收益。保留正文中 Combined 加 MLP fallback 后完整动作简化为同质性阈值的说明。

**Claim 影响：** 明确应用边界，不改变规则、动作或计分。

### L1：GPR 的系数数目

**文件：** `sections_applied/09_reproducibility_appendix.tex`。

**原文：** “ten trainable PPR-initialized coefficients”。

**修改后：** “ten propagation steps with eleven trainable PPR-initialized coefficients for orders zero through ten”。

**理由：** 实现使用 `np.arange(steps + 1)`，包含零阶项。仅更正文档，未改模型。

**Claim 影响：** 无结果影响。

### L2：两跳统计的采样边界

**文件：** `sections_applied/03_decision_protocol.tex`。

**原文：** “the two-hop statistic uses training-labeled endpoint pairs”。

**修改后：** 明确在训练节点诱导子图上抽取 50,000 条有序长度二路径；中心权重为 d(d−1)，均匀抽取不同邻居，路径的三个节点均属训练集。不同抽样可重复，端点相邻的三角形路径仍计入。随机种子为 split seed + 200,000，并说明无有效路径时缺失。

**理由：** 使描述对应现有实现，而不是让读者误以为只约束端点，或计算严格图距离二。

**Claim 影响：** 完善可重建性，无结果影响。

### L3：划分的最小样本约束

**文件：** `sections_applied/03_decision_protocol.tex`。

**原文：** “at least one validation and two test examples”。

**修改后：** “Each class must contain at least three nodes, with at least one assigned to each of the training, validation, and test sets.”

**理由：** 对应代码保证；每类仅三节点时，测试集只有一个。

**Claim 影响：** 纠正方法事实，不改变已生成划分。

### L4：重连强度与区间的实验单位

**文件：** `sections_applied/05_edge_intervention.tex`。

**原文：** “verified double-edge swaps randomize the graph”；图注仅写 “Differences and confidence intervals are percentage points.”

**修改后：** 说明每条原始边对应五次成功交换、随机化种子 split seed + 100,000；表注说明每个固定数据集上的十个配对种子差和 10,000 次 percentile bootstrap，区别于正文三个数据集层面的描述性区间。

**理由：** 让干预强度和不确定性单位可直接核查。

**Claim 影响：** 无数值、统计方法或推断范围变化。

### L5：摘要与真实 audit 的字节绑定

**文件：** `scripts/summarize_reviewer_appendix.py`、`tests/test_reviewer_appendix_summary.py`、`results/diagnostic/route_a_prospective_v2/analysis/reviewer_appendix_summary.json`；新增 provenance amendment 与旧摘要副本。

**原记录：** `source_audit_sha256=4bf33130bad67d2a6330abb1e3efb2ebf51b6fc6e50984b8c67d73610df0c131`。

**修改后：** 从当前 audit 重新计算决策统计，使用旧摘要中四类数据集计数作为明确标记的元数据，要求所有非 provenance 字段与旧摘要精确一致后，绑定实际 audit SHA-256：`3c5aa160ac420536fd3f0be58b880fc99b3fe5912215a54d149e4cb62905b3bd`。

**理由：** 直接手改哈希无法证明统计来自该输入。新路径检查真实输入字节，拒绝过期绑定、输入字节改变、科学字段改变及覆盖既有输出。旧摘要原字节保留，原哈希来源仍未查明，不能推断为已证明的私有或未脱敏版本。

**Claim 影响：** 所有科学字段不变。数据集节点/边/类/特征计数本轮没有重新加载原始数据验证。

### L6：可用性声明与逐表重构路径

**文件：** `main_applied_intelligence.tex`、附录 A、`README.md`；新增 `reconstruction-index.md`。

**原文：** “The accompanying research archive contains source code, configurations, compact summaries, and retained unit-level records ...”

**修改后：** 指定 `applied-intelligence-research-companion-2026-09-28.zip`，说明 README、VERSION、SHA256SUMS、四个原证据包和来源修正；索引覆盖全部 22 个表标签及图，列明所需输入、脚本、数据和历史 Git 的限制。

**理由：** 让读者区分摘要重聚合、原始记录重构、checkpoint replay 与重新训练，避免把 Git 源码 ZIP 宣称为全部运行证据。README 同时纠正旧的“尚未投 TMLR”和过宽诊断表述。

**Claim 影响：** 提高交付可核验性；未新增独立重放全部模型的保证。

## 验证与保留边界

- 本机项目环境中 89 项针对性检查通过；最初使用内置 Python 时，两模块因缺少 SciPy 导入失败，保留该失败日志，再用已有项目环境完成同一组检查。
- Applied 正文 claim audit 通过。
- 模块化源码和投稿合并源码均编译成功，PDF 文本逐页一致。新 PDF 41 页，旧版 40 页；检查全部页面缩略图，并放大核查新增策略表和可用性声明。最终日志无未定义引用、重复标签或溢出框；Springer 模板仍产生 underfull 排版提示。
- 20 个未改方法说明的表，数值 token 完全一致；另外两表仅增加决策阶段说明和修正 GPR 系数数目。14 个 display equation、43 个 citation key、参考文献文件、标题和摘要均保持不变。
- 训练实现、权威配置、旧 `sections/`、`sections_tmlr/` 和旧 PDF/修订报告保持原样。历史稿件中的两处已知方法措辞不被倒改；本报告记录它们的更正。
- 独立的八项收口检查确认方法文字与现有实现一致、摘要科学字段不变、22 个表标签都有重构索引。
- 最终 companion 的打包/从包解压验证记录单独保存在交付目录，不能用上述测试数量替代该验证。

## 仍然存在的真实限制

本轮没有解决也没有掩盖 MLP 与完整图组合总调参预算不同、简单规则代表性有限、全量预处理稳健性不在此稿证据内、完整 checkpoint replay 需额外工件等限制。两项主要意见通过更精确的贡献和用途论证回应，不能据此承诺 Applied Intelligence 的创新性门槛或录用结果。

旧的来源哈希为何产生仍未查明；当前公开输入与摘要的一致性已经修复并加回归检查。此修订可以进入作者最终审阅和投稿准备，但本轮未上传期刊、未运行新远端 CI，也未合并主分支。

完整原文—新文差异见 `review-revision-2026-09-28.diff`；机器核验见 `review-revision-verification-2026-09-28.json`。无需重新确认既有作者、邮箱、机构、无资助与无利益冲突信息；正式投稿操作仍属于单独步骤。
