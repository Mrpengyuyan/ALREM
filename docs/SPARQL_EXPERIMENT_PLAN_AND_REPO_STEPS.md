# 多语言 Text-to-SPARQL + ALREM 实验方案与仓库修改执行步骤

## 1. 目标与边界

### 1.1 论文实验目标
- 验证 ALREM 在两阶段多语言 Text-to-SPARQL 任务上的有效性。
- 证明收益来自 rank 分配策略，而非参数量、解码策略或评测口径差异。
- 补齐任务层 baseline（ICL），避免实验叙事退化为“仅 LoRA 工程比较”。

### 1.2 本轮工作边界（先方案后改码）
- 当前文档只定义实验和改造计划。
- 暂不执行新的代码修改（训练逻辑、评测逻辑功能不变）。

---

## 2. 实验总设计

### 2.1 两阶段训练主线
- Stage 1：LC-QuAD 2.0（英文）学习 SPARQL 结构。
- Stage 2：QALD-9-plus（多语言）继续训练适配。
- Test：QALD-9-plus 官方测试集（当前语言集合：`en,de,es,ru`）。

### 2.2 Baseline 两层结构

#### A. 方法层 baseline（已具备）
- Uniform LoRA
- Parameter-matched LoRA
- ALREM-main
- ALREM-strong（强瓶颈）
- ALREM-reverse-sandwich（反三明治）

#### B. 任务层 baseline（待补）
- ICL zero-shot
- ICL few-shot（固定 `k` 和 `seed`）
- 文献 reported baseline（不可复现实验单独标注）

### 2.3 核心研究问题（RQ）
- RQ1：ALREM 相比 Uniform 是否提升 EA/ER/NormEM/F1/CLC？
- RQ2：参数量匹配后（matched）ALREM 是否仍保持优势？
- RQ3：strong/reverse 消融是否支持 sandwich 假设？
- RQ4：ALREM 与 ICL 基线相比是否有稳定收益？

---

## 3. 公平比较统一协议（必须）

### 3.1 数据统一
- 同一测试文件：`data/sparql/qald9plus_test.jsonl`
- 同一语言集合：`en,de,es,ru`
- few-shot 示例仅来自 train/dev，不得来自 test

### 3.2 推理配置统一
- 统一 `max_new_tokens`
- 统一 `do_sample/temperature/top_p`
- 统一随机种子（如涉及采样）

### 3.3 评测链统一
- 同一后处理规则
- 同一执行器（cache-first / offline_only 语义一致）
- 同一指标实现：EA / Executable Rate / Normalized EM / Answer F1 / CLC

### 3.4 输出格式统一
- 各方法生成统一 `predictions.jsonl` schema：
  - `idx`
  - `qid`
  - `language`
  - `question`
  - `gold_sparql`
  - `pred_sparql`
  - `generation_time_sec`
  - `mode`（`adapter` / `icl_zero` / `icl_fewshot`）

---

## 4. 实验执行矩阵

### 4.1 主结果（必跑）
- Stage1+Stage2 Uniform
- Stage1+Stage2 Param-matched
- Stage1+Stage2 ALREM-main

### 4.2 方法消融（必跑）
- ALREM-strong
- ALREM-reverse-sandwich

### 4.3 任务 baseline（必跑）
- ICL zero-shot
- ICL few-shot（建议先固定 `k=4`）

### 4.4 结果产出
- 主表：总体指标（EA/ER/EM/F1/CLC）
- 分语言表：en/de/es/ru
- 消融表：main vs strong vs reverse
- 错误分布表：generation/execution/wrong-answer 分类

---

## 5. 仓库修改执行步骤（下一阶段将按此实施）

## 5.1 修改原则
- 最小增量改动，不重写现有主链路。
- 生成入口可分离，评测出口统一。
- 每步可回滚、可测试、可追溯。

### 5.2 文件级改造清单

#### Step 1：统一评测出口能力补齐
- 文件：`src/eval_sparql.py`
- 动作：
  - 增加 `--predictions_file` 模式（仅评测，不生成）
  - 复用现有 `compute_all_metrics(...)`
- 目的：
  - ICL 与 Adapter 共享同一评测实现

#### Step 2：新增 ICL 生成入口
- 文件：`src/run_icl_baseline.py`（新增）
- 动作：
  - 支持 zero-shot / few-shot
  - 加载 base model（不加载 adapter）
  - 生成统一 `predictions.jsonl`
- 目的：
  - 引入任务层 baseline，保持与现有训练代码解耦

#### Step 3：新增 ICL 运行脚本与配置
- 文件：
  - `scripts/run_icl.sh`（新增）
  - `configs/sparql_icl_zero_shot.yaml`（新增）
  - `configs/sparql_icl_few_shot.yaml`（新增）
- 动作：
  - Linux 一键运行 ICL baseline
  - 配置固定语言集合、解码参数、cache 路径

#### Step 4：测试与文档补齐
- 文件：
  - `tests/test_icl_baseline.py`（新增）
  - `README.md`（更新）
- 动作：
  - 验证 few-shot 无 test 泄漏、schema 合法、seed 可复现
  - 增加 ICL 路径说明与统一评测规则

---

## 6. 执行顺序与检查点

### Phase B-1：实现前检查
- 确认当前 `main` 干净（除 `data/` 外）
- 新建功能分支：`feat/sparql-icl-unified-eval`

### Phase B-2：增量开发
1. 改 `eval_sparql.py`（仅加评测入口，不改指标逻辑）
2. 新增 `run_icl_baseline.py`
3. 新增 `run_icl.sh` + ICL configs
4. 新增测试与 README

### Phase B-3：本地验证
- `python -m compileall src scripts tests`
- `python -m pytest -q tests`
- ICL 最小 smoke（小样本）
- Adapter 最小 smoke（复用既有脚本）

### Phase B-4：提交与合并
- 按功能拆分 commit（建议 2~4 个）
- PR 审核后合并到 `main`

---

## 7. 验收标准（DoD）
- ICL/Adapter 均能输出统一 predictions 文件。
- ICL/Adapter 均走同一评测出口并得到指标文件。
- 所有既有测试通过，新增测试通过。
- 无 test 泄漏（few-shot 样本来源受控）。
- 结果可复现（seed、config、log、run_report 可追踪）。

---

## 8. 主要风险与规避
- 风险：ICL 与 Adapter 解码参数不一致导致比较偏差  
  - 规避：配置层显式固定，日志中打印参数
- 风险：few-shot 示例混入 test  
  - 规避：强制示例池路径只允许 train/dev
- 风险：离线缓存不完整导致评测中断或偏差  
  - 规避：沿用现有 offline cache 校验策略

---

## 9. 当前状态说明
- 本文档已落地，作为后续改造执行依据。
- 代码功能本轮未新增（仅新增计划文档）。
