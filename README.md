# NIHR 课题申请书自动评分系统

基于本地大语言模型（Ollama）的 NIHR 课题申请书结构化评分流水线。输入 PDF，输出各维度结构化评分、证据来源及评审理由。

---

## 整体思路

### 为什么不直接把全文丢给模型打分？

申请书动辄数万字，超出模型的上下文窗口。即便塞进去，模型也很难在注意力被稀释的情况下逐条核对每个评分子项的证据。

本系统的解决方案是：**先把文档拆成小块（Chunk Pool），再用两阶段流水线做定向检索 + 打分**，让模型每次只看与当前评分维度最相关的内容。

---

## 流水线架构

```
PDF
 │
 ▼
[1] 解析（Parser）
 │   按章节提取文本，生成结构化 JSON
 │
 ▼
[2] 构建 Chunk Pool
 │   每个章节文本按字符数切块（≤1200字/块）
 │   每块分配唯一 chunk_id，记录所属章节
 │
 ▼
[3] Stage 1 — 信念积累（Belief Accumulation）
 │   逐章节扫描，让模型找出与各评分子项相关的证据块
 │   输出：每个子项的 good/bad 证据块 ID + 含义解读
 │   结果汇总为 belief_state（全局信念状态）
 │
 ▼
[4] Stage 2 — 最终打分（Final Scoring）
 │   对每个评分大项，从 belief_state 推断相关章节
 │   拼接该大项的"精简版申请书文本"发给模型
 │   模型对每个 signal 打 0-5 分，并给出 pros/drawbacks
 │
 ▼
[5] 聚合输出
     各 signal → 子项（sub_criterion）→ 大项（section）→ overall
     计算加权平均分，附带 doc_type 专项排除逻辑
```

---

## 关键模块说明

### 解析层（Parser）

位于 `src/all_type_parser/`。自动识别 PDF 类型并路由到对应解析器：

| 文件 | 适用类型 | doc_type |
|---|---|---|
| `fellowships_parser.py` | NIHR Fellowship（博士/博士后） | `fellowship` |
| `RfPB_parser.py` | Research for Patient Benefit Stage 2 | `rfpb` |
| `all_other_parser.py` | 其他格式兜底 | `unknown` |

解析输出为结构化 JSON，顶层包含 `doc_type` 字段，用于后续评分时的专项适配。

RfPB 与 Fellowship 的结构差异：
- Fellowship：固定大框（蓝色方框）出现在每页顶部，章节分隔明确
- RfPB Stage 2：蓝色方框可出现在页面任意位置，同一页可能有多个章节，需逐行检测

---

### Chunk Pool（`src/pool/build_pool.py`）

将解析后的 JSON 中所有文本字段切成固定大小的块（默认 ≤ 1200 字符）。每块记录：

- `chunk_id`：格式为 `sec{章节缩写}__{序号}_{子序号}`，如 `secdrp__001_a`
- `parser_section`：所属章节名（如 `Detailed Research Plan`）
- `source_path`：原始路径（如 `APPLICATION DETAILS > Detailed Research Plan`）

同时生成两个派生块：
- **Application Context**：汇总申请书元数据（标题、申请人、机构等）
- **Application Form Analysis**：基于解析结果的结构性统计（词数、列表数、重复率、章节间语义重叠度等），专供表单质量维度使用

---

### Stage 1 — 信念积累

**设计动机**：申请书中同一段内容可能同时与多个评分维度相关（例如 CV 既关系到研究经验，也关系到领导力轨迹）。Stage 1 通过全文扫描，预先建立"哪些块与哪些子项相关"的映射，供 Stage 2 精准检索。

**执行逻辑**：
- 遍历所有应用章节（不含 Application Form）
- 每次把当前章节的所有 chunk 文本 + 当前全局信念状态 + 完整评分标准发给模型
- 模型输出 findings，格式为：
  ```json
  {
    "sub_id": "g.4",
    "evidence": {
      "good_evidence_ids": ["secac__001_a"],
      "bad_evidence_ids": []
    },
    "implication": "CV 中列举了 7 篇发表论文及多项获批课题，支持研究产出维度"
  }
  ```
- Finding 按子项合并到 `belief_state.subcriteria_beliefs`，跨章节累积

**扁平化设计**：每个 finding 直接对应一个 `sub_id`，不再嵌套 signal 层级，减少 token 消耗，让模型在有限上下文内覆盖更多子项。

---

### Stage 2 — 最终打分

**执行逻辑**：
- 对每个评分大项（general / proposed_research / sites_support 等），从 belief_state 中反查出相关的解析章节
- 从 Chunk Pool 中取出这些章节的完整文本，拼成"精简申请书"
- 将精简申请书 + 该大项的完整评分标准 + belief_state 摘要一起发给模型
- 模型对每个 signal 打 0-5 分，给出 pros（优势）和 drawbacks（不足）

**检索策略**（`SECTION_TO_PARSER_SECTIONS`）：每个评分大项预设了应参考的解析章节列表，例如：
- `proposed_research` → 科学摘要、详细研究计划、与上阶段的变化（RfPB 专有）等
- `training_development` → 培训与发展、支持与导师制等
- `general` → 不限定章节（全局检索）

---

### 评分标准（`criteria_points.json`）

六个评分大项，每项下设若干子项（sub_criterion），每个子项下设若干 signal：

| 大项 | key | 说明 |
|---|---|---|
| 综合素质 | `general` | 申请人经验、产出、领导力等 |
| 研究方案 | `proposed_research` | 研究问题、方法设计、可行性等 |
| 培训与发展 | `training_development` | 培训计划、导师、职业发展支持 |
| 机构与支持 | `sites_support` | 机构资质、督导安排、研究文化 |
| 患者与公众参与 | `wpcc` | PPI 设计、代表性、参与深度 |
| 申请表质量 | `application_form` | 结构完整性、逻辑性、格式规范 |

打分计算路径：signal（0-5）→ 子项加权平均 → 换算为 0-10 → 大项加权平均 → overall

---

### doc_type 适配机制

不同类型的课题申请书评分维度不完全相同。系统通过 `doc_type` 字段实现专项排除：

**RfPB（Research for Patient Benefit）**

RfPB 是面向已有研究基础的项目基金，不是个人职业发展基金。与 Fellowship 的核心区别：
- 申请书不要求职业发展陈述、培训计划等个人叙事内容
- 没有培训/发展章节（Training & Development）

因此对 `doc_type=rfpb` 的申请书自动执行：

| 排除对象 | 排除范围 | 原因 |
|---|---|---|
| `g.1` 优秀申请通用特征 | 不计入 general 得分 | 含"培训计划"等 Fellowship 专属 signal |
| `g.2` 为什么需要此奖项 | 不计入 general 得分 | 询问个人职业发展需求，与项目基金无关 |
| `training_development` | 不计入 overall 得分 | RfPB 申请表无对应章节，强行计入只会拉低总分 |

排除的子项依然会被打分并出现在输出中（带 `excluded_reason: "not_applicable_for_doc_type"` 标记），仅不参与均值计算，方便人工审查。

---

## 目录结构

```
.
├── criteria_points.json          # 评分标准定义
├── qwen3_ollama.py               # 主入口（Ollama 后端）
├── score_experiments.ipynb       # 评分稳定性实验 Notebook
├── src/
│   ├── all_type_parser/
│   │   ├── all_type_parser.py    # 解析器路由
│   │   ├── fellowships_parser.py # Fellowship PDF 解析
│   │   ├── RfPB_parser.py        # RfPB Stage 2 PDF 解析
│   │   └── pdf_utils.py          # PDF 工具函数
│   ├── pool/
│   │   └── build_pool.py         # Chunk Pool 构建
│   └── scoring/
│       └── pipeline.py           # 两阶段评分流水线
└── data/
    ├── successful/               # 测试用申请书 PDF
    └── experiments/              # 稳定性实验数据
```

---

## 运行方式

```bash
# 1. 解析 PDF（自动识别类型）
python -m src.all_type_parser.all_type_parser path/to/application.pdf

# 2. 评分（需本地 Ollama 运行对应模型）
OLLAMA_MODEL=qwen3.5:35b python qwen3_ollama.py \
  data/successful/json_data/IC00029_RfPB.json \
  --criteria criteria_points.json \
  --out data/successful/json_data/IC00029_RfPB_scored.json
```

稳定性实验（重复跑多次、多文件对比）使用 `score_experiments.ipynb`，支持：
- 同一 PDF 多次评分的方差统计
- A/B 两组申请书的分布对比及假设检验

---

## 输出结构

评分 JSON 的顶层结构：

```json
{
  "doc_id": "IC00029_RfPB",
  "run_info": { "scorer_model": "qwen3.5:35b", "ran_at_utc": "..." },
  "pool_lookup": { "chunk_id": { "text": "...", "parser_section": "..." } },
  "belief_state": { "subcriteria_beliefs": { "pr.1": { "good_evidence_ids": [...] } } },
  "features": {
    "general": {
      "score_10": 7.33,
      "sub_criteria": [
        {
          "sub_id": "g.3",
          "score_10": 8.0,
          "counts_toward_section_average": true,
          "signals": [ { "sid": "g.3.a", "score": 4 } ],
          "pros": "...",
          "drawbacks": "...",
          "evidence": [ { "id": "secac__001_a", "text": "..." } ]
        }
      ]
    }
  },
  "overall": { "score_10": 8.44, "final_score_0to100": 84.4 },
  "debug": {
    "doc_type": "rfpb",
    "excluded_sections": ["training_development"],
    "excluded_sub_ids": ["g.1", "g.2"]
  }
}
```
