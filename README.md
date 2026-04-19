# 低显存 Transformer 机器翻译课程项目

这是一个面向课程作业的、**端到端从头训练**的机器翻译项目模板。项目默认使用 **Multi30k 英语→德语 (en→de)** 平行语料，采用 **PyTorch 原生模块实现的轻量级 Encoder-Decoder Transformer**，支持在**低显存（目标 1GB 显存以内）**设备上完成完整流程：

- 数据集下载与校验
- 文本预处理与词表构建
- Transformer 模型从头训练
- 对照实验自动执行
- BLEU / chrF 指标输出
- 翻译效果演示
- 训练曲线绘制

> 设计目标：不是追求 SOTA，而是追求“**低显存可跑通、流程完整、适合写课程实验报告**”。

---

## 1. 项目特性

- **严格从头训练**：不使用任何预训练模型、预训练词向量或外部翻译权重。
- **纯本地闭环**：数据准备完成后，可在断网环境下完成训练、评估与推理。
- **低显存优先**：默认配置针对约 `0.9GB` 显存峰值预算设计。
- **轻量依赖**：核心依赖只有 PyTorch、tqdm、matplotlib，`sacrebleu` 仅为可选项。
- **对照实验现成可用**：默认提供 3 组实验（基线 + 2 个轻量消融）。
- **文档友好**：`docs/` 中附带实验报告参考模板与设计说明。

---

## 2. 推荐目录结构

```text
mt_transformer_course_project/
├─ configs/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ docs/
├─ outputs/
├─ scripts/
├─ src/
├─ tests/
├─ README.md
└─ requirements.txt
```

---

## 3. 环境配置

### 3.1 Python 版本

推荐：**Python 3.10+**

### 3.2 安装依赖

#### 方案 A：直接安装 requirements（最简单）

```bash
pip install -r requirements.txt
```

#### 方案 B：先按本机 CUDA/CPU 环境安装 PyTorch，再安装其他依赖

如果你的 `pip install -r requirements.txt` 无法正确安装适配本机 CUDA 的 PyTorch，请先参考 PyTorch 官方安装页安装 `torch`，再执行：

```bash
pip install tqdm matplotlib sacrebleu
```

> 说明：`sacrebleu` 是**可选**依赖。若未安装，项目会自动回退到内置 BLEU/chrF 实现，仍可完整运行。

---

## 4. 数据集准备

项目默认使用 **Multi30k en→de**。你只需要执行本地脚本即可自动下载并校验。

> GitHub 仓库归档说明：仓库只保留代码、配置、文档和文本实验结果；原始数据、预处理数据、模型权重和图片曲线不会提交。数据目录中的占位说明见 `data/README.md`，训练产物归档规则见 `outputs/README.md`。

> 当前脚本已经改成“**顶部常量写死 + 直接运行**”模式。常见路径和配置都写在各自 Python 文件最上方，通常直接运行即可；若你想切换到 quick/full 配置，只需要改脚本顶部常量。

数据来源（建议在实验报告中注明）：

- 数据集仓库：`https://github.com/multi30k/dataset`
- Task 1 原始文件目录：`https://github.com/multi30k/dataset/tree/master/data/task1/raw`
- 本项目脚本实际拉取的原始文件基地址：`https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/`

### 4.1 下载 + 校验原始数据

```bash
python scripts/download_multi30k.py
```

完成后会得到：

```text
data/raw/multi30k/
├─ train.en.gz
├─ train.de.gz
├─ val.en.gz
├─ val.de.gz
├─ test_2016_flickr.en.gz
├─ test_2016_flickr.de.gz
└─ download_manifest.json
```

### 4.2 预处理 + 构建词表

```bash
python scripts/prepare_data.py
```

完成后会得到：

```text
data/processed/multi30k_en_de/
├─ train.src
├─ train.tgt
├─ valid.src
├─ valid.tgt
├─ test.src
├─ test.tgt
├─ src_vocab.json
├─ tgt_vocab.json
└─ meta.json
```

---

## 5. 训练流程

### 5.1 快速冒烟版（推荐先跑）

```bash
python scripts/train.py
```

用途：
- 检查环境是否正常
- 检查预处理文件是否可读
- 快速验证训练/评测/推理链路

如果你要跑快速冒烟版，请先把 `scripts/train.py` 顶部的 `CONFIG_PATH` 改成 `configs/quick_demo_en_de.json`。

### 5.2 完整课程作业版

```bash
python scripts/train.py
```

训练完成后，会在 `outputs/baseline_tiny_en_de/` 下生成：

- `checkpoints/best.pt`：最佳模型
- `history.csv`：训练日志
- `test_metrics.json`：测试集指标
- `predictions_test.txt`：测试集翻译输出
- `resolved_config.json`：本次训练的完整配置

---

## 6. 对照实验

默认提供 3 组实验：

1. `baseline`：复用正式训练得到的基线结果
2. `learned_pos`：学习式位置编码
3. `label_smoothing`：标签平滑训练策略

运行方式：

```bash
python scripts/run_experiments.py
```

输出：

```text
outputs/experiments/
├─ baseline/
├─ learned_pos/
├─ label_smoothing/
├─ summary.csv
└─ summary.md
```

---

## 7. 单独评估模型

如果你已经训练过模型，可直接评估：

```bash
python scripts/evaluate.py
```

---

## 8. 翻译演示

### 8.1 使用内置示例句子

```bash
python scripts/translate_demo.py
```

### 8.2 翻译单句

将 `scripts/translate_demo.py` 顶部的 `SINGLE_TEXT` 改成你的句子后，直接运行：

```bash
python scripts/translate_demo.py
```

### 8.3 批量翻译文件

将 `scripts/translate_demo.py` 顶部的 `INPUT_FILE` / `OUTPUT_FILE` 改成目标文件后，直接运行：

```bash
python scripts/translate_demo.py
```

---

## 9. 绘制训练曲线

```bash
python scripts/plot_curves.py
```

输出文件示例：

- `outputs/baseline_tiny_en_de/loss_curve.png`
- `outputs/baseline_tiny_en_de/bleu_curve.png`

---

## 10. 一键串行执行（可选）

如果你想用一个脚本串起来执行，可使用：

```bash
python scripts/run_pipeline.py
```

默认流程：
1. 检查原始数据是否存在
2. 若不存在且你没有传 `--skip_download`，自动下载
3. 执行预处理
4. 执行训练
5. 输出测试指标
6. 运行演示翻译

---

## 11. 低显存策略说明

本项目默认配置已经针对低显存做了控制：

- Tiny Transformer：`d_model=128, enc=2, dec=2, nhead=4`
- 直接吃满 4GB 以内显存余量：`batch_size=48`
- 梯度累积简化为：`grad_accum_steps=1`
- 最大序列长度限制：`max_len=40`
- 可选混合精度：CUDA 环境自动启用
- 训练过程记录峰值显存（若有 CUDA）

### 11.1 如果仍然显存不足，可按顺序尝试：

1. 将 `batch_size` 从 8 改为 4
2. 将 `grad_accum_steps` 适当增大
3. 将 `max_len` 从 40 改为 32
4. 将 `d_model` 从 128 改为 96
5. 将 `dim_feedforward` 从 256 改为 192

---

## 12. 核心配置说明

完整配置文件示例见 `configs/full_tiny_en_de.json`。

最关键的几项：

- `data.max_len`：句子最大长度（含 BOS/EOS）
- `data.max_src_vocab / max_tgt_vocab`：词表上限
- `model.position_type`：`sinusoidal` 或 `learned`
- `train.label_smoothing`：标签平滑系数
- `runtime.memory_budget_mb`：显存预算提示值（仅用于日志提示）

---

## 13. 常见问题

### Q1：为什么不直接用更大的数据集？
因为本项目的目标是**快速跑通课程作业**，而不是追求竞赛级效果。Multi30k 体量小、来源稳定、文本干净，更适合低显存设备与课程实验。

### Q2：为什么没有用 BPE / SentencePiece？
为了减少依赖、降低复杂度，并保证你拿到压缩包后可以直接本地跑通，本项目采用了**轻量正则分词 + 词表截断**方案。对于课程作业，这种设计在“实现成本 / 结果可解释性 / 可复现性”之间更均衡。

### Q3：指标和 sacreBLEU 会完全一致吗？
如果你安装了 `sacrebleu`，项目会优先使用它；否则会自动回退到内置实现。课程作业通常使用项目内统一指标即可。如果你追求与论文更强一致性，建议额外安装 `sacrebleu`。

### Q4：我不想下载数据，能直接跑吗？
可以运行 `tests/smoke_test.py` 做玩具数据冒烟测试，但正式实验仍建议使用 Multi30k。

---

## 14. 建议的课程实验报告写法

你可以直接参考 `docs/REPORT_TEMPLATE.md`，重点写：

- 数据集来源与规模
- 预处理策略
- 模型结构与参数量
- 训练策略与低显存适配方法
- 对照实验设计与控制变量
- BLEU / chrF 结果分析
- 翻译样例展示与误差分析

---

## 15. 参考执行顺序（推荐）

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 下载数据
python scripts/download_multi30k.py

# 3) 预处理
python scripts/prepare_data.py

# 4) 冒烟训练
# 如果你要 quick 版，把 train.py 顶部 CONFIG_PATH 改成 quick_demo_en_de.json
python scripts/train.py

# 5) 正式训练
# 把 train.py 顶部 CONFIG_PATH 改回 full_tiny_en_de.json
python scripts/train.py

# 6) 跑三组对照实验（baseline 复用正式训练结果）
python scripts/run_experiments.py

# 7) 绘制曲线
python scripts/plot_curves.py

# 8) 翻译演示
python scripts/translate_demo.py
```

---

## 16. 免责声明

本项目的目标是：

- **保证流程完整**
- **尽量低显存**
- **适合课程实验与报告撰写**

它并不以追求最优翻译质量为首要目标。如果你后续想继续提升效果，可以考虑：

- 使用 BPE / SentencePiece
- 使用 beam search
- 扩大模型维度或层数
- 换用 IWSLT14 等更大的平行语料
