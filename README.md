# JayChou Lyrics Generator 歌词生成器（RNN / GRU，PyTorch）

基于 **PyTorch** 在周杰伦歌词语料上训练一个文本生成模型，支持：

- 训练语料：`data/jaychou_lyrics.txt`
- 已训练模型权重：`model/text_model.pth`
- 训练可视化与生成样例：`可视化/`
- 训练与生成主程序：`gru_lyrics_generator.py`

---

## 结果展示（Results）

### 训练曲线（Training Curves）
![](可视化/loss_curve.png)
  可以看到，随着训练的进行，训练损失和验证损失均迅速下降，并在后期趋于平稳，说明模型逐步收敛。整体来看，模型训练过程稳定，没有明显的过拟合现象。

### 生成样例（Generation Samples）
> 说明：以下为 2 个较成功样例 + 1 个失败样例（用于展示模型边界与常见问题）。  
> 生成属于采样式输出（受随机种子/temperature/top-k 等影响），同起始词可能生成不同文本。

| 成功案例 1 | 成功案例 2 |
| --- | --- |
| ![](可视化/测试样例1.png) | ![](可视化/测试样例2.png) |

**失败案例（边界示例）**：  
![](可视化/测试样例3.png)

---

## 核心方法（Core Methods）

### 数据集与预处理
- 语料文件：`data/jaychou_lyrics.txt`
- 分词：使用 `jieba` 按行分词
- 特殊 token：
  - `<unk>`：未知词
  - `<sep>`：行分隔符（每行末尾追加，用于分段）

### Data Cleaning（数据清洗）

为了提升歌词生成质量、减少无意义 token，本项目在训练前对原始语料 `data/jaychou_lyrics.txt` 做了规则化清洗，主要包括：

- 移除非歌词元信息与标记：如 `END`、`music/MUSIC`、`rap/` 前缀、版权/来源提示等。
- 移除人名/署名等与生成目标无关的内容，避免模型在生成时插入“作者/艺人姓名”。
- 去除注释与舞台提示类文本：例如翻译说明、括注段落、特殊符号标记行等。
- 清理特殊符号与异常格式：例如 `☆` 前缀、`**`、方括号注释等，降低模型输出噪声字符的概率。
- （可选/分阶段）清理英文口癖与混入外语（如 `Coffee/tea/me` 等），使输出更偏向中文歌词风格。
- （可选）清理过度重复标点（如 `……`、`。。`），让输出更整洁。

清洗目标是让模型尽可能学习“歌词正文”的语言模式，减少生成结果出现标签词、英文碎片、异常符号和与语义无关的噪声。

### 模型：TextGenerator（Embedding + GRU）
- 词向量：`Embedding(vocab_size, 128)`
- 主体：`GRU(128 -> 256, num_layers=1, batch_first=True)`
- 输出：`Linear(256 -> vocab_size)`（逐 token 预测下一个 token）

### 训练策略（Training Setup）
- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam(lr=1e-3)`
- 学习率调度：`ReduceLROnPlateau(factor=0.5, patience=4)`
- 稳定训练：
  - 梯度裁剪：`clip_grad_norm_(max_norm=1.0)`
  - Early Stopping（patience=10，按验证集 loss）
- 训练/验证拆分：按 9:1 切分，并固定随机种子保证可复现

### 生成策略（Sampling Strategy）
- temperature：0.7
- top-k：12
- repetition penalty：1.15（窗口 20）
- 多候选采样：生成多个候选，按平均 log prob 选更稳定的输出

---

## 模型优化过程说明（阶段记录）
> TODO：你提到“从经典 RNN 到 GRU + 其他手段”的迭代优化过程，后续可以在这里详细记录（对面试很加分）。
- 阶段 1：基础 RNN（TODO）
- 阶段 2：GRU 替换与训练稳定性改进（TODO）
- 阶段 3：采样策略与重复惩罚、多候选筛选（TODO）
- 结论：TODO

---

## 环境与安装（Environment & Installation）
- Python 3.8+（建议）
- 依赖安装：

```bash
pip install -r requirements.txt
```

> 提示：`torch` 在不同平台（CPU/CUDA）安装方式不同。如安装遇到问题，建议按 PyTorch 官方指引选择对应版本。  
> 我的环境参考：Windows + `torch 2.7.1+cu118`，`jieba 0.42.1`，`numpy 1.26.4`。

---

## 可复现性说明（Reproducibility）
- 语料已包含：`data/jaychou_lyrics.txt`
- 已训练权重已包含：`model/text_model.pth`
- 因生成采用采样策略，输出会有一定随机性。

---

## 快速开始（Quick Start）

### 1) 直接生成（默认）
运行：

```bash
python gru_lyrics_generator.py
```

默认会调用：

```python
evaluate("雨下", 50)
```

### 2) 训练（可选）
在 `gru_lyrics_generator.py` 主入口中将：

```python
#train()
```

改为：

```python
train()
```

然后运行：

```bash
python gru_lyrics_generator.py
```

训练完成后会保存到：

- `model/text_model.pth`

---

## 项目结构（Project Structure）
```text
.
├─ data/
│  └─ jaychou_lyrics.txt
├─ model/
│  └─ text_model.pth
├─ 可视化/
│  ├─ loss_curve.png
│  ├─ 测试样例1.png
│  ├─ 测试样例2.png
│  └─ 测试样例3.png
├─ gru_lyrics_generator.py
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ .gitignore
```

---

## 说明（Notes）
- 代码会自动选择设备：有 CUDA 则用 GPU，否则用 CPU。
- 起始词必须在词典中，否则会提示“起始词不在词典中”。

---

## License
MIT License（见 `LICENSE`）。
