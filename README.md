# rnn-jaychou-lyrics-generator 歌词生成器

基于 PyTorch 的歌词生成项目：使用周杰伦歌词训练语言模型（GRU/RNN），给定**起始词**与**生成长度**，自动生成歌词文本。

---

## 项目结构

- `gru_lyrics_generator.py`：训练与生成主程序
- `data/jaychou_lyrics.txt`：训练语料（已做规则化清洗，见下文）
- `model/text_model.pth`：训练好的模型参数（可直接用于生成）
- `可视化/`：训练曲线与生成效果截图
- `requirements.txt`：Python 依赖
- `.gitignore`：忽略规则

---

## 环境依赖与安装

建议 Python 3.8+。

安装依赖：

```bash
pip install -r requirements.txt
```

### PyTorch 安装说明
本项目依赖 PyTorch。不同设备（CPU / 不同 CUDA 版本）安装方式不同：

- 若你是 **CPU-only**：按 PyTorch 官网选择 CPU 版本安装即可
- 若你有 **NVIDIA GPU**：按 PyTorch 官网选择与你 CUDA 对应的版本

> 我的训练环境参考（仅供对齐复现，不要求一致）：  
> Windows + `torch 2.7.1+cu118`，`jieba 0.42.1`，`numpy 1.26.4`

---

## 快速开始：生成歌词（使用已训练模型）

仓库已包含训练好的权重 `model/text_model.pth`，可直接生成。

确保以下文件存在：
- `data/jaychou_lyrics.txt`
- `model/text_model.pth`

运行：

```bash
python gru_lyrics_generator.py
```

默认会在脚本末尾调用：

```python
evaluate("雨下", 50)
```

参数说明：
- `start_word`：起始词（必须在词表中；若不在词典中会提示并停止）
- `sentence_length`：生成长度（生成多少个 token）

---

## 训练模型（可选）

如果你想重新训练，在 `gru_lyrics_generator.py` 中把 `train()` 打开：

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

## Data Cleaning（数据清洗）

为了提升歌词生成质量、减少无意义 token，本项目在训练前对原始语料 `data/jaychou_lyrics.txt` 做了规则化清洗，主要包括：

- 移除非歌词元信息与标记：如 `END`、`music/MUSIC`、`rap/` 前缀、版权/来源提示等。
- 移除人名/署名等与生成目标无关的内容，避免模型在生成时插入“作者/艺人姓名”。
- 去除注释与舞台提示类文本：例如翻译说明、括注段落、特殊符号标记行等。
- 清理特殊符号与异常格式：例如 `☆` 前缀、`**`、方括号注释等，降低模型输出噪声字符的概率。
- （可选/分阶段）清理英文口癖与混入外语（如 `Coffee/tea/me` 等），使输出更偏向中文歌词风格。
- （可选）清理过度重复标点（如 `……`、`。。`），让输出更整洁。

清洗目标是让模型尽可能学习“歌词正文”的语言模式，减少生成结果出现标签词、英文碎片、异常符号和与语义无关的噪声。

---

## 训练曲线（Loss）

下图为训练过程的 loss 曲线（用于观察收敛情况与训练稳定性）：

![训练曲线（loss）](可视化/loss_curve.png)

可以看到，随着训练的进行，训练损失和验证损失均迅速下降，并在后期趋于平稳，说明模型逐步收敛。整体来看，模型训练过程稳定，没有明显的过拟合现象。

---

## 生成效果示例（可视化）

下面展示 3 个生成样例截图（2 个相对成功的案例 + 1 个失败案例），用于直观观察模型输出质量。

> 说明：生成属于“采样式”输出（非固定答案），同一起始词在不同随机种子/采样参数下可能生成不同文本。

### 测试样例 1（成功案例）

![测试样例1](可视化/测试样例1.png)

### 测试样例 2（成功案例）

![测试样例2](可视化/测试样例2.png)

### 测试样例 3（失败案例）

![测试样例3](可视化/测试样例3.png)

---

## 实现要点（简述）

- 分词：使用 `jieba` 对歌词逐行分词；每行末尾追加特殊分隔符 `<sep>`。
- 模型：Embedding + GRU + Linear（逐 token 预测下一个 token）。
- 训练：交叉熵损失，Adam 优化；包含学习率调度与 early stopping；梯度裁剪提升 RNN 类模型训练稳定性。
- 生成：temperature + top-k 采样 + repetition penalty，并使用多候选采样挑选更稳定的输出。

---

## 常见问题

1) **提示：起始词不在词典中**  
说明起始词在训练语料中未出现（或分词粒度不同）。请尝试更常见的词/短语，或调整起始词的分词方式。

2) **找不到 `model/text_model.pth` 或 `data/jaychou_lyrics.txt`**  
请确认你是在仓库根目录运行脚本，且 `data/`、`model/` 目录与文件存在。

---

## License

本仓库包含 `LICENSE` 文件，具体以仓库内许可证文本为准。
