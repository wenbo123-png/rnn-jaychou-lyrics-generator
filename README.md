# rnn-jaychou-lyrics-generator

基于 PyTorch 的 RNN（GRU）歌词生成小项目：使用周杰伦歌词训练语言模型，给定**起始词**与**生成长度**，自动生成歌词文本。

## 项目结构

- `gru_lyrics_generator.py`：训练与生成主程序
- `data/jaychou_lyrics.txt`：训练语料（周杰伦歌词文本）
- `model/text_model.pth`：训练好的模型参数（可直接用于生成）
- `可视化/`：训练过程与效果截图（loss 曲线、生成样例等）
- `requirements.txt`：Python 依赖
- `.gitignore`：忽略规则

## 环境依赖

- Python 3.8+（建议）
- 依赖安装：

```bash
pip install -r requirements.txt
```

> 说明：`torch` 在不同系统/显卡环境下安装方式可能不同。如安装失败，请参考 PyTorch 官网选择 CPU/CUDA 对应版本。

## 快速开始：直接生成歌词（使用已训练模型）

仓库已包含训练好的权重 `model/text_model.pth`，你可以直接运行生成：

1. 确保存在：
   - `data/jaychou_lyrics.txt`
   - `model/text_model.pth`

2. 运行：

```bash
python gru_lyrics_generator.py
```

在代码末尾可修改生成参数，例如：

```python
evaluate("雨下", 50)
```

- 第一个参数：起始词（必须在词表中，否则会提示不在词典中）
- 第二个参数：生成长度（生成多少个 token）

## 训练模型（可选）

如需重新训练，请在 `gru_lyrics_generator.py` 中把训练入口打开：

```python
# train()
```

然后运行：

```bash
python gru_lyrics_generator.py
```

训练结束会保存权重到：

- `model/text_model.pth`

## 备注

- 本项目使用 jieba 对中文歌词做分词，并用 `<sep>` 作为行分隔符。
- 生成阶段包含温度采样、top-k 采样与重复惩罚等策略，以减少“复读”和提升可读性。

## License

本仓库包含 `LICENSE` 文件，具体以仓库内许可证文本为准。
