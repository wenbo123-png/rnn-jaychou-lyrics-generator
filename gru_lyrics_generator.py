"""
案例：
    RNN案例，基于周杰伦歌词来训练模型，用给定的起始词，结合长度，来进行AI歌词生成。
"""

# 导入相关模块
import torch
import jieba
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np

# CUDA设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义特殊词：未知词和分隔符
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"

# 检查GPU是否生效
def check_cuda():
    print(f"当前设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到可用GPU，使用CPU运行。")

# 设置随机种子
def set_seed(seed=42):
    # 固定随机种子，保证训练尽可能可复现
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed)

# todo:1.获取数据，进行分词，获取词表
def build_vocab():
    # 用字典做词去重，提升效率
    unique_words, word_to_index, corpus_idx = [], {}, []

    def add_word(word):
        if word not in word_to_index:
            word_to_index[word] = len(unique_words)
            unique_words.append(word)
        return word_to_index[word]

    # 先加入特殊token，避免预测阶段出现未知词报错
    add_word(UNK_TOKEN)
    add_word(SEP_TOKEN)

    # 遍历数据集，分词后转为索引序列
    for line in open("./data/jaychou_lyrics.txt", 'r', encoding="utf-8"):
        words = jieba.lcut(line)
        for word in words:
            corpus_idx.append(add_word(word))
        # 每行结尾追加分隔符，替代对空格token的隐式依赖
        corpus_idx.append(word_to_index[SEP_TOKEN])

    # 统计语料中去重后的词的数量
    word_count = len(unique_words)
    # 返回结果：去重词列表，词表，去重后词的数量，歌词文本索引表示
    return unique_words, word_to_index, word_count, corpus_idx

# todo:2.数据预处理，构建数据集
class LyricsDataset(torch.utils.data.Dataset):
    # 初始化词索引，词个数等
    def __init__(self, corpus_idx, num_chars):
        # 获取文档中词的索引、数量、每句中词的个数
        self.corpus_idx = corpus_idx
        self.word_count = len(corpus_idx)
        self.num_chars = num_chars
        # 句子数量
        self.number = self.word_count // self.num_chars

    # 获取数据集的长度,当使用len()时，自动调用该方法
    def __len__(self):
        return self.number

    # 获取数据集中的一个样本,当使用obj[index]时，自动调用该方法
    def __getitem__(self, idx):        # 获取第idx个样本的输入和输出
        start = min(max(idx,0), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        # 输入值
        x = self.corpus_idx[start:end]
        # 输出值
        y = self.corpus_idx[start+1:end+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)     # 返回张量形式

# todo:3.搭建RNN神经网络
class TextGenerator(nn.Module):
    # 初始化方法，定义RNN层和全连接层
    def __init__(self,unique_word_count):
        super().__init__()
        self.embedding_dim = 128
        self.hidden_size = 256
        self.num_layers = 1
        # 词嵌入层：语料中词的数量，词向量的维度
        self.emb = nn.Embedding(unique_word_count, self.embedding_dim)
        # GRU层：词向量维度，隐藏层维度，网络层数
        self.rnn = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        # 输出层（全连接层）:输入维度为隐藏层维度，输出维度为语料中词的数量
        self.out = nn.Linear(self.hidden_size, unique_word_count)

    # 前向传播方法
    def forward(self,inputs,hidden):
        # 词嵌入层处理
        embd = self.emb(inputs)      # [batch_size, seq_len, embedding_dim]
        # RNN层处理
        rnn_output, hidden = self.rnn(embd, hidden)    # [batch_size, seq_len, hidden_size]
        # 输出层处理：对每个时间步做词表分类
        logits = self.out(rnn_output)                  # [batch_size, seq_len, vocab_size]
        return logits, hidden

    # 隐藏层的初始化方法
    def init_hidden(self, batch_size):
        #隐藏层初始化：网络层数，批量大小，隐藏层向量维度
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))

# todo:4.模型训练
def train():
    set_seed(42)
    # 构建词典
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    # 获取数据集
    lyrics = LyricsDataset(corpus_idx, num_chars=48)
    # 划分训练集和验证集，用于学习率调度和早停
    val_size = max(1, int(len(lyrics) * 0.1))
    train_size = len(lyrics) - val_size
    train_dataset, val_dataset = random_split(
        lyrics,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # 构建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    # 创建模型对象
    model = TextGenerator(unique_word_count).to(device)
    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float("inf")
    best_checkpoint = None
    early_stop_patience = 10
    no_improve_epochs = 0

    # 模型训练
    epochs = 120
    for epoch in range(epochs):
        model.train()
        # 定义变量记录: 本轮开始训练时间, 迭代(批次)次数, 训练总损失.
        start, iter_num, total_loss = time.time(), 0, 0.0
        # 遍历数据集，后台会自动调用__getitem__方法获取输入和输出
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            # 获取隐藏层初始值
            hidden = model.init_hidden(x.size(0)).to(device)
            # 前向传播
            logits, hidden = model(x, hidden)
            # 计算损失
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))  # 展平
            # 梯度清零+反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，提升RNN类模型训练稳定性
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # 累计损失和迭代次数
            total_loss += loss.item()
            iter_num += 1
        train_loss = total_loss / iter_num

        # 验证集评估
        model.eval()
        val_total_loss, val_iter_num = 0.0, 0
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                hidden = model.init_hidden(x.size(0)).to(device)
                logits, hidden = model(x, hidden)
                val_loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                val_total_loss += val_loss.item()
                val_iter_num += 1
        avg_val_loss = val_total_loss / val_iter_num

        # 按验证集损失调整学习率
        scheduler.step(avg_val_loss)

        # 保存当前最优模型并更新早停计数
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "embedding_dim": model.embedding_dim,
                    "hidden_size": model.hidden_size,
                    "num_layers": model.num_layers,
                },
                "unique_words": unique_words,
                "word_to_index": word_to_index,
            }
        else:
            no_improve_epochs += 1

        # 从优化器里读取当前学习率，方便观察学习率调整情况
        current_lr = optimizer.param_groups[0]["lr"]
        # 打印本轮的训练信息
        print(
            f'epoch: {epoch+1}, time: {time.time()-start:.2f}s, '
            f'train_loss: {train_loss:.4f}, val_loss: {avg_val_loss:.4f}, lr: {current_lr:.6f}'
        )

        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}, best_val_loss: {best_val_loss:.4f}")
            break

    # 保存模型参数和必要元信息，保证训练/推理词表一致
    checkpoint = best_checkpoint if best_checkpoint is not None else {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "embedding_dim": model.embedding_dim,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
        },
        "unique_words": unique_words,
        "word_to_index": word_to_index,
    }
    torch.save(checkpoint, './model/text_model.pth')

# todo:5.模型预测
def evaluate(start_word,sentence_length):
    temperature = 0.7
    top_k = 12
    repetition_penalty = 1.15
    num_candidates = 4
    repeat_window = 20

    # 构建词典
    unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
    checkpoint = torch.load('./model/text_model.pth', map_location=device)
    # 优先使用checkpoint中的词表，避免推理和训练词表不一致
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        unique_words = checkpoint.get("unique_words", unique_words)
        word_to_index = checkpoint.get("word_to_index", word_to_index)
        unique_word_count = len(unique_words)
        state_dict = checkpoint["model_state_dict"]
    else:
        # 兼容旧版本：仅保存state_dict
        state_dict = checkpoint
        unique_word_count = state_dict["emb.weight"].shape[0]
    # 获取模型
    model = TextGenerator(unique_word_count).to(device)
    # 加载模型参数
    model.load_state_dict(state_dict)
    model.eval()
    # 起始词不存在时给提示并停止，避免回退到 <unk> 后继续“乱生成”
    if start_word not in word_to_index:
        print(f"提示：起始词“{start_word}”不在词典中，请更换一个已知词。")
        return
    # 将输入的起始词转换为索引
    word_idx = word_to_index[start_word]
    # 多候选采样：保留创作感的同时，用平均log概率选更稳定的结果
    def generate_one_candidate(start_idx):
        hidden = model.init_hidden(1).to(device)
        output_idx = [start_idx]
        current_idx = start_idx
        total_log_prob = 0.0

        with torch.no_grad():
            for _ in range(sentence_length):
                input_tensor = torch.tensor([[current_idx]], dtype=torch.long).to(device)
                logits, hidden = model(input_tensor, hidden)
                next_logits = logits[0, -1] / max(temperature, 1e-6)   # 温度缩放

                # 重复惩罚：降低近期已生成词再次被采样的概率，减少复读和跑偏
                if repetition_penalty is not None and repetition_penalty > 1.0:
                    recent_tokens = output_idx[-max(1, repeat_window):]
                    for token_id in set(recent_tokens):
                        if next_logits[token_id] < 0:
                            next_logits[token_id] *= repetition_penalty
                        else:
                            next_logits[token_id] /= repetition_penalty

                if top_k is not None and top_k > 0:
                    k = min(top_k, next_logits.shape[-1])
                    topk_logits, topk_indices = torch.topk(next_logits, k)
                    probs = torch.softmax(topk_logits, dim=-1)
                    sampled_pos = torch.multinomial(probs, num_samples=1).item()
                    current_idx = topk_indices[sampled_pos].item()
                    total_log_prob += torch.log(probs[sampled_pos] + 1e-12).item()
                else:
                    probs = torch.softmax(next_logits, dim=-1)
                    current_idx = torch.multinomial(probs, num_samples=1).item()
                    total_log_prob += torch.log(probs[current_idx] + 1e-12).item()

                output_idx.append(current_idx)

        avg_log_prob = total_log_prob / max(sentence_length, 1)
        return output_idx, avg_log_prob

    candidates = []
    for _ in range(max(1, num_candidates)):
        c_output_idx, c_score = generate_one_candidate(word_idx)
        candidates.append((c_score, c_output_idx))
    output_idx = max(candidates, key=lambda x: x[0])[1]
    # 将输出索引转换为词并打印：隐藏 <sep> 字面量，用空格展示分段
    for idx in output_idx:
        token = unique_words[idx]
        if token == SEP_TOKEN:
            print(' ', end='')
            continue
        if token == UNK_TOKEN:
            continue
        print(token, end='')
    print()

# todo:6.测试
if __name__ == "__main__":
    check_cuda()
    # 训练模型
    #train()
    # 模型预测
    evaluate("雨下", 50)
