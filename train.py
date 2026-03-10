"""
训练循环 - 从零实现
实现前向传播、反向传播（数值梯度）和优化
"""

import numpy as np
from config import cfg
from model import Qwen3Model, cross_entropy_loss
from optimizer import AdamW
from tokenizer import SimpleTokenizer


class Dataset:
    """
    数据集类：处理文本数据，生成训练批次

    语言模型的训练数据格式：
        输入 x: [token_0, token_1, ..., token_{n-1}]
        目标 y: [token_1, token_2, ..., token_n]

        即：用前 n-1 个 token 预测第 n 个 token
    """

    def __init__(self, text, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len  # 每个样本的序列长度

        # 训练分词器（建立词表）
        tokenizer.train(text)

        # 将整个文本编码为 token ID 序列
        # ids: 一维数组，形状 [num_tokens]
        self.ids = np.array(tokenizer.encode(text), dtype=np.int64)

    def __len__(self):
        """返回数据集中的样本数量"""
        # 每个样本需要 seq_len + 1 个 token（输入 + 目标）
        return max(1, len(self.ids) - self.seq_len)

    def get_batch(self, batch_size=4):
        """
        获取一个训练批次

        参数：
            batch_size: 批次大小

        返回：
            x: 输入序列，形状 [batch, seq_len]
            y: 目标序列，形状 [batch, seq_len]

        举例：
            假设 ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]，seq_len = 3

            随机选择起始位置 i = 2:
                x = [2, 3, 4]  # 输入
                y = [3, 4, 5]  # 目标（输入右移一位）

            这样模型学习：给定 [2,3,4]，预测 [3,4,5]
        """
        # 随机选择 batch_size 个起始位置
        # randint(0, len(self), batch_size) 生成 [0, len(self)) 范围内的随机整数
        indices = np.random.randint(0, len(self), batch_size)

        # 构建输入序列 x
        # 对于每个起始位置 i，取 ids[i:i+seq_len]
        # np.stack 将多个序列堆叠成一个批次
        # 结果形状: [batch_size, seq_len]
        x = np.stack([self.ids[i : i + self.seq_len] for i in indices])

        # 构建目标序列 y
        # 目标是输入右移一位：ids[i+1:i+seq_len+1]
        # 这就是"下一个 token 预测"的核心
        y = np.stack([self.ids[i + 1 : i + self.seq_len + 1] for i in indices])

        return x, y


def forward_pass(model, input_ids, targets=None):
    """
    前向传播：计算模型输出和损失

    参数：
        model: Qwen3 模型
        input_ids: 输入 token ID，形状 [batch, seq_len]
        targets: 目标 token ID，形状 [batch, seq_len]

    返回：
        logits: 模型输出，形状 [batch, seq_len, vocab_size]
        loss: 交叉熵损失（如果提供了 targets）
    """
    logits = model.forward(input_ids)
    loss = None
    if targets is not None:
        loss = cross_entropy_loss(logits, targets)
    return logits, loss


def backward_pass(model, input_ids, targets, eps=1e-5):
    """
    反向传播：使用数值梯度计算参数梯度

    数值梯度原理：
        ∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)

    这是最简单但最慢的梯度计算方法：
        - 对每个参数，计算两个前向传播
        - 总共需要 2 * num_parameters 次前向传播
        - 对于 54M 参数的模型，每次更新需要 108M 次前向传播！

    实际训练中应该使用：
        - 解析梯度（手动推导）
        - 自动微分（PyTorch/TensorFlow）

    参数：
        model: 模型
        input_ids: 输入
        targets: 目标
        eps: 差分步长

    返回：
        grads: 梯度列表，与参数一一对应
        base_loss: 当前损失值
    """
    grads = []
    params = model.parameters()

    # 计算当前损失（用于监控，不用于梯度）
    _, base_loss = forward_pass(model, input_ids, targets)

    # 对每个参数计算梯度
    for param in params:
        # 初始化梯度数组，形状与参数相同
        grad = np.zeros_like(param)

        # np.nditer: 多维数组迭代器
        # flags=['multi_index']: 提供当前元素的索引
        # op_flags=[['readwrite']]: 允许读写
        it = np.nditer(param, flags=["multi_index"], op_flags=[["readwrite"]])

        # 遍历参数的每个元素
        while not it.finished:
            idx = it.multi_index  # 当前元素的索引，如 (i, j, k)
            old_val = param[idx]  # 保存原始值

            # 计算 L(θ + ε)
            param[idx] = old_val + eps
            _, loss_plus = forward_pass(model, input_ids, targets)

            # 计算 L(θ - ε)
            param[idx] = old_val - eps
            _, loss_minus = forward_pass(model, input_ids, targets)

            # 恢复原始值
            param[idx] = old_val

            # 中心差分公式
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)

            it.iternext()  # 移动到下一个元素

        grads.append(grad)

    return grads, base_loss


def train(model, dataset, optimizer, num_steps=100, log_interval=10):
    """
    训练循环

    训练步骤：
        1. 获取一个批次数据
        2. 前向传播 + 反向传播（计算梯度）
        3. 更新参数

    参数：
        model: 模型
        dataset: 数据集
        optimizer: 优化器
        num_steps: 训练步数
        log_interval: 日志打印间隔
    """
    losses = []

    for step in range(num_steps):
        # 步骤1: 获取批次数据
        input_ids, targets = dataset.get_batch(batch_size=2)

        # 步骤2: 计算梯度
        grads, loss = backward_pass(model, input_ids, targets)

        # 步骤3: 更新参数
        optimizer.step(grads)

        losses.append(loss)

        # 打印日志
        if step % log_interval == 0:
            avg_loss = (
                np.mean(losses[-log_interval:])
                if len(losses) >= log_interval
                else np.mean(losses)
            )
            print(f"Step {step:4d} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")

    return losses


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    """
    文本生成（自回归生成）

    自回归生成过程：
        1. 将 prompt 编码为 token ID
        2. 循环生成新 token：
           a. 前向传播得到 logits
           b. 取最后一个位置的 logits
           c. 应用 temperature 缩放
           d. 采样下一个 token
           e. 将新 token 添加到序列
        3. 解码生成文本

    参数：
        model: 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数

    Temperature 的作用：
        - temperature > 1: 更随机，更多样化
        - temperature = 1: 正常采样
        - temperature < 1: 更确定，更保守
        - temperature -> 0: 趋近于贪婪解码（选概率最大的）

    公式：
        probs = softmax(logits / temperature)

        当 temperature 很小时，logits/temperature 变大，
        softmax 输出更尖锐，最大概率的 token 更容易被选中
    """
    # 将 prompt 编码为 token ID
    # 形状: [1, seq_len]（batch_size=1）
    input_ids = np.array([tokenizer.encode(prompt)], dtype=np.int64)

    for _ in range(max_new_tokens):
        # 如果序列太长，截断到最大位置编码长度
        if input_ids.shape[1] > cfg.max_position_embeddings:
            input_ids = input_ids[:, -cfg.max_position_embeddings :]

        # 前向传播
        # logits 形状: [1, seq_len, vocab_size]
        logits = model.forward(input_ids)

        # 取最后一个位置的 logits
        # 形状: [vocab_size]
        next_token_logits = logits[0, -1, :] / temperature

        # 计算 softmax 概率
        # 数值稳定版本：减去最大值
        probs = np.exp(next_token_logits - np.max(next_token_logits))
        probs = probs / np.sum(probs)

        # 根据概率分布采样下一个 token
        # np.random.choice: 按概率 p 从 range(len(probs)) 中采样
        next_token = np.random.choice(len(probs), p=probs)

        # 将新 token 添加到序列
        # [[next_token]] 形状: [1, 1]
        # concatenate 后 input_ids 形状: [1, seq_len+1]
        input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

    # 解码生成文本
    return tokenizer.decode(input_ids[0].tolist())


def main():
    """主函数：演示训练和生成"""
    print("=" * 50)
    print("Qwen3 From Scratch - Training Demo")
    print("=" * 50)

    # 训练语料（重复 10 次增加数据量）
    corpus = (
        """
    Once upon a time, there was a little girl named Alice.
    She lived in a small house near a big forest.
    One day, she decided to explore the forest.
    She walked and walked until she found a rabbit hole.
    The rabbit hole was deep and dark.
    Alice was curious, so she jumped into the hole.
    She fell down, down, down into a magical world.
    In this world, she met many strange creatures.
    There was a white rabbit with a pocket watch.
    There was a smiling cat that could disappear.
    There was a mad hatter who loved tea parties.
    Alice had many adventures in this wonderland.
    She grew tall and she grew small.
    She played croquet with the Queen of Hearts.
    Finally, she woke up and realized it was all a dream.
    """
        * 10
    )

    # 创建分词器和数据集
    tokenizer = SimpleTokenizer(vocab_size=1000)
    dataset = Dataset(corpus, tokenizer, seq_len=32)

    print(f"\nCorpus size: {len(corpus)} chars")
    print(f"Tokenized: {len(dataset.ids)} tokens")
    print(f"Vocab size: {tokenizer.vocab_size_actual}")
    print(f"Dataset batches: {len(dataset)}")

    # 创建模型
    model = Qwen3Model(cfg)
    params = model.parameters()
    print(f"\nModel parameters: {sum(p.size for p in params):,}")

    # 创建优化器
    optimizer = AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    # 训练
    print("\nStarting training...")
    print("-" * 50)

    losses = train(model, dataset, optimizer, num_steps=50, log_interval=10)

    print("-" * 50)
    print(f"Final loss: {losses[-1]:.4f}")

    # 生成文本
    print("\nGenerating text...")
    prompt = "once upon a time"
    generated = generate(model, tokenizer, prompt, max_new_tokens=30, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()
