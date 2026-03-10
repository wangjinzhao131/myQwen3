"""
Qwen3 模型实现 - 从零实现
只使用 numpy 进行矩阵运算，不依赖 PyTorch/TensorFlow
"""

import numpy as np
from config import cfg
from tensor_ops import softmax, silu, rms_norm, apply_rope, causal_mask


class Linear:
    """
    线性层（全连接层）

    数学公式: y = x @ W + b

    这是神经网络最基本的组件，所有矩阵乘法都在这里发生

    参数:
        in_features: 输入特征维度
        out_features: 输出特征维度
        bias: 是否使用偏置
    """

    def __init__(self, in_features, out_features, bias=True):
        # 权重初始化：Xavier/He 初始化
        # scale = sqrt(2/in_features) 适合 ReLU/SiLU 激活函数
        # 这样初始化可以让前向传播时方差保持稳定
        scale = np.sqrt(2.0 / in_features)

        # 权重矩阵：形状 [in_features, out_features]
        # 注意：PyTorch 是 [out_features, in_features]，我们用 [in, out] 更直观
        # np.random.randn 生成标准正态分布 N(0,1)
        self.weight = (
            np.random.randn(in_features, out_features).astype(np.float32) * scale
        )

        # 偏置：形状 [out_features]，初始化为 0
        self.bias = np.zeros(out_features, dtype=np.float32) if bias else None

    def __call__(self, x):
        """
        前向传播：矩阵乘法

        输入 x: 形状 [..., in_features]
        输出 y: 形状 [..., out_features]

        矩阵乘法详解:
            x @ weight
            x: [..., in_features]
            weight: [in_features, out_features]
            结果: [..., out_features]

        举例:
            x: [batch=2, seq=10, in=512]
            weight: [512, 2048]
            x @ weight: [2, 10, 2048]

        广播机制:
            - 最后两维做矩阵乘法
            - 前面的维度自动广播
        """
        if self.bias is not None:
            # x @ weight: [..., out_features]
            # + bias: [out_features] 自动广播到 [..., out_features]
            return x @ self.weight + self.bias
        return x @ self.weight

    def parameters(self):
        """返回所有可学习参数"""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Embedding:
    """
    词嵌入层

    将离散的 token ID 转换为连续的向量表示

    本质上是一个查找表：
        embedding[id] -> 向量

    参数:
        num_embeddings: 词汇表大小（有多少个不同的 token）
        embedding_dim: 嵌入向量维度
    """

    def __init__(self, num_embeddings, embedding_dim):
        # 嵌入矩阵：形状 [vocab_size, hidden_size]
        # 每一行是一个 token 的向量表示
        # 初始化为小随机值（标准差 0.02）
        self.embedding = (
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        )

    def __call__(self, x):
        """
        查找嵌入向量

        输入 x: token ID 数组，形状 [batch, seq_len]
        输出: 嵌入向量，形状 [batch, seq_len, hidden_size]

        原理:
            embedding[x] 使用 numpy 的高级索引
            对于 x 中的每个 ID，返回 embedding 矩阵的对应行

        举例:
            x = [[1, 5, 3],      # shape (2, 3)
                 [2, 0, 7]]
            embedding: [10000, 512]

            embedding[x] 会返回:
            - embedding[1], embedding[5], embedding[3] 作为第一行
            - embedding[2], embedding[0], embedding[7] 作为第二行
            结果 shape: (2, 3, 512)
        """
        return self.embedding[x]

    def parameters(self):
        return [self.embedding]


class GroupedQueryAttention:
    """
    分组查询注意力 (Grouped Query Attention, GQA)

    GQA 是 Multi-Head Attention 的优化版本：
        - 标准 MHA: 每个 head 有独立的 Q, K, V
        - GQA: 多个 Q head 共享同一组 K, V

    优点:
        - 减少 KV Cache 大小（推理时更省内存）
        - 保持接近 MHA 的性能

    举例:
        num_heads = 16 (Q 的头数)
        num_kv_heads = 2 (K, V 的头数)
        每个 KV head 被 8 个 Q head 共享 (16/2=8)
    """

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        self.hidden_size = hidden_size  # 隐藏层维度，如 512
        self.num_heads = num_heads  # Q 的头数，如 16
        self.num_kv_heads = num_kv_heads  # K, V 的头数，如 2
        self.head_dim = head_dim  # 每个头的维度，如 32

        # Q 投影：hidden_size -> num_heads * head_dim
        # 例如: 512 -> 16 * 32 = 512
        self.q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)

        # K 投影：hidden_size -> num_kv_heads * head_dim
        # 例如: 512 -> 2 * 32 = 64
        self.k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)

        # V 投影：hidden_size -> num_kv_heads * head_dim
        self.v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)

        # 输出投影：num_heads * head_dim -> hidden_size
        self.o_proj = Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, position_ids, attention_mask=None):
        """
        GQA 前向传播

        输入:
            x: 形状 [batch, seq_len, hidden_size]
            position_ids: 位置索引 [seq_len]

        输出:
            形状 [batch, seq_len, hidden_size]

        步骤:
            1. 线性投影得到 Q, K, V
            2. 应用 RoPE 位置编码
            3. 扩展 K, V（GQA 的关键）
            4. 计算注意力分数
            5. 应用因果掩码
            6. Softmax 归一化
            7. 加权求和
            8. 输出投影
        """
        batch_size, seq_len, _ = x.shape

        # ========== 步骤1: 线性投影 ==========
        # Q 投影: [batch, seq, hidden] @ [hidden, num_heads*head_dim]
        # 结果: [batch, seq, num_heads*head_dim]
        q = self.q_proj(x)

        # 重塑为 [batch, seq, num_heads, head_dim]
        # 这样每个 head 有独立的 head_dim 维度
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # K, V 同理，但头数更少
        k = self.k_proj(x).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(x).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        # ========== 步骤2: 转置为注意力计算格式 ==========
        # 从 [batch, seq, heads, dim] 转为 [batch, heads, seq, dim]
        # 这样可以在 heads 维度上并行计算
        q = np.transpose(q, (0, 2, 1, 3))  # [batch, num_heads, seq, head_dim]
        k = np.transpose(k, (0, 2, 1, 3))  # [batch, num_kv_heads, seq, head_dim]
        v = np.transpose(v, (0, 2, 1, 3))  # [batch, num_kv_heads, seq, head_dim]

        # ========== 步骤3: 应用 RoPE 位置编码 ==========
        # RoPE 只应用于 Q 和 K，不应用于 V
        # 因为位置信息需要在计算注意力分数时体现
        q, k = apply_rope(q, k, position_ids, cfg.rope_theta)

        # ========== 步骤4: GQA 扩展 K, V ==========
        # GQA 的核心：将 K, V 复制多份以匹配 Q 的头数
        # num_kv_groups = num_heads / num_kv_heads
        # 例如: 16 / 2 = 8，每个 KV head 被复制 8 次
        num_kv_groups = self.num_heads // self.num_kv_heads

        if num_kv_groups > 1:
            # np.repeat: 沿 axis=1（heads 维度）复制
            # k: [batch, num_kv_heads, seq, dim] -> [batch, num_heads, seq, dim]
            # 例如: [1, 2, 10, 32] -> [1, 16, 10, 32]
            k = np.repeat(k, num_kv_groups, axis=1)
            v = np.repeat(v, num_kv_groups, axis=1)

        # ========== 步骤5: 计算注意力分数 ==========
        # 注意力公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

        # Q @ K^T: [batch, heads, seq, dim] @ [batch, heads, dim, seq]
        # 结果: [batch, heads, seq, seq] - 每个位置对每个位置的注意力分数
        #
        # 矩阵乘法详解:
        #   q: [batch, heads, seq_q, dim]
        #   k^T: [batch, heads, dim, seq_k]
        #   q @ k^T: [batch, heads, seq_q, seq_k]
        #   其中 (seq_q, dim) @ (dim, seq_k) = (seq_q, seq_k)
        attn_weights = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))

        # 缩放：除以 sqrt(head_dim)
        # 为什么缩放？防止点积过大导致 softmax 梯度消失
        # 当 dim 很大时，点积的方差也很大，softmax 会变得很尖锐
        attn_weights = attn_weights / np.sqrt(self.head_dim)

        # ========== 步骤6: 应用因果掩码 ==========
        # 因果掩码：确保位置 i 只能看到位置 0 到 i
        # mask: [1, 1, seq, seq]，下三角为 True
        mask = causal_mask(seq_len)

        # np.where: mask 为 True 的位置保持原值，否则设为 -1e9
        # -1e9 在 softmax 后会变成接近 0 的概率
        attn_weights = attn_weights + np.where(mask, 0, -1e9)

        # ========== 步骤7: Softmax 归一化 ==========
        # 在最后一个维度（seq_k）上做 softmax
        # 结果：每行是一个概率分布，和为 1
        attn_probs = softmax(attn_weights, axis=-1)

        # ========== 步骤8: 加权求和 ==========
        # attn_probs @ V: [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, dim]
        # 结果: [batch, heads, seq_q, dim]
        #
        # 这一步的含义：
        #   对于每个 query 位置，用注意力概率对 V 进行加权求和
        #   得到该位置的上下文表示
        attn_output = np.matmul(attn_probs, v)

        # ========== 步骤9: 恢复形状 ==========
        # 转置回 [batch, seq, heads, dim]
        attn_output = np.transpose(attn_output, (0, 2, 1, 3))

        # 展平为 [batch, seq, heads*dim] = [batch, seq, hidden_size]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # ========== 步骤10: 输出投影 ==========
        # 最后一个线性层，将多头输出合并
        return self.o_proj(attn_output)

    def parameters(self):
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.o_proj.parameters()
        )


class SwiGLUFFN:
    """
    SwiGLU 前馈神经网络

    SwiGLU = Swish + GLU (Gated Linear Unit)

    数学公式:
        FFN_SwiGLU(x) = (Swish(x @ W1) * (x @ W3)) @ W2
                      = (SiLU(x @ W1) * (x @ W3)) @ W2

    与标准 FFN 的区别:
        标准 FFN: ReLU(x @ W1) @ W2
        SwiGLU: 多了一个门控机制 (x @ W3)

    参数量:
        标准 FFN: 2 * hidden * intermediate
        SwiGLU: 3 * hidden * intermediate (多了 W3)

    优点:
        - 门控机制允许模型选择性地传递信息
        - SiLU 比 ReLU 更平滑，梯度更稳定
    """

    def __init__(self, hidden_size, intermediate_size):
        # W1: 上投影，hidden -> intermediate
        # 用于计算"激活值"
        self.w1 = Linear(hidden_size, intermediate_size, bias=False)

        # W3: 门控投影，hidden -> intermediate
        # 用于计算"门控值"
        self.w3 = Linear(hidden_size, intermediate_size, bias=False)

        # W2: 下投影，intermediate -> hidden
        # 输出层
        self.w2 = Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        """
        SwiGLU 前向传播

        输入 x: [batch, seq, hidden]
        输出: [batch, seq, hidden]

        计算过程:
            1. gate = x @ W1 -> [batch, seq, intermediate]
            2. gate = SiLU(gate)  # 激活函数
            3. up = x @ W3 -> [batch, seq, intermediate]
            4. hidden = gate * up  # 逐元素相乘（门控）
            5. output = hidden @ W2 -> [batch, seq, hidden]

        门控的含义:
            - gate 决定哪些信息应该通过
            - up 提供实际的信息内容
            - gate * up: 选择性地传递信息
        """
        # x @ W1: [batch, seq, intermediate]
        # silu(...): 应用 SiLU 激活函数
        gate = silu(self.w1(x))

        # x @ W3: [batch, seq, intermediate]
        up = self.w3(x)

        # 门控：逐元素相乘
        # gate * up: [batch, seq, intermediate]
        hidden = gate * up

        # 下投影回 hidden_size
        return self.w2(hidden)

    def parameters(self):
        return self.w1.parameters() + self.w3.parameters() + self.w2.parameters()


class TransformerBlock:
    """
    Transformer 块

    结构 (Pre-Norm):
        x -> RMSNorm -> Attention -> + -> RMSNorm -> FFN -> + -> output
        |                              |                         |
        +------------------------------+-------------------------+

    Pre-Norm vs Post-Norm:
        - Pre-Norm: 先归一化再进入子层（Qwen3 使用）
        - Post-Norm: 先进入子层再归一化（原始 Transformer）
        - Pre-Norm 训练更稳定，梯度传播更好
    """

    def __init__(
        self, hidden_size, num_heads, num_kv_heads, intermediate_size, rms_norm_eps
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # 注意力层
        self.attention = GroupedQueryAttention(
            hidden_size, num_heads, num_kv_heads, hidden_size // num_heads
        )

        # 前馈网络
        self.ffn = SwiGLUFFN(hidden_size, intermediate_size)

        # 两个 RMSNorm 的可学习权重
        # 初始化为全 1，训练时会学习
        self.input_layernorm_weight = np.ones(hidden_size, dtype=np.float32)
        self.post_attention_layernorm_weight = np.ones(hidden_size, dtype=np.float32)

    def forward(self, x, position_ids, attention_mask=None):
        """
        Transformer 块前向传播

        输入输出形状相同: [batch, seq, hidden]

        残差连接的作用:
            - 缓解梯度消失
            - 允许梯度直接流向浅层
            - 使深层网络更容易训练
        """
        # ========== 第一个子层: Attention ==========
        residual = x  # 保存输入用于残差连接

        # Pre-Norm: 先归一化
        x = rms_norm(x, self.input_layernorm_weight, cfg.rms_norm_eps)

        # 注意力计算
        x = self.attention.forward(x, position_ids, attention_mask)

        # 残差连接
        x = x + residual

        # ========== 第二个子层: FFN ==========
        residual = x  # 保存输入用于残差连接

        # Pre-Norm
        x = rms_norm(x, self.post_attention_layernorm_weight, cfg.rms_norm_eps)

        # 前馈网络
        x = self.ffn(x)

        # 残差连接
        x = x + residual

        return x

    def parameters(self):
        params = self.attention.parameters() + self.ffn.parameters()
        params.extend(
            [self.input_layernorm_weight, self.post_attention_layernorm_weight]
        )
        return params


class Qwen3Model:
    """
    Qwen3 语言模型

    架构:
        Token Embedding -> N × TransformerBlock -> RMSNorm -> LM Head

    这是一个因果语言模型 (Causal Language Model)：
        - 给定前文，预测下一个 token
        - 使用交叉熵损失训练
    """

    def __init__(self, config):
        self.config = config

        # 词嵌入层
        # vocab_size 个 token，每个 token 用 hidden_size 维向量表示
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)

        # N 个 Transformer 块
        self.layers = []
        for _ in range(config.num_hidden_layers):
            self.layers.append(
                TransformerBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.intermediate_size,
                    config.rms_norm_eps,
                )
            )

        # 最终的 RMSNorm
        self.norm = np.ones(config.hidden_size, dtype=np.float32)

        # 语言模型头：hidden_size -> vocab_size
        # 输出每个 token 的 logits（未归一化的概率）
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        """
        模型前向传播

        输入:
            input_ids: token ID 数组，形状 [batch, seq_len]

        输出:
            logits: 形状 [batch, seq_len, vocab_size]
            logits[b, s, v] 表示批次 b 中位置 s 预测 token v 的分数

        流程:
            1. Token Embedding: [batch, seq] -> [batch, seq, hidden]
            2. 通过 N 个 Transformer 块
            3. 最终 RMSNorm
            4. LM Head: [batch, seq, hidden] -> [batch, seq, vocab]
        """
        batch_size, seq_len = input_ids.shape

        # 位置索引: [0, 1, 2, ..., seq_len-1]
        position_ids = np.arange(seq_len, dtype=np.int64)[np.newaxis, :]

        # ========== 步骤1: 词嵌入 ==========
        # [batch, seq] -> [batch, seq, hidden]
        hidden_states = self.embed_tokens(input_ids)

        # ========== 步骤2: 通过 Transformer 块 ==========
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, position_ids, attention_mask)

        # ========== 步骤3: 最终归一化 ==========
        hidden_states = rms_norm(hidden_states, self.norm, self.cfg_rms_norm_eps)

        # ========== 步骤4: LM Head ==========
        # [batch, seq, hidden] -> [batch, seq, vocab]
        logits = self.lm_head(hidden_states)

        return logits

    @property
    def cfg_rms_norm_eps(self):
        return self.config.rms_norm_eps

    def parameters(self):
        """返回所有可学习参数"""
        params = self.embed_tokens.parameters()
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend([self.norm])
        params.extend(self.lm_head.parameters())
        return params


def cross_entropy_loss(logits, targets):
    """
    交叉熵损失函数

    数学公式:
        loss = -log(softmax(logits)[target])

    用于语言模型训练：
        - logits: 模型输出的词汇表概率分布
        - targets: 真实的下一个 token ID

    参数:
        logits: 形状 [batch, seq, vocab]
        targets: 形状 [batch, seq]

    返回:
        标量损失值
    """
    batch_size, seq_len, vocab_size = logits.shape

    # 展平为 [batch*seq, vocab]
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # ========== 数值稳定的 Softmax ==========
    # 减去最大值防止溢出
    logits_max = np.max(logits_flat, axis=-1, keepdims=True)
    logits_exp = np.exp(logits_flat - logits_max)

    # 计算概率
    probs = logits_exp / np.sum(logits_exp, axis=-1, keepdims=True)

    # ========== 提取目标 token 的概率 ==========
    # 使用高级索引：probs[i, targets_flat[i]] 取出每个样本目标 token 的概率
    # np.arange(batch*seq) 生成 [0, 1, 2, ..., batch*seq-1]
    # 这相当于对每个样本，取出其目标 token 的概率
    target_probs = probs[np.arange(batch_size * seq_len), targets_flat]

    # ========== 计算负对数似然 ==========
    # -log(p) + 1e-9 防止 log(0)
    loss = -np.mean(np.log(target_probs + 1e-9))

    return loss


def test_model():
    """测试模型是否能正常运行"""
    model = Qwen3Model(cfg)
    input_ids = np.random.randint(0, cfg.vocab_size, (2, 10))
    logits = model.forward(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: {(2, 10, cfg.vocab_size)}")


if __name__ == "__main__":
    test_model()
