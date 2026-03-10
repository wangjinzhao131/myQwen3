"""
核心张量运算 - 从零实现
只使用 numpy 进行矩阵乘法，不依赖任何深度学习框架
"""

import numpy as np


def softmax(x, axis=-1):
    """
    Softmax 函数：将向量转换为概率分布

    数学公式: softmax(x_i) = exp(x_i) / sum(exp(x_j))

    为什么要减去最大值？
    - 数值稳定性：防止 exp(大数) 溢出
    - 例如: exp(1000) 会溢出，但 exp(1000-1000) = exp(0) = 1

    参数:
        x: 输入张量，形状 [..., dim]
        axis: 在哪个维度上计算 softmax，默认最后一个维度

    举例:
        x = [[1, 2, 3],          # shape (2, 3)
             [1, 2, 3]]
        softmax(x) = [[0.09, 0.24, 0.67],
                      [0.09, 0.24, 0.67]]
    """
    # 第一步：减去最大值（数值稳定性）
    # keepdims=True 保持维度，例如 (2,3) -> (2,1)，方便广播
    x_max = np.max(x, axis=axis, keepdims=True)  # shape: [..., 1]
    x_shifted = x - x_max  # 广播减法，每个元素减去对应的最大值

    # 第二步：计算指数
    exp_x = np.exp(x_shifted)  # shape 与 x 相同

    # 第三步：归一化（除以总和）
    # sum 后 shape 变成 [...]，keepdims=True 保持 [..., 1]
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

    # 广播除法，每个元素除以对应位置的和
    return exp_x / sum_exp  # shape 与 x 相同


def sigmoid(x):
    """
    Sigmoid 函数：S型激活函数

    数学公式: σ(x) = 1 / (1 + exp(-x))

    特性:
        - 输出范围: (0, 1)
        - 当 x -> +∞, σ(x) -> 1
        - 当 x -> -∞, σ(x) -> 0
        - 当 x = 0, σ(x) = 0.5

    用于 SwiGLU 激活函数中的门控机制
    """
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    """
    SiLU (Swish) 激活函数

    数学公式: SiLU(x) = x * sigmoid(x)

    特性:
        - 平滑、非单调
        - 在负值区域有轻微的负输出
        - 比 ReLU 在深度网络中表现更好

    用于 SwiGLU: gate * SiLU(up)
    """
    return x * sigmoid(x)


def rms_norm(x, weight, eps=1e-6):
    """
    RMSNorm (Root Mean Square Normalization)

    与 LayerNorm 的区别:
        - LayerNorm: 减去均值，除以标准差
        - RMSNorm: 只除以均方根，不减均值
        - RMSNorm 计算更快，效果相近

    数学公式:
        RMS(x) = sqrt(mean(x^2) + ε)
        output = (x / RMS) * weight

    参数:
        x: 输入张量，形状 [..., hidden_size]
        weight: 可学习的缩放参数，形状 [hidden_size]
        eps: 防止除零的小常数

    举例:
        x = [[1, 2, 3, 4]]           # shape (1, 4)
        mean(x^2) = (1+4+9+16)/4 = 7.5
        RMS = sqrt(7.5 + 1e-6) ≈ 2.74
        x_norm = x / 2.74 = [[0.365, 0.73, 1.095, 1.46]]
        output = x_norm * weight     # weight 是可学习参数
    """
    # 计算均方值：mean(x^2)
    # axis=-1 表示在最后一个维度（hidden_size）上计算
    # keepdims=True 保持形状 [..., 1]，方便后续广播
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)  # shape: [..., 1]

    # 计算 RMS = sqrt(mean(x^2) + ε)
    rms = np.sqrt(mean_sq + eps)  # shape: [..., 1]

    # 归一化：x / RMS
    x_norm = x / rms  # 广播除法，shape 与 x 相同

    # 缩放：乘以可学习权重
    # weight shape: [hidden_size]，会自动广播到 [..., hidden_size]
    return x_norm * weight


def rotate_half(x):
    """
    RoPE 的核心操作：旋转一半维度

    将向量分成两半，然后交换并取负

    输入: x = [x1, x2, x3, x4, x5, x6, x7, x8]
    输出: [-x5, -x6, -x7, -x8, x1, x2, x3, x4]

    这等价于旋转矩阵:
        [cos θ, -sin θ]
        [sin θ,  cos θ]

    用于 RoPE 的旋转位置编码
    """
    # 获取最后一维的一半位置
    half = x.shape[-1] // 2

    # 分成两半
    x1 = x[..., :half]  # 前一半: [x1, x2, x3, x4]
    x2 = x[..., half:]  # 后一半: [x5, x6, x7, x8]

    # 拼接：[-x2, x1]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, position_ids, theta=10000.0):
    """
    RoPE (Rotary Position Embedding) - 旋转位置编码

    核心思想:
        - 将位置信息编码为旋转角度
        - 不同位置有不同的旋转角度
        - 相对位置可以通过角度差计算

    数学公式:
        对于位置 p 和维度 d:
        angle[p, d] = p / (theta^(2d/D))
        cos_angle = cos(angle)
        sin_angle = sin(angle)

        旋转后的向量:
        q_rotated = q * cos_angle + rotate_half(q) * sin_angle

    参数:
        q: Query 张量，形状 [batch, num_heads, seq_len, head_dim]
        k: Key 张量，形状 [batch, num_kv_heads, seq_len, head_dim]
        position_ids: 位置索引，形状 [seq_len]
        theta: 基础频率，默认 10000

    举例 (简化版):
        假设 head_dim=4, seq_len=2, theta=10000

        position_ids = [0, 1]

        频率计算:
        inv_freq = 1 / (10000^[0/4, 2/4]) = [1.0, 0.1]

        角度计算:
        angles = [[0*1.0, 0*0.1],    # 位置 0
                  [1*1.0, 1*0.1]]    # 位置 1
               = [[0, 0],
                  [1, 0.1]]

        然后扩展到 head_dim 维度，计算 cos 和 sin
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # 步骤1: 计算逆频率
    # 公式: inv_freq[d] = 1 / (theta^(2d/D))
    # 只计算一半维度，因为后面会复制
    # 例如 head_dim=32, 则计算 d=0,2,4,...,30 共16个频率
    dim_indices = np.arange(
        0, head_dim, 2, dtype=np.float32
    )  # [0, 2, 4, ..., head_dim-2]
    inv_freq = 1.0 / (theta ** (dim_indices / head_dim))  # shape: [head_dim//2]

    # 步骤2: 计算每个位置的频率
    # positions shape: [seq_len]
    positions = position_ids.reshape(-1)  # 展平为 1D

    # 外积: positions × inv_freq
    # 结果 shape: [seq_len, head_dim//2]
    # freqs[p, d] = positions[p] * inv_freq[d]
    freqs = np.outer(positions, inv_freq)

    # 步骤3: 扩展到完整维度
    # 将 [seq_len, head_dim//2] 复制拼接为 [seq_len, head_dim]
    # 例如: [[a, b]] -> [[a, b, a, b]]
    emb = np.concatenate([freqs, freqs], axis=-1)  # shape: [seq_len, head_dim]

    # 步骤4: 计算 cos 和 sin
    cos = np.cos(emb).astype(q.dtype)  # shape: [seq_len, head_dim]
    sin = np.sin(emb).astype(q.dtype)  # shape: [seq_len, head_dim]

    # 步骤5: 扩展维度以匹配 q 和 k
    # 从 [seq_len, head_dim] 扩展到 [1, 1, seq_len, head_dim]
    # 这样可以与 [batch, num_heads, seq_len, head_dim] 广播
    cos = cos[np.newaxis, np.newaxis, :, :]  # shape: [1, 1, seq_len, head_dim]
    sin = sin[np.newaxis, np.newaxis, :, :]  # shape: [1, 1, seq_len, head_dim]

    # 步骤6: 应用旋转
    # 公式: q_rotated = q * cos + rotate_half(q) * sin
    # 这里使用了复数旋转的等价形式
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    return q_rotated, k_rotated


def causal_mask(size):
    """
    因果掩码 (Causal Mask) - 用于自回归语言模型

    确保位置 i 只能关注位置 0 到 i（不能看到未来）

    生成一个下三角矩阵:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    1 表示可以关注，0 表示不能关注

    参数:
        size: 序列长度

    返回:
        mask: 形状 [1, 1, size, size]，方便广播
    """
    # np.tril: 取下三角，上三角置0
    mask = np.tril(np.ones((size, size), dtype=np.bool_))

    # 扩展维度: [size, size] -> [1, 1, size, size]
    # 方便与 [batch, num_heads, seq_len, seq_len] 的注意力权重广播
    return mask[np.newaxis, np.newaxis, :, :]


class Tensor:
    """
    简单的张量类（预留用于自动微分）

    当前实现只存储数据，自动微分功能待完善
    """

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None  # 梯度存储

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
