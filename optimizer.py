"""
Adam 优化器实现 - 从零实现
基于 Kingma & Ba (2014) 论文，包含偏差修正
"""

import numpy as np


class Adam:
    """
    Adam 优化器 (Adaptive Moment Estimation)

    核心思想：
        - 结合 Momentum（动量）和 RMSprop（自适应学习率）
        - 为每个参数维护两个移动平均：一阶矩（梯度）和二阶矩（梯度平方）
        - 使用偏差修正来抵消初始化为 0 带来的偏差

    数学公式：
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          # 一阶矩（梯度的指数移动平均）
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         # 二阶矩（梯度平方的指数移动平均）
        m̂_t = m_t / (1 - β₁^t)                       # 偏差修正后的一阶矩
        v̂_t = v_t / (1 - β₂^t)                       # 偏差修正后的二阶矩
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)        # 参数更新

    参数：
        params: 参数列表，每个参数是 numpy 数组
        lr: 学习率 α，默认 0.001
        betas: (β₁, β₂)，默认 (0.9, 0.95)
        eps: 数值稳定性常数 ε，默认 1e-8
        weight_decay: L2 正则化系数，默认 0
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.lr = lr  # 学习率
        self.beta1, self.beta2 = betas  # 一阶和二阶矩的衰减率
        self.eps = eps  # 防止除零
        self.weight_decay = weight_decay  # L2 正则化系数

        # 一阶矩：梯度的指数移动平均
        # 形状与参数相同，初始化为 0
        self.m = {}

        # 二阶矩：梯度平方的指数移动平均
        # 形状与参数相同，初始化为 0
        self.v = {}

        # 时间步：用于计算偏差修正因子
        self.t = 0

        self.params = params
        # 为每个参数初始化 m 和 v
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(p)  # 与参数形状相同的零数组
            self.v[i] = np.zeros_like(p)

    def step(self, grads):
        """
        执行一步参数更新

        参数：
            grads: 梯度列表，与 params 一一对应

        更新过程详解：

        1. 偏差修正因子：
           - 初始时 m_0 = 0，导致 m_t 偏向于 0
           - 修正因子 1/(1-β^t) 会放大早期的 m_t
           - 随着训练进行（t 增大），修正因子趋近于 1

        2. 一阶矩更新（类似 Momentum）：
           m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
           - β₁ = 0.9 意味着历史梯度占 90%，当前梯度占 10%
           - 平滑梯度，减少震荡

        3. 二阶矩更新（类似 RMSprop）：
           v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
           - β₂ = 0.95 意味着历史梯度平方占 95%
           - 记录梯度的"不确定性"或"变化幅度"

        4. 参数更新：
           θ = θ - α * m̂ / (√v̂ + ε)
           - m̂ 决定更新方向
           - √v̂ 决定每个参数的学习率缩放
           - 梯度变化大的参数，学习率自动减小
           - 梯度变化小的参数，学习率自动增大
        """
        self.t += 1  # 时间步 +1

        # 计算偏差修正因子
        # β^t 会随着 t 增大而趋近于 0，所以修正因子趋近于 1
        bias_correction1 = 1 - self.beta1**self.t  # 一阶矩修正因子
        bias_correction2 = 1 - self.beta2**self.t  # 二阶矩修正因子

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue

            # ========== 更新一阶矩 ==========
            # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
            # 这是一个加权平均：90% 历史 + 10% 当前
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # ========== 更新二阶矩 ==========
            # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
            # grad**2 是逐元素平方
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # ========== 偏差修正 ==========
            # m̂ = m_t / (1 - β₁^t)
            # 修正初始化为 0 带来的偏差
            m_hat = self.m[i] / bias_correction1

            # v̂ = v_t / (1 - β₂^t)
            v_hat = self.v[i] / bias_correction2

            # ========== 计算更新步长 ==========
            # 分母：√v̂ + ε
            # ε 防止 v̂ 为 0 时除零
            denom = np.sqrt(v_hat) + self.eps

            # 更新：θ = θ - α * m̂ / denom
            # 注意：这是逐元素运算
            # 每个参数有自己的"自适应学习率"：α / (√v̂ + ε)
            param -= self.lr * m_hat / denom

            # ========== L2 正则化（可选）==========
            # 标准 Adam 的 weight decay：直接减去 λ * θ
            # 这与 AdamW 不同（见下）
            if self.weight_decay > 0:
                param -= self.lr * self.weight_decay * param

        return self.params


class AdamW:
    """
    AdamW 优化器（解耦的权重衰减）

    与标准 Adam 的区别：
        - Adam: weight decay 与梯度更新耦合
        - AdamW: weight decay 独立于梯度更新

    为什么 AdamW 更好？
        - Adam 中，weight decay 被自适应学习率除，导致正则化不一致
        - AdamW 直接对参数乘以 (1 - αλ)，正则化更稳定

    数学公式：
        θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)    # 梯度更新
        θ_t = θ_t * (1 - αλ)                      # 解耦的权重衰减

    参数：
        weight_decay: 权重衰减系数 λ，默认 0.01
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = {}  # 一阶矩
        self.v = {}  # 二阶矩
        self.t = 0  # 时间步

        self.params = params
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(p)
            self.v[i] = np.zeros_like(p)

    def step(self, grads):
        """
        AdamW 参数更新

        与 Adam 的唯一区别在于 weight decay 的处理方式：

        Adam:
            param -= lr * weight_decay * param  # 减法

        AdamW:
            param *= (1 - lr * weight_decay)    # 乘法（等价但更稳定）

        乘法形式的优点：
            - 与学习率解耦
            - 正则化强度不依赖于自适应学习率
            - 在深度学习中通常效果更好
        """
        self.t += 1

        # 偏差修正因子
        bias_correction1 = 1 - self.beta1**self.t
        bias_correction2 = 1 - self.beta2**self.t

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue

            # 一阶矩更新
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # 二阶矩更新
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # 偏差修正
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            # 计算更新
            denom = np.sqrt(v_hat) + self.eps
            param -= self.lr * m_hat / denom

            # ========== 解耦的权重衰减（AdamW 的关键）==========
            # 直接对参数进行衰减，不依赖于梯度
            # 等价于：param = param - lr * weight_decay * param
            # 但乘法形式更稳定
            if self.weight_decay > 0:
                param *= 1 - self.lr * self.weight_decay

        return self.params


def numerical_gradient(f, x, eps=1e-5):
    """
    数值梯度计算（有限差分法）

    数学原理：
        ∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)

    这是中心差分公式，比单侧差分更精确

    参数：
        f: 目标函数，接受参数 x，返回标量
        x: 参数数组
        eps: 差分步长，默认 1e-5

    返回：
        grad: 与 x 形状相同的梯度数组

    注意：
        - 这是 O(n) 复杂度，n 是参数数量
        - 对于大模型，计算非常慢
        - 实际训练应使用解析梯度或自动微分
    """
    grad = np.zeros_like(x)  # 初始化梯度数组

    # np.nditer: 多维迭代器，遍历数组的每个元素
    # flags=['multi_index']: 提供当前元素的多维索引
    # op_flags=['readwrite']: 允许修改数组元素
    it = np.nditer(x, flags=["multi_index"], op_flags=[["readwrite"]])

    while not it.finished:
        idx = it.multi_index  # 当前元素的索引，如 (i, j, k)
        old_val = x[idx]  # 保存原始值

        # 计算 f(x + ε)
        x[idx] = old_val + eps
        f_plus = f(x)

        # 计算 f(x - ε)
        x[idx] = old_val - eps
        f_minus = f(x)

        # 恢复原始值
        x[idx] = old_val

        # 中心差分公式
        grad[idx] = (f_plus - f_minus) / (2 * eps)

        it.iternext()  # 移动到下一个元素

    return grad
