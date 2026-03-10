"""
Core tensor operations implemented from scratch
Using numpy only for matrix operations - no ML libraries
"""

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x):
    return x * sigmoid(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rms_norm(x, weight, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, position_ids, theta=10000.0):
    batch_size, num_heads, seq_len, head_dim = q.shape

    positions = position_ids.reshape(-1)
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.outer(positions, inv_freq)

    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb).astype(q.dtype)
    sin = np.sin(emb).astype(q.dtype)

    cos = cos[np.newaxis, np.newaxis, :, :]
    sin = sin[np.newaxis, np.newaxis, :, :]

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return np.concatenate([-x2, x1], axis=-1)

    q_real = q * cos + rotate_half(q) * sin
    k_real = k * cos + rotate_half(k) * sin

    return q_real, k_real


def causal_mask(size):
    mask = np.tril(np.ones((size, size), dtype=np.bool_))
    return mask[np.newaxis, np.newaxis, :, :]


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
