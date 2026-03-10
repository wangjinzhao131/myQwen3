"""
Adam Optimizer Implementation - From Scratch
Based on Kingma & Ba (2014) with bias correction
"""

import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (uncentered variance)
        self.t = 0  # Timestep

        self.params = params
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(p)
            self.v[i] = np.zeros_like(p)

    def step(self, grads):
        self.t += 1

        bias_correction1 = 1 - self.beta1**self.t
        bias_correction2 = 1 - self.beta2**self.t

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            denom = np.sqrt(v_hat) + self.eps
            param -= self.lr * m_hat / denom

            if self.weight_decay > 0:
                param -= self.lr * self.weight_decay * param

        return self.params


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = {}
        self.v = {}
        self.t = 0

        self.params = params
        for i, p in enumerate(params):
            self.m[i] = np.zeros_like(p)
            self.v[i] = np.zeros_like(p)

    def step(self, grads):
        self.t += 1

        bias_correction1 = 1 - self.beta1**self.t
        bias_correction2 = 1 - self.beta2**self.t

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            denom = np.sqrt(v_hat) + self.eps
            param -= self.lr * m_hat / denom

            if self.weight_decay > 0:
                param *= 1 - self.lr * self.weight_decay

        return self.params


def compute_gradients(loss, params, forward_fn, inputs, targets):
    pass


def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        x[idx] = old_val + eps
        f_plus = f(x)
        x[idx] = old_val - eps
        f_minus = f(x)
        x[idx] = old_val
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad
