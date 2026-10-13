"""气动数据生成：用理论公式生成 (Ma, alpha, AR) -> (CL, CD) 训练数据"""
import numpy as np


def generate_data(n_samples: int = 20000, seed: int = 42):
    """
    生成气动数据集。
    输入特征：[马赫数 Ma, 迎角 alpha(deg), 展弦比 AR]
    输出标签：[升力系数 CL, 阻力系数 CD]
    """
    rng = np.random.default_rng(seed)

    Ma = rng.uniform(0.3, 0.85, n_samples)
    alpha = rng.uniform(-4, 14, n_samples)
    AR = rng.uniform(5, 13, n_samples)

    # 后掠角固定 25°
    sweep = np.deg2rad(25.0)
    e = 0.80 * np.cos(sweep)

    a0 = 2 * np.pi * np.cos(sweep)
    CL_alpha = a0 / (1 + a0 / (np.pi * AR * e))
    CL_alpha_deg = CL_alpha * np.pi / 180

    alpha_0 = -2.0
    CL = CL_alpha_deg * (alpha - alpha_0)

    CL_max = np.minimum(1.4 + 0.3 * np.log10(AR) * np.cos(sweep), 1.8)
    CL = np.minimum(CL, CL_max)

    CD0 = np.where(
        Ma < 0.7, 0.018,
        np.where(Ma < 0.85,
                 0.018 + 0.06 * (Ma - 0.7) ** 2,
                 0.025 + 0.1 * (Ma - 0.85))
    )

    k = 1.0 / (np.pi * AR * e)
    CD = CD0 + k * CL ** 2

    X = np.stack([Ma, alpha, AR], axis=1).astype(np.float32)
    y = np.stack([CL, CD], axis=1).astype(np.float32)

    return X, y


def normalize(X, y):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0) + 1e-8

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    stats = {
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_mean, "y_std": y_std,
    }
    return X_norm, y_norm, stats