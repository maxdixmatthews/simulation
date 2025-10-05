import numpy as np
import pandas as pd
from scipy.special import softmax

def simulate_mnlogit(
    n_samples: int,
    n_classes: int,
    n_features: int,
    *,
    base_rates=None,          # e.g. [0.6,0.3,0.1]; if None => uniform
    coef_scale: float = 1.0,  # std for coefficients
    class_sep: float = 1.0,   # scales logits (↑ easier, ↓ harder)
    intercept: bool = True,
    n_informative: int | None = None,  # if None => all features informative
    feature_scale: float = 1.0,
    flip_y: float = 0.0,               # label noise fraction in [0,1]
    seed: int | None = None,           # if None => hash of params (deterministic)
    return_probas: bool = False,
):
    """Deterministic multinomial-logit simulator with minimal dependencies."""
    rng = np.random.default_rng(seed)

    # ----- sizes / basic checks -----
    K, p, n = int(n_classes), int(n_features), int(n_samples)
    if K < 2: raise ValueError("n_classes must be >= 2")
    if p < 1: raise ValueError("n_features must be >= 1")
    n_inf = p if n_informative is None else int(n_informative)
    if not (1 <= n_inf <= p): raise ValueError("n_informative must be in [1, n_features]")

    # ----- class priors -----
    if base_rates is None:
        pri = np.full(K, 1.0 / K)
    else:
        pri = np.asarray(base_rates, float)
        if pri.shape != (K,) or not np.isclose(pri.sum(), 1.0):
            raise ValueError("base_rates must be length K and sum to 1")

    # ----- features: Gaussian -----
    X = rng.normal(0.0, feature_scale, size=(n, p))

    # ----- parameters -----
    W = np.zeros((p, K))
    W[:n_inf, :] = rng.normal(0.0, coef_scale, size=(n_inf, K))
    W *= class_sep
    b = np.log(np.clip(pri, 1e-12, None)) - (0 if not intercept else np.mean(np.log(np.clip(pri,1e-12,None))))
    if not intercept: b = np.zeros(K)

    # ----- probabilities & labels -----
    P = softmax(X @ W + b, axis=1)
    cumP = np.cumsum(P, axis=1)
    y = (rng.random(n)[:, None] > cumP).sum(axis=1).astype(int)

    # optional label noise (flip to a different class uniformly)
    if flip_y > 0:
        flips = rng.random(n) < float(flip_y)
        if flips.any():
            r = rng.integers(0, K-1, size=flips.sum())
            yf = y[flips]
            y[flips] = r + (r >= yf)

    df = pd.DataFrame(X, columns=[f"X{j}" for j in range(p)])
    df["Y"] = y 
    if return_probas:
        for k in range(K): df[f"p{k}"] = P[:, k]

    params = {"W": W, "b": b, "seed_used": seed}
    return df, params