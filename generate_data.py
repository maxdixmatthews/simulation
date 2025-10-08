import numpy as np
import pandas as pd
from scipy.special import softmax
import statsmodels.api as sm
import zlib
from numpy.random import Generator, PCG64

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

def generate_column(hashed_seed, n_samples):
    # Select distribution type randomly between normal, poisson, gamma
    rg = Generator(PCG64(hashed_seed))
    dist_type = rg.choice(['normal', 'poisson', 'gamma', 'laplace'])
    if dist_type == 'normal':
        # change all params to be a dictionary
        # select mean and stddev randomly skewed towards lower values
        mean = rg.normal(loc=0, scale=2)
        stddev = rg.exponential(scale=1)
        params = (np.around(mean,2), np.around(stddev,2))
        samples =  rg.normal(loc=mean, scale=stddev, size=n_samples)
    elif dist_type == 'poisson':
        # select lambda randomly skewed towards lower values and positive
        lam = rg.exponential(scale=4)
        params = (np.around(lam,2),)
        samples = rg.poisson(lam=lam, size=n_samples)
    elif dist_type == 'gamma':
        # select shape and scale randomly skewed towards lower values and positive
        shape = rg.exponential(scale=2)
        scale = rg.exponential(scale=1)
        params = (np.around(shape,2), np.around(scale,2))
        samples = rg.gamma(shape=shape, scale=scale, size=n_samples)
    elif dist_type == 'laplace':
        mean = rg.normal(loc=0, scale=1)
        scale = rg.exponential(scale=1)
        params = (np.around(mean,2), np.around(scale,2))
        samples = rg.laplace(loc=mean, scale=scale, size=n_samples)
    return dist_type + f"{params}", np.around(samples, 2)

def generate_mlr_data(seed, n_features, n_samples, n_classes, cov_beta, intercepts):
    np.set_printoptions(legacy='1.25')
    hashed_seed = int(zlib.crc32((seed*n_features).to_bytes(8, "big")))
    X = np.empty((n_samples, n_features))
    dist_types = []
    for i in range(0,n_features): 
        hashed_seed = int(zlib.crc32((hashed_seed+1).to_bytes(8, "big")))
        dist_type, new_col = generate_column(hashed_seed, n_samples)
        X[:, i] = new_col
        print(f"Col{i+1}: {dist_type}")
        dist_types.append(f"Col{i+1}: {dist_type}")

    w = np.array(cov_beta)  

    b = np.array(intercepts)[None, :].T            # scalar intercept

    X_new = X                    # shape (n, p)
    X_new_c = sm.add_constant(X_new) # adds intercept column as first col

    # Build a dummy GLM just so we can use its predict; no fitting
    glm = sm.GLM(endog=np.arange(n_classes), exog=np.zeros((n_classes, X_new_c.shape[1])))

    beta = np.hstack([b, w]).T          # shape (p+1, K-1)
    S = glm.predict(params=beta, exog=X_new_c)  # shape (n,)

    row_max = np.maximum(0.0, S.max(axis=1, keepdims=True))  # include baseline (0) in the max
    exp_non_baseline   = np.exp(S - row_max)                             # non-baseline terms
    base    = np.exp(-row_max)                                # baseline term (exp(0 - row_max))
    denom   = base + exp_non_baseline.sum(axis=1, keepdims=True)

    P_nonbase = exp_non_baseline / denom
    P_base    = base  / denom
    proba     = np.concatenate([P_base, P_nonbase], axis=1)   
    y_hat = proba.argmax(axis=1)    
    y = y_hat
    counts = np.bincount(y_hat, minlength=4)      # a is your 1D array of 0/1/2/3
    freq   = counts / counts.sum()
    return X, y, dist_types, proba, freq

def check_hashes(X, y, x_hash, y_hash, full_hash):
    full_dataset = np.column_stack((X, y))
    full_dataset_hash = int(zlib.crc32(full_dataset.tobytes()))
    if int(zlib.crc32(X.tobytes())) != x_hash:
        raise ValueError(f"X hash mismatch: {int(zlib.crc32(X.tobytes()))} != {x_hash}")
    if int(zlib.crc32(y.tobytes())) != y_hash:
        raise ValueError(f"y hash mismatch: {int(zlib.crc32(y.tobytes()))} != {y_hash}")
    if full_dataset_hash != full_hash:
        raise ValueError(f"Full dataset hash mismatch: {full_dataset_hash} != {full_hash}")
