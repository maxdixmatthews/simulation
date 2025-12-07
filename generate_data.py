import numpy as np
import pandas as pd
from scipy.special import softmax
import statsmodels.api as sm
import zlib
from numpy.random import Generator, PCG64
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_curve, precision_recall_curve, classification_report, confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef, jaccard_score, hamming_loss)
import json
import sqlalchemy as sa

def generate_column(hashed_seed, n_samples, new_col=None, extra_conditions=""):
    # Select distribution type randomly between normal, poisson, gamma
    rg = Generator(PCG64(hashed_seed))
    if new_col is not None:
        dist_type = rg.choice(['normal', 'poisson', 'gamma', 'laplace', 'binomial', 'normal_binomial', 'last_square', 'last_binomial'], p=[0.2, 0.05, 0.1, 0.15, 0.15, 0.2, 0.05, 0.1])
        # dist_type = rg.choice(['normal', 'poisson', 'gamma', 'laplace', 'binomial', 'normal_binomial'])
    else:
        dist_type = rg.choice(['normal', 'poisson', 'gamma', 'laplace', 'binomial', 'normal_binomial'], p=[0.3, 0.1, 0.1, 0.2, 0.25, 0.05])
    if "easy_cols" in extra_conditions:
        dist_type = rg.choice(['normal', 'poisson', 'gamma', 'laplace', 'binomial'], p=[0.3, 0.1, 0.15, 0.2, 0.25])

    if dist_type == 'normal':
        # select mean and stddev randomly skewed towards lower values
        mean = rg.normal(loc=0, scale=2)
        stddev = rg.exponential(scale=1)
        params = (np.around(mean,2), np.around(stddev,2))
        samples =  rg.normal(loc=mean, scale=stddev, size=n_samples)
    elif dist_type == 'poisson':
        # select lambda randomly skewed towards lower values and positive
        lam = rg.exponential(scale=2)
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
    elif dist_type == 'binomial':
        p = rg.uniform(low=0.2, high=0.9, size=None)
        # params = np.around(p,2)
        params = (np.around(p,2),)
        samples = rg.binomial(1, p, size=n_samples)
    elif dist_type == 'normal_binomial':
        p = rg.uniform(low=0.2, high=0.9, size=None)
        params = (np.around(p,2),)
        samples = rg.normal(loc=2, scale=0.2, size=n_samples) * rg.binomial(1, p, size=n_samples)
    elif dist_type == 'last_square':
        params = ""
        samples = new_col ** 2 + rg.normal(loc=0, scale=1)
    elif dist_type == 'last_binomial':
        p = rg.uniform(low=0.2, high=0.9, size=1)
        params = (np.around(p,2),)
        samples = new_col * rg.binomial(1, p, size=n_samples)

    return dist_type + f"{params}", np.around(samples, 2)

def generate_mlr_data(seed, n_features, n_samples, n_classes, cov_beta, intercepts, extra_conditions=""):
    np.set_printoptions(legacy='1.21')
    hashed_seed = int(zlib.crc32((seed*n_features).to_bytes(8, "big")))
    rg = Generator(PCG64(hashed_seed))
    X = np.empty((n_samples, n_features))
    dist_types = []
    new_col = None
    for i in range(0,n_features): 
        hashed_seed = int(zlib.crc32((hashed_seed+1).to_bytes(8, "big")))
        dist_type, new_col = generate_column(hashed_seed, n_samples, new_col, extra_conditions)
        X[:, i] = new_col
        dist_types.append(f"Col{i+1}: {dist_type}")
    w = np.array(cov_beta)
    b = np.array(intercepts)[None, :].T            # scalar intercept
    X_new = X                    # shape (n, p)
    X_new_c = sm.add_constant(X_new) # adds intercept column as first col

    # Build a dummy GLM just so we can use its predict; no fitting
    p = X_new_c.shape[1]
    glm = sm.GLM(endog=np.arange(n_classes), exog=np.zeros((n_classes, X_new_c.shape[1])))
    beta = np.hstack([b, w]).T          # shape (p+1, K-1)
    S = glm.predict(params=beta, exog=X_new_c)  # shape (n,)

    # check if string starts with "add_logit_noise"
    if "add_logit_noise" in extra_conditions:
        sigma = float(extra_conditions.split("=")[1])
        S = S + rg.normal(0.0, sigma, size=S.shape)

    row_max = np.maximum(0.0, S.max(axis=1, keepdims=True))  # include baseline (0) in the max
    exp_non_baseline   = np.exp(S - row_max)                             # non-baseline terms
    base    = np.exp(-row_max)                                # baseline term (exp(0 - row_max))
    denom   = base + exp_non_baseline.sum(axis=1, keepdims=True)

    P_nonbase = exp_non_baseline / denom
    P_base    = base  / denom
    proba     = np.concatenate([P_base, P_nonbase], axis=1)   

    # rg = np.random.default_rng(seed)
    y_hat = np.array([rg.choice(proba.shape[1], p=row) for row in proba], dtype=np.int64)  # shape (10,)
    # y_hat = proba.argmax(axis=1)
    y = y_hat
    counts = np.bincount(y_hat, minlength=n_classes)      
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

    
import ast
from collections import deque

def _parse_tree(obj):
    """Return (by_union, U) where by_union[S] = (L, R) using frozenset labels."""
    if isinstance(obj, tuple):
        t = obj
    elif isinstance(obj, str):
        t = ast.literal_eval(obj)
    else:
        raise TypeError("Input must be a tuple or a string representing a tuple.")

    def is_tuple_of_ints(x):
        return isinstance(x, tuple) and all(isinstance(i, int) for i in x)

    splits = []
    def walk(x):
        if isinstance(x, tuple):
            if len(x) == 2 and all(is_tuple_of_ints(s) for s in x):
                splits.append((x[0], x[1]))
            else:
                for y in x:
                    walk(y)
    walk(t)
    if not splits:
        raise ValueError("No (left,right) splits found.")

    pair_sets = [(frozenset(l), frozenset(r)) for l, r in splits]
    U = frozenset(set().union(*[l | r for l, r in pair_sets]))
    by_union = { (l | r): (l, r) for l, r in pair_sets }
    if U not in by_union:
        raise ValueError("No root split whose union equals all labels.")
    return by_union, U

def depth_first_splits(obj):
    """Preorder list of splits as ((left tuple),(right tuple))."""
    by, U = _parse_tree(obj)
    out = []
    def dfs(S):
        if len(S) == 1: return
        if S not in by: raise ValueError(f"Missing split for subset {tuple(sorted(S))}.")
        L, R = by[S]
        out.append((tuple(sorted(L)), tuple(sorted(R))))
        dfs(L); dfs(R)
    dfs(U)
    return out

# def breadth_first_splits(obj):
#     """Level-order list of splits as ((left tuple),(right tuple))."""
#     by, U = _parse_tree(obj)
#     out, q = [], deque([U])
#     while q:
#         S = q.popleft()
#         if len(S) == 1: continue
#         if S not in by: raise ValueError(f"Missing split for subset {tuple(sorted(S))}.")
#         L, R = by[S]
#         out.append((tuple(sorted(L)), tuple(sorted(R))))
#         q.append(L); q.append(R)
#     return out


def bfs_splits(tree):
    # normalise splits: ints, sorted, bigger side left
    norm = []
    for L, R in tree:
        L = tuple(sorted(int(c) for c in L))
        R = tuple(sorted(int(c) for c in R))
        if (len(R), R) > (len(L), L):
            L, R = R, L
        norm.append((L, R))

    by = {frozenset(L + R): (L, R) for (L, R) in norm}
    root = set().union(*(set(L) | set(R) for (L, R) in norm))

    out, q = [], deque([root])
    while q:
        U = q.popleft()
        if len(U) <= 1:
            continue
        L, R = by[frozenset(U)]
        out.append((L, R))
        q.append(set(L))
        q.append(set(R))
    return tuple(out)

def newick(obj):
    """Newick string '(left,right);' with leaf labels."""
    by, U = _parse_tree(obj)
    def rec(S):
        if len(S) == 1: return str(next(iter(S)))
        if S not in by: raise ValueError(f"Missing split for subset {tuple(sorted(S))}.")
        L, R = by[S]
        return f"({rec(L)},{rec(R)})"
    return rec(U) + ";"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_nd_data(seed, n_features, n_samples, n_classes, cov_list, nd_structure, nd_params=None, extra_conditions=None):
    np.set_printoptions(legacy='1.21')
    covariates_list = cov_list
    if nd_params is not None:
        nd_structure = list(nd_params.keys())
        covariates_list = list(nd_params.values())
        nd_node_desc = {}
        for index, i in enumerate(nd_structure):
            combined = tuple(sorted(i[0] + i[1]))
            nd_node_desc[combined] = i
    else:
        nd_params = {}
        nd_node_desc = {}
        for index, i in enumerate(nd_structure):
            nd_params[i] = covariates_list[index]
            combined = tuple(sorted(i[0] + i[1]))
            nd_node_desc[combined] = i

    hashed_seed = int(zlib.crc32((seed*n_features).to_bytes(8, "big")))
    rg = Generator(PCG64(hashed_seed))
    X = np.empty((n_samples, n_features))
    dist_types = []
    new_col = None
    for i in range(0,n_features): 
        hashed_seed = int(zlib.crc32((hashed_seed+1).to_bytes(8, "big")))
        dist_type, new_col = generate_column(hashed_seed, n_samples, new_col, extra_conditions)
        X[:, i] = new_col
        dist_types.append(f"Col{i+1}: {dist_type}")
        
    columns = [f'p{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)

    Y = []
    for index, row in df.iterrows():
        mutliplyer = [1.0] + row.tolist()
        curr_layer = nd_structure[0]
        while(True):
            cov = nd_params.get(curr_layer)
            res = np.array(cov, dtype=float) * np.array(mutliplyer)
            bern_input = sigmoid(sum(res))
            bern_output = rg.binomial(1, bern_input)
            node_winner = curr_layer[bern_output]
            curr_layer = nd_node_desc.get(node_winner, node_winner)
            if curr_layer not in nd_params: 
                Y.append(curr_layer[0])
                break

    counts = np.bincount(Y, minlength=n_classes)      
    freq   = counts / counts.sum()

    return X, np.array(Y), dist_types, freq

def all_nd_data(row):
    dataset_name = row['name']
    simulation_id = row['simulation_id']
    seed = int(float(row['seed']))
    n_samples = int(float(row['n_samples']))
    n_classes = int(row['n_classes'])
    n_features = int(row['n_features'])
    x_hash = int(row['x_hash'])
    y_hash = int(row['y_hash'])
    full_dataset_hash = int(row['full_dataset_hash'])
    extra_conditions = str(row['extra_condition'])
    cov = ast.literal_eval(row['covariates'])
    nd_structure = ast.literal_eval(row['generated_nd'])
    nd_params = ast.literal_eval(row['nd_params'])

    best_prev_tree = tuple(bfs_splits(tuple(nd_structure)))

    X, y, dist_types, freq = generate_nd_data(seed, n_features, n_samples, n_classes, cov, nd_structure, nd_params=nd_params, extra_conditions=extra_conditions)
    # gd.check_hashes(X, y, x_hash, y_hash, full_dataset_hash)
    df = pd.DataFrame(X, columns=[f"p{i+1}" for i in range(n_features)])
    # df['Y'] = y
    X, X_test, y, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

    categories = tuple(np.unique(y))
    return_dict = {
        "X": X, 
        "y": y,
        "categories": categories,
        "X_test": X_test,
        "y_test":y_test,
        "best_prev_tree":best_prev_tree
    }
    return return_dict

def all_non_nd(row):
    dataset_name = row['name']
    simulation_id = row['simulation_id']
    seed = int(float(row['seed']))
    n_samples = int(float(row['n_samples']))
    n_classes = int(row['n_classes'])
    n_features = int(row['n_features'])
    cov_beta = json.loads(row['covariates'])
    intercepts = json.loads(row['intercept'])
    x_hash = int(row['x_hash'])
    y_hash = int(row['y_hash'])
    full_dataset_hash = int(row['full_dataset_hash'])
    extra_conditions = str(row['extra_condition'])
    # given_best_tree = row['selected_tree']
    test_size = 0.2

    best_prev_tree = None

    X, y, dist_types, proba, freq = generate_mlr_data(seed, n_features, n_samples, n_classes, cov_beta, intercepts, extra_conditions=extra_conditions)
    df = pd.DataFrame(X, columns=[f"p{i+1}" for i in range(n_features)])
    X, X_test, y, y_test = train_test_split(df, y, test_size=test_size, random_state=seed)

    categories = tuple(np.unique(y))
    return_dict = {
        "X": X, 
        "y": y,
        "categories": categories,
        "X_test": X_test,
        "y_test":y_test,
        "best_prev_tree":best_prev_tree
    }
    return return_dict

# def bfs_splits(tree):
#     # normalise each split: sort labels; bigger side left (tie → lexicographic)
#     S = [ (tuple(sorted(L)), tuple(sorted(R))) for (L,R) in tree ]
#     S = [ (max(a,b, key=lambda t:(len(t), t)), min(a,b, key=lambda t:(len(t), t))) for a,b in S ]

#     # map subset → split; find root label set
#     by = { frozenset((*L, *R)): (L, R) for (L, R) in S }
#     root = set().union(*[ set(L)|set(R) for (L,R) in S ])

#     # BFS order
#     out, q = [], deque([root])
#     while q:
#         U = q.popleft()
#         if len(U) <= 1: 
#             continue
#         L, R = by[frozenset(U)]
#         out.append((L, R))
#         q.append(set(L)); q.append(set(R))
#     return tuple(out)

def calculate_metrics(config, y_true, y_pred):
    """
    """
    confusion_matrix_ls = str(confusion_matrix(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    matthews_corr_coef = matthews_corrcoef(y_true, y_pred)
    jaccard_macro = jaccard_score(y_true, y_pred, average='macro')
    hamming_loss_db = hamming_loss(y_true, y_pred)
    
    metrics_dict = {
        "accuracy":[accuracy],        
        "precision_macro":[precision_macro], 
        "precision_micro":[precision_micro], 
        "recall_macro":[recall_macro], 
        "recall_micro":[recall_micro], 
        "f1_macro":[f1_macro], 
        "f1_micro":[f1_micro], 
        "balanced_accuracy":[balanced_accuracy], 
        "cohen_kappa":[cohen_kappa], 
        "matthews_corrcoef":[matthews_corr_coef], 
        "jaccard_macro":[jaccard_macro], 
        "hamming_loss":[hamming_loss_db], 
        "confusion_matrix":[confusion_matrix_ls],
    }
    return pd.DataFrame(metrics_dict)

def df_upsert_postgres(data_frame, table_name, engine, schema=None, match_columns=None, insert_only=False):
    """
    Perform an "upsert" on a PostgreSQL table from a DataFrame.
    Constructs an INSERT … ON CONFLICT statement, uploads the DataFrame to a
    temporary table, and then executes the INSERT.
    Parameters
    ----------
    data_frame : pandas.DataFrame
        The DataFrame to be upserted.
    table_name : str
        The name of the target table.
    engine : sqlalchemy.engine.Engine
        The SQLAlchemy Engine to use.
    schema : str, optional
        The name of the schema containing the target table.
    match_columns : list of str, optional
        A list of the column name(s) on which to match. If omitted, the
        primary key columns of the target table will be used.
    insert_only : bool, optional
        On conflict do not update. (Default: False)
    """
    table_spec = ""
    if schema:
        table_spec += '"' + schema.replace('"', '""') + '".'
    table_spec += '"' + table_name.replace('"', '""') + '"'

    df_columns = list(data_frame.columns)
    if not match_columns:
        insp = sa.inspect(engine)
        match_columns = insp.get_pk_constraint(table_name, schema=schema)[
            "constrained_columns"
        ]
    columns_to_update = [col for col in df_columns if col not in match_columns]
    insert_col_list = ", ".join([f'"{col_name}"' for col_name in df_columns])
    stmt = f"INSERT INTO {table_spec} ({insert_col_list})\n"
    stmt += f"SELECT {insert_col_list} FROM temp_table\n"
    match_col_list = ", ".join([f'"{col}"' for col in match_columns])
    stmt += f"ON CONFLICT ({match_col_list}) DO "
    if insert_only:
        stmt += "NOTHING"
    else:
        stmt += "UPDATE SET\n"
        stmt += ", ".join(
            [f'"{col}" = EXCLUDED."{col}"' for col in columns_to_update]
        )

    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS temp_table")
        conn.exec_driver_sql(
            f"CREATE TEMPORARY TABLE temp_table AS SELECT * FROM {table_spec} WHERE false"
        )
        data_frame.to_sql("temp_table", conn, if_exists="append", index=False)
        conn.exec_driver_sql(stmt)
   