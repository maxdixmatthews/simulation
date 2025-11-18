import numpy as np
import pandas as pd
import generate_data as gd
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations
import math
from scipy.special import factorial2
import random
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from numpy.random import Generator, PCG64
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np
from bokeh.plotting import figure, show, save, output_file
from bokeh.layouts import column as bcol, row as brow   
from bokeh.models import ColumnDataSource, Div
from bokeh.palettes import Category10
from graphviz import Source
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import silhouette_score
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Inferno256, Blues, BuGn, RdYlGn
from bokeh.transform import linear_cmap, factor_mark
from bokeh.models import LinearColorMapper, ColorBar, Title
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from statsmodels.tools.sm_exceptions import PerfectSeparationError, ConvergenceWarning

global cache

def v2_defined_all_trees(n: int):
    """Return a list of all ND trees over labels 0..n-1 (each tree is a tuple of splits)."""
    return list(_gen(tuple(range(n))))

def _gen(labels):
    labels = tuple(sorted(labels))
    if len(labels) <= 1:
        yield ()
        return
    s = labels[0]  # canonical rule: the smallest label stays on the 'left' side
    for r in range(1, len(labels)):
        for rest in combinations(labels[1:], r - 1):
            left  = tuple(sorted((s,) + rest))
            right = tuple(x for x in labels if x not in left)
            # normalize pair so children are unordered: larger side first, then lexicographic
            pair  = tuple(sorted((left, right), key=lambda t: (-len(t), t)))

            L = ((),) if len(left)  == 1 else tuple(_gen(left))
            R = ((),) if len(right) == 1 else tuple(_gen(right))

            for lt in L:
                for rt in R:
                    # pre-order DFS (shortest code). Swap to (lt+rt+(pair,)) if you want post-order.
                    yield (pair,) + lt + rt

def _canon_split(left_set, right_set):
    """Canonicalize a split (A,B) so (A,B)==(B,A) for hashing/caching."""
    A = tuple(sorted(left_set))
    B = tuple(sorted(right_set))
    return (A, B) if A <= B else (B, A)

def tree_signatures(tree):
    """
    Tree is a tuple/list of splits from defined_all_trees:
      ((A1,B1), (A2,B2), ..., (A_{n-1},B_{n-1}))
    Return a frozenset of canonicalized splits.
    """
    return frozenset(_canon_split(a, b) for (a, b) in tree)

def split_loglik_star_sm(X, y, split, maxiter=200, eps=1e-12):
    """
    ℓ*(split) using statsmodels Logit (unpenalized MLE).
    Returns the node log-likelihood at the MLE (float).
    Degenerate targets (all 0/1) → return 0.0 (supremum in the limit).
    """
    A, B = split
    mask = np.isin(y, A + B)
    if not np.any(mask):
        return 0.0
    s = np.isin(y[mask], B).astype(int)

    if s.min() == s.max():
        return 0.0

    try: 
        # Direct log-likelihood at the MLE:
        lr = LogisticRegression(penalty=None, solver="newton-cholesky", max_iter=maxiter)
        lr.fit(X[mask], s)
        p = lr.predict_proba(X[mask])[:, 1]
        return float(-log_loss(s, p, normalize=False))
    except PerfectSeparationError:
        print("Perfect Separation Error.")
        return 0.0
    except Exception as e:
        raise e
        # Any numerical prob: treat as saturated/flat
        print("Failed to do normal so we are fitting regularized.")
        p = np.clip(sm.Logit(s, Xsub).fit_regularized(alpha=1e-6, L1_wt=0.0, maxiter=maxiter).predict(), eps, 1-eps)
        return float((s*np.log(p) + (1-s)*np.log(1-p)).sum())

def delta_loglik(X, y, sig_T, sig_Tstar):
    """Compute log L(T*) - log L(T) by only evaluating splits in the symmetric difference."""
    add = sig_Tstar - sig_T
    add_vals = []
    rem = sig_T - sig_Tstar
    rem_vals = []
    
    for s in add:
        if s not in cache:
            log_lik = split_loglik_star_sm(X, y, s)
            cache[s] = log_lik
        else: 
            log_lik = cache[s]
        add_vals.append(log_lik)
    for s in rem:
        if s not in cache:
            log_lik = split_loglik_star_sm(X, y, s)
            cache[s] = log_lik
        else: 
            log_lik = cache[s]
        rem_vals.append(log_lik)
    
    return sum(add_vals) - sum(rem_vals)

def non_unif_local_swap(T, rg, alpha = 0.5):
    all_splits = list(T)
    eligible_splits = [x for x in all_splits if len(x[0]) > 1 or len(x[1]) > 1]

    node_to_perform_on = random.choices(
        eligible_splits,
        weights=[1 / (len(a) + len(b))**alpha for a, b in eligible_splits],
        k=1
    )[0]
    # local_item_swap
    L = list(node_to_perform_on[0])
    R = list(node_to_perform_on[1])
    left_swap = random.sample(L, 1)[0]
    right_swap = random.sample(R, 1)[0]
    # print(f"Swapping {left_swap} with {right_swap}")
    # print(f"Original node: {node_to_perform_on}")
    L.remove(left_swap)
    R.remove(right_swap)
    new_L = L + [right_swap]
    new_R = R + [left_swap]
    new_node = (tuple(new_L), tuple(new_R))

    # Now cascade the changes down the tree
    T = list(T)
    S   = set(new_node[0]) | set(new_node[1])
    U = [set(L)|set(R) for (L,R) in T]
    desc_idx = [k for k in range(0, len(T)) if U[k].issubset(S)]

    for k in desc_idx:
        L, R = T[k]
        A = np.concatenate([np.asarray(L), np.asarray(R)])  
        ma, mb = (A == left_swap), (A == right_swap)
        if ma.any() or mb.any():
            A[ma], A[mb] = right_swap, left_swap
            nL = len(L)
            c = [sorted(A[:nL]), sorted(A[nL:])]
            c.sort(key=len, reverse=True)
            l_sub = c[0]
            r_sub = c[1]
            T[k] = (tuple(l_sub), tuple(r_sub))

    return gd.bfs_splits(T), 0.0

def non_unif_entire_node_swap(T, rg, alpha = 0.5):
    total_possible = 0
    # all_splits = gd.breadth_first_splits(T)
    all_splits = list(T)
    eligible_splits = [x for x in all_splits if len(x[0]) > 1 or len(x[1]) > 1]
    
    node_to_perform_on = random.choices(
        eligible_splits,
        weights=[1 / (len(a) + len(b))**alpha for a, b in eligible_splits],
        k=1
    )[0]
    total_possible += math.comb(len(eligible_splits), 1)
    node_choices = len(eligible_splits)
    S = set(node_to_perform_on[0]) | set(node_to_perform_on[1])
    original_nodes = [ (L,R) for (L,R) in all_splits if (set(L)|set(R)).issubset(S) and (len(L)+len(R) >= 2) ]

    # entire node swap
    L = list(node_to_perform_on[0])
    R = list(node_to_perform_on[1])
    all_classes = set(L + R)
    number_of_classes = len(all_classes)
    left_split_size = random.choice(range(1,number_of_classes))
    # total_possible += factorial2(2*number_of_classes - 3)
    new_L = sorted(random.sample(list(all_classes), left_split_size))
    new_R = sorted(list(set(all_classes) - set(new_L)))
    new_node = (tuple(new_L), tuple(new_R))
    T = list(T)
    T.remove(node_to_perform_on)
    T.append(new_node)
    g_top = 1.0 / ((number_of_classes - 1) * math.comb(number_of_classes, left_split_size))

    # Now cascade the changes down the tree
    T = list(T)
    S   = set(new_node[0]) | set(new_node[1])

    rm = {i for i,(L,R) in enumerate(T) if (set(L)|set(R)).issubset(S)}
    T  = [x for i,x in enumerate(T) if i not in rm]

    def grow(classes):
        classes = list(classes)
        n = len(classes)
        if n <= 1:
            return [], 1.0
        l_split_size = random.choice(range(1,n))                  
        A = set(random.sample(classes, l_split_size))
        B = set(classes) - A
        c = [A, B]
        c.sort(key=len, reverse=True)
        A = c[0]
        B = c[1]
        left_splits,  pL = grow(A)
        right_splits, pR = grow(B)
        node = (tuple(sorted(A)), tuple(sorted(B)))
        return [node] + left_splits + right_splits, (1.0 / ((n - 1) * math.comb(n, l_split_size))) * pL * pR
    
    left_desc,  p_left  = grow(new_node[0])
    right_desc, p_right = grow(new_node[1])
    T += [new_node] + left_desc + right_desc
    # T = gd.breadth_first_splits(tuple(T))

    # print(f"Swapping entire node {node_to_perform_on} with {new_node}")
    # 5) full probability g(T*|T)
    E_star = sum(1 for L,R in T if len(L)>1 or len(R)>1)
    g_rev = 1.0 / E_star
    for (L,R) in original_nodes:              
        mm, kk = len(L)+len(R), len(L)       
        g_rev *= 1.0 / ((mm - 1) * math.comb(mm, kk))

    g_fwd = (1.0 / node_choices) * g_top * p_left * p_right
    return gd.bfs_splits(T), g_fwd, g_rev

def tree_loglik(X, y, T):
# sum node log-likelihoods (cached), using existing split scorer
    s = 0.0
    for split in T:
        if split not in cache:
            cache[split] = split_loglik_star_sm(X, y, split)
        s += cache[split]
    return s

def local_swap(T, rg):
    all_splits = list(T)
    eligible_splits = [x for x in all_splits if len(x[0]) > 1 or len(x[1]) > 1]
    # print(eligible_splits)
    node_to_perform_on = random.sample(eligible_splits, 1)[0]
    # local_item_swap
    L = list(node_to_perform_on[0])
    R = list(node_to_perform_on[1])
    left_swap = random.sample(L, 1)[0]
    right_swap = random.sample(R, 1)[0]
    # print(f"Swapping {left_swap} with {right_swap}")
    # print(f"Original node: {node_to_perform_on}")
    L.remove(left_swap)
    R.remove(right_swap)
    new_L = L + [right_swap]
    new_R = R + [left_swap]
    new_node = (tuple(new_L), tuple(new_R))

    # Now cascade the changes down the tree
    T = list(T)
    S   = set(new_node[0]) | set(new_node[1])
    U = [set(L)|set(R) for (L,R) in T]
    desc_idx = [k for k in range(0, len(T)) if U[k].issubset(S)]

    for k in desc_idx:
        L, R = T[k]
        A = np.concatenate([np.asarray(L), np.asarray(R)])  
        ma, mb = (A == left_swap), (A == right_swap)
        if ma.any() or mb.any():
            A[ma], A[mb] = right_swap, left_swap
            nL = len(L)
            c = [sorted(A[:nL]), sorted(A[nL:])]
            c.sort(key=len, reverse=True)
            l_sub = c[0]
            r_sub = c[1]
            T[k] = (tuple(l_sub), tuple(r_sub))

    return gd.bfs_splits(T), 0.0

def entire_node_swap(T, rg):
    total_possible = 0
    # all_splits = gd.breadth_first_splits(T)
    all_splits = list(T)
    eligible_splits = [x for x in all_splits if len(x[0]) > 1 or len(x[1]) > 1]
    node_to_perform_on = random.sample(eligible_splits, 1)[0]
    total_possible += math.comb(len(eligible_splits), 1)
    node_choices = len(eligible_splits)
    S = set(node_to_perform_on[0]) | set(node_to_perform_on[1])
    original_nodes = [ (L,R) for (L,R) in all_splits if (set(L)|set(R)).issubset(S) and (len(L)+len(R) >= 2) ]

    # entire node swap
    L = list(node_to_perform_on[0])
    R = list(node_to_perform_on[1])
    all_classes = set(L + R)
    number_of_classes = len(all_classes)
    left_split_size = random.choice(range(1,number_of_classes))
    # total_possible += factorial2(2*number_of_classes - 3)
    new_L = sorted(random.sample(list(all_classes), left_split_size))
    new_R = sorted(list(set(all_classes) - set(new_L)))
    new_node = (tuple(new_L), tuple(new_R))
    T = list(T)
    T.remove(node_to_perform_on)
    T.append(new_node)
    g_top = 1.0 / ((number_of_classes - 1) * math.comb(number_of_classes, left_split_size))

    # Now cascade the changes down the tree
    T = list(T)
    S   = set(new_node[0]) | set(new_node[1])

    rm = {i for i,(L,R) in enumerate(T) if (set(L)|set(R)).issubset(S)}
    T  = [x for i,x in enumerate(T) if i not in rm]

    def grow(classes):
        classes = list(classes)
        n = len(classes)
        if n <= 1:
            return [], 1.0
        l_split_size = random.choice(range(1,n))                  
        A = set(random.sample(classes, l_split_size))
        B = set(classes) - A
        c = [A, B]
        c.sort(key=len, reverse=True)
        A = c[0]
        B = c[1]
        left_splits,  pL = grow(A)
        right_splits, pR = grow(B)
        node = (tuple(sorted(A)), tuple(sorted(B)))
        return [node] + left_splits + right_splits, (1.0 / ((n - 1) * math.comb(n, l_split_size))) * pL * pR
    
    left_desc,  p_left  = grow(new_node[0])
    right_desc, p_right = grow(new_node[1])
    T += [new_node] + left_desc + right_desc
    # T = gd.breadth_first_splits(tuple(T))

    # print(f"Swapping entire node {node_to_perform_on} with {new_node}")
    # 5) full probability g(T*|T)
    E_star = sum(1 for L,R in T if len(L)>1 or len(R)>1)
    g_rev = 1.0 / E_star
    for (L,R) in original_nodes:              
        mm, kk = len(L)+len(R), len(L)       
        g_rev *= 1.0 / ((mm - 1) * math.comb(mm, kk))

    g_fwd = (1.0 / node_choices) * g_top * p_left * p_right
    return gd.bfs_splits(T), g_fwd, g_rev

def init_tree_n(n_classes, rng):

    def grow(classes):
        classes = list(classes)
        n = len(classes)
        if n <= 1:
            return []
        l_split_size = random.choice(range(1,n))                  
        A = set(random.sample(classes, l_split_size))
        B = set(classes) - A
        left_splits = grow(A)
        right_splits= grow(B)
        node = (tuple(sorted(A)), tuple(sorted(B)))
        return [node] + left_splits + right_splits

    return grow(range(0, n_classes))

def mh_dep_non_unif_node_trees(X, y, n_samples, categories, rng_seed=42, alpha = 0.5, eta_thresh=0.6):
    """
    trees: list of ND trees (tuples (left_set, right_set, left_child, right_child))
    g: proposal pmf over the library (shape M), fixed (independence)
    X (n,p), y (n,)
    Returns: indices trace (n_samples,), optional cache for reuse/diagnostics
    """
    rng = Generator(PCG64(rng_seed))

    T = tuple(init_tree_n(categories, rng))
    T = tuple(gd.breadth_first_splits(T))
    trace = np.empty(n_samples, dtype=object)
    all_tested_trees = set()
    all_tested_trees.add(T)

    for t in range(n_samples):
        eta = rng.uniform(0, 1)
        if eta < eta_thresh:
            T_star, gdiff = non_unif_local_swap(T, rng, alpha)
        else:
            T_star, gfwd, grev = non_unif_entire_node_swap(T, rng, alpha)
            gdiff = np.log(grev) - np.log(gfwd)
        all_tested_trees.add(T_star)
        dLL  = delta_loglik(X, y, tree_signatures(T), tree_signatures(T_star))  # likelihood function
        logR = dLL + gdiff   # MH accept log-ratio
        if np.log(rng.random()) < min(0.0, logR):
            T = T_star
        trace[t] = T
    return trace, all_tested_trees, cache

def mh_dep_non_unif_simmulated_annealing(X, y, n_samples, categories, rng_seed=42, start_beta=0.05, end_beta=1.1, alpha = 0.5, eta_thresh=0.5):
    """
    trees: list of ND trees (tuples (left_set, right_set, left_child, right_child))
    g: proposal pmf over the library (shape M), fixed (independence)
    X (n,p), y (n,)
    Returns: indices trace (n_samples,), optional cache for reuse/diagnostics
    """
    rng = Generator(PCG64(rng_seed))

    T = tuple(init_tree_n(categories, rng))
    T = tuple(gd.breadth_first_splits(T))
    trace = np.empty(n_samples, dtype=object)
    all_tested_trees = set()
    all_tested_trees.add(T)

    for t in range(n_samples):
        beta = start_beta + (end_beta - start_beta) * (t / max(1, n_samples-1))

        eta = rng.uniform(0, 1)

        if eta < eta_thresh:
            T_star, gdiff = non_unif_local_swap(T, rng, 5*beta)
        else:
            T_star, gfwd, grev = non_unif_entire_node_swap(T, rng, 5*beta)
            gdiff = np.log(grev) - np.log(gfwd)
        all_tested_trees.add(T_star)
        dLL  = delta_loglik(X, y, tree_signatures(T), tree_signatures(T_star))  # likelihood function
        logR = beta * dLL + gdiff 
        if np.log(rng.random()) < min(0.0, logR):
            T = T_star
        trace[t] = T
    return trace, all_tested_trees, cache

def check_label_fit(labels, y, coords):

    df_eval = pd.DataFrame({
        "score": y,
        "cluster": pd.Categorical(labels)   # no manual one-hot; just mark as categorical
    })

    # OLS with cluster fixed effects
    ols = smf.ols("score ~ C(cluster)", data=df_eval).fit()
    r2_in_sample = float(ols.rsquared)

    # ANOVA table (Type II)
    anova = sm.stats.anova_lm(ols, typ=2)

    # Sums of squares and dfs
    ss_between = float(anova.loc["C(cluster)", "sum_sq"])
    df_between = float(anova.loc["C(cluster)", "df"])
    ss_within  = float(anova.loc["Residual",   "sum_sq"])
    df_within  = float(anova.loc["Residual",   "df"])

    # Mean square within (pooled within-cluster variance estimate)
    ms_within = ss_within / df_within

    # Effect sizes:
    #   η²  = SS_between / (SS_between + SS_within)
    #   ω²  = (SS_between - df_between * MS_within) / (SS_total + MS_within)
    ss_total = ss_between + ss_within
    eta_sq   = ss_between / ss_total
    omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within)

    # Optional: per-cluster descriptive stats
    per_cluster = df_eval.groupby("cluster")["score"].agg(["count", "mean", "std"]).sort_index()
    # print(per_cluster)

    # ----------------------------- (B) Mixed-effects ICC (random intercepts) -----------------------------
    # Model: score_ij = μ + u_j + ε_ij, with u_j ~ N(0, σ_u^2) for cluster j, ε_ij ~ N(0, σ^2)
    # ICC = σ_u^2 / (σ_u^2 + σ^2)
    md = sm.MixedLM.from_formula("score ~ 1", groups="cluster", data=df_eval)
    m  = md.fit(reml=True)  # REML for variance components

    # Between-cluster variance (random intercept variance)
    var_cluster = float(m.cov_re.iloc[0, 0])
    # Within-cluster (residual) variance
    var_resid   = float(m.scale)

    icc = var_cluster / (var_cluster + var_resid)

    sil = silhouette_score(coords, labels, metric="euclidean")

    return_values = {
        "r2_in_sample": r2_in_sample,
        "eta_sq": eta_sq,
        "omega_sq": omega_sq,
        "icc": icc,
        "sil": sil
    }

    return return_values

def embed_tree(df, embeding):
    if embeding == "bfs":
        # newicks = [str(breadth_first_splits(t)) for t in df["tree"]]
        df["tree_encoded"] = df["tree"].apply(lambda x: str(gd.bfs_splits(x)))
    elif embeding == "dfs":
        # newicks = [str(depth_first_splits(t)) for t in df["tree"]]
        df["tree_encoded"] = df["tree"].apply(lambda x: str(gd.depth_first_splits(x)))
    else:
        embeding = "newick"
        # newicks = [newick(t) for t in df["tree"]]
        df["tree_encoded"] = df["tree"].apply(gd.newick)
    # newicks = sorted(newicks)
    # df['tree_encoded'] = newicks
    df = df.sort_values(by='tree_encoded')
    return df

def html_similarity_scatter(df, embeding="newick", ngram_range=(2,4), n_clusters=4, to_html=None, analyzer=None, tokenize=None, vec=None, cards_per_row=None):

    # Newick → TF-IDF → cosine similarity
    df = embed_tree(df, embeding)
    newicks = df["tree_encoded"]

    # --- unify vectorizer so we can get feature names later ---
    if analyzer is not None:
        vec_used = TfidfVectorizer(analyzer=analyzer, lowercase=False, norm="l2")
    elif vec is not None:
        vec_used = vec
    elif tokenize is not None:
        vec_used = TfidfVectorizer(analyzer="word", token_pattern=tokenize)
    else:
        vec_used = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, norm="l2")

    X = vec_used.fit_transform(newicks)
    feature_names = np.array(vec_used.get_feature_names_out())

    S = cosine_similarity(X)                         # similarity (absolute; no reordering)
    D = 1.0 - S                                      # distance for embedding

    # MDS: finds 2D coords whose Euclidean distances best match the given distance matrix
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=0).fit_transform(D)
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=0).fit_predict(coords)

    all_vals = check_label_fit(labels, df["score"].to_numpy(float), coords)
    r2_in_sample, eta_sq, omega_sq, icc, sil = all_vals["r2_in_sample"], all_vals["eta_sq"], all_vals["omega_sq"], all_vals["icc"], all_vals["sil"]

    print(f"In-sample R^2 = {r2_in_sample:.3f} — fraction of variance in accuracy explained by cluster labels in this sample (fixed-effects OLS).")
    print(f"η² = {eta_sq:.3f} — ANOVA effect size: proportion of total accuracy variance attributable to differences between clusters (in-sample).")
    print(f"ω² = {omega_sq:.3f} — bias-corrected effect size estimating the population proportion of accuracy variance explained by cluster membership.")
    print(f"ICC = {icc:.3f} — intraclass correlation from a random-intercepts model: share of accuracy variance due to between-cluster differences (0=no signal, 1=all between-cluster).")
    print(f"Silhouette = {sil:.3f} — mean of (b-a)/max(a,b), comparing a=dist to cluster center to b=nearest-other-cluster distance (−1 bad, 0 overlap, +1 well-separated).")

    cluster_str = labels.astype(str)

    # color by accuracy (min → max gradient)
    scores = df["score"].to_numpy(float)
    vmin, vmax = float(scores.min()), float(scores.max())
    out = df[["tree", "score"]]
    out[embeding] = newicks
    out["cluster"] = labels.astype(str)

    cluster_levels = sorted(set(cluster_str))
    marker_list = [
        "circle", "square", "triangle", "diamond", "inverted_triangle",
        "star", "hex", "circle_cross", "circle_x", "circle_dot",
        "square_cross", "square_x", "square_dot",
        "cross", "x", "asterisk", "star", "hex",
        "plus", "dash", "dot",
        "diamond_cross", "diamond_dot",
        "triangle_dot", "triangle_pin",
        "star_dot", "square_pin",
    ][:len(cluster_levels)]
    marker_map = dict(zip(cluster_levels, marker_list))
    marker_col = [marker_map[c] for c in cluster_str]  # one marker per point

    src = ColumnDataSource(dict(
        x=coords[:,0], y=coords[:,1],
        score=df["score"].astype(float).to_numpy(),
        cluster=labels, newick=newicks,  marker=marker_col, 
        tree=[str(t) for t in df["tree"]],
    ))
    p = figure(title=f"Trees (MDS of cosine on {embeding}) with {n_clusters}-clusters", width=800, height=520,
            tools="pan,wheel_zoom,reset,save")
    mapper = linear_cmap("score", palette=list(RdYlGn.get(8))[::-1], low=src.data["score"].min(), high=src.data["score"].max())

    p.scatter("x", "y", source=src, size=9, marker="marker", color=mapper, line_color="#222", alpha=0.9)

    adj_r2 = float(r2_in_sample)
    w2     = float(omega_sq)
    sil_v  = float(sil)
    def tag(v, good, ok):
        return "🟢 good" if v >= good else ("🟡 fine" if v >= ok else "🔴 bad")

    r2_tag  = tag(adj_r2, good=0.50, ok=0.20)
    w2_tag  = tag(w2,     good=0.50, ok=0.20)
    sil_tag = tag(sil_v,  good=0.50, ok=0.25)

    acc_div = Div(text=f"""
    <div style="border:1px solid #ddd;border-radius:8px;padding:10px;margin-top:6px;background:#fafafa; font-size:13px;">
    <h4 style="margin:0 0 6px 0;">Accuracy Fit</h4>
    <div style="margin-bottom:6px;">
        <strong>Adj&nbsp;R²:</strong> {adj_r2:.3f} &nbsp;{r2_tag}<br>
        <meter min="0" max="1" low="0.20" high="0.50" optimum="0.70" value="{adj_r2:.4f}" style="width:100%"></meter>
    </div>
    <div>
        <strong>ω²:</strong> {w2:.3f} &nbsp;{w2_tag}<br>
        <meter min="0" max="1" low="0.20" high="0.50" optimum="0.70" value="{w2:.4f}" style="width:100%"></meter>
    </div>
    <div style="margin-top:6px;"><small>
        R²: share of accuracy variance explained by cluster labels (in-sample). 
        <br>
        ω²: is bias-corrected R².
    </small></div>
    </div>
    """, width=int(p.width/2)-12)

    geo_div = Div(text=f"""
    <div style="border:1px solid #ddd;border-radius:8px;padding:10px;margin-top:6px;background:#fafafa; font-size:13px;">
    <h4 style="margin:0 0 6px 0;">Geometric Fit</h4>
    <div>
        <strong>Silhouette:</strong> {sil_v:.3f} &nbsp;{sil_tag}<br>
        <meter min="-1" max="1" low="0.25" high="0.50" optimum="0.7" value="{sil_v:.4f}" style="width:100%"></meter>
    </div>
    <div style="margin-top:6px;"><small>
        Silhouette: mean of (b-a)/max(a,b), comparing a=dist to cluster center to b=nearest-other-cluster distance.
    </small></div>
    </div>
    """, width=int(p.width/2)-12)

    cbar = ColorBar(color_mapper=mapper.transform)
    p.add_layout(cbar, 'right')

    print(f"In-sample R^2 = {r2_in_sample:.3f} — fraction of variance in accuracy explained by cluster labels in this sample (fixed-effects OLS).")
    print(f"η² = {eta_sq:.3f} — ANOVA effect size: proportion of total accuracy variance attributable to differences between clusters (in-sample).")
    print(f"ω² = {omega_sq:.3f} — bias-corrected effect size estimating the population proportion of accuracy variance explained by cluster membership.")
    print(f"ICC = {icc:.3f} — intraclass correlation from a random-intercepts model: share of accuracy variance due to between-cluster differences (0=no signal, 1=all between-cluster).")
    print(f"Silhouette = {sil:.3f} — mean of (b-a)/max(a,b), comparing a=dist to cluster center to b=nearest-other-cluster distance (−1 bad, 0 overlap, +1 well-separated).")

    p.add_tools(HoverTool(tooltips=[("acc","@score{0.000}"),("tree","@newick")]))

    X = X.tocsr(); fn = feature_names; L = labels; mm = marker_map
    cols = cards_per_row if cards_per_row is not None else 4

    # 1) ω² per feature across all clusters
    g = pd.Series(L, name='g')
    w2 = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        y = pd.Series(X[:, j].toarray().ravel(), name='y')
        aov = sm.stats.anova_lm(ols('y ~ C(g)', data=pd.concat([y, g], axis=1)).fit(), typ=2)
        ss, df = aov.loc['C(g)', ['sum_sq','df']]
        ssr, dfr = aov.loc['Residual', ['sum_sq','df']]
        ms = ssr / dfr
        w2[j] = max((ss - df*ms) / (ss + ssr + ms), 0.0)   # ω²

    # simple icon map for your existing marker names
    icon = {"circle":"●","square":"■","triangle":"▲","diamond":"◆","inverted_triangle":"▼","star":"✶","hex":"⬢",
            "circle_cross":"⊕","circle_x":"⊗","circle_dot":"◉","square_cross":"⊞","square_x":"☒","square_dot":"▣",
            "cross":"✚","x":"✕","asterisk":"✱","plus":"＋","dash":"–","dot":"•","diamond_cross":"⟐","diamond_dot":"◈",
            "triangle_dot":"▲","triangle_pin":"▲","star_dot":"✶","square_pin":"■"}

    # 2) Cards per cluster: pick terms by in-cluster mean, show ω² bars
    cards = []
    for c in range(n_clusters):
        idx = np.where(L == c)[0]
        if not idx.size: continue
        mc = X[idx].mean(0).A1
        top = mc.argsort()[-6:][::-1]           # show common-in-cluster terms
        vmax = float(w2[top].max()) or 1.0
        rows = ''.join(
            f'<div style="display:flex;gap:8px;align-items:center;margin:4px 0;">'
            f'<span style="min-width:150px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{fn[i]}</span>'
            f'<meter min="0" max="{vmax:.6f}" value="{w2[i]:.6f}" style="flex:1;height:14px;"></meter>'
            f'<span style="min-width:56px;text-align:right;">{w2[i]:.2f}</span></div>'
            for i in top
        )
        mname = mm[str(c)]
        cards.append(
            f'<div style="border:1px solid #ccc;border-radius:12px;padding:12px;background:#F8FAFD;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<b>Cluster {c}</b><span><small>n={idx.size}</small>&nbsp;<span style="font-size:22px;">{icon.get(mname,"•")}</span></span></div>'
            f'{rows}</div>'
        )

    cluster_div = Div(text=
    f'<div style="border:1px solid #ddd;border-radius:12px;padding:12px;margin-top:8px;background:#fff;">'
    f'<h4 style="margin:0 0 8px 0;">Cluster Content Summary — ω² (variance explained by clusters)</h4>'
    f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);gap:12px;">{"".join(cards)}</div>'
    f'<div style="margin-top:6px;font-size:12px;"><em>ω²: bias-corrected ANOVA R² per term (higher = more of that term’s variance is due to cluster differences).</em></div>'
    f'</div>', width=p.width)

    layout = bcol(p, brow(acc_div, geo_div), cluster_div, sizing_mode="stretch_width")

    if to_html is not None:
        output_file(to_html); 
        save(layout)
        show(layout)

    return S, coords

def viz_mh_trace(traces, burn=0, top_k=20, html_path=None, best_known=None, loglik_fn=None, notes=""):
    try: L = min(len(t) for t in traces); arr = np.stack([np.asarray(t[:L], object) for t in traces], 0)
    except Exception: arr = np.asarray(traces, object)[None, :]

    flat = arr.ravel()
    uniq_trees, inv = np.unique(flat, return_inverse=True)
    arr_id = inv.reshape(arr.shape)

    accs = [(tr[1:] != tr[:-1]).mean() if tr.size > 1 else np.nan for tr in arr_id]
    overall = float(np.nanmean(accs)) if len(accs) else np.nan

    post = arr_id[:, burn:].ravel()
    uniq, counts = np.unique(post, return_counts=True)
    order = counts.argsort()[::-1][:top_k]
    top_ids, probs = uniq[order], counts[order] / counts.sum()
    labels = [str(int(i)) for i in top_ids]
    top_trees = [uniq_trees[int(i)] for i in top_ids]

    _ll = (lambda T: float(loglik_fn(T))) if loglik_fn else (lambda T: None)
    bk_ll = _ll(best_known) if (best_known is not None) else None
    top_lls = [_ll(T) for T in top_trees]
    if loglik_fn and any(ll is not None for ll in top_lls):
        j = int(np.nanargmax(top_lls)); best_found_id, best_found_ll, best_found_tree = int(top_ids[j]), top_lls[j], top_trees[j]
    else:
        best_found_id, best_found_ll, best_found_tree = int(top_ids[0]), None, top_trees[0]

    pal = Category10[10]

    p1 = figure(title="Trace (tree id)", width=720, height=240,
                tools="xpan,xwheel_zoom,reset,hover",
                tooltips=[("iter","@it"),("id","@idx"),("chain","@ch")])
    for c, tr in enumerate(arr_id):
        it = np.arange(tr.size); src = ColumnDataSource(dict(it=it, idx=tr, ch=[str(c)]*tr.size))
        p1.line('it','idx', source=src, alpha=0.85, line_width=2, color=pal[c%10])
        p1.circle('it','idx', source=src, size=3, alpha=0.6, color=pal[c%10])

    def _acf(x, L=50):
        x = np.asarray(x, float); x -= x.mean(); den = (x*x).sum() or 1.0
        K = min(L, max(1, x.size-1)); return np.array([1.0] + [float(np.dot(x[:-k], x[k:]) / den) for k in range(1, K+1)])
    p2 = figure(title="Autocorrelation", width=720, height=220, x_axis_label="lag",
                tools="xpan,xwheel_zoom,reset,hover", tooltips=[("lag","@lag"),("acf","@acf{0.000}")])
    for c, tr in enumerate(arr_id):
        ac = _acf(tr); src = ColumnDataSource(dict(lag=np.arange(ac.size), acf=ac))
        p2.line('lag','acf', source=src, line_width=2, alpha=0.9, color=pal[c%10])

    src2 = ColumnDataSource(dict(idx=labels, p=probs))
    p3 = figure(x_range=labels, title=f"Posterior mass (top {top_k})", width=720, height=260,
                tools="hover", tooltips=[("id","@idx"),("mass","@p{0.000}")])
    p3.vbar(x='idx', top='p', source=src2, width=0.9)

    ac_labels = [f"ch{c}" for c in range(arr_id.shape[0])]
    src3 = ColumnDataSource(dict(ch=ac_labels, a=[float(a) for a in accs]))
    p4 = figure(x_range=ac_labels, title=f"Acceptance (overall {overall:.3f})", width=720, height=200,
                tools="hover", tooltips=[("chain","@ch"),("acc","@a{0.000}")])
    p4.vbar(x='ch', top='a', source=src3, width=0.7)

    def _fmt_tree(t): s = str(t); return s if len(s) <= 800 else s[:797] + "…"
    def _fmt_tree_lines(t):
        try: return "\n".join(str((tuple(L), tuple(R))) for (L,R) in t)
        except Exception: return _fmt_tree(t)

    def _dot_svg(T):
        try:
            by = {frozenset((*L,*R)):(L,R) for (L,R) in T}
            root = set().union(*[set(L)|set(R) for (L,R) in T])
            dot = ["digraph ND{rankdir=TB;node[shape=box,fontsize=9];"]; seen=set()
            def rec(U):
                Ufs=frozenset(U)
                if Ufs in seen: return
                seen.add(Ufs)
                name="n_"+"_".join(map(str,sorted(U))); lab="{"+",".join(map(str,sorted(U)))+"}"
                dot.append(f'{name}[label="{lab}"];')
                if len(U)<=1: return
                L,R=by[frozenset(U)]
                for V in (set(L),set(R)):
                    v="n_"+"_".join(map(str,sorted(V))); dot.append(f"{name}->{v};"); rec(V)
            rec(root); dot.append("}")
            return Source("\n".join(dot)).pipe(format="svg").decode("utf-8")
        except Exception:
            return "<!-- graphviz failed -->"

    bk_svg = _dot_svg(best_known) if best_known is not None else ""
    bf_svg = _dot_svg(best_found_tree) if best_found_tree is not None else ""

    bk_txt = "" if best_known is None else f"<h3>Best-known</h3><pre>{_fmt_tree_lines(best_known)}</pre><br>"
    bf_txt = f"<h3>Best-found (id {best_found_id}{'' if best_found_ll is None else f', ll {best_found_ll:.3f}'})</h3><pre>{_fmt_tree_lines(best_found_tree)}</pre>"

    tops = "".join(f"<li><b>id {int(i)}</b> (p={p:.3f}{'' if (top_lls[k] is None) else f', ll {top_lls[k]:.3f}'})<br><code>{_fmt_tree(top_trees[k])}</code></li>"
                   for k,(i,p) in enumerate(zip(top_ids, probs)))

    div_summary = Div(text=f"<h2>MH Summary</h2>{'' if bk_ll is None else f'<p><b>Best-known ll:</b> {bk_ll:.3f}</p>'}{notes}")
    div_best = Div(text=f"<table><tr>"
                        f"<td valign='top'>{bk_txt}{bk_svg}</td>"
                        f"<td style='width:20px'></td>"
                        f"<td valign='top'>{bf_txt}{bf_svg}</td>"
                        f"</tr></table>")
    div_list = Div(text=f"<h3>Top {top_k} trees</h3><ol>{tops}</ol>")

    layout = brow(bcol(p1, p2, p3, p4), bcol(div_summary, div_best, div_list))  # <-- use aliases here
    if html_path: output_file(html_path, title="MH Diagnostics"); save(layout)
    else: show(layout)

    return {
        "ids_map": {int(i): uniq_trees[int(i)] for i in top_ids},
        "top_ids": top_ids, "top_probs": probs, "top_trees": top_trees,
        "acc": accs, "overall": overall,
        "best_found_id": best_found_id, "best_found_ll": best_found_ll, "best_found_tree": best_found_tree,
        "best_known_ll": bk_ll
    }