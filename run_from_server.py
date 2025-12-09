import database_helpers as db 
import markov_methods as mh
import generate_data as gd
import config
import time
import ast
import pandas as pd
import sqlalchemy as sa
import psycopg2
from itertools import product
from datetime import datetime
import zlib
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from joblib import dump, load
import os
from sklearn.preprocessing import StandardScaler
import random
import sys
import psutil, os
import traceback
from sklearn.decomposition import PCA

from scipy.ndimage import rotate, shift

def run_model(job: dict) -> pd.DataFrame:
    run_id = job["id"]
    base = job["models"]
    dataset = job["dataset"]
    train_seed = int(job["seed"])
    simulation_flag = job["simulation_flag"]
    train_test_ratio = job["train_test_ratio"]
    alpha = job["alpha"]
    eta_threshold = job["eta_threshold"]
    node_weight_power = job["node_weight_power"]
    chain_length = job["chain_length"]
    extra_conditions = job["extra_conditions"]
    rng_seed = ast.literal_eval(job["rng_seed"]) if job["rng_seed"] is not None else None
    if "hospital" in dataset: 
        df_train = pd.read_csv(f"hospital/train_{dataset}.csv")
        X = df_train.drop(["y"], axis=1)
        y = df_train["y"]
        df_test = pd.read_csv(f"hospital/test_{dataset}.csv")
        X_test = df_test.drop(["y"], axis=1)
        y_test = df_test["y"]
    elif int(simulation_flag) == 0:
        df = pd.read_csv(f"data/{dataset}.csv")
        X, X_test, y, y_test = train_test_split(
            df.drop(["y"], axis=1), df["y"], test_size=train_test_ratio, random_state=train_seed
        )

    if "pca" in str(extra_conditions):
        pass
    
    if "no_scaling" not in str(extra_conditions):
        scaler = StandardScaler().fit(X)
        X, X_test = scaler.transform(X), scaler.transform(X_test)
    y = np.asarray(y, dtype=int)                      
    y_test = np.asarray(y_test, dtype=int)                      
    # categories = tuple(np.unique(y))

    categories = tuple(int(c) for c in np.unique(y))  # force plain Python ints
    config.X = X
    config.y = y
    config.X_test = X_test
    config.y_test = y_test
    config.categories = categories
    
    def _one_sa(seed):
        global cache
        trace, trees, cache2 = mh.mh_dep_non_unif_simmulated_annealing(X, y, chain_length, len(categories), rng_seed=seed, start_beta=0.00001, end_beta=0.05)
        return trace, trees

    def _one(seed):
        global cache
        trace, trees, cache2 = mh.mh_dep_non_unif_node_trees(X, y, chain_length, len(categories), rng_seed=seed, alpha=alpha, eta_thresh=eta_threshold, model=base)
        return trace, trees

    def _one_lda(seed):
        global cache
        trace, trees, cache2 = mh.mh_dep_non_unif_node_trees_lda(X, y, chain_length, len(categories), rng_seed=seed, alpha=alpha, eta_thresh=eta_threshold)
        return trace, trees

    config.node_weight_power = node_weight_power
    start = time.perf_counter()
    if base == "lda":
        return_traces = Parallel(n_jobs=-1, backend="threading")(delayed(_one_lda)(s) for s in rng_seed)
        algo_type = "MH_lda_"
        tree_func = mh.tree_loglik_lda
        cur_algo = "MH"
    elif base == "lr":
        return_traces = Parallel(n_jobs=-1, backend="threading")(delayed(_one)(s) for s in rng_seed)
        algo_type = "MH_lr_"
        tree_func = mh.tree_loglik
        cur_algo = "MH"
    elif base == "svm":
        return_traces = Parallel(n_jobs=-1, backend="threading")(delayed(_one)(s) for s in rng_seed)
        algo_type = "MH_svm_"
        tree_func = mh.tree_loglik
        cur_algo = "MH"
    train_timer = round(time.perf_counter()-start,3)
    artifacts_path = r"C:\Users\maxdi\UWA\RES-MATHS-NDS-P002363 - Documents\General\models"
    
    traces = [trace[0] for trace in return_traces]
    return_all_trees = [tree[1] for tree in return_traces]
    all_trees = set()
    return_traces = [all_trees | trace_trees for trace_trees in return_all_trees]
    trees = []
    for chain in traces:
        trees.extend(chain)
    # mh.viz_mh_trace(traces, X, y, burn=0, top_k=20, html_path=f"{artifacts_path}\\plots\\trace_{algo_type}_{dataset}_{run_id}.html", best_known=None, loglik_fn=tree_func, model=base)
    unique_trees = list(dict.fromkeys(trees))
    def _score_tree(T):
        return tree_func(X, y, T, base)

    scores = Parallel(n_jobs=-1, backend="threading")(
        delayed(_score_tree)(T) for T in unique_trees
    )
    embeding = "bfs"
    df_all = pd.DataFrame({"tree": unique_trees, "score": scores})
    df_all = df_all.sort_values("score", ascending=False)
    NUMBER_OF_CLASSES = len(categories)

    split_analyzer = mh.make_split_analyzer(NUMBER_OF_CLASSES)
    vec = TfidfVectorizer(analyzer=split_analyzer)
    try:
        pass
        # info_df, coords = mh.html_similarity_scatter(df_all, embeding=embeding, vec=vec, to_html=f"{artifacts_path}\\plots\\{embeding}_{algo_type}_{dataset}_{run_id}.html")
    except Exception as e:
        print("Could not create similarity plot.")
        print(traceback.format_exc())
    trace_name = f"traces_{run_id}.joblib"
    # dump(traces, artifacts_path + f"\\trace\\{trace_name}")

    config_name = f"config_{run_id}.joblib"
    rss_bytes = psutil.Process(os.getpid()).memory_info().rss
    
    if rss_bytes < 2 * 1024**3:
        pass
        # config_state = {name: getattr(config, name) for name in ["model_cache", "cache", "categories"]}
    else:
        pass
        # config_state = {name: getattr(config, name) for name in ["cache", "categories"]}
        
    # dump(config_state, artifacts_path + f"\\config\\{config_name}", compress=3)
      
    df_name = f"df_{run_id}.joblib"
    # dump(df_all, artifacts_path + f"\\all_trees\\{df_name}")

    name = "nd" + f"_dataset={dataset}"+ f"_model={base}" + f"_seed={train_seed}" + f"_ratio={train_test_ratio}" + f"_nd_method={cur_algo}" + f"_chain_len={chain_length}"+ f"_alpha={alpha}"+ f"_eta={eta_threshold}"+ f"_gamma={node_weight_power}_"+str(extra_conditions)
    common_dict = {
        "id": run_id,
        "name": name,
        "models": base,
        "dataset": dataset,
        "simulation_flag": simulation_flag,
        "seed": train_seed,
        "rng_seed": str(rng_seed), 
        "train_test_ratio": train_test_ratio,
        "alpha": alpha,
        "eta_threshold": eta_threshold,
        "node_weight_power": node_weight_power,
        "trace_link": trace_name,
        "config_link": config_name,
        "all_trees_link": df_name,
        "chain_length": chain_length,
        "nd_traversal_type":cur_algo
    }
    
    # 1) best single tree
    start = time.perf_counter()
    nd_architecture = "single_nd"
    best_tree = df_all.sort_values("score", ascending=False).iloc[0]["tree"]
    y_hat = mh.nd_predict(X_test, best_tree, categories, base=base)
    loss = mh.nd_logloss(X_test, config.y_test, best_tree, categories, base=base)
    prediction_timer = round(time.perf_counter()-start,3)
    run_timestamp =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    postgres_df = gd.calculate_metrics(None, y_test, y_hat)
    postgres_df = postgres_df.assign(**common_dict)

    postgres_df["train_time_seconds"] = train_timer
    postgres_df["test_time_seconds"] = prediction_timer
    postgres_df["nd_architecture"] = nd_architecture
    postgres_df["run_timestamp"] = run_timestamp
    postgres_df["log_loss"] = loss
    postgres_df["node_weight_power"] = node_weight_power
    postgres_df["nd_structure"] = str(best_tree)
    postgres_df["model_structure"] = ""
    postgres_df["extra_conditions"] = extra_conditions
    postgres_df["notes"] = ""
    
    engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
    gd.df_upsert_postgres(postgres_df, "test_nd_model_registry", engine, schema=None, match_columns=["id", "models","dataset","seed","rng_seed","extra_conditions","nd_architecture","nd_traversal_type"], insert_only=False)
    engine = None
    
    # 1b) TRAIN best single tree
    start = time.perf_counter()
    nd_architecture = "single_nd"
    best_tree = df_all.sort_values("score", ascending=False).iloc[0]["tree"]
    y_hat = mh.nd_predict(X, best_tree, categories, base=base)
    loss = mh.nd_logloss(X, config.y, best_tree, categories, base=base)
    prediction_timer = round(time.perf_counter()-start,3)
    run_timestamp =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    postgres_df = gd.calculate_metrics(None, y, y_hat)
    postgres_df = postgres_df.assign(**common_dict)

    postgres_df["train_time_seconds"] = train_timer
    postgres_df["test_time_seconds"] = prediction_timer
    postgres_df["nd_architecture"] = nd_architecture
    postgres_df["run_timestamp"] = run_timestamp
    postgres_df["log_loss"] = loss
    postgres_df["node_weight_power"] = node_weight_power
    postgres_df["nd_structure"] = str(best_tree)
    postgres_df["model_structure"] = ""
    postgres_df["extra_conditions"] = extra_conditions
    postgres_df["notes"] = ""
    
    engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
    gd.df_upsert_postgres(postgres_df, "train_nd_model_registry", engine, schema=None, match_columns=["id", "models","dataset","seed","rng_seed","extra_conditions","nd_architecture","nd_traversal_type"], insert_only=False)
    engine = None 
    
    # 2) bayesian model averaging
    nd_architecture = "bma"
    # thin = 10
    # thin = max(10, chain_length/(10**3))
    start = time.perf_counter()
    trees_post, weights_post = mh.collect_posterior_trees(traces, burn_in=1_000, thin=10)
    proba_bma, _ = mh.bma_predict_proba(X_test, trees_post, weights_post, categories, base=base)
    idx_bma = proba_bma.argmax(axis=1)
    y_hat = np.array(categories)[idx_bma]
    loss = mh.bma_logloss(X_test, config.y_test, trees_post, weights_post, categories, base=base)
    run_timestamp =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    prediction_timer = round(time.perf_counter()-start,3)
    postgres_df = gd.calculate_metrics(None, y_test, y_hat)
    postgres_df = postgres_df.assign(**common_dict)

    postgres_df["train_time_seconds"] = train_timer
    postgres_df["test_time_seconds"] = prediction_timer
    postgres_df["nd_architecture"] = nd_architecture
    postgres_df["run_timestamp"] = run_timestamp
    postgres_df["log_loss"] = loss
    postgres_df["node_weight_power"] = node_weight_power
    postgres_df["nd_structure"] = ""
    postgres_df["model_structure"] = ""
    postgres_df["extra_conditions"] = extra_conditions
    postgres_df["notes"] = ""

    engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
    gd.df_upsert_postgres(postgres_df, "test_nd_model_registry", engine, schema=None, match_columns=["id", "models","dataset","seed","rng_seed","extra_conditions","nd_architecture","nd_traversal_type"], insert_only=False)
    engine = None

    # 2b) TRAIN bayesian model averaging
    nd_architecture = "bma"
    start = time.perf_counter()
    trees_post, weights_post = mh.collect_posterior_trees(traces, burn_in=1_000, thin=10)
    proba_bma, _ = mh.bma_predict_proba(X_test, trees_post, weights_post, categories, base=base)
    idx_bma = proba_bma.argmax(axis=1)
    y_hat = np.array(categories)[idx_bma]
    loss = mh.bma_logloss(X_test, config.y_test, trees_post, weights_post, categories, base=base)
    run_timestamp =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    prediction_timer = round(time.perf_counter()-start,3)
    postgres_df = gd.calculate_metrics(None, y_test, y_hat)
    postgres_df = postgres_df.assign(**common_dict)

    postgres_df["train_time_seconds"] = train_timer
    postgres_df["test_time_seconds"] = prediction_timer
    postgres_df["nd_architecture"] = nd_architecture
    postgres_df["run_timestamp"] = run_timestamp
    postgres_df["log_loss"] = loss
    postgres_df["node_weight_power"] = node_weight_power
    postgres_df["nd_structure"] = ""
    postgres_df["model_structure"] = ""
    postgres_df["extra_conditions"] = extra_conditions
    postgres_df["notes"] = ""

    engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
    gd.df_upsert_postgres(postgres_df, "train_nd_model_registry", engine, schema=None, match_columns=["id", "models","dataset","seed","rng_seed","extra_conditions","nd_architecture","nd_traversal_type"], insert_only=False)
    engine = None

    # 3) Pseudo MH trace
    nd_architecture = "pseudo_bma"
    trees_post, weights_post = mh.trees_weights_from_df(df_all)
    proba_pseudo_bma, _ = mh.bma_predict_proba(X_test, trees_post, weights_post, categories, base=base)
    idx_pseudo_bma = proba_pseudo_bma.argmax(axis=1)
    y_hat_pseudo_bma = np.array(categories)[idx_pseudo_bma]
    loss_pseudo_bma = mh.bma_logloss(X_test, config.y_test, trees_post, weights_post, categories, base=base)
    run_timestamp =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    prediction_timer = round(time.perf_counter()-start,3)
    postgres_df = gd.calculate_metrics(None, y_test, y_hat_pseudo_bma)
    postgres_df = postgres_df.assign(**common_dict)

    postgres_df["train_time_seconds"] = train_timer
    postgres_df["test_time_seconds"] = prediction_timer
    postgres_df["nd_architecture"] = nd_architecture
    postgres_df["run_timestamp"] = run_timestamp
    postgres_df["log_loss"] = loss_pseudo_bma
    postgres_df["node_weight_power"] = node_weight_power
    postgres_df["nd_structure"] = ""
    postgres_df["model_structure"] = ""
    postgres_df["extra_conditions"] = extra_conditions
    postgres_df["notes"] = ""

    engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
    gd.df_upsert_postgres(postgres_df, "test_nd_model_registry", engine, schema=None, match_columns=["id", "models","dataset","seed","rng_seed","extra_conditions","nd_architecture","nd_traversal_type"], insert_only=False)
    engine = None

    # 3b) TRAIN Pseudo MH trace
    nd_architecture = "pseudo_bma"
    trees_post, weights_post = mh.trees_weights_from_df(df_all)
    proba_pseudo_bma, _ = mh.bma_predict_proba(X, trees_post, weights_post, categories, base=base)
    idx_pseudo_bma = proba_pseudo_bma.argmax(axis=1)
    y_hat_pseudo_bma = np.array(categories)[idx_pseudo_bma]
    loss_pseudo_bma = mh.bma_logloss(X, config.y, trees_post, weights_post, categories, base=base)
    run_timestamp =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    prediction_timer = round(time.perf_counter()-start,3)
    postgres_df = gd.calculate_metrics(None, y, y_hat_pseudo_bma)
    postgres_df = postgres_df.assign(**common_dict)

    postgres_df["train_time_seconds"] = train_timer
    postgres_df["test_time_seconds"] = prediction_timer
    postgres_df["nd_architecture"] = nd_architecture
    postgres_df["run_timestamp"] = run_timestamp
    postgres_df["log_loss"] = loss_pseudo_bma
    postgres_df["node_weight_power"] = node_weight_power
    postgres_df["nd_structure"] = ""
    postgres_df["model_structure"] = ""
    postgres_df["extra_conditions"] = extra_conditions
    postgres_df["notes"] = ""

    engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
    gd.df_upsert_postgres(postgres_df, "train_nd_model_registry", engine, schema=None, match_columns=["id","models","dataset","seed","rng_seed","extra_conditions","nd_architecture","nd_traversal_type"], insert_only=False)
    engine = None

    return postgres_df

def worker_loop():
    password = os.environ['ML_POSTGRESS_URL'].split(':')[2].split("@")[0]
    host = "2m43izca46y.db.cloud.edu.au"
    database = "max"
    while True:
        engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
        conn = psycopg2.connect(host=host, dbname=database, user="max", password=password)
        job = db.get_next_job(conn)
        if job is None:
            time.sleep(30)
            continue

        job_id = job["id"]
        print(job)

        try:
            postgres_df = run_model(job)
            conn.close()
            conn = psycopg2.connect(host=host, dbname=database, user="max", password=password)
            db.mark_job_done(conn, job_id)
        except Exception as e:
            print("WE FAILED!")
            print(traceback.format_exc())
            conn.close()
            conn = psycopg2.connect(host=host, dbname=database, user="max", password=password)
            db.mark_job_failed(conn, job_id, str(e))
            # raise e

if __name__ == "__main__":
    worker_loop()