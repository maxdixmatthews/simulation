import psycopg2
from itertools import product
from datetime import datetime
import zlib
import psycopg2
import json 
import pandas as pd
import generate_data as gd
import sqlalchemy as sa
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import ast
import config
import numpy as np
# Add to the nd_job_queue
password = os.environ['ML_POSTGRESS_URL'].split(':')[2].split("@")[0]
host = "2m43izca46y.db.cloud.edu.au"
database = "max"
engine = sa.create_engine(f"postgresql://max:{password}@{host}/{database}")
conn = psycopg2.connect(
    host=host,
    dbname=database,
    user="max",
    password=password,
)

# --- define the grid --------------------------------------------

dataset_ls = ["handwritten_digits", "letter_dataset"]
dataset_ls = ["mi_comps", "hmnist_28_28_L", "hmnist_28_28_RGB", "defungi_64_grayscale"]
dataset_ls = ["coarse5_yhospital_model", "coarse4_yhospital_model", "coarse3_yhospital_model", "coarse2_yhospital_model", "coarse1_yhospital_model", "20types_yhospital_model"]
dataset_ls = ["deskew_mnist"]
dataset_ls = ["defungi_seg", "mi_comps", "coarse5_yhospital_model", "coarse4_yhospital_model", "coarse3_yhospital_model", "coarse2_yhospital_model", "coarse1_yhospital_model"]
base_model = "svm"      # or whatever you’re using
simulation_flag = 0
train_test_ratios = [0.2, 0.4, 0.5]

seed_values = [42, 43, 44, 45, 46]             # example seeds
rng_seed_values = [5, 11, 42, 50, 60, 70, 80, 90, 100, 120, 130, 190]          # or something else if you prefer

alphas = [0.001, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0, 1.2, 1.50, 2, 3]                         # put more values here if you want
eta_thresholds = [0.001, 0.2, 0.5, 0.7, 0.8, 0.9, 0.9999]                 # idem
node_weight_powers = [0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1]
chain_lengths = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
extra_conditions_vals = [0]
cur_algo = ["MH"]

# --- build jobs as a cartesian product --------------------------

jobs = []

for (
    seed,
    alpha,
    eta,
    nwp,
    chain_len,
    extra_cond,
    algo,
    dataset,
    train_test_ratio
) in product(
    seed_values,
    alphas,
    eta_thresholds,
    node_weight_powers,
    chain_lengths,
    extra_conditions_vals,
    cur_algo,
    dataset_ls,
    train_test_ratios
):
    # simple id; make it whatever you want as long as it's unique
    name = "nd" + f"_dataset={dataset}"+ f"_model={base_model}" + f"_seed={seed}" + f"_ratio={train_test_ratio}" + f"_nd_method={algo}" + f"_chain_len={chain_len}"+ f"_alpha={alpha}"+ f"_eta={eta}"+ f"_gamma={nwp}_"+str(extra_cond)
    run_timestamp =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    id = int(zlib.crc32(name.encode()))
    
    jobs.append(
        (
            id,          # id (PK in target table as well)
            base_model,          # models
            dataset,
            simulation_flag,
            train_test_ratio,
            int(seed),
            str(rng_seed_values),       # stored as text; worker can ast.literal_eval
            float(alpha),
            float(eta),
            float(nwp),
            int(chain_len),
            int(extra_cond),
        )
    )

# --- insert into nd_job_queue as pending ------------------------
job_columns = [
    "id",
    "models",
    "dataset",
    "simulation_flag",
    "train_test_ratio",
    "seed",
    "rng_seed",
    "alpha",
    "eta_threshold",
    "node_weight_power",
    "chain_length",
    "extra_conditions",
]
 
jobs_df = pd.DataFrame(jobs, columns=job_columns)
jobs_df["status"] = "pending"

# 2) Show all rows with a duplicated id
dupes = jobs_df[jobs_df["id"].duplicated(keep=False)]
print(dupes)

gd.df_upsert_postgres(jobs_df,"nd_job_queue",engine,schema="public", match_columns=["id"], insert_only=False)

conn.close()
