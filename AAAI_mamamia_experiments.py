import os, re
import sys
import time
import random as rand
import pickle

import warnings

import pandas as pd
import numpy as np
from numpy.random import default_rng
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')
# import disjoint_set

import mst

import privbayes

sys.path.append('private_gsd/')
from utils.utils_data import Dataset, Domain
from stats import Marginals, ChainedStatistics
from models import GSD
from jax.random import PRNGKey

# from collections import Counter

from util import *
from determine_focal_points import *
from conduct_attacks import *

import psutil

###################------------------------------------#
#### CONSTANTS ####------------------------------------#
###################------------------------------------#

min_HH_size = 5

# DIR = "/Users/golobs/Documents/GradSchool/"
DIR = "/home/golobs/"

FPs_directory = DIR + "focalpoints/"
# FPs_directory = DIR + "experiment_artifacts/focalpoints/"

FP_completed_file = FPs_directory + "FP_completed_file.txt"

rng = default_rng()

n_FP_shadowruns = 50
n_sizes = [100, 316, 1_000, 3_162, 10_000, 31_623]
# n_sizes = [100, 316]
# t_sizes = [10, 18, 32, 56, 100, 178]
t_sizes = [10, 18]
epsilons = [round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 2)]
epsilons_2 = [round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 3)]
# epsilons = [.1, 1]
sdgs = ["mst", "priv", "gsd", "rap"]
# sdgs = ["mst", "priv"]

# experiment parameters
expA = SimpleNamespace(
    r=30,
    n=10_000,
    t=100,
    exclude={},
)

expB = SimpleNamespace(
    r=30,
    eps=10,
    exclude={"gsd": [31_623]}
)


expD = SimpleNamespace(
    r=30,
    n=1000,
    exclude={}
)



def main():
    task = sys.argv[1]
    if task == "shadowmodel":
        shadow_model()
    elif task == "attack":
        mama_mia()
    elif task == "status":
        print_status()
    elif task == "mkdirs":
        make_directory_structure()
    else:
        print("No known command given.")



def shadow_model():

    # FPs_completed = Path(FP_completed_file).read_text() if Path(FP_completed_file).exists() else ""
    if not Path(FP_completed_file).exists():
        with open(FP_completed_file, "w") as f:
            f.writelines("sdg, epsilon, N, data\n")

    sdg_methods = {
        "mst": determine_mst_marginals,
        "priv": determine_privbayes_conditionals,
        "gsd": determine_gsd_marginals,
        "rap": determine_rap_queries,
        # "rap2": determine_rap_queries,
        # (Config("cali", rap_k=5, rap_top_q=30), determine_rap_queries),
    }

    experiment_methods = {
        "A": shadow_model_experiment_A,
        "B": shadow_model_experiment_B,
        "D": shadow_model_experiment_D,
    }

    # example command: "python3 AAAI_.py shadowmodel A mst 3.16"
    experiment = sys.argv[2]
    sdg = sys.argv[3]
    experiment_method = experiment_methods[experiment]
    sdg_method = sdg_methods[sdg]

    experiment_method(sdg, sdg_method)


def fp_filename(sdg, epsilon, n, data):
    return f"focalpoints/FP4_{sdg}_e{fo(epsilon)}_n{n}_{data}"


def shadow_model_experiment_A(sdg, sdg_method):
    param = sys.argv[4]
    cfg = Config("snake")
    _, aux, columns, meta, _ = get_data(cfg)
    epsilon = float(param)

    filename = fp_filename(sdg, epsilon, expA.n, "snake")
    runtime_filename = filename + "_runtime"
    runtime = load_artifact(runtime_filename) or {"time": 0, "num_sets": 0}

    for _ in tqdm(range(n_FP_shadowruns)):
        start = time.process_time()
        fps = sdg_method(cfg, aux, columns, cfg.categorical_columns, meta, epsilon, expA.n, filename)
        end = time.process_time()
        runtime["time"] += (end - start)
        runtime["num_sets"] += 1

    print(f"completed FP modelling for experiment A, e{epsilon}, n{expA.n}")
    with open(FP_completed_file, "a") as f:
        f.writelines(f"{sdg}, {fo(epsilon)}, {expA.n}, snake\n")
    dump_artifact(runtime, runtime_filename)



def shadow_model_experiment_B(sdg, sdg_method):
    param = sys.argv[4]
    cfg = Config("snake")
    _, aux, columns, meta, _ = get_data(cfg)
    n_size = int(param)

    filename = fp_filename(sdg, expB.eps, n_size, "snake")
    runtime_filename = filename + "_runtime"
    runtime = load_artifact(runtime_filename) or {"time": 0, "num_sets": 0}

    for _ in tqdm(range(n_FP_shadowruns)):
        start = time.process_time()
        fps = sdg_method(cfg, aux, columns, cfg.categorical_columns, meta, expB.eps, n_size, filename)
        end = time.process_time()
        runtime["time"] += (end - start)
        runtime["num_sets"] += 1

    print(f"completed FP modelling for experiment B, e{expB.eps}, n{n_size}")
    with open(FP_completed_file, "a") as f:
        f.writelines(f"{sdg}, {fo(expB.eps)}, {n_size}, snake\n")
    dump_artifact(runtime, runtime_filename)




def shadow_model_experiment_D(sdg, sdg_method):
    param = sys.argv[4]
    skip_snake = len(sys.argv) >= 6

    snake_cfg = Config("snake")
    cali_cfg = Config("cali")
    _, snake_aux, snake_columns, snake_meta, _ = get_data(snake_cfg)
    _, cali_aux, cali_columns, cali_meta, _ = get_data(cali_cfg)

    eps = float(param)

    filename_snake = fp_filename(sdg, eps, expD.n, "snake")
    filename_cali = fp_filename(sdg, eps, expD.n, "cali")
    runtime_filename_snake = filename_snake + "_runtime"
    runtime_filename_cali = filename_cali + "_runtime"
    runtime_snake = load_artifact(runtime_filename_snake) or {"time": 0, "num_sets": 0}
    runtime_cali = load_artifact(runtime_filename_cali) or {"time": 0, "num_sets": 0}

    if not skip_snake:
        for _ in tqdm(range(n_FP_shadowruns)):
            start = time.process_time()
            fps = sdg_method(snake_cfg, snake_aux, snake_columns, snake_cfg.categorical_columns, snake_meta, eps, expD.n, filename_snake)
            end = time.process_time()
            runtime_snake["time"] += (end - start)
            runtime_snake["num_sets"] += 1
        print(f"completed FP modelling for experiment D, e{eps}, n{expD.n}, snake")
    else:
        print(f"...skipping snake for experiment D, e{eps}, n{expD.n}")

    for _ in tqdm(range(n_FP_shadowruns)):
        start = time.process_time()
        fps = sdg_method(cali_cfg, cali_aux, cali_columns, cali_cfg.categorical_columns, cali_meta, eps, expD.n, filename_cali)
        end = time.process_time()
        runtime_cali["time"] += (end - start)
        runtime_cali["num_sets"] += 1
    print(f"completed FP modelling for experiment D, e{eps}, n{expD.n}, cali")

    with open(FP_completed_file, "a") as f:
        f.writelines(f"{sdg}, {fo(eps)}, {expD.n}, snake\n")
        f.writelines(f"{sdg}, {fo(eps)}, {expD.n}, cali\n")
    dump_artifact(runtime_snake, runtime_filename_snake)
    dump_artifact(runtime_cali, runtime_filename_cali)




def mama_mia():
    assert False, "Not yet implemented!"


def make_directory_structure():
    assert False, "Not yet implemented!"



def print_status():
    FPs_completed = open(FP_completed_file, "r").readlines()

    print("\nexperiment A")
    for sdg in sdgs:
        for eps in epsilons:
            if f"{sdg}, {fo(eps)}, {expA.n}, snake\n" not in FPs_completed and eps not in expA.exclude.get(sdg, []):
                print(f"\t{sdg}, e{fo(eps)}, n{expA.n}, snake", end="...")
                progress = max((load_artifact(fp_filename(sdg, eps, expA.n, "snake")) or {".": 0}).values())
                print(f"{progress} / {n_FP_shadowruns}")
        print()


    print("\nexperiment B")
    for sdg in sdgs:
        for n in n_sizes:
            if f"{sdg}, {fo(expB.eps)}, {n}, snake\n" not in FPs_completed and n not in expB.exclude.get(sdg, []):
                print(f"\t{sdg}, e{fo(expB.eps)}, n{n}, snake", end="...")
                progress = max((load_artifact(fp_filename(sdg, expB.eps, n, "snake")) or {".": 0}).values())
                print(f"{progress} / {n_FP_shadowruns}")
        print()


    print("\nexperiment D")
    for sdg in sdgs:
        for eps in epsilons_2:
            for data in ["snake", "cali"]:
                if f"{sdg}, {fo(eps)}, {expD.n}, {data}\n" not in FPs_completed and eps not in expD.exclude.get(sdg, []):
                    print(f"\t{sdg}, e{fo(eps)}, n{expD.n}, {data}", end="...")
                    progress = max((load_artifact(fp_filename(sdg, eps, expD.n, data)) or {".": 0}).values())
                    print(f"{progress} / {n_FP_shadowruns}")
                    # print("NOT LAUNCHED")
        print()










#
# def dump_artifact_3(artifact, name):
#     pickle_file = open(name, 'wb')
#     pickle.dump(artifact, pickle_file)
#     pickle_file.close()
#
#
# def load_artifact_3(name):
#     try:
#         pickle_file = open(name, 'rb')
#         artifact = pickle.load(pickle_file)
#         pickle_file.close()
#         return artifact
#     except:
#         return None

#
# def sample_targets(aux, num_targets):
#     hh_counts = aux['HHID'].value_counts()
#     candidate_households = hh_counts[hh_counts >= min_HH_size].index
#
#     set_MI_target_ids = pd.Series(candidate_households).sample(n=num_targets).values
#     single_MI_target_ids = pd.Series(aux[~aux.HHID.isin(set_MI_target_ids)].index).sample(n=num_targets).values
#
#     return single_MI_target_ids, set_MI_target_ids
#
#
# def sample_train(aux, labels_location, s, n):
#     set_MI_label_matrix = load_artifact(labels_location + "label_matrix_setMI")
#     single_MI_label_matrix = load_artifact(labels_location + "label_matrix_singleMI")
#
#     set_MI_targets = set_MI_label_matrix.columns
#     single_MI_targets = single_MI_label_matrix.columns
#
#     set_MI_members = set_MI_targets[set_MI_label_matrix.iloc[s, :]]
#     single_MI_members = single_MI_targets[single_MI_label_matrix.iloc[s, :]]
#
#     all_members = pd.concat([
#         aux[aux.index.isin(single_MI_members)],
#         aux[aux.HHID.isin(set_MI_members)]
#     ])
#
#     num_non_targets = n - all_members.shape[0]
#     D_train = pd.concat([aux.sample(n=num_non_targets), all_members])
#
#     return D_train.sample(frac=1)  # shuffle in members


def fo(eps):
    return '{0:.2f}'.format(eps)









main()

