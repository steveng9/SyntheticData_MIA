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

# FPs_directory = DIR + "focalpoints/"
# results_directory = DIR + "focalpoints/"
FPs_directory = DIR + "experiment_artifacts/focalpoints/"
results_directory = "mamamia_results/"

FP_completed_file = FPs_directory + "FP_completed_file.txt"
attack_completed_file = DIR + "experiment_artifacts/" + results_directory + "attack_completed_file.txt"

rng = default_rng()

n_FP_shadowruns = 50
n_sizes = [100, 316, 1_000, 3_162, 10_000, 31_623]
# n_sizes = [100, 316]
# t_sizes = [10, 18, 32, 56, 100, 178]
t_sizes = [10, 18]
epsilons = [round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 2)]
epsilons_2 = [round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 3)]
# epsilons = [.1, 1]
sdgs = ["mst", "priv", "gsd"]
# sdgs = ["mst", "priv"]




def main():
    task = sys.argv[1]
    if task == "shadowmodel":
        shadow_model()
    elif task == "attack":
        mama_mia()
    elif task == "status":
        print_status()
    elif task == "attack_status":
        print_attack_status()
    elif task == "mkdirs":
        make_directory_structure()
    else:
        print("No known command given.")


def fp_filename(sdg, epsilon, n, data):
    return f"focalpoints/FP5_{sdg}_e{fo(epsilon)}_n{n}_{data}"


def attack_results_filename(location, sdg, epsilon, n, data, overlap, set_MI):
    return f"{location}results_{sdg}_e{fo(epsilon)}_n{n}_{data}_o{overlap}_set{set_MI}"


def fo(eps):
    return '{0:.2f}'.format(eps)


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
    if not Path(attack_completed_file).exists():
        with open(attack_completed_file, "w") as f:
            f.writelines("sdg, epsilon, N, data, overlap, setMI\n")

    experiment = sys.argv[2]
    sdg = sys.argv[3]

    sdg_methods = {
        "mst": attack_mst,
        "priv": attack_privbayes,
        "gsd": attack_gsd,
        "rap": attack_rap
    }

    experiment_methods = {
        "A": attack_experiment_A,
        "B": attack_experiment_B,
        "D": attack_experiment_D,
    }

    # example command: "python3 AAAI_.py attack D mst 3.16 <overlap>True <setMI>False"
    experiment_method = experiment_methods[experiment]
    sdg_method = sdg_methods[sdg]

    experiment_method(sdg, sdg_method)


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
        for eps in epsilons:
            for data in ["snake", "cali"]:
                if f"{sdg}, {fo(eps)}, {expD.n}, {data}\n" not in FPs_completed and eps not in expD.exclude.get(sdg, []):
                    print(f"\t{sdg}, e{fo(eps)}, n{expD.n}, {data}", end="...")
                    progress = max((load_artifact(fp_filename(sdg, eps, expD.n, data)) or {".": 0}).values())
                    print(f"{progress} / {n_FP_shadowruns}")
                    # print("NOT LAUNCHED")
        print()



def print_attack_status(location=results_directory, completed_file=attack_completed_file):
    attacks_completed = open(completed_file, "r").readlines()
    completed = []

    print("\nexperiment A")
    for sdg in sdgs:
        for eps in epsilons:
            if f"{sdg}, {fo(eps)}, {expA.n}, snake, True, False\n" not in attacks_completed and eps not in expA.exclude.get(sdg, []):
                print(f"\t{sdg}, e{fo(eps)}, n{expA.n}, snake", end="...")
                progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, eps, expA.n, "snake", True, False)) or {".": []}).values()])
                print(f"{progress} / {C.n_runs}")
            else:
                completed.append(f"{sdg}, {fo(eps)}, {expA.n}, snake, True, False")

        print()


    print("\nexperiment B")
    for sdg in sdgs:
        for n in n_sizes:
            if f"{sdg}, {fo(expB.eps)}, {n}, snake, True, False\n" not in attacks_completed and n not in expB.exclude.get(sdg, []):
                print(f"\t{sdg}, e{fo(expB.eps)}, n{n}, snake", end="...")
                progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, expB.eps, n, "snake", True, False)) or {".": []}).values()])
                print(f"{progress} / {C.n_runs}")
            else:
                completed.append(f"{sdg}, {fo(expB.eps)}, {n}, snake, True, False")
        print()


    print("\nexperiment D")
    for sdg in sdgs:
        for eps in epsilons:
            for data in ["snake", "cali"]:
                if f"{sdg}, {fo(eps)}, {expD.n}, {data}, True, False\n" not in attacks_completed and eps not in expD.exclude.get(sdg, []):
                    print(f"\t{sdg}, e{fo(eps)}, n{expD.n}, {data}", end="...")
                    progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, eps, expD.n, data, True, False)) or {".": []}).values()])
                    print(f"{progress} / {C.n_runs}")
                    # print("NOT LAUNCHED")
                else:
                    completed.append(f"{sdg}, {fo(eps)}, {expD.n}, {data}, True, False")
        print()


    print("\nnon-overlapping")
    for sdg in sdgs:
        for eps in epsilons:
            for data in ["snake", "cali"]:
                if f"{sdg}, {fo(eps)}, {expD.n}, {data}, False, False\n" not in attacks_completed and eps not in expD.exclude.get(sdg, []):
                    print(f"\t{sdg}, e{fo(eps)}, n{expD.n}, {data}", end="...")
                    progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, eps, expD.n, data, False, False)) or {".": []}).values()])
                    print(f"{progress} / {C.n_runs}")
                    # print("NOT LAUNCHED")
                else:
                    completed.append(f"{sdg}, {fo(eps)}, {expD.n}, {data}, False, False")
        print()


    print("\nset MI")
    for sdg in sdgs:
        for eps in epsilons:
            for data in ["snake", "cali"]:
                if f"{sdg}, {fo(eps)}, {expD.n}, {data}, True, True\n" not in attacks_completed and eps not in expD.exclude.get(sdg, []):
                    print(f"\t{sdg}, e{fo(eps)}, n{expD.n}, {data}", end="...")
                    progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, eps, expD.n, data, True, True)) or {".": []}).values()])
                    print(f"{progress} / {C.n_runs}")
                    # print("NOT LAUNCHED")
                else:
                    completed.append(f"{sdg}, {fo(eps)}, {expD.n}, {data}, True, True")
        print()

    print("\ncompleted:")
    for c in completed:
        print(c)





def attack_experiment_A(sdg, sdg_method):
    epsilon = float(sys.argv[4])
    overlap = sys.argv[5] == "True"
    set_MI = sys.argv[6] == "True"
    cfg = Config("snake", set_MI=set_MI, train_size=expA.n, overlapping_aux=overlap, check_arbitrary_fps=False)
    _, full_aux, columns, meta, _ = get_data(cfg)

    results_filename = attack_results_filename(results_directory, sdg, epsilon, expA.n, "snake", overlap, set_MI)
    # runtime_filename = results_filename + "_runtime"
    # runtime = load_artifact(runtime_filename) or {"time": 0, "num_runs": 0}

    results = load_artifact(results_filename) or {
        "KDE_MA": [],
        "KDE_AUC": [],
        "KDE_time": [],
        "MM_MA": [],
        "MM_AUC": [],
        "MM_MA_weighted": [],
        "MM_AUC_weighted": [],
        "MM_time": [],
        "MM_arbitrary_MA": [],
        "distance": []
    }

    fps = load_artifact(fp_filename(sdg, epsilon, expA.n, "snake"))

    for run in tqdm(range(C.n_runs)):
        target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(cfg, full_aux, columns)
        aux = full_aux if overlap else full_aux[~full_aux.index.isin(train.index)]

        # start = time.process_time()
        kde_ma, kde_auc, kde_time, mm_ma, mm_auc, mm_ma_w, mm_auc_w, mm_time, mm_arbitrary_ma, distance = \
            sdg_method(cfg, meta, aux, columns, train, epsilon, targets, target_ids, membership, kde_sample_seed, fps)
        # end = time.process_time()

        if kde_ma is not None: results["KDE_MA"].append(kde_ma)
        if kde_auc is not None: results["KDE_AUC"].append(kde_auc)
        if kde_time is not None: results["KDE_time"].append(kde_time)
        if mm_ma is not None: results["MM_MA"].append(mm_ma)
        if mm_auc is not None: results["MM_AUC"].append(mm_auc)
        if mm_ma_w is not None: results["MM_MA_weighted"].append(mm_ma_w)
        if mm_auc_w is not None: results["MM_AUC_weighted"].append(mm_auc_w)
        if mm_time is not None: results["MM_time"].append(mm_time)
        if distance is not None: results["distance"].append(distance)
        if mm_arbitrary_ma is not None: results["MM_arbitrary_MA"].append(mm_arbitrary_ma)

        # save off intermediate results
        dump_artifact(results, results_filename)
        # runtime["time"] += (end - start)
        # runtime["num_sets"] += 1

    print(f"completed MAMA-MIA attack for experiment A, e{epsilon}, n{expA.n}, snake")
    with open(attack_completed_file, "a") as f:
        f.writelines(f"{sdg}, {fo(epsilon)}, {expA.n}, snake, {overlap}, {set_MI}\n")
    # dump_artifact(runtime, runtime_filename)






def attack_experiment_B(sdg, sdg_method):
    n = int(sys.argv[4])
    overlap = sys.argv[5] == "True"
    set_MI = sys.argv[6] == "True"
    cfg = Config("snake", set_MI=set_MI, train_size=n, overlapping_aux=overlap, check_arbitrary_fps=False)
    _, full_aux, columns, meta, _ = get_data(cfg)

    results_filename = attack_results_filename(results_directory, sdg, expB.eps, n, "snake", overlap, set_MI)
    # runtime_filename = results_filename + "_runtime"
    # runtime = load_artifact(runtime_filename) or {"time": 0, "num_runs": 0}

    results = load_artifact(results_filename) or {
        "KDE_MA": [],
        "KDE_AUC": [],
        "KDE_time": [],
        "MM_MA": [],
        "MM_AUC": [],
        "MM_MA_weighted": [],
        "MM_AUC_weighted": [],
        "MM_time": [],
        "MM_arbitrary_MA": [],
        "distance": []
    }

    fps = load_artifact(fp_filename(sdg, expB.eps, n, "snake"))

    for run in tqdm(range(C.n_runs)):
        target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(cfg, full_aux, columns)
        aux = full_aux if overlap else full_aux[~full_aux.index.isin(train.index)]

        # start = time.process_time()
        kde_ma, kde_auc, kde_time, mm_ma, mm_auc, mm_ma_w, mm_auc_w, mm_time, mm_arbitrary_ma, distance = \
            sdg_method(cfg, meta, aux, columns, train, expB.eps, targets, target_ids, membership, kde_sample_seed, fps)
        # end = time.process_time()

        if kde_ma is not None: results["KDE_MA"].append(kde_ma)
        if kde_auc is not None: results["KDE_AUC"].append(kde_auc)
        if kde_time is not None: results["KDE_time"].append(kde_time)
        if mm_ma is not None: results["MM_MA"].append(mm_ma)
        if mm_auc is not None: results["MM_AUC"].append(mm_auc)
        if mm_ma_w is not None: results["MM_MA_weighted"].append(mm_ma_w)
        if mm_auc_w is not None: results["MM_AUC_weighted"].append(mm_auc_w)
        if mm_time is not None: results["MM_time"].append(mm_time)
        if distance is not None: results["distance"].append(distance)
        if mm_arbitrary_ma is not None: results["MM_arbitrary_MA"].append(mm_arbitrary_ma)

        # save off intermediate results
        dump_artifact(results, results_filename)
        # runtime["time"] += (end - start)
        # runtime["num_sets"] += 1

    print(f"completed MAMA-MIA attack for experiment B, {fo(expB.eps)}, {n}, snake")
    with open(attack_completed_file, "a") as f:
        f.writelines(f"{sdg}, {fo(expB.eps)}, {n}, snake, {overlap}, {set_MI}\n")
    # dump_artifact(runtime, runtime_filename)




def attack_experiment_D(sdg, sdg_method):
    datasets = ["snake", "cali"]
    epsilon = float(sys.argv[4])
    overlap = sys.argv[5] == "True"
    set_MI = sys.argv[6] == "True"
    if len(sys.argv) >= 8:
        datasets = [sys.argv[7]]



    for data in datasets:
        cfg = Config(data, set_MI=set_MI, train_size=expD.n, overlapping_aux=overlap, check_arbitrary_fps=True)
        _, full_aux, columns, meta, _ = get_data(cfg)

        results_filename = attack_results_filename(results_directory, sdg, epsilon, expD.n, data, overlap, set_MI)
        # runtime_filename = results_filename + "_runtime"
        # runtime = load_artifact(runtime_filename) or {"time": 0, "num_runs": 0}

        results = load_artifact(results_filename) or {
            "KDE_MA": [],
            "KDE_AUC": [],
            "KDE_time": [],
            "MM_MA": [],
            "MM_AUC": [],
            "MM_MA_weighted": [],
            "MM_AUC_weighted": [],
            "MM_time": [],
            "MM_arbitrary_MA": [],
            "distance": []
        }

        fps = load_artifact(fp_filename(sdg, epsilon, expD.n, data))

        for run in tqdm(range(C.n_runs)):
            target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(cfg, full_aux, columns)
            aux = full_aux if overlap else full_aux[~full_aux.index.isin(train.index)]

            # start = time.process_time()
            kde_ma, kde_auc, kde_time, mm_ma, mm_auc, mm_ma_w, mm_auc_w, mm_time, mm_arbitrary_ma, distance = \
                sdg_method(cfg, meta, aux, columns, train, epsilon, targets, target_ids, membership, kde_sample_seed, fps)
            # end = time.process_time()

            if kde_ma is not None: results["KDE_MA"].append(kde_ma)
            if kde_auc is not None: results["KDE_AUC"].append(kde_auc)
            if kde_time is not None: results["KDE_time"].append(kde_time)
            if mm_ma is not None: results["MM_MA"].append(mm_ma)
            if mm_auc is not None: results["MM_AUC"].append(mm_auc)
            if mm_ma_w is not None: results["MM_MA_weighted"].append(mm_ma_w)
            if mm_auc_w is not None: results["MM_AUC_weighted"].append(mm_auc_w)
            if mm_time is not None: results["MM_time"].append(mm_time)
            if distance is not None: results["distance"].append(distance)
            if mm_arbitrary_ma is not None: results["MM_arbitrary_MA"].append(mm_arbitrary_ma)

            # save off intermediate results
            dump_artifact(results, results_filename)
            # runtime["time"] += (end - start)
            # runtime["num_sets"] += 1

        print(f"completed MAMA-MIA attack for experiment D, e{epsilon}, n{expD.n}, {data}")
        with open(attack_completed_file, "a") as f:
            f.writelines(f"{sdg}, {fo(epsilon)}, {expD.n}, {data}, {overlap}, {set_MI}\n")
    # dump_artifact(runtime, runtime_filename)













if __name__ == '__main__':
    main()

