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

import csv
import zipfile

sys.path.append('private_gsd/')
from utils.utils_data import Dataset, Domain
from stats import Marginals, ChainedStatistics
from models import GSD
from jax.random import PRNGKey

# from collections import Counter

from util import *
from determine_focal_points import *
from conduct_attacks import *
from conduct_attacks_on_wrong_sdg import mst_attack_on_wrong_synth, privbayes_attack_on_wrong_synth, gsd_attack_on_wrong_synth,mst_synthesize, privbayes_synthesize, gsd_synthesize

import psutil

###################------------------------------------#
#### CONSTANTS ####------------------------------------#
###################------------------------------------#

min_HH_size = 5

DIR = "/Users/golobs/Documents/GradSchool/Thesis/"
# DIR = "/home/golobs/"
# DIR = "/"

# FPs_directory = DIR + "focalpoints/"
# results_directory = DIR + "focalpoints/"
FPs_directory = DIR + "experiment_artifacts/focalpoints/"
results_directory = "satml25-rebuttal/mamamia_results/"

FP_completed_file = FPs_directory + "FP_completed_file.txt"
attack_completed_file = DIR + "experiment_artifacts/" + results_directory + "attack_completed_file.txt"

rng = default_rng()

n_FP_shadowruns = 15
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
    elif task == "midst_submission":
        midst_submission()
        zip_submission()
    elif task == "attack_wrong_sdg":
        mama_mia_wrong()
    elif task == "status":
        print_status()
    elif task == "attack_status":
        print_attack_status()
    elif task == "mkdirs":
        make_directory_structure()
    else:
        print("No known command given.")


def fp_filename(sdg, epsilon, n, data, n_bins):
    return f"focalpoints/FP4_{sdg}_e{fo(epsilon)}_n{n}_{data}_b{n_bins}"


def attack_results_filename(location, sdg, epsilon, n, data, overlap, set_MI):
    return f"{location}results_{sdg}_e{fo(epsilon)}_n{n}_{data}_o{overlap}_set{set_MI}_gen100k"


def wrong_attack_results_filename(location, sdg_synth, sdg_attack, epsilon, n):
    return f"{location}results_WRONGATTACK_s-{sdg_synth}_a-{sdg_attack}_e{fo(epsilon)}_n{n}"


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
        "E": shadow_model_experiment_E,
    }

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



def shadow_model_experiment_E(sdg, sdg_method):
    param = sys.argv[4]
    for eps in [10, 100, 1000]:
        print("\n\n\n\neps:", eps)
        # for n_bins in [4, 7, 10, 15, 20, 30, 50]:
        for n_bins in [40, 60, 80]:
            print("\n\n\tn_bins:", n_bins)
            C.n_bins = n_bins
            berka_cfg = Config("berka")
            _, berka_aux, berka_columns, berka_meta, _ = get_data(berka_cfg)
            filename_berka = fp_filename(sdg, eps, expE.n, "berka", C.n_bins)

            for _ in tqdm(range(n_FP_shadowruns)):
                fps = sdg_method(berka_cfg, berka_aux, berka_columns, berka_cfg.categorical_columns, berka_meta, eps, expE.n, filename_berka)
            print(f"\t\tcompleted FP modelling for experiment E, e{eps}, b{n_bins}, n{expE.n}, berka")



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
        "E": attack_experiment_E,
    }

    # example command: "python3 AAAI_.py attack D mst 3.16 <overlap>True <setMI>False"
    experiment_method = experiment_methods[experiment]
    sdg_method = sdg_methods[sdg]

    experiment_method(sdg, sdg_method)


def mama_mia_wrong():
    sdg_attack = sys.argv[2]
    sdg_synth = sys.argv[3]

    sdg_synths = {
        "mst": mst_synthesize,
        "priv": privbayes_synthesize,
        "gsd": gsd_synthesize,
    }
    sdg_synth_method = sdg_synths[sdg_synth]

    sdg_attacks = {
        "mst": mst_attack_on_wrong_synth,
        "priv": privbayes_attack_on_wrong_synth,
        "gsd": gsd_attack_on_wrong_synth,
    }
    sdg_attack_method = sdg_attacks[sdg_attack]

    attack_experiment_A_wrong(sdg_synth, sdg_synth_method, sdg_attack, sdg_attack_method)

def midst_submission():
    n_bins = 30
    C.n_bins = n_bins
    epsilon = 1000
    overlap = True
    set_MI = False
    data = "berka"
    sdg = "mst"

    cfg = Config(data, set_MI=set_MI, train_size=expE.n, overlapping_aux=overlap, check_arbitrary_fps=False)
    _, aux, columns, meta, _ = get_data(cfg)

    # fps = load_artifact(fp_filename(sdg, epsilon, expE.n, data, n_bins))
    fps = load_artifact("berka_fps")
    scaler = load_artifact("berka_std_scaler")


    for phase in ["dev", "final"]:
        dir_ = f"/Users/golobs/PycharmProjects/MIDSTModels/data/tabddpm_black_box/{phase}/"
        model_folders = [item for item in os.listdir(dir_) if os.path.isdir(os.path.join(dir_, item))]
        for model_folder in sorted(model_folders, key=lambda d: int(d.split('_')[1])):
            path = os.path.join(dir_, model_folder)

            print(path)
            synth = pd.read_csv(path + "/trans_synthetic.csv")
            synth = pd.DataFrame(scaler.transform(synth), columns=columns)
            synth = discretize_continuous_features_equaldepth(synth, "berka")

            targets = pd.read_csv(path + "/challenge_with_id.csv")
            targets.drop(columns=["trans_id", "account_id"], inplace=True)
            targets = pd.DataFrame(scaler.transform(targets), columns=columns)
            targets = discretize_continuous_features_equaldepth(targets, "berka")
            target_ids = np.array(targets.index)

            predictions = custom_mst_SUBMISSION_attack_for_berka(cfg, epsilon, aux, synth, targets, target_ids, fps)

            with open(os.path.join(path, "prediction.csv"), mode="w", newline="") as file:
                writer = csv.writer(file)
                for value in list(predictions):
                    writer.writerow([value])


def zip_submission():
    with zipfile.ZipFile(f"black_box_single_table_submission2.zip", 'w') as zipf:
        for phase in ["dev", "final"]:
            dir_ = f"/Users/golobs/PycharmProjects/MIDSTModels/data/tabddpm_black_box/{phase}/"
            model_folders = [item for item in os.listdir(dir_) if os.path.isdir(os.path.join(dir_, item))]
            for model_folder in sorted(model_folders, key=lambda d: int(d.split('_')[1])):
                path = os.path.join(dir_, model_folder)
                if not os.path.isdir(path): continue

                file = os.path.join(path, "prediction.csv")
                if os.path.exists(file):
                    # arcname = f"black_box_single_table_submission/tabddpm_black_box/{phase}/{model_folder}/prediction.csv"
                    arcname = f"tabddpm_black_box/{phase}/{model_folder}/prediction.csv"
                    print(arcname)

                    zipf.write(file, arcname=arcname)
                else:
                    raise FileNotFoundError(f"`prediction.csv` not found in {path}.")


def print_status():
    # FPs_completed = open(FP_completed_file, "r").readlines()
    FPs_completed = []

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
    # attacks_completed = open(completed_file, "r").readlines()
    attacks_completed = []
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
        "distance": [],
        "KDE_ROC": [],
        "MM_ROC": [],
    }

    fps = load_artifact(fp_filename(sdg, epsilon, expA.n, "snake"))

    for run in tqdm(range(C.n_runs)):
        target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(cfg, full_aux, columns)
        aux = full_aux if overlap else full_aux[~full_aux.index.isin(train.index)]

        # start = time.process_time()
        kde_ma, kde_auc, kde_time, mm_ma, mm_auc, mm_ma_w, mm_auc_w, mm_time, mm_arbitrary_ma, distance, kde_roc, mm_roc = \
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
        if kde_roc is not None: results["KDE_ROC"].append(kde_roc)
        if mm_roc is not None: results["MM_ROC"].append(mm_roc)

        # save off intermediate results
        dump_artifact(results, results_filename)
        # runtime["time"] += (end - start)
        # runtime["num_sets"] += 1

    print(f"completed MAMA-MIA attack for experiment A, e{epsilon}, n{expA.n}, snake")
    with open(attack_completed_file, "a") as f:
        f.writelines(f"{sdg}, {fo(epsilon)}, {expA.n}, snake, {overlap}, {set_MI}\n")
    # dump_artifact(runtime, runtime_filename)


def attack_experiment_A_wrong(sdg_synth, sdg_synth_method, sdg_attack, sdg_attack_method):
    epsilon = float(sys.argv[4])
    cfg = Config("snake", set_MI=False, train_size=expA.n, overlapping_aux=True, check_arbitrary_fps=False)
    _, full_aux, columns, meta, _ = get_data(cfg)

    results_filename = wrong_attack_results_filename(results_directory, sdg_synth, sdg_attack, epsilon, expA.n)

    results = load_artifact(results_filename) or {
        "MM_AUC": [],
        "MM_AUC_weighted": [],
    }

    fps = load_artifact(fp_filename(sdg_attack, epsilon, expA.n, "snake"))

    for run in tqdm(range(C.n_runs)):
        target_ids, targets, membership, train, _ = sample_experimental_data(cfg, full_aux, columns)

        synth = sdg_synth_method(cfg, meta, columns, train, epsilon)
        aux = full_aux

        kde_ma, kde_auc, kde_time, mm_ma, mm_auc, mm_ma_w, mm_auc_w, mm_time, mm_arbitrary_ma, distance, kde_roc, mm_roc = \
            sdg_attack_method(cfg, meta, aux, columns, train, epsilon, targets, target_ids, membership, 0, fps, synth)

        if mm_auc is not None: results["MM_AUC"].append(mm_auc)
        if mm_auc_w is not None: results["MM_AUC_weighted"].append(mm_auc_w)

        dump_artifact(results, results_filename)

    print(f"completed MAMA-MIA WRONG attack ({sdg_attack} attack --> {sdg_attack} data), e{epsilon}, n{expA.n}, snake")




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


def attack_experiment_E(sdg, sdg_method):

    for eps in [1000, 100, 10]:
        print("\n\n\n\neps:", eps)
        # for n_bins in [4, 7, 10, 15, 20, 30, 40, 50, 60, 80]:
        for n_bins in [30, 40, 50, 60, 80]:
            print("\n\n\tn_bins:", n_bins)
            C.n_bins = n_bins
            epsilon = eps
            overlap = True
            set_MI = False
            data = "berka"

            cfg = Config(data, set_MI=set_MI, train_size=expE.n, overlapping_aux=overlap, check_arbitrary_fps=False)
            _, aux, columns, meta, _ = get_data(cfg)

            fps = load_artifact(fp_filename(sdg, epsilon, expE.n, data, n_bins))
            scaler = load_artifact("berka_std_scaler")

            ave_tpr = 0
            ave_auc = 0
            for model_num in tqdm(range(1, 31)):
                # print(model_num, end=" ")
                dir_ = f"/Users/golobs/PycharmProjects/MIDSTModels/data/tabddpm_black_box/train/tabddpm_{model_num}/"

                train = pd.read_csv(dir_ + "train_with_id.csv")
                train.drop(columns=["trans_id", "account_id"], inplace=True)
                train = pd.DataFrame(scaler.transform(train), columns=columns)
                train = discretize_continuous_features_equaldepth(train, "berka")

                synth = pd.read_csv(dir_ + "trans_synthetic.csv")
                synth = pd.DataFrame(scaler.transform(synth), columns=columns)
                synth = discretize_continuous_features_equaldepth(synth, "berka")

                targets = pd.read_csv(dir_ + "challenge_with_id.csv")
                targets.drop(columns=["trans_id", "account_id"], inplace=True)
                targets = pd.DataFrame(scaler.transform(targets), columns=columns)
                targets = discretize_continuous_features_equaldepth(targets, "berka")
                target_ids = np.array(targets.index)
                membership = pd.read_csv(dir_ + "challenge_label.csv").values

                _, _, _, _, _, _, mm_auc_w, _, _, _, _, mm_roc = \
                    sdg_method(cfg, meta, aux, columns, train, epsilon, targets, target_ids, membership, 0, fps, synth=synth)

                tpr = get_tpr_at_fpr(*mm_roc)[0]

                # print("\tauc:", mm_auc_w, "\ttpr:", tpr)
                ave_tpr += tpr
                ave_auc += mm_auc_w

            ave_tpr /= 30
            ave_auc /= 30
            print(f"ave MAMA-MIA attack for eps: {epsilon} n_bins: {n_bins} \tauc:{ave_auc} \ttpr:{ave_tpr}")











if __name__ == '__main__':
    main()

