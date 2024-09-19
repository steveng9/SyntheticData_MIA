

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
# from utils.utils_data import Dataset, Domain
from private_gsd.utils.utils_data import Dataset, Domain
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
# DIR = "/home/golobs/"
DIR = "/"

shadowsets_directory = DIR + "shadowsets/"
# shadowsets_directory = "/home/golobs/shadowsets/"
# shadowsets_directory = "/home/golobs/shadowsets_cali/"
# shadowsets_directory = "/home/golobs/shadowsets_new_GSD/"

label_matrix_filename = "label_matrix"

label_assigned_filename = shadowsets_directory + "label_matrices_assigned.txt"

rng = default_rng()

DATA = "snake"
n_sizes = [100, 316, 1_000, 3_162, 10_000, 31_623]
t_sizes = [10, 18, 32, 56, 100, 178]
epsilons = [round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 2)]
# epsilons = [.1, 1]
sdgs = ["mst", "priv", "gsd"]
# sdgs = ["mst", "gsd"]

# experiment parameters
expA = SimpleNamespace(
    s=500,
    r=30,
    n=10_000,
    t=100,
    exclude={},
)

expB = SimpleNamespace(
    s=500,
    r=30,
    eps=10,
    exclude={"gsd": [31_623]}
)

expC = SimpleNamespace(
    s=4_000,
    r=30,
    n=1000,
    t=32,
    eps=10,
    exclude={},
)

expD = SimpleNamespace(
    s=500,
    r=30,
    n=1_000,
    t=32,
    exclude={},
)



def main():
    cfg = Config(DATA)
    if DATA == "snake":
        _, aux, _, meta, _ = snake_data(cfg)
    else:
        _, aux, _, meta, _ = california_data(cfg)

    task = sys.argv[1]
    if task == "assign":
        determine_target_assignments(aux)
    elif task == "gen":
        generate_shadowsets(aux, meta, cfg)
    elif task == "status":
        print_status()
    elif task == "del":
        delete_shadowsets()
    elif task == "mkdirs":
        make_directory_structure()
    else:
        print("No known command given.")

def make_directory_structure():
    if not os.path.exists(shadowsets_directory):
        os.mkdir(shadowsets_directory)

    print("making directory structure for experiment A")
    directory = shadowsets_directory + "expA/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    for eps in epsilons:
        directory = shadowsets_directory + f"expA/e{fo(eps)}/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        for sdg in sdgs:
            directory_ = directory + f"{sdg}/"
            if not os.path.exists(directory_):
                print(f"creating {directory_}")
                os.mkdir(directory_)
                dump_artifact({"time": 0.0, "num_sets": 0}, directory_ + "runtime")

    print("making directory structure for experiment B")
    directory = shadowsets_directory + "expB/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    for n, t in zip(n_sizes, t_sizes):
        directory = shadowsets_directory + f"expB/n{n}/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        for sdg in sdgs:
            directory_ = directory + f"{sdg}/"
            if not os.path.exists(directory_):
                print(f"creating {directory_}")
                os.mkdir(directory_)
                dump_artifact({"time": 0.0, "num_sets": 0}, directory_ + "runtime")

    print("making directory structure for experiment C")
    directory = shadowsets_directory + "expC/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = shadowsets_directory + "expC/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    for sdg in sdgs:
        directory_ = directory + f"{sdg}/"
        if not os.path.exists(directory_):
            print(f"creating {directory_}")
            os.mkdir(directory_)
            dump_artifact({"time": 0.0, "num_sets": 0}, directory_ + "runtime")

    print("making directory structure for experiment D")
    directory = shadowsets_directory + "expD/"
    if not os.path.exists(directory):
        os.mkdir(directory)

    for eps in epsilons:
        directory = shadowsets_directory + f"expD/e{fo(eps)}/"
        if not os.path.exists(directory):
            os.mkdir(directory)
        for sdg in sdgs:
            directory_ = directory + f"{sdg}/"
            if not os.path.exists(directory_):
                print(f"creating {directory_}")
                os.mkdir(directory_)
                dump_artifact({"time": 0.0, "num_sets": 0}, directory_ + "runtime")



def delete_shadowsets():
    if len(sys.argv) == 2 or sys.argv[2].upper() == "A":
        print()
        for eps in epsilons:
            delete_shadowsets_in_dir("A", f"e{fo(eps)}")

    if len(sys.argv) == 2 or sys.argv[2].upper() == "B":
        print()
        for n in n_sizes:
            delete_shadowsets_in_dir("B", f"n{n}")

    if len(sys.argv) == 2 or sys.argv[2].upper() == "C":
        print()
        delete_shadowsets_in_dir("C", "")

    if len(sys.argv) == 2 or sys.argv[2].upper() == "D":
        print()
        for eps in epsilons:
            delete_shadowsets_in_dir("D", f"e{fo(eps)}")



def delete_shadowsets_in_dir(experiment, sub_experiment):
    for sdg in sdgs:
        location = shadowsets_directory + f"exp{experiment}/{sub_experiment}/{sdg}/"

        for f in os.listdir(location):
            if re.search("^s\d+\.parquet$", f):
                os.remove(os.path.join(location, f))
            if re.search("^s\d+\.csv$", f):
                os.remove(os.path.join(location, f))
            if re.search("^s\d+_train_ids$", f):
                os.remove(os.path.join(location, f))

        dump_artifact({"time": 0.0, "num_sets": 0}, location + "runtime")

        print(f"cleared: {location}")


def determine_target_assignments(aux):
    make_directory_structure()

    experiment = sys.argv[2].upper()
    label_assigned = Path(label_assigned_filename).read_text() if Path(label_assigned_filename).exists() else ""
    assert experiment not in label_assigned, f"Labels already assigned for experiment {experiment} on this VM!"

    if experiment == "A":
        assign_experiment_A(aux)
    if experiment == "B":
        assign_experiment_B(aux)
    if experiment == "C":
        assign_experiment_C(aux)
    if experiment == "D":
        assign_experiment_D(aux)

    with open(label_assigned_filename, "a") as f:
        f.write(experiment)


def generate_shadowsets(aux, meta, cfg):
    assert os.path.exists(shadowsets_directory)
    experiment = sys.argv[2].upper()
    assert experiment in Path(
        label_assigned_filename).read_text(), f"Labels not yet assigned for experiment {experiment} on this VM!"

    if experiment == "A":
        gen_experiment_A(aux, meta, cfg)
        print("finished ~A thread.")
    if experiment == "B":
        gen_experiment_B(aux, meta, cfg)
        print("finished ~B thread.")
    if experiment == "C":
        gen_experiment_C(aux, meta, cfg)
        print("finished ~C thread.")
    if experiment == "D":
        gen_experiment_D(aux, meta, cfg)
        print("finished ~D thread.")


def print_status():
    print("memory stats:")
    print(
        f"total: {psutil.virtual_memory().total // 1_000_000} MB,    available: {psutil.virtual_memory().available // 1_000_000} MB,    used: {psutil.virtual_memory().percent} %")

    if len(sys.argv) == 2 or sys.argv[2].upper() == "A":
        print()
        print("----------------------")
        total_needed = 0
        total_generated = 0
        for eps in epsilons:
            generated, needed = stats_for_experiment("A", f"e{fo(eps)}", expA)
            total_needed += needed
            total_generated += generated
        print(
            f"experiment A completed: {total_generated} / {total_needed}... {round(total_generated / total_needed * 100)} %")

    if len(sys.argv) == 2 or sys.argv[2].upper() == "B":
        print()
        print("----------------------")
        total_needed = 0
        total_generated = 0
        for n in n_sizes:
            generated, needed = stats_for_experiment("B", f"n{n}", expB)
            total_needed += needed
            total_generated += generated
        print(
            f"experiment B completed: {total_generated} / {total_needed}... {round(total_generated / total_needed * 100)} %")

    if len(sys.argv) == 2 or sys.argv[2].upper() == "C":
        print()
        print("----------------------")
        total_generated, total_needed = stats_for_experiment("C", "", expC)
        print(
            f"experiment C completed: {total_generated} / {total_needed}... {round(total_generated / total_needed * 100)} %")

    if len(sys.argv) == 2 or sys.argv[2].upper() == "D":
        print()
        print("----------------------")
        total_needed = 0
        total_generated = 0
        for eps in epsilons:
            generated, needed = stats_for_experiment("D", f"e{fo(eps)}", expD)
            total_needed += needed
            total_generated += generated
        print(
            f"experiment D completed: {total_generated} / {total_needed}... {round(total_generated / total_needed * 100)} %")


def stats_for_experiment(experiment, sub_experiment, vars):
    print(f"experiment {experiment}, {sub_experiment}")

    total_generated = 0
    total_needed = 0
    for sdg in sdgs:
        total_shadowsets_completed = 0
        for s in range(vars.s + vars.r):
            total_needed += 1
            shadowset_filename = shadowsets_directory + f"exp{experiment}/{sub_experiment}/{sdg}/s{s}.parquet"
            if os.path.isfile(shadowset_filename):
                total_shadowsets_completed += 1
                total_generated += 1

        percent_done = round(total_shadowsets_completed / (vars.s + vars.r) * 100)
        print(f"\t\t{sdg}: {total_shadowsets_completed} / {(vars.s + vars.r)} shadowsets completed... {percent_done} %")

    return total_generated, total_needed


def assign_experiment_A(aux):
    for eps in epsilons:
        directory = shadowsets_directory + f"expA/e{fo(eps)}/"
        single_MI_targets, set_MI_targets = sample_targets(aux, expA.t)
        dump_artifact(create_label_matrix(expA.r, expA.s, single_MI_targets),
                      directory + label_matrix_filename + f"_singleMI")
        dump_artifact(create_label_matrix(expA.r, expA.s, set_MI_targets),
                      directory + label_matrix_filename + f"_setMI")


def assign_experiment_B(aux):
    for n, t in zip(n_sizes, t_sizes):
        directory = shadowsets_directory + f"expB/n{n}/"
        single_MI_targets, set_MI_targets = sample_targets(aux, t)
        dump_artifact(create_label_matrix(expB.r, expB.s, single_MI_targets),
                      directory + label_matrix_filename + f"_singleMI")
        dump_artifact(create_label_matrix(expB.r, expB.s, set_MI_targets),
                      directory + label_matrix_filename + f"_setMI")


def assign_experiment_C(aux):
    directory = shadowsets_directory + "expC/"
    single_MI_targets, set_MI_targets = sample_targets(aux, expC.t)
    dump_artifact(create_label_matrix(expC.r, expC.s, single_MI_targets),
                  directory + label_matrix_filename + f"_singleMI")
    dump_artifact(create_label_matrix(expC.r, expC.s, set_MI_targets), directory + label_matrix_filename + f"_setMI")


def assign_experiment_D(aux):
    for eps in epsilons:
        directory = shadowsets_directory + f"expD/e{fo(eps)}/"
        single_MI_targets, set_MI_targets = sample_targets(aux, expD.t)
        dump_artifact(create_label_matrix(expD.r, expD.s, single_MI_targets),
                      directory + label_matrix_filename + f"_singleMI")
        dump_artifact(create_label_matrix(expD.r, expD.s, set_MI_targets),
                      directory + label_matrix_filename + f"_setMI")


def gen_matrix_2(r, s, targets):
    t = len(targets)
    assert not (t % 2 or s % 2 or r % 2)
    print("config 2: half of the targets are in each shadowset")
    label_list = [np.sort(rng.choice(t, size=t // 2, replace=False)) for _ in range(s + r)]
    label_matrix_T = np.zeros((s + r, t), dtype=bool)
    for i in range(s + r):
        label_matrix_T[i][label_list[i]] = True
    label_matrix = np.swapaxes(label_matrix_T, 0, 1)

    return pd.DataFrame(label_matrix_T, columns=targets)


def gen_experiment_A(aux, meta, cfg):
    finished = False
    while not finished:
        finished = True

        for eps in sorted(epsilons, key=lambda
                _: rand.random()):  # make random so the tasks are distributed more evenly across each process
            incomplete = get_shadowsets_incomplete(shadowsets_directory + f"expA/e{fo(eps)}/", eps, expA)
            if len(incomplete) > 0:
                finished = False
                generate_shadowset_for_each_SDG(cfg, aux, meta, shadowsets_directory + f"expA/e{fo(eps)}/",
                                                rand.choice(incomplete), expA.n, eps, expA.exclude)


def gen_experiment_B(aux, meta, cfg):
    finished = False
    while not finished:
        finished = True

        for n in sorted(n_sizes, key=lambda
                _: rand.random()):  # make random so the tasks are distributed more evenly across each process
            incomplete = get_shadowsets_incomplete(shadowsets_directory + f"expB/n{n}/", n, expB)
            if len(incomplete) > 0:
                finished = False
                generate_shadowset_for_each_SDG(cfg, aux, meta, shadowsets_directory + f"expB/n{n}/",
                                                rand.choice(incomplete), n, expB.eps, expB.exclude)


def gen_experiment_C(aux, meta, cfg):
    finished = False
    while not finished:
        finished = True
        incomplete = get_shadowsets_incomplete(shadowsets_directory + f"expC/", None, expC)
        if len(incomplete) > 0:
            finished = False
            generate_shadowset_for_each_SDG(cfg, aux, meta, shadowsets_directory + f"expC/", rand.choice(incomplete),
                                            expC.n, expC.eps, expC.exclude)


def gen_experiment_D(aux, meta, cfg):
    finished = False
    while not finished:
        finished = True

        for eps in sorted(epsilons, key=lambda _: rand.random()):  # make random so the tasks are distributed more evenly across each process launched
            incomplete = get_shadowsets_incomplete(shadowsets_directory + f"expD/e{fo(eps)}/", eps, expD)
            if len(incomplete) > 0:
                finished = False
                generate_shadowset_for_each_SDG(cfg, aux, meta, shadowsets_directory + f"expD/e{fo(eps)}/", rand.choice(incomplete), expD.n, eps, expD.exclude)


def generate_shadowset_for_each_SDG(cfg, aux, meta, location, s, n, eps, exclude):
    gen_fns = {"mst": gen_mst, "priv": gen_priv, "gsd": gen_gsd}

    for sdg in sdgs:
        if n in exclude.get(sdg, []) or eps in exclude.get(sdg, []):
            print(f"\tskipping {sdg}, n{n}, e{eps}")
            continue
        if len(sys.argv) > 3 and sdg != sys.argv[3]:
            print(f"\tskipping {sdg}, n{n}, e{eps}")
            continue

        shadowset_filename = location + f"{sdg}/s{s}.parquet"
        # TODO: make option to only do one sdg
        if not os.path.isfile(shadowset_filename):
            print(f"\tgenerating for {sdg}, n{n}, e{eps}, #{s}")
            D_train = sample_train(aux, location, s, n)

            start = time.process_time()
            synth = gen_fns[sdg](cfg, D_train, meta, n, eps)
            end = time.process_time()

            if synth is not None:
                synth.to_parquet(shadowset_filename, index=False)
                dump_artifact(D_train.index.values, location + f"{sdg}/s{s}_train_ids")

            # record runtime
            # (there is a small race condition, if two processes edit this file at once.
            # This isn't very consequential, because both 'time' and 'num_sets' in the runtime dict
            # will be overwritten. So the aggregate, average time wouldn't be incorrect.)
            runtime = load_artifact(location + f"{sdg}/runtime")
            runtime["time"] += (end - start)
            runtime["num_sets"] += 1
            dump_artifact(runtime, location + f"{sdg}/runtime")


def dump_artifact(artifact, name):
    pickle_file = open(name, 'wb')
    pickle.dump(artifact, pickle_file)
    pickle_file.close()


def load_artifact(name):
    try:
        pickle_file = open(name, 'rb')
        artifact = pickle.load(pickle_file)
        pickle_file.close()
        return artifact
    except:
        return None


def sample_targets(aux, num_targets):
    hh_counts = aux['HHID'].value_counts()
    candidate_households = hh_counts[hh_counts >= min_HH_size].index

    set_MI_target_ids = pd.Series(candidate_households).sample(n=num_targets).values
    single_MI_target_ids = pd.Series(aux[~aux.HHID.isin(set_MI_target_ids)].index).sample(n=num_targets).values

    return single_MI_target_ids, set_MI_target_ids


def sample_train(aux, labels_location, s, n):
    set_MI_label_matrix = load_artifact(labels_location + "label_matrix_setMI")
    single_MI_label_matrix = load_artifact(labels_location + "label_matrix_singleMI")

    set_MI_targets = set_MI_label_matrix.columns
    single_MI_targets = single_MI_label_matrix.columns

    set_MI_members = set_MI_targets[set_MI_label_matrix.iloc[s, :]]
    single_MI_members = single_MI_targets[single_MI_label_matrix.iloc[s, :]]

    all_members = pd.concat([
        aux[aux.index.isin(single_MI_members)],
        aux[aux.HHID.isin(set_MI_members)]
    ])

    num_non_targets = n - all_members.shape[0]
    D_train = pd.concat([aux.sample(n=num_non_targets), all_members])

    return D_train.sample(frac=1)  # shuffle in members


def get_shadowsets_incomplete(location, experiment_param, vars):
    num_sets = vars.s + vars.r
    incomplete = []
    for s in range(num_sets):
        for sdg in sdgs:
            if experiment_param not in vars.exclude.get(sdg, []) and not os.path.isfile(location + f"{sdg}/s{s}.parquet"):
                # then this shadowset # needs to be generated for at least one of the sdgs
                incomplete.append(s)
                break

    return incomplete


def fo(eps):
    return '{0:.2f}'.format(eps)


def gen_mst(cfg, train, meta, n_synth, eps):
    columns = [m["name"] for m in meta]
    mst_gen = mst.MST(dataset=train[columns], metadata=meta, size=n_synth, epsilon=eps)
    try:
        mst_gen.run()
        synth = mst_gen.output
        if "age" in columns:
            synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
        return synth
    except ValueError:
        print(f"Error in MST for e{eps}, n{n_synth}")
        return None


def gen_priv(cfg, train, meta, n_synth, eps):
    columns = [m["name"] for m in meta]
    privbayes_gen = privbayes.PRIVBAYES(dataset=train[columns], metadata=meta, size=n_synth, epsilon=eps)
    privbayes_gen.run()
    synth = privbayes_gen.output
    if "age" in columns:
        synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
    return synth


def gen_gsd(cfg, train, meta, n_synth, eps):
    columns = [m["name"] for m in meta]
    encoded_train = encode_data_all_numeric(cfg, train, minmax_encode_catg=False)

    GSD_config = {feature_meta['name']: len(feature_meta['representation']) if feature_meta[
                                                                                   'name'] in cfg.categorical_columns else 1 \
                  for feature_meta in meta}
    domain = Domain.fromdict(GSD_config)
    data = Dataset(encoded_train, domain)
    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=cfg.gsd_k, bins=cfg.gsd_bins)
    stat_module = ChainedStatistics([marginal_module2])
    stat_module.fit(data)

    seed = rand.randint(0, 1000)
    delta = 1.0 / len(data) ** 2

    algo = GSD(
        domain=data.domain,
        print_progress=False,
        stop_early=True,
        num_generations=cfg.gsd_generations,
        population_size_muta=50,
        population_size_cross=50,
        sparse_statistics=True,
        data_size=train.shape[0]
    )

    # try:
    encoded_synth, query_ids = algo.fit_dp(PRNGKey(seed), stat_module=stat_module, epsilon=eps, delta=delta)
    synth = decode_data_from_numeric(cfg, encoded_synth.df, minmax_encode_catg=False)
    if "age" in columns:
        synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
    return synth
    # except:
    #     print(f"Error in GSD for e{eps}, n{n_synth}")
    #     return None


create_label_matrix = gen_matrix_2
main()
