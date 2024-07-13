import sys
import time
import random as rand

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')
# import disjoint_set

import mst
import privbayes

sys.path.append('private_gsd/')
# from utils.utils_data import Dataset, Domain
# from stats import Marginals, ChainedStatistics
# from models import GSD
# from jax.random import PRNGKey


# from collections import Counter

from util import *
from determine_focal_points import *
from conduct_attacks import *


def main():

    ### 1. Shadow Modelling
    # determine_all_FP()

    ### 2. measuring densities, and scoring attack
    attack()


def determine_all_FP():
    cali_cfg = Config("cali")
    snake_cfg = Config("snake")

    _, cali_aux, cali_columns, cali_meta, _ = get_data(cali_cfg)
    _, snake_aux, snake_columns, snake_meta, _ = get_data(snake_cfg)

    methods = [
        # (cali_cfg, determine_gsd_marginals),
        # (cali_cfg, determine_mst_marginals),
        # (cali_cfg, determine_privbayes_conditionals),
        (cali_cfg, determine_rap_queries),
        # (Config("cali", rap_k=5, rap_top_q=30), determine_rap_queries),
        # (snake_cfg, determine_gsd_marginals),
        # (snake_cfg, determine_mst_marginals),
        # (snake_cfg, determine_privbayes_conditionals),
        (snake_cfg, determine_rap_queries),
        # (Config("snake", rap_k=5, rap_top_q=30), determine_rap_queries),
    ]

    for i in range(C.n_shadow_runs):
        print("\n\nRUN: ", i)
        for eps in C.shadow_epsilons:
            print(eps)
            for cfg, sdg_method in methods:
                if cfg.data_name == "cali":
                    sdg_method(cfg, cali_aux, cali_columns, cali_cfg.categorical_columns, cali_meta, eps)
                elif cfg.data_name == "snake":
                    sdg_method(cfg, snake_aux, snake_columns, snake_cfg.categorical_columns, snake_meta, eps)


def attack():

    datasets = [
        "snake",
        "cali",
    ]
    set_MIs = [
        True,
        False,
    ]
    sdgs = {
        "MST": attack_mst,
        "PrivBayes": attack_privbayes,
        # "GSD": attack_gsd,
        # "RAP": attack_rap
    }
    train_sizes = {
        # 100: 10,
        # 316: 26,
        # 1_000: 64,
        # 3_162: 160,
        10_000: 400,
        31_622: 1000,
    }
    # train_sizes = Config(None).train_sizes

    # epsilons = [round(10 ** x, 2) for x in np.arange(-1, 3.1)]

    for r in tqdm(range(C.n_runs)):
        print(f"\n\n\nRUN {r}")
        for t, tar in train_sizes.items():
            print("\n\nTRAIN SIZE: ", t, tar)
            for dataset in datasets:
                for set_MI in set_MIs:
                    if dataset == "cali" and t > 20_000: continue

                    attack_experiments(sdgs, Config(
                        dataset,
                        train_size=t,
                        set_MI=set_MI,
                        # overlapping_aux=False,
                    ))


main()
