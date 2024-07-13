from encode_data import *

import sys
import os
import pandas as pd
import numpy as np
import json
import math
from tqdm import tqdm

from pandas.api.types import is_numeric_dtype
from scipy import stats
from matplotlib import pyplot
from functools import reduce

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder

import warnings
warnings.filterwarnings("ignore")
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

sys.path.append('relaxed-adaptive-projection/relaxed_adaptive_projection/')
import rap
import logging
import os
import sys
import configargparse
import pandas as pd
import itertools
from jax import numpy as jnp, random

from relaxed_adaptive_projection import RAPConfiguration, RAP
from relaxed_adaptive_projection.constants import Norm, ProjectionInterval
from utils_data import data_sources, ohe_to_categorical




# Experimental Configuration
class Config:
    def __init__(
            self,
            data_name,
            n_runs_MA=30,
            train_sizes={100: 10, 316: 26, 1_000: 64, 3162: 160, 10_000: 400, 31_622: 1000},
            train_size=1_000,
            set_MI=False,
            household_min_size=5,
            # epsilons=list(reversed([round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 3)])),
            epsilons=[round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 3)],
            check_arbitrary_fps=False,

            # weight threshold (values) is based epsilon (keys)
            fp_weight_thresholds={.01: .5, 1: .6, 10: .7, 100: .8, 1000: .85},

            overlapping_aux=True,

            # RAP parameters
            rap_k=3,
            rap_epochs=7,
            rap_top_q=50,
            rap_all_queries=False,
            rap_iterations=1500,
            rap_use_FP_threshold=4,
            num_kways=64,
            use_subset_of_kways=False,
            seed=None,

            # GSD parameters
            gsd_k=2,
            gsd_bins=[2, 4, 8, 16, 32],
            gsd_generations=20_0,
    ):
        self.data_name = data_name
        self.n_runs_MA = n_runs_MA
        self.train_sizes = train_sizes
        self.train_size = train_size
        self.synth_size = train_size
        self.num_targets = train_sizes[train_size]
        self.num_members = train_sizes[train_size] // 2
        self.set_MI = set_MI
        self.household_min_size = household_min_size
        self.epsilons = epsilons
        self.check_arbitrary_fps = check_arbitrary_fps
        self.fp_weight_thresholds = fp_weight_thresholds
        self.overlapping_aux = overlapping_aux
        self.rap_k = rap_k
        self.rap_epochs = rap_epochs
        self.rap_top_q = rap_top_q
        self.rap_all_queries = rap_all_queries
        self.rap_iterations = rap_iterations
        self.rap_use_FP_threshold = rap_use_FP_threshold
        self.num_kways = num_kways
        self.use_subset_of_kways = use_subset_of_kways
        self.seed = seed
        self.gsd_k = gsd_k
        self.gsd_bins = gsd_bins
        self.gsd_generations = gsd_generations

    def get_filename(self, task, use_RAP_config=False, overlap=True):
        MI_type = 'set' if self.set_MI else 'single'
        filename = f"{task}3_results_{self.data_name}_{C.n_bins}_{MI_type}MI_{self.train_size}"
        if use_RAP_config:
            filename += f"_RAP_{self.rap_top_q}_{self.rap_k}_{self.rap_epochs}_{self.num_kways}_{self.use_subset_of_kways}"
        if not overlap:
            filename += "_nonoverlap"
        return filename




############################################______________________________________________
############################################______________________________________________
############################################______________________________________________


def determine_weight_threshold(cfg, eps, fp_weights):
    # choose highest weight threshold of when to incorporate focal-point based on epsilon
    return max(fp_weights.values()) * max([t for e, t in cfg.fp_weight_thresholds.items() if eps >= e])


def make_FP_filename(cfg, sdg, eps, specify_epsilon=True, try_epsilons=C.shadow_epsilons):
    eps = min([e for e in try_epsilons if eps <= e])

    # filename = f"FP2_{cfg.data_name}_{sdg}_{'{0:.2f}'.format(eps)}_{C.n_bins}_False"
    filename = f"FP2_{cfg.data_name}_{sdg}_{'{0:.2f}'.format(eps)}_10_False"
    if not specify_epsilon:
        filename = f"FP2_{cfg.data_name}_{sdg}_100000.00_{C.n_bins}_False"
    if sdg == "RAP":
        filename += f"_50_{cfg.rap_k}_10_2000"
        # filename += f"_{cfg.rap_top_q}_{cfg.rap_k}_{cfg.rap_epochs}_{cfg.rap_iterations}"
        # filename = "FP_snake_RAP_100000.00_10_3_True_70_3_10_64_False"
    if sdg == "GSD":
        filename += f"_{cfg.gsd_k}"

    print(filename)
    return filename

def membership_advantage(y_true, scores):
    y_pred = scores > .5
    sample_weight = 2 * np.abs(0.5 - scores)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    ma = tpr - fpr
    ma = (ma + 1) / 2
    return ma

def area_under_curve(y_true, predictions):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, predictions)
        return auc(fpr, tpr)
    except ValueError:
        return None

def activate_1(p_rel, confidence=1, center=True) -> np.ndarray:
    logs = np.log(p_rel)
    median = np.median(logs) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (logs - median)))
    return probabilities

def activate_2(p_rel, confidence=1, center=True) -> np.ndarray:
    zscores = stats.zscore(p_rel)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities

def activate_3(p_rel, confidence=1, center=True) -> np.ndarray:
    logs = np.log(p_rel)
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities

def activate_4(p_rel, confidence=1, center=True) -> np.ndarray:
    median = np.median(p_rel) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (p_rel - median)))
    return probabilities


def get_data(cfg):
    if cfg.data_name == "cali":
        return california_data(cfg)
    if cfg.data_name == "snake":
        return snake_data(cfg)
    return None


def sample_experimental_data(cfg, aux, columns):
    # determine all 'candidate' clusters of minimum size for "set membership inference"
    hh_counts = aux['HHID'].value_counts()
    candidate_households = hh_counts[hh_counts >= cfg.household_min_size].index

    target_ids = pd.Series(candidate_households).sample(n=cfg.num_targets) \
        if cfg.set_MI else pd.Series(aux.index).sample(n=cfg.num_targets).values
    targets = aux[aux['HHID'].isin(target_ids)] \
        if cfg.set_MI else aux[aux.index.isin(target_ids)]
    member_ids = pd.Series(target_ids).sample(n=cfg.num_members).values
    members = aux[aux['HHID'].isin(member_ids)] \
        if cfg.set_MI else aux[aux.index.isin(member_ids)]

    # aux_sample_no_targets = aux[~aux.isin(target_ids)].sample(n=cfg.train_size)
    aux_sample_no_targets = aux[aux.merge(targets.drop_duplicates(), how='left', indicator=True)["_merge"] == "left_only"].sample(n=(cfg.train_size-members.shape[0]))
    membership = np.array([1 if c in member_ids else 0 for c in target_ids.tolist()])
    train = pd.concat([aux_sample_no_targets, members]).sample(frac=1)  # sample() to shuffle targets in

    return target_ids, targets, membership, train, np.random.randint(10000)


def get_queries(columns_domain, kway_attrs, N=None):
    col_map = {col: i for i, col in enumerate(columns_domain.keys())}

    feat_pos = []
    cur = 0
    for f, sz in enumerate(columns_domain.values()):
        feat_pos.append(list(range(cur, cur + sz)))
        cur += sz

    queries = []
    if N is None:
        for feat in kway_attrs:
            queries.append([feat_pos[col_map[col]] for col in feat])

        num_queries = sum([reduce(lambda x, y: x * y, [len(i) for i in q], 1) for q in queries])
        return queries, num_queries

    for feat in kway_attrs:
        positions = []
        for col in feat:
            i = col_map[col]
            positions.append(feat_pos[i])
        for tup in itertools.product(*positions):
            queries.append(tup)

    num_queries = len(queries) if N == -1 else N

    return np.array(queries, np.int64)[:num_queries], num_queries

def get_rap_synth(cfg, columns, train_encoded, eps, columns_domain):

    ### This code is copy-pasted from Relaxed-Adaptive-Projection Library, with unused
    ### segments removed, and constants added that are used by authors (Aydore et al., 2021)

    stat_module = __import__("statistickway")
    n_prime = cfg.synth_size
    seed = 0
    categorical_consistency = True
    key = random.PRNGKey(seed)
    n, d = train_encoded.shape
    delta = 1 / n ** 2

    # consider all k-way queries
    kway_attrs = [p for p in itertools.combinations(columns, cfg.rap_k)]
    if cfg.use_subset_of_kways and len(kway_attrs) > cfg.num_kways:
        prng = np.random.RandomState(cfg.seed) if cfg.seed is not None else np.random
        kway_attrs = [kway_attrs[i] for i in prng.choice(len(kway_attrs), cfg.num_kways, replace=False)]
    kway_compact_queries, _ = get_queries(columns_domain, kway_attrs)

    all_statistic_fn = stat_module.preserve_statistic(kway_compact_queries)
    true_statistics = all_statistic_fn(train_encoded)

    projection_interval = None

    epochs = min(cfg.rap_epochs, jnp.ceil(len(true_statistics) / cfg.rap_top_q).astype(jnp.int32)) \
        if not cfg.rap_all_queries else 1


    if cfg.rap_all_queries:
        # ensure consistency w/ top_q for one-shot case (is this correct?)
        cfg.rap_top_q = len(true_statistics)

    if categorical_consistency:
        feats_csum = jnp.array([0] + list(columns_domain.values())).cumsum()
        feats_idx = [list(range(feats_csum[i], feats_csum[i + 1])) for i in range(len(feats_csum) - 1)]
    else:
        feats_idx = None

    algorithm_configuration = RAPConfiguration(
        num_points=n,
        num_generated_points=n_prime,
        num_dimensions=d,
        statistic_function=all_statistic_fn,
        preserve_subset_statistic=stat_module.preserve_subset_statistic,
        get_queries=get_queries,
        get_sensitivity=stat_module.get_sensitivity,
        verbose=False,
        silent=False,
        epochs=epochs,
        iterations=cfg.rap_iterations,
        epsilon=eps,
        delta=delta,
        norm=Norm('L2'),
        projection_interval=projection_interval,
        optimizer_learning_rate=.001,
        lambda_l1=0,
        k=cfg.rap_k,
        top_q=cfg.rap_top_q,
        use_all_queries=cfg.rap_all_queries,
        rap_stopping_condition=1e-7,
        initialize_binomial=False,
        feats_idx=feats_idx,
    )

    key, subkey = random.split(key)
    rap = RAP(algorithm_configuration, key=key)
    key, subkey = random.split(subkey)
    queries_used = rap.train(train_encoded, columns_domain, kway_attrs, key, train_last_epoch=True)

    # get synthetic dataset

    # if categorical_consistency:
    Dprime_ohe = rap.generate_rounded_dataset(key)
    Dprime_catg = pd.DataFrame(data=ohe_to_categorical(Dprime_ohe, feats_idx), columns=list(columns))

    all_synth_statistics_ohe = all_statistic_fn(Dprime_ohe)
    max_final_ohe = np.max(np.absolute(true_statistics - all_synth_statistics_ohe))
    l1_final_ohe = np.linalg.norm(true_statistics - all_synth_statistics_ohe, ord=1) / float(len(kway_attrs))
    l2_final_ohe = np.linalg.norm(true_statistics - all_synth_statistics_ohe, ord=2)
    print("\tFinal rounded max abs error", max_final_ohe)
    print("\tFinal rounded L1 error", l1_final_ohe)
    print("\tFinal rounded L2 error", l2_final_ohe)

    names = ["epsilon", "max_final_ohe", "l2_final_ohe", "l1_final_ohe"]
    res = [eps, max_final_ohe, l2_final_ohe, l1_final_ohe]
    results = pd.DataFrame([res], columns=names)

    synth_file_name = DATA_DIR + f"Thesis/experiment_artifacts/rap_synth_eps{'{0:.2f}'.format(eps)}_{cfg.data_name}"
    results_file_name = DATA_DIR + f"Thesis/experiment_artifacts/rap_synth_results_{cfg.data_name}"

    if os.path.exists(results_file_name):
        results_prev = pd.read_csv(results_file_name)
        results = results._append(results_prev, sort=False)
    results.to_csv(results_file_name, index=False)

    # synth = rap.D_prime
    synth = pd.DataFrame(Dprime_ohe)
    synth.to_csv(synth_file_name, index=False)

    return synth, Dprime_catg, queries_used

def wasserstein_distance(cfg, d1, d2, columns, encode_d2=True):
    d1 = pd.DataFrame(binarize_discrete_features_evenly(cfg, d1, columns)[0])
    d2 = pd.DataFrame(binarize_discrete_features_evenly(cfg, d2, columns)[0] if encode_d2 else d2)
    wd = 0
    ratio = d1.shape[0] / d2.shape[0]
    for col in d1.columns:
        wd += abs(d1[col].sum() - d2[col].sum()*ratio) / d1.shape[0]
    return wd


def moderate_recursively(x):
    new_min = .2
    new_max = 5
    c_min = np.emath.logn(new_min, x.min())
    c_max = np.emath.logn(new_max, x.max())
    c = (c_min + c_max) / 2
    return x**(1/c)

def moderate_sequentially(x):
    new_min = .9
    c_min = np.emath.logn(new_min, x.min())
    c_max = np.emath.logn(1/new_min, x.max())
    c = (c_min + c_max) / 2
    return x**(1/c)


def generate_arbitrary_FPs(columns, num, size_min, size_max):
    FPs = {}
    n = len(columns)
    all_combos = [list(itertools.combinations(columns, l)) for l in range(size_min, size_max+1)]
    num_combos = sum(len(combos) for combos in all_combos)
    while len(FPs) < num and len(FPs) < num_combos:
        size_choice = np.random.randint(size_min, size_max + 1)
        pick = all_combos[size_choice - size_min][np.random.randint(math.comb(n, size_choice))]
        FPs[pick] = 1
    return FPs

def save_off_intermediate_FPs(cfg, eps, FPs, sdg):
    filename = make_FP_filename(cfg, sdg, eps)
    FP_frequencies = load_artifact(filename) or {}
    for FP in FPs:
        FP_frequencies[FP] = FP_frequencies.get(FP, 0) + 1

    FP_frequencies = dict(sorted(FP_frequencies.items(), key=lambda x: -x[1]))
    dump_artifact(FP_frequencies, filename)

def plot_output(scores):
    if C.verbose and False:
        bins = np.linspace(0, 1, 50)
        pyplot.hist(scores, bins)
        pyplot.legend(loc='upper right')
        pyplot.show()

def score_attack(cfg, A, num_queries_used, targets, target_ids, membership, activation_fn=activate_3):
    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if cfg.set_MI else targets.index.values,
        'A': pd.Series(A / np.maximum(np.array([1] * targets.shape[0]), num_queries_used))
    })

    scores = []
    grouped_predictions = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped_predictions.get_group(hhid).A.mean())
    activated_scores = activation_fn(np.array(scores))

    MA = membership_advantage(membership, activated_scores)
    AUC = area_under_curve(membership, activated_scores)

    return predictions, MA, AUC


