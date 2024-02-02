import sys
import pandas as pd
import numpy as np
import pickle
import json
import math
from tqdm import tqdm

from pandas.api.types import is_numeric_dtype
from scipy import stats
from matplotlib import pyplot
from types import SimpleNamespace
from functools import reduce

from sklearn.metrics import confusion_matrix
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



DATA_DIR = "/Users/golobs/Documents/GradSchool/"


# Experimental Configuration
class Config:
    def __init__(
            self,
            data_name,
            n_runs_MA=10,
            train_sizes={100: 10, 316: 26, 1_000: 64, 3162: 160, 10_000: 400, 31_622: 1000},
            train_size=1_000,
            set_MI=True,
            household_min_size=4,
            epsilons=list(reversed([round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 3)])),

            # RAP parameters
            rap_k=3,
            rap_epochs=10,
            rap_top_q=200,
            rap_all_queries=False,
            rap_iterations=2000,
            rap_use_FP_threshold=2,
            num_kways=64,
            use_subset_of_kways=False,
            seed=None,
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
        self.rap_k = rap_k
        self.rap_epochs = rap_epochs
        self.rap_top_q = rap_top_q
        self.rap_all_queries = rap_all_queries
        self.rap_iterations = rap_iterations
        self.rap_use_FP_threshold = rap_use_FP_threshold
        self.num_kways = num_kways
        self.use_subset_of_kways = use_subset_of_kways
        self.seed = seed

    def get_filename(self, task, use_RAP_config=False):
        MI_type = 'set' if self.set_MI else 'single'
        filename = f"{task}_results_{self.data_name}_{C.n_bins}_{MI_type}MI_{self.train_size}"
        if use_RAP_config:
            filename += f"_RAP_{self.rap_top_q}_{self.rap_k}_{self.rap_epochs}_{self.num_kways}_{self.use_subset_of_kways}"
        return filename


# constants
C = SimpleNamespace(
    verbose=False,
    n_bins=10,
    n_runs=50,

    # shadow modelling
    shadow_epsilons=[round(10 ** x, 2) for x in np.arange(-1, 3.1)],
    n_shadow_runs=30,
    shadow_train_size=10_000,
    shadow_synth_size=10_000,

    # KDE
    use_categorical_features=True,
    samples_append_targets=False,
    rap_bucket_numeric=True,
)




############################################______________________________________________
############################################______________________________________________
############################################______________________________________________

def make_FP_filename(cfg, sdg, eps):
    filename = f"FP_{cfg.data_name}_{sdg}_{'{0:.2f}'.format(eps)}_{C.n_bins}_{cfg.rap_k}_{cfg.set_MI}"
    if sdg == "RAP":
        filename += f"_{cfg.rap_top_q}_{cfg.rap_k}_{cfg.rap_epochs}_{cfg.num_kways}_{cfg.use_subset_of_kways}"
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

def dump_artifact(artifact, name):
    pickle_file = open(DATA_DIR + f'Thesis/experiment_artifacts/{name}', 'wb')
    pickle.dump(artifact, pickle_file)
    pickle_file.close()

def load_artifact(name):
    try:
        pickle_file = open(DATA_DIR + f'Thesis/experiment_artifacts/{name}', 'rb')
        artifact = pickle.load(pickle_file)
        pickle_file.close()
        return artifact
    except:
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

def convert_finite_ordered_to_numeric(df):
    meta = pd.read_json(DATA_DIR + "SNAKE/meta.json")
    df_new = df.copy()
    for col in ['gradeatn', 'faminc']:
        df_new[col] = df[col].apply(lambda x: meta[meta['name'] == col].representation.values[0].index(x))
        df_new[col] = df_new[col].astype(np.int64)
    return df_new

def revert_finite_ordered_to_numeric(df):
    meta = pd.read_json(DATA_DIR + "SNAKE/meta.json")
    df_new = df.copy()
    for col in ['gradeatn', 'faminc']:
        upper = len(meta[meta['name'] == col].representation.values[0]) - 1
        df_new[col] = df[col].apply(lambda x: meta[meta['name'] == col].representation.values[0][max(min(x, upper), 0)])
        df_new[col] = df_new[col].astype(np.str_)
    return df_new

def fit_continuous_features_equaldepth(aux_data, name):
    n_per_basket = aux_data.shape[0] // C.n_bins
    thresholds = {}
    for col in aux_data.columns:
        vals = sorted(aux_data[col].values)
        thresholds[col] = [vals[i] for i in range(0, aux_data.shape[0], n_per_basket)]
    dump_artifact(thresholds, f"{name}_thresholds_for_continuous_features_{C.n_bins}")

def discretize_continuous_features_equaldepth(data, name):
    thresholds = load_artifact(f"{name}_thresholds_for_continuous_features_{C.n_bins}")
    data_copy = pd.DataFrame()
    for col in data.columns:
        data_copy[col] = np.digitize(data[col].values, thresholds[col])
    return data_copy


def fit_discrete_features_evenly(name, aux_data, meta, columns):
    meta = pd.DataFrame(meta)
    columns_encodings = {}
    columns_domain = {}
    for col in columns:
        col_data = aux_data[col]
        if is_numeric_dtype(col_data) and C.rap_bucket_numeric:

            # if n_bins doesn't divide the values nicely, then this logic more evenly
            # distributes the data to bins than the above logic
            splits = np.array_split(sorted(col_data.values), C.n_bins)
            basket_edges = [0]
            for i in range(1, C.n_bins):
                # don't duplicate basket edges when basket is overfull
                basket_edges.append(splits[i][0] if splits[i][0] > basket_edges[i-1] else basket_edges[i-1]+1)
            columns_encodings[col] = basket_edges
            columns_domain[col] = C.n_bins
        else:
            categories = meta[meta["name"] == col].representation.values[0]
            ohe = OneHotEncoder(categories=[categories]).fit(np.reshape(col_data.to_numpy(), (-1, 1)))
            columns_encodings[col] = ohe
            columns_domain[col] = len(categories)

    dump_artifact(columns_encodings, f"{name}_thresholds_for_discrete_features_{C.n_bins}bins")
    dump_artifact(columns_domain, f"{name}_ohe_domain_{C.n_bins}bins")


def binarize_discrete_features_evenly(cfg, data, columns):
    columns_encodings = load_artifact(f"{cfg.data_name}_thresholds_for_discrete_features_{C.n_bins}bins")
    columns_domain = load_artifact(f"{cfg.data_name}_ohe_domain_{C.n_bins}bins")

    ohe_data = []
    for col in columns:
        col_data = data[col]
        col_encoding = columns_encodings[col]
        if is_numeric_dtype(col_data) and C.rap_bucket_numeric:
            bins = np.digitize(col_data, col_encoding)
            ohe_data.append(np.eye(C.n_bins)[bins - 1])
        else:
            ohe_data.append(col_encoding.transform(np.reshape(col_data.to_numpy(), (-1, 1))).toarray())

    return np.hstack(ohe_data), columns_domain

def decode_rap_synth(cfg, columns, meta, synth):
    meta = pd.DataFrame(meta)
    columns_encodings = load_artifact(f"{cfg.data_name}_thresholds_for_discrete_features_{C.n_bins}bins")
    synth_decoded = pd.DataFrame()

    for i, col in enumerate(columns):
        enc = columns_encodings[col]
        if isinstance(enc, list):
            # decode back into number from bucket
            enc.append(np.inf)
            bins = synth[col].values
            min_val = int(meta[meta["name"] == col].representation.values[0][0])
            max_val = int(meta[meta["name"] == col].representation.values[0][-1])
            domain = list(zip(
                np.array([max(a, b) for a, b in zip(np.take(enc, bins), [min_val]*synth.shape[0])]),
                np.array([min(c, d) for c, d in zip(np.take(enc, bins+1), [max_val]*synth.shape[0])])
            ))
            synth_decoded[col] = np.array([np.random.randint(lower, max(upper, lower+1)) for lower, upper in domain])
        else:
            # de-onehot encode category
            synth_decoded[col] = enc.categories_[0][synth[col]]

    return synth_decoded


def fit_KDE_encoder(data_name, aux, meta, numerical_columns, catg_columns):
    meta = pd.DataFrame(meta)
    catg_encoder = OrdinalEncoder(categories=[meta[meta["name"] == col].representation.values[0] for col in catg_columns]).fit(aux[catg_columns])

    scalars = {col: MinMaxScaler().fit(aux[[col]]) for col in numerical_columns}
    dump_artifact((catg_encoder, catg_columns), f"KDE_{data_name}_catg_encoder")
    dump_artifact((scalars, numerical_columns), f"KDE_{data_name}_numerical_encoder_bins{C.n_bins}")


def encode_data_for_KDE(cfg, aux, synth, targets, target_ids, sample_seed):
    # encode data
    # -----------------------------
    target_exclusion_list = aux.HHID.isin(target_ids) if cfg.set_MI else aux.index.isin(target_ids)
    aux_sample = aux[~target_exclusion_list].sample(n=synth.shape[0], random_state=sample_seed) # TODO: see if density is n-independent (try huge n for aux_sample)
    if C.samples_append_targets: aux_sample = pd.concat([aux_sample, targets]).sample(frac=1) # shuffle

    numeric_encoders, numeric_columns = load_artifact(f"KDE_{cfg.data_name}_numerical_encoder_bins{C.n_bins}")
    encoded_synth = pd.DataFrame()
    encoded_targets = pd.DataFrame()
    encoded_aux_sample = pd.DataFrame()
    for col in numeric_columns:
        encoded_synth[[col]] = numeric_encoders[col].transform(synth[[col]])
        encoded_targets[[col]] = numeric_encoders[col].transform(targets[[col]])
        encoded_aux_sample[[col]] = numeric_encoders[col].transform(aux_sample[[col]])
    if C.use_categorical_features:
        catg_encoder, catg_columns = load_artifact(f"KDE_{cfg.data_name}_catg_encoder")
        if len(catg_columns) > 0:
            encoded_synth[catg_columns] = MinMaxScaler().fit_transform(catg_encoder.transform(synth[catg_columns]))
            encoded_targets[catg_columns] = MinMaxScaler().fit_transform(catg_encoder.transform(targets[catg_columns]))
            encoded_aux_sample[catg_columns] = MinMaxScaler().fit_transform(catg_encoder.transform(aux_sample[catg_columns]))

    return encoded_synth, encoded_targets, encoded_aux_sample

def get_data(cfg):
    if cfg.data_name == "cali":
        return california_data(cfg)
    if cfg.data_name == "snake":
        return snake_data()
    return None

def california_data(cfg):
    columns = [str(x) for x in range(9)]
    meta = [{"name": col, "representation": list(range(C.n_bins))} for col in columns] # TODO is this range correct?
    aux_original = pd.DataFrame(StandardScaler().fit_transform(fetch_california_housing(as_frame=True).frame.sample(frac=1)), columns=columns)

    fit_continuous_features_equaldepth(aux_original, "cali")
    aux = discretize_continuous_features_equaldepth(aux_original, "cali")
    fit_discrete_features_evenly("cali", aux, pd.DataFrame(meta), columns)
    fit_KDE_encoder("cali", aux, meta, columns, [])

    aux["HHID"] = np.hstack([[i]*cfg.household_min_size for i in range(math.ceil(aux.shape[0] / cfg.household_min_size))])[:aux.shape[0]]
    meta = [{'name': str(col), 'type': 'finite/ordered', 'representation': range(C.n_bins)} for col in columns]
    return None, aux, columns, meta, "cali"


def snake_data():
    with open(DATA_DIR + "SNAKE/meta.json") as f:
        meta = json.load(f)
    # meta = pd.read_json(DATA_DIR + "SNAKE/meta.json")

    aux = pd.read_parquet(DATA_DIR + "SNAKE/base.parquet")
    aux['HHID'] = aux.index
    aux.index = range(aux.shape[0])
    columns = aux[np.take(aux.columns, range(15))].columns
    numeric_columns = ['age', 'ownchild', 'hoursut']
    catg_columns = [col for col in columns if col not in numeric_columns]

    fit_discrete_features_evenly("snake", aux, pd.DataFrame(meta), columns)
    fit_KDE_encoder("snake", aux, meta, numeric_columns, catg_columns)

    return None, aux, columns, meta, "snake"


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

def save_off_intermediate_FPs(cfg, eps, all_FPs, sdg):
    filename = make_FP_filename(cfg, sdg, eps)
    FP_frequencies = load_artifact(filename) or {}
    for FPs in all_FPs:
        for FP in FPs:
            attrs = np.array(FP).tolist()
            attrs.sort()
            FP_tuple = tuple(attrs)
            FP_frequencies[FP_tuple] = FP_frequencies.get(FP_tuple, 0) + 1

    FP_frequencies = dict(sorted(FP_frequencies.items(), key=lambda x: -x[1]))
    dump_artifact(FP_frequencies, filename)

def plot_output(scores):
    if C.verbose:
        bins = np.linspace(0, 1, 50)
        pyplot.hist(scores, bins)
        pyplot.legend(loc='upper right')
        pyplot.show()