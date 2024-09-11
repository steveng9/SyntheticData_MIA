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
from utils.utils_data import Dataset, Domain
from stats import Marginals, ChainedStatistics
from models import GSD
from jax.random import PRNGKey


from collections import Counter

from util import *




def determine_mst_marginals(cfg, aux, columns, _, meta, eps, n_size, filename=None):
    cliques = []
    gen = mst.MST(
        dataset=aux[columns].sample(n=n_size),
        metadata=meta,
        size=n_size,
        epsilon=eps,
    )
    gen.run()

    for FP in gen.cliques:
        attrs = np.array(FP).tolist()
        attrs.sort()
        cliques.append(tuple(attrs))
    save_off_intermediate_FPs(cfg, eps, cliques, "MST", filename=filename)
    return cliques



def determine_privbayes_conditionals(cfg, aux, columns, _, meta, eps, n_size, filename=None):
    conditionals = []
    gen = privbayes.PRIVBAYES(
        dataset=aux[columns].sample(n=n_size),
        metadata=meta,
        size=n_size,
        epsilon=eps,
    )
    gen.run()

    for FP in gen.conditionals:
        attrs = np.array(FP).tolist()
        child, parents = attrs[0], attrs[1:]
        parents.sort()
        conditionals.append(tuple([child]+parents))
    save_off_intermediate_FPs(cfg, eps, conditionals, "PrivBayes", filename=filename)
    return conditionals


def determine_gsd_marginals(cfg, aux, columns, catg_cols, meta, eps, n_size, filename=None):
    train = aux[columns + ['HHID']].sample(n=n_size)

    GSD_config = {feature_meta['name']: len(feature_meta['representation']) if feature_meta['name'] in catg_cols else 1 \
                  for feature_meta in meta}

    domain = Domain.fromdict(GSD_config)
    encoded_train = encode_data_all_numeric(cfg, train, minmax_encode_catg=False)
    data = Dataset(encoded_train, domain)

    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=cfg.gsd_k, bins=cfg.gsd_bins)
    all_possible_queries = marginal_module2.queries
    stat_module = ChainedStatistics([marginal_module2])
    stat_module.fit(data)

    delta = 1.0 / len(data) ** 2

    # TODO pull out constants
    algo = GSD(domain=data.domain,
               print_progress=True,
               stop_early=True,
               num_generations=cfg.gsd_generations,
               population_size_muta=50,
               population_size_cross=50,
               sparse_statistics=True,
               data_size=n_size
               )

    query_ids = algo.fit_dp(PRNGKey(rand.randint(0, 1000)), stat_module=stat_module, epsilon=eps, delta=delta, only_determine_fps=True)

    FPs = [tuple([round(float(x), 3) for x in all_possible_queries[query_id]]) for query_id in query_ids]
    save_off_intermediate_FPs(cfg, eps, FPs, "GSD", filename=filename)
    return FPs


def determine_rap_queries(cfg, aux, columns, _, meta, eps, n_size, filename=None):

    ### This code is copy-pasted from the Relaxed-Adaptive-Projection Library, with unused
    ### segments removed, and constants added that are used by authors (Aydore et al., 2021)

    n_prime = cfg.synth_size
    seed = 0
    categorical_consistency = True
    aux_encoded, columns_domain = binarize_discrete_features_evenly(cfg, aux, columns)

    key = random.PRNGKey(seed)
    # sample rows from auxiliary data
    D = aux_encoded[np.random.choice(aux.shape[0], n_size)]
    n, d = D.shape
    delta = 1 / n ** 2

    stat_module = __import__("statistickway")

    # First select random k-way marginals from the dataset
    kway_attrs = [p for p in itertools.combinations(columns, cfg.rap_k)]
    kway_compact_queries, _ = get_queries(columns_domain, kway_attrs)
    all_statistic_fn = stat_module.preserve_statistic(kway_compact_queries)
    true_statistics = all_statistic_fn(D)

    # projection_interval = ProjectionInterval(*args.project) if args.project else None
    projection_interval = None

    epochs = min(cfg.rap_epochs, np.ceil(len(true_statistics) / cfg.rap_top_q).astype(np.int32)) \
        if not cfg.rap_all_queries else 1

    if cfg.rap_all_queries:
        # ensure consistency w/ top_q for one-shot case (is this correct?)
        cfg.rap_top_q = len(true_statistics)

    # Initial analysis
    print("Number of queries: {}".format(len(true_statistics)))
    print("Number of epochs: {}".format(epochs))
    print("Epsilon: {}".format(eps))

    if categorical_consistency:
        feats_csum = np.array([0] + list(columns_domain.values())).cumsum()
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
    # growing number of sanitized statistics to preserve
    key, subkey = random.split(subkey)
    queries = rap.train(D, columns_domain, kway_attrs, key, train_last_epoch=False)

    queries_sorted = []
    for FP in queries:
        attrs = np.array(FP).tolist()
        attrs.sort()
        queries_sorted.append(tuple(attrs))
    save_off_intermediate_FPs(cfg, eps, queries_sorted, "RAP", filename=filename)
    return queries_sorted

