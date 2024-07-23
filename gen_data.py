import os
import sys
import random as rand

from util import *

sys.path.append('private_gsd/')

from utils.utils_data import Dataset, Domain
from stats import Marginals, ChainedStatistics
from models import GSD
from jax.random import PRNGKey



def determine_fps_GDS():
    data_size = 316

    snake_cfg = Config("snake", train_size=data_size)
    (numeric_cols, catg_cols), snake_aux, snake_columns, snake_meta, _ = get_data(snake_cfg)
    target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(snake_cfg, snake_aux, snake_columns)

    GSD_config = {feature_meta['name']: len(feature_meta['representation']) if feature_meta['name'] in catg_cols else 1 \
                  for feature_meta in snake_meta}

    domain = Domain.fromdict(GSD_config)
    encoded_train = encode_data_all_numeric(snake_cfg, train, minmax_encode_catg=False)
    data = Dataset(encoded_train, domain)


    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])

    stat_module = ChainedStatistics([marginal_module2])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()



    seed = 0
    delta = 1.0 / len(data) ** 2

    all_fp_counts = {}
    for i, eps in enumerate(reversed([.01, .1, 1, 30, 1000])):
        print(f"\n\n{eps}")
        dump_artifact({}, "GSD_FP_counts")

        for j in range(10):
            try:
                algo = GSD(domain=data.domain,
                           print_progress=True,
                           stop_early=True,
                           num_generations=10000,
                           population_size_muta=50,
                           population_size_cross=50,
                           sparse_statistics=True,
                           data_size=data_size
                           )

                seed = rand.randint(0, 1000)
                encoded_synth = algo.fit_dp(PRNGKey(seed), stat_module=stat_module, epsilon=eps, delta=delta).df

            except BaseException:
                print(f"xxxxx {j}")

        fp_counts = load_artifact("GSD_FP_counts")
        all_fp_counts[eps] = fp_counts



    # # todo: make this more efficient! OR move it to before saving the original FP_counts file
    # fp_weights = {tuple([round(float(x), 3) for x in all_possible_queries[query_id]]): weight \
    #               for query_id, weight in fp_id_weights.items()}
    #
    # dump_artifact(fp_weights, make_FP_filename(cfg, "GSD", eps))
    # sys.exit()


    dump_artifact(all_fp_counts, f"GSD_FP_counts_all")



#### gen synthetic data
#### test data quality


def test_data_quality():
    data_size = 316

    snake_cfg = Config("snake", train_size=data_size)
    (numeric_cols, catg_cols), snake_aux, snake_columns, snake_meta, _ = get_data(snake_cfg)
    target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(snake_cfg, snake_aux,
                                                                                       snake_columns)

    GSD_config = {feature_meta['name']: len(feature_meta['representation']) if feature_meta['name'] in catg_cols else 1 \
                  for feature_meta in snake_meta}

    domain = Domain.fromdict(GSD_config)
    encoded_train = encode_data_all_numeric(snake_cfg, train, minmax_encode_catg=False)
    data = Dataset(encoded_train, domain)

    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])

    stat_module = ChainedStatistics([marginal_module2])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()


    seed = 0
    delta = 1.0 / len(data) ** 2
    # eps = 1
    wd = []
    for i, eps in enumerate([.1, .3, 1, 3, 10, 30, 100, 300, 1000]):
    # for i, eps in enumerate(reversed([.1, .3, 1, 3, 10, 30, 100, 300, 1000])):

        # Choose Private-GSD parameters
        algo = GSD(domain=data.domain,
                   print_progress=True,
                   stop_early=True,
                   num_generations=10000,
                   population_size_muta=50,
                   population_size_cross=50,
                   sparse_statistics=True,
                   data_size=data_size
                   )

        encoded_synth = algo.fit_dp(PRNGKey(i), stat_module=stat_module, epsilon=eps, delta=delta).df


        synth = decode_data_from_numeric(snake_cfg, encoded_synth, minmax_encode_catg=False)
        # print(synth[np.take(synth.columns, range(0, 5))].head())
        # print(synth[np.take(synth.columns, range(5, 10))].head())
        # print(synth[np.take(synth.columns, range(10, 15))].head())


        ### test quality of synthetic data
        distance = wasserstein_distance(snake_cfg, train, synth, snake_columns)
        print(f"\n\n{eps}, {distance}")
        wd.append(distance)

    print(wd)


    # import jax.numpy as jnp
    #
    # errors = jnp.abs(true_stats - stat_fn(sync_data.to_numpy()))
    # print(f'GSD: eps={eps:.2f}, seed={seed}'
    #       f'\t max error = {errors.max():.5f}'
    #       f'\t avg error = {errors.mean():.5f}')


def test_attack_gsd_data_part1():
    data_size = 1_000

    snake_cfg = Config("snake", train_size=data_size, set_MI=False)
    (numeric_cols, catg_cols), snake_aux, snake_columns, snake_meta, _ = get_data(snake_cfg)
    target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(snake_cfg, snake_aux, snake_columns)

    GSD_config = {feature_meta['name']: len(feature_meta['representation']) if feature_meta['name'] in catg_cols else 1 \
                  for feature_meta in snake_meta}

    domain = Domain.fromdict(GSD_config)
    encoded_train = encode_data_all_numeric(snake_cfg, train, minmax_encode_catg=False)
    data = Dataset(encoded_train, domain)

    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=2, bins=[2, 4, 8, 16, 32])

    stat_module = ChainedStatistics([marginal_module2])
    stat_module.fit(data)

    true_stats = stat_module.get_all_true_statistics()
    stat_fn = stat_module._get_workload_fn()


    seed = rand.randint(0, 1000)
    delta = 1.0 / len(data) ** 2
    eps = 1000
    # for i, eps in enumerate([.1, .3, 1, 3, 10, 30, 100, 300, 1000]):
    # for i, eps in enumerate(reversed([.1, .3, 1, 3, 10, 30, 100, 300, 1000])):

    # Choose Private-GSD parameters
    algo = GSD(domain=data.domain,
               print_progress=True,
               stop_early=True,
               num_generations=20_000,
               population_size_muta=50,
               population_size_cross=50,
               sparse_statistics=True,
               data_size=data_size
               )

    encoded_synth, query_ids = algo.fit_dp(PRNGKey(seed), stat_module=stat_module, epsilon=eps, delta=delta)
    encoded_synth = encoded_synth.df
    synth = decode_data_from_numeric(snake_cfg, encoded_synth, minmax_encode_catg=False)

    dump_artifact(encoded_synth, f"test_gsd_encoded_synth_{data_size}_e{eps}_{snake_cfg.set_MI}")
    dump_artifact(synth, f"test_gsd_synth_{data_size}_e{eps}_{snake_cfg.set_MI}")
    dump_artifact(target_ids, f"test_gsd_attack_{data_size}_target_ids_e{eps}_{snake_cfg.set_MI}")
    dump_artifact(membership, f"test_gsd_attack_{data_size}_membership_e{eps}_{snake_cfg.set_MI}")

    all_possible_queries = marginal_module2.queries
    FPs = [tuple([round(float(x), 3) for x in all_possible_queries[query_id]]) for query_id in query_ids]
    dump_artifact(FPs, f"test_gsd_attack_{data_size}_FPs_e{eps}_{snake_cfg.set_MI}")


def test_attack_gsd_data_part2():
    data_size = 1_000
    eps = 1000

    cfg = Config("snake", train_size=data_size, set_MI=False)
    (numeric_cols, catg_cols), aux, columns, meta, _ = get_data(cfg)
    encoded_aux = encode_data_all_numeric(cfg, aux, minmax_encode_catg=False)
    encoded_synth = load_artifact(f"test_gsd_encoded_synth_{data_size}_e{eps}_{cfg.set_MI}")
    synth = load_artifact(f"test_gsd_synth_{data_size}_e{eps}_{cfg.set_MI}")
    target_ids = load_artifact(f"test_gsd_attack_{data_size}_target_ids_e{eps}_{cfg.set_MI}")
    membership = load_artifact(f"test_gsd_attack_{data_size}_membership_e{eps}_{cfg.set_MI}")
    targets = aux[aux['HHID'].isin(target_ids)] \
        if cfg.set_MI else aux[aux.index.isin(target_ids)]
    encoded_targets = encode_data_all_numeric(cfg, targets, minmax_encode_catg=False)


    GSD_config = {feature_meta['name']: len(feature_meta['representation']) if feature_meta['name'] in catg_cols else 1 \
                  for feature_meta in meta}
    domain = Domain.fromdict(GSD_config)
    all_possible_queries = Marginals.get_all_kway_combinations(domain, k=2, bins=[2, 4, 8, 16, 32]).queries

    tailored_ma = run_gsd_tailored(cfg, all_possible_queries, encoded_aux, encoded_synth, eps, encoded_targets, target_ids, membership)


def run_gsd_tailored(cfg, all_possible_queries, encoded_aux, encoded_synth, eps, encoded_targets, target_ids, membership):
    fp_weights = load_artifact(make_FP_filename(cfg, "GSD", eps))

    # start_time = time.process_time()
    scores, tailored_ma = run_gsd_tailored_get_MA(cfg, encoded_aux, encoded_synth, encoded_targets, target_ids, membership, fp_weights)
    # end_time = time.process_time()
    plot_output(scores)
    print(f"MA: {tailored_ma}")

    # only choose conditionals up to a certain # of parents
    # max_conditional_size = 5
    # max_conditional_size = max([len(k) for k in conditionals_weights.keys()])
    # arbitrary_conditionals = generate_arbitrary_FPs(columns, len(columns)-1, 1, max_conditional_size)
    # _, arbitrary_ma = run_priv_tailored_get_MA(cfg, aux, synth, targets, target_ids, membership, arbitrary_conditionals)

    return tailored_ma


def run_gsd_tailored_get_MA(cfg, aux, synth, targets, target_ids, membership, fp_weights):
    A = np.array([0.0] * targets.shape[0])
    num_queries_used = np.array([0] * targets.shape[0])

    threshold = max(fp_weights.values()) * .8
    for i, (fp, weight) in enumerate(fp_weights.items()):
        if i % 500 == 0: print(f"I: {i}", end=" ")
        # if i > 200: break
        if weight < threshold: continue

        fp_feature_indeces = [int(idx) for idx in list(fp)[ : cfg.gsd_k]]
        fp_feature_uppers = list(fp)[cfg.gsd_k : cfg.gsd_k*2]
        fp_feature_lowers = list(fp)[cfg.gsd_k*2 : cfg.gsd_k*3]

        k_idx = range(cfg.gsd_k)
        def in_bin(row):
            return [(fp_feature_lowers[j] <= row[j]) * (row[j] < fp_feature_uppers[j]) for j in k_idx]

        # todo: make more efficient
        D_synth = pd.DataFrame(np.swapaxes(in_bin(synth.iloc[:, fp_feature_indeces].T.values),0,1)).value_counts(normalize=True)
        D_aux = pd.DataFrame(np.swapaxes(in_bin(aux.iloc[:, fp_feature_indeces].T.values),0,1)).value_counts(normalize=True)

        default_val = 1e-10
        target_vals = np.swapaxes(in_bin(targets.iloc[:, fp_feature_indeces].T.values), 0, 1)
        # A += np.array([weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val)) for val in target_vals])
        # A += np.array([D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val)) for val in target_vals])

        for w, val in enumerate(target_vals):
            if val.all():
                A[w] += D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val))
                # A[w] += weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val))
                num_queries_used[w] += 1

    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if cfg.set_MI else targets.index.values,
        'A': pd.Series(A / np.maximum(np.array([1] * targets.shape[0]), num_queries_used))
    })

    scores = []
    grouped_predictions = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped_predictions.get_group(hhid).A.mean())
    activated_scores = activate_3(np.array(scores))

    return activated_scores, membership_advantage(membership, activated_scores)


# determine_fps_GDS()
# test_data_quality()
# test_attack_gsd_data_part1()
# test_attack_gsd_data_part2()
