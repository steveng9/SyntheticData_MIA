import sys
import time

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')
# import disjoint_set
import mst
import privbayes


from collections import Counter

from util import *


def main():

    # Shadow Modelling
    # determine_all_FP()

    # measuring densities, and scoring attack
    MA_all()


def MA_all():
    t, tar = list(Config(None).train_sizes.items())[3]
    sdgs = {"MST": run_mst, "PrivBayes": run_privbayes, "RAP": run_rap}
    print()
    print()
    print("TRAIN SIZE: ", t, tar)
    for r in tqdm(range(C.n_runs)):
        if t < 20_000:
            MA_experiment(Config("cali", train_size=t, set_MI=True, rap_top_q=50, rap_iterations=2000, rap_use_FP_threshold=5), sdgs)
            MA_experiment(Config("snake", train_size=t), sdgs)
        MA_experiment(Config("snake", train_size=t, set_MI=False), sdgs)
        MA_experiment(Config("snake", train_size=t), sdgs)

def determine_all_FP():
    cali_cfg = Config("cali")
    snake_cfg = Config("snake")
    _, cali_aux, cali_columns, cali_meta, _ = get_data(cali_cfg)
    _, snake_aux, snake_columns, snake_meta, _ = get_data(snake_cfg)

    for i in range(C.n_shadow_runs):
        print("\nRUN: ", i)
        for eps in C.shadow_epsilons:
            print(eps)
            determine_mst_marginals(cali_cfg, cali_aux, cali_columns, cali_meta, eps)
            determine_mst_marginals(snake_cfg, snake_aux, snake_columns, snake_meta, eps)
            determine_privbayes_conditionals(cali_cfg, cali_aux, cali_columns, cali_meta, eps)
            determine_privbayes_conditionals(snake_cfg, snake_aux, snake_columns, snake_meta, eps)
            determine_rap_queries(cali_cfg, cali_aux, cali_columns, cali_meta, eps)
            determine_rap_queries(snake_cfg, snake_aux, snake_columns, snake_meta, eps)

def MA_experiment(cfg, sdgs):

    _, aux, columns, meta, _ = get_data(cfg)
    results_filename = cfg.get_filename("MA", use_RAP_config=True)

    results = load_artifact(results_filename) or {eps: {
        sdg: {
            "KDE_MA": [],
            "KDE_time": [],
            "custom_MA": [],
            "custom_time": [],
            "arbitrary_MA": [],
            "distance": []
        } for sdg in sdgs.keys()
    } for eps in cfg.epsilons}

    print(f"\n{cfg.data_name}, {cfg.set_MI}, EPS: ", end="")
    for eps in cfg.epsilons:
        print(eps, end=", ")

        # use same sample for each sdg
        target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(cfg, aux, columns)

        for sdg, run_fn in sdgs.items():

            kde_ma, kde_time, tailored_ma, tailored_time, arbitrary_ma, distance = run_fn(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed)

            if kde_ma is not None:
                results[eps][sdg]["KDE_MA"].append(kde_ma)
                results[eps][sdg]["KDE_time"].append(kde_time)
            if tailored_ma is not None:
                results[eps][sdg]["custom_MA"].append(tailored_ma)
                results[eps][sdg]["custom_time"].append(tailored_time)
                results[eps][sdg]["arbitrary_MA"].append(arbitrary_ma)
                results[eps][sdg]["distance"].append(distance)

            # save off intermediate results
            dump_artifact(results, results_filename)


## Shadow Modelling

def determine_mst_marginals(cfg, aux, columns, meta, eps):
    all_cliques = []
    gen = mst.MST(
        dataset=aux[columns].sample(n=C.shadow_train_size),
        metadata=meta,
        size=C.shadow_synth_size,
        epsilon=eps,
    )
    gen.run()
    all_cliques.append(gen.cliques)
    save_off_intermediate_FPs(cfg, eps, all_cliques, "MST")

def determine_privbayes_conditionals(cfg, aux, columns, meta, eps):
    all_conditionals = []
    gen = privbayes.PRIVBAYES(
        dataset=aux[columns].sample(n=C.shadow_train_size),
        metadata=meta,
        size=C.shadow_synth_size,
        epsilon=eps,
    )
    gen.run()
    all_conditionals.append(gen.conditionals)
    save_off_intermediate_FPs(cfg, eps, all_conditionals, "PrivBayes")

def determine_rap_queries(cfg, aux, columns, meta, eps):

    ### This code is copy-pasted from the Relaxed-Adaptive-Projection Library, with unused
    ### segments removed, and constants added that are used by authors (Aydore et al., 2021)

    n_prime = cfg.synth_size
    seed = 0
    categorical_consistency = True
    aux_encoded, columns_domain = binarize_discrete_features_evenly(cfg, aux, columns)

    key = random.PRNGKey(seed)
    # sample rows from auxiliary data
    D = aux_encoded[np.random.choice(aux.shape[0], C.shadow_train_size)]
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

    save_off_intermediate_FPs(cfg, eps, [queries], "RAP")


## DOMIAS Proper

def kde_get_ma(cfg, aux, synth, targets, target_ids, membership, sample_seed):
    encoded_synth, encoded_targets, encoded_aux_sample = encode_data_for_KDE(cfg, aux, synth, targets, target_ids, sample_seed)

    # Find density at each target in synth and in base
    try:
        start_time = time.process_time()
        density_synth = stats.gaussian_kde(encoded_synth.values.transpose(1, 0))
        density_aux = stats.gaussian_kde(encoded_aux_sample.values.transpose(1, 0))

        p_synth_evaluated = density_synth.evaluate(encoded_targets.to_numpy().transpose(1, 0))
        p_aux_evaluated = density_aux.evaluate(encoded_targets.to_numpy().transpose(1, 0))

        # Score
        A = p_synth_evaluated / (p_aux_evaluated + 1e-20)
        predictions = pd.DataFrame({
            'hhid': targets['HHID'].values if cfg.set_MI else targets.index.values,
            'A': pd.Series(A)
        })

        scores = []
        households = predictions.groupby("hhid")
        for hhid in target_ids.tolist():
            scores.append(households.get_group(hhid).A.mean())
        activated_scores = activate_3(np.array(scores))

        end_time = time.process_time()

        return membership_advantage(membership, activated_scores), end_time - start_time
    except np.linalg.LinAlgError:
        print("Error in calculating KDE.")
        return None, None


## MAMA-MIA

### MST attack

def run_mst(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed):
    mst_gen = mst.MST(dataset=train[columns], metadata=meta, size=cfg.synth_size, epsilon=eps)
    try:
        mst_gen.run()
        synth = mst_gen.output

        if cfg.data_name == "snake":
            synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
        elif cfg.data_name == "cali":
            synth = synth.astype(int)

        kde_ma, kde_time = kde_get_ma(cfg, aux, synth, targets, target_ids, membership, kde_sample_seed)
        tailored_ma, arbitrary_ma, tailored_time = run_mst_tailored(cfg, columns, aux, synth, eps, targets, target_ids, membership)
        wd = wasserstein_distance(cfg, train, synth, columns)

        return kde_ma, kde_time, tailored_ma, tailored_time, arbitrary_ma, wd
    except:
        print("Error in running MST.")
        return None, None, None, None, None, None

def run_mst_tailored(cfg, columns, aux, synth, eps, targets, target_ids, membership):
    marginals_weights = load_artifact(make_FP_filename(cfg, "MST", eps))
    start_time = time.process_time()
    scores, tailored_ma = run_mst_tailored_get_MA(cfg, aux, synth, targets, target_ids, membership, marginals_weights)
    end_time = time.process_time()
    plot_output(scores)

    arbitrary_marginals = generate_arbitrary_FPs(columns, len(columns)-1, 2, 2)
    _, arbitrary_ma = run_mst_tailored_get_MA(cfg, aux, synth, targets, target_ids, membership, arbitrary_marginals)

    return tailored_ma, arbitrary_ma, end_time - start_time

def run_mst_tailored_get_MA(cfg, aux, synth, targets, target_ids, membership, marginals_weights):
    A = np.array([0.0]*targets.shape[0])
    for marginal, weight in marginals_weights.items():
        marginal_list = list(marginal)
        D_synth = synth[marginal_list].value_counts(normalize=True)
        D_aux = aux[marginal_list].value_counts(normalize=True)

        default_val = 1e-10
        A += np.array([weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val)) for val in targets[marginal_list].values])

    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if cfg.set_MI else targets.index.values,
        'A': pd.Series(A)
    })

    scores = []
    grouped_predictions = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped_predictions.get_group(hhid).A.mean())
    activated_scores = activate_3(np.array(scores))

    return activated_scores, membership_advantage(membership, activated_scores)


### PrivBayes attack

def run_privbayes(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed):

    # generate synthetic test data
    privbayes_gen = privbayes.PRIVBAYES(dataset=train[columns], metadata=meta, size=cfg.synth_size, epsilon=eps)
    privbayes_gen.run()
    synth = privbayes_gen.output

    if cfg.data_name == "snake":
        synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
    elif cfg.data_name == "cali":
        synth = synth.astype(int)

    # conduct experiments
    kde_ma, kde_time = kde_get_ma(cfg, aux, synth, targets, target_ids, membership, kde_sample_seed)
    tailored_ma, arbitrary_ma, tailored_time = run_priv_tailored(cfg, columns, aux, synth, eps, targets, target_ids, membership)
    wd = wasserstein_distance(cfg, train, synth, columns)

    return kde_ma, kde_time, tailored_ma, tailored_time, arbitrary_ma, wd

def run_priv_tailored(cfg, columns, aux, synth, eps, targets, target_ids, membership):
    conditionals_weights = load_artifact(make_FP_filename(cfg, "PrivBayes", eps))
    start_time = time.process_time()
    scores, tailored_ma = run_priv_tailored_get_MA(cfg, aux, synth, targets, target_ids, membership, conditionals_weights)
    end_time = time.process_time()
    plot_output(scores)

    # only choose conditionals up to a certain # of parents
    max_conditional_size = 5
    # max_conditional_size = max([len(k) for k in conditionals_weights.keys()])
    arbitrary_conditionals = generate_arbitrary_FPs(columns, len(columns)-1, 1, max_conditional_size)
    _, arbitrary_ma = run_priv_tailored_get_MA(cfg, aux, synth, targets, target_ids, membership, arbitrary_conditionals)

    return tailored_ma, arbitrary_ma, end_time - start_time

def run_priv_tailored_get_MA(cfg, aux, synth, targets, target_ids, membership, conditionals_weights):
    A = np.array([0.0] * targets.shape[0])
    for conditional, weight in conditionals_weights.items():
        conditional_list = list(conditional)
        child, parents = conditional_list[0], conditional_list[1:]

        # get conditional values
        synth_groups = synth.groupby(parents if parents != [] else lambda _: True)
        aux_groups = aux.groupby(parents if parents != [] else lambda _: True)
        target_vals = targets[[child] + parents].values

        D_synth = synth_groups[child].value_counts(normalize=True)
        D_aux = aux_groups[child].value_counts(normalize=True)

        default_val = 1e-10
        ratio_conditionals = []
        for child_val, *parent_vals in target_vals:
            parent_vals = tuple(parent_vals)
            synth_conditional = D_synth.get(parent_vals or True, default=pd.Series(dtype=np.float64)).get(child_val, default=default_val)
            # no "get()" default needed for base since target is in base
            aux_conditional = D_aux.get(parent_vals or True).get(child_val)
            ratio_conditionals.append(weight * synth_conditional / aux_conditional)

        A += np.array(ratio_conditionals)

    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if cfg.set_MI else targets.index.values,
        'A': pd.Series(A)
    })

    scores = []
    grouped_predictions = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped_predictions.get_group(hhid).A.mean())
    activated_scores = activate_3(np.array(scores))

    return activated_scores, membership_advantage(membership, activated_scores)

### RAP Attack

def run_rap(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed):

    # generate synthetic test data
    # catg_synth, synth, queries_used, train, targets, target_ids, membership = load_artifact(f"{cfg.data_name}_{'{0:.2f}'.format(eps)}_temp_synth_stuff")

    train_encoded, columns_domain = binarize_discrete_features_evenly(cfg, train, columns)
    synth, catg_synth, queries_used = get_rap_synth(cfg, columns, train_encoded, eps, columns_domain)
    # save_off_intermediate_FPs(cfg, eps, [queries_used], "RAP")
    dump_artifact((catg_synth, synth, queries_used, train, targets, target_ids, membership), f"{cfg.data_name}_{'{0:.2f}'.format(eps)}_temp_synth_stuff")

    # conduct experiments
    synth_decoded = decode_rap_synth(cfg, columns, meta, catg_synth)
    kde_ma, kde_time = kde_get_ma(cfg, aux, synth_decoded, targets, target_ids, membership, kde_sample_seed)
    tailored_ma, arbitrary_ma, tailored_time = run_rap_tailored(cfg, columns, aux, synth, eps, targets, target_ids, membership)
    wd = wasserstein_distance(cfg, train, synth, columns, encode_d2=False)
    wd_2 = wasserstein_distance(cfg, train, synth_decoded, columns)

    assert wd_2 == wd, "Error in WD calculation!"

    # SANITY CHECKS
    # sanity_ma_train = tailored_rap_get_ma(data_name, columns, aux, pd.DataFrame(train_encoded), targets, target_ids, membership)
    # sanity_ma_verification_queries, _, _ = run_rap_tailored(cfg, columns, aux, synth, targets, target_ids, membership, verification_queries=queries_used)
    # print(f"\nReal: {tailored_ma}, Sanity: {sanity_ma_verification_queries}")

    return kde_ma, kde_time, tailored_ma, tailored_time, arbitrary_ma, wd

def run_rap_tailored(cfg, columns, aux, synth, eps, targets, target_ids, membership, verification_queries=None):
    queries_weights = load_artifact(make_FP_filename(cfg, "RAP", eps)) if verification_queries is None else \
        {tuple(q.tolist()): cfg.rap_use_FP_threshold for q in verification_queries}

    aux_encoded = pd.DataFrame(binarize_discrete_features_evenly(cfg, aux, columns)[0])
    targets_encoded = pd.DataFrame(binarize_discrete_features_evenly(cfg, targets, columns)[0])

    start_time = time.process_time()
    scores, tailored_ma = run_rap_tailored_get_MA(cfg, aux_encoded, synth, targets, targets_encoded, target_ids, membership, queries_weights)
    end_time = time.process_time()

    assert (aux_encoded.columns == targets_encoded.columns).all()
    assert len(aux_encoded.columns) == synth.shape[1]

    arbitrary_queries = generate_arbitrary_FPs(aux_encoded.columns, cfg.rap_top_q * cfg.rap_epochs, cfg.rap_k, cfg.rap_k)
    _, arbitrary_FP_ma = run_rap_tailored_get_MA(cfg, aux_encoded, synth, targets, targets_encoded, target_ids, membership, arbitrary_queries)

    plot_output(scores)

    return tailored_ma, arbitrary_FP_ma, end_time - start_time


def run_rap_tailored_get_MA(cfg, aux_encoded, synth, targets, targets_encoded, target_ids, membership, queries_weights):

    A = np.array([0.0]*targets_encoded.shape[0])
    for query, weight in queries_weights.items():
        if weight < cfg.rap_use_FP_threshold: continue
        query_columns = list(query)
        D_synth = synth[query_columns].value_counts(normalize=True)
        D_aux = aux_encoded[query_columns].value_counts(normalize=True)

        default_val = 1e-6
        A += np.array([weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val)) for val in targets_encoded[query_columns].values])

    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if cfg.set_MI else targets.index.values,
        'A': pd.Series(A)
    })

    scores = []
    grouped_predictions = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped_predictions.get_group(hhid).A.mean())
    activated_scores = activate_3(np.array(scores))

    return activated_scores, membership_advantage(membership, activated_scores)



main()
