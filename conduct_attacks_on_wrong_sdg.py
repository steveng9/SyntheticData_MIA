from conduct_attacks import *


def mst_synthesize(cfg, meta, columns, train, eps):
    mst_gen = mst.MST(dataset=train[columns], metadata=meta, size=cfg.synth_size, epsilon=eps)
    try:
        mst_gen.run()
        synth = mst_gen.output
        if cfg.data_name == "snake":
            synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
        elif cfg.data_name == "cali":
            synth = synth.astype(int)
        return synth
    except ValueError:
        print("Error in running MST.")
        return None


def privbayes_synthesize(cfg, meta, columns, train, eps):
    privbayes_gen = privbayes.PRIVBAYES(dataset=train[columns], metadata=meta, size=cfg.synth_size, epsilon=eps)
    privbayes_gen.run()
    synth = privbayes_gen.output
    if cfg.data_name == "snake":
        synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
    elif cfg.data_name == "cali":
        synth = synth.astype(int)
    return synth


def gsd_synthesize(cfg, meta, columns, train, eps):
    encoded_train = encode_data_all_numeric(cfg, train, minmax_encode_catg=False)
    GSD_config = {feature_meta['name']: len(feature_meta['representation']) if feature_meta['name'] in cfg.categorical_columns else 1 \
                  for feature_meta in meta}
    domain = Domain.fromdict(GSD_config)
    data = Dataset(encoded_train, domain)
    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=cfg.gsd_k, bins=cfg.gsd_bins)
    stat_module = ChainedStatistics([marginal_module2])
    stat_module.fit(data)
    # true_stats = stat_module.get_all_true_statistics()
    # stat_fn = stat_module._get_workload_fn()
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
    try:
        # GENERATE and SAVE synthetic test data
        encoded_synth, query_ids = algo.fit_dp(PRNGKey(seed), stat_module=stat_module, epsilon=eps, delta=delta)
        encoded_synth = encoded_synth.df
        synth = decode_data_from_numeric(cfg, encoded_synth, minmax_encode_catg=False)
        if cfg.data_name == "snake":
            synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
        elif cfg.data_name == "cali":
            synth = synth.astype(int)
        return synth
    except:
        print("Error in running GSD.")
        return None




def mst_attack_on_wrong_synth(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed, fps, synth):
    try:
        kde_ma, kde_auc, kde_time, kde_roc = kde_get_ma(cfg, aux, synth, targets, target_ids, membership, kde_sample_seed)
        tailored_ma, tailored_auc, arbitrary_ma, tailored_time, tailored_roc = run_all_mst_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, fps)
        wd = wasserstein_distance(cfg, train, synth, columns)

        return kde_ma, kde_auc, kde_time, None, None, tailored_ma, tailored_auc, tailored_time, arbitrary_ma, wd, kde_roc, tailored_roc
    except ValueError:
        print("Error in running MST.")
        return None, None, None, None, None, None, None, None, None, None, None


def privbayes_attack_on_wrong_synth(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed, fps, synth):
    # conduct experiments
    kde_ma, kde_auc, kde_time, kde_roc = kde_get_ma(cfg, aux, synth, targets, target_ids, membership, kde_sample_seed)
    tailored_ma, tailored_auc, arbitrary_ma, tailored_time, tailored_roc = run_all_privbayes_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, fps)
    wd = wasserstein_distance(cfg, train, synth, columns)

    return kde_ma, kde_auc, kde_time, None, None, tailored_ma, tailored_auc, tailored_time, arbitrary_ma, wd, kde_roc, tailored_roc


def gsd_attack_on_wrong_synth(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed, fps, synth):
    encoded_aux = encode_data_all_numeric(cfg, aux, minmax_encode_catg=False)
    encoded_train = encode_data_all_numeric(cfg, train, minmax_encode_catg=False)
    encoded_targets = encode_data_all_numeric(cfg, targets, minmax_encode_catg=False)
    encoded_synth = encode_data_all_numeric(cfg, synth, minmax_encode_catg=False)

    GSD_config = {feature_meta['name']: len(feature_meta['representation']) if feature_meta['name'] in cfg.categorical_columns else 1 \
                  for feature_meta in meta}
    domain = Domain.fromdict(GSD_config)
    data = Dataset(encoded_train, domain)
    marginal_module2 = Marginals.get_all_kway_combinations(data.domain, k=cfg.gsd_k, bins=cfg.gsd_bins)
    # stat_module = ChainedStatistics([marginal_module2])
    # stat_module.fit(data)
    # # true_stats = stat_module.get_all_true_statistics()
    # # stat_fn = stat_module._get_workload_fn()
    #
    # seed = rand.randint(0, 1000)
    # delta = 1.0 / len(data) ** 2
    #
    # algo = GSD(
    #     domain=data.domain,
    #     print_progress=False,
    #     stop_early=True,
    #     num_generations=cfg.gsd_generations,
    #     population_size_muta=50,
    #     population_size_cross=50,
    #     sparse_statistics=True,
    #     data_size=train.shape[0]
    # )
    #
    try:
        # # LOAD saved synthetic test data
        # # encoded_synth, synth, train, encoded_targets, targets, target_ids, membership, query_ids = load_artifact(f"{cfg.data_name}_{'{0:.2f}'.format(eps)}_GSD_temp_synth_stuff")
        #
        # # GENERATE and SAVE synthetic test data
        # encoded_synth, query_ids = algo.fit_dp(PRNGKey(seed), stat_module=stat_module, epsilon=eps, delta=delta)
        # encoded_synth = encoded_synth.df
        # synth = decode_data_from_numeric(cfg, encoded_synth, minmax_encode_catg=False)
        # dump_artifact((encoded_synth, synth, train, encoded_targets, targets, target_ids, membership, query_ids),
        #               f"{cfg.data_name}_{'{0:.2f}'.format(eps)}_GSD_temp_synth_stuff")
        #
        # if cfg.data_name == "snake":
        #     synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
        # elif cfg.data_name == "cali":
        #     synth = synth.astype(int)

        all_possible_queries = marginal_module2.queries
        kde_ma, kde_auc, kde_time, kde_roc = kde_get_ma(cfg, aux, synth, targets, target_ids, membership, kde_sample_seed)
        tailored_ma, tailored_auc, tailored_ma_w, tailored_auc_w, arbitrary_ma, tailored_time, tailored_roc = run_all_gsd_experiments(cfg, all_possible_queries, encoded_aux, encoded_synth, eps, encoded_targets, target_ids, membership, fps)
        wd = wasserstein_distance(cfg, train, synth, columns)

        return kde_ma, kde_auc, kde_time, tailored_ma, tailored_auc, tailored_ma_w, tailored_auc_w, tailored_time, arbitrary_ma, wd, kde_roc, tailored_roc
    except:
        print("Error in running GSD.")
        return None, None, None, None, None, None, None, None, None, None