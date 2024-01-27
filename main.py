import sys

import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')
# import disjoint_set
import mst
import privbayes


from collections import Counter

from util import *

# sys.path.append('/Users/golobs/PycharmProjects/SynthCity_Pategan/synthcity/src/')
# from synthcity.plugins.privacy.plugin_pategan import PATEGAN
# from synthcity.plugins.privacy import plugin_pategan_attack
# from synthcity.plugins.core.models.tabular_encoder import TabularEncoder
# from synthcity.plugins import Plugins


def determine_mst_marginals(name):
    _, aux, columns, meta, _ = get_data(name)
    synth_size = 10000

    for i, eps in enumerate(C.shadow_epsilons):
        all_cliques = []
        print(eps)
        for _ in tqdm(range(C.n_shadow_runs)):
            gen = mst.MST(
                dataset=aux[columns].sample(n=C.shadow_train_size),
                metadata=meta,
                size=synth_size,
                epsilon=eps,
            )
            gen.run()
            all_cliques.append(gen.cliques)

        marginal_frequencies = {}
        for cliques in all_cliques:
            for clique in cliques:
                attrs = list(clique)
                attrs.sort()
                attrs = ', '.join(attrs)
                marginal_frequencies[attrs] = marginal_frequencies.get(attrs, 0) + 1

        marginal_frequencies = dict(sorted(marginal_frequencies.items(), key=lambda x: -x[1]))
        dump_artifact(marginal_frequencies, f"FP_{name}_MST_eps{eps}_bins{C.n_baskets}")

def determine_privbayes_conditionals(name):
    _, aux, columns, meta, _ = get_data(name)
    synth_size = 10000

    for i, eps in enumerate(C.shadow_epsilons):
        print(eps)
        all_conditionals = []
        for _ in tqdm(range(C.n_shadow_runs)):
            gen = privbayes.PRIVBAYES(
                dataset=aux[columns].sample(n=C.shadow_train_size),
                metadata=meta,
                size=synth_size,
                epsilon=eps,
            )
            gen.run()
            all_conditionals.append(gen.conditionals)

        conditional_frequencies = {}
        for conditionals in all_conditionals:
            for c in conditionals:
                attrs = list(c)
                attrs.sort()
                attrs = ', '.join(attrs)
                conditional_frequencies[attrs] = conditional_frequencies.get(attrs, 0) + 1

        conditional_frequencies = dict(sorted(conditional_frequencies.items(), key=lambda x: -x[1]))
        dump_artifact(conditional_frequencies, f"FP_{name}_PrivBayes_eps{eps}_bins{C.n_baskets}")

def determine_rap_queries(name):
    _, aux, columns, meta, _ = get_data(name)
    n_prime = 1000
    seed = 0
    categorical_consistency = True
    aux_encoded, columns_domain = binarize_discrete_features_evenly(name, aux, columns)

    for i, eps in enumerate(C.shadow_epsilons):
        all_queries = []
        print(eps)
        for _ in tqdm(range(C.n_shadow_runs)):

            key = random.PRNGKey(seed)
            # sample rows from auxiliary data
            D = aux_encoded[np.random.choice(aux.shape[0], C.shadow_train_size)]
            n, d = D.shape
            delta = 1 / n ** 2

            stat_module = __import__("statistickway")

            # First select random k-way marginals from the dataset
            kway_attrs = [p for p in itertools.combinations(columns, C.rap_k)]
            kway_compact_queries, _ = get_queries(columns_domain, kway_attrs)
            all_statistic_fn = stat_module.preserve_statistic(kway_compact_queries)
            true_statistics = all_statistic_fn(D)

            # projection_interval = ProjectionInterval(*args.project) if args.project else None
            projection_interval = None

            epochs = min(C.rap_epochs, np.ceil(len(true_statistics) / C.rap_top_q).astype(np.int32)) \
                if not C.rap_all_queries else 1

            if C.rap_all_queries:
                # ensure consistency w/ top_q for one-shot case (is this correct?)
                C.rap_top_q = len(true_statistics)

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
                iterations=C.rap_iterations,
                epsilon=eps,
                delta=delta,
                norm=Norm('L2'),
                projection_interval=projection_interval,
                optimizer_learning_rate=.001,
                lambda_l1=0,
                k=C.rap_k,
                top_q=C.rap_top_q,
                use_all_queries=C.rap_all_queries,
                rap_stopping_condition=1e-7,
                initialize_binomial=False,
                feats_idx=feats_idx,
            )

            key, subkey = random.split(key)
            rap = RAP(algorithm_configuration, key=key)
            # growing number of sanitized statistics to preserve
            key, subkey = random.split(subkey)
            queries = rap.train(D, columns_domain, kway_attrs, key, train_last_epoch=False)
            all_queries.append(queries)

            save_off_intermediate_queries(name, eps, all_queries)


def attack_new_data_PATEGAN():

    # process
    verbose = True
    save_data = True

    # experiments
    training_noise = False
    n_runs = 1
    epsilon = 1000
    train_size = 1000
    synth_size = 1500
    aux_sample_size = synth_size
    exclude_targets_to_aux_train = True
    num_targets = 100
    num_members = 50
    HH_size = 5
    HH_max = 20
    discrete_columns = ['agechild', 'citistat', 'female', 'married', 'wbhaom', 'cow1', 'ftptstat', 'statefips', 'mind16', 'mocc10']

    # architecture
    duplicate_args = True
    pategan_synth_args = {
        "epsilon": epsilon,
        "n_iter": 1,
        "encoder_max_clusters": 10,
        "weight_concentration_prior": 1e4,
        "batch_size": 60,
        "n_iter_gan": 4_000,

        "generator_n_units_latent": 20,

        "generator_n_iter": 7,
        "generator_n_layers_hidden": 2,
        "generator_n_units_hidden": 300,
        "generator_lr": 1e-5,

        "discriminator_n_iter": 7,
        "discriminator_n_layers_hidden": 2,
        "discriminator_n_units_hidden": 300,
        "discriminator_lr": 1e-5,
    }
    pategan_attack_args = {
        "epsilon": epsilon,
        "n_iter": 1,
        "encoder_max_clusters": 6,
        "batch_size": 64,
        "n_iter_gan": 60,

        "generator_n_iter": 5,
        "generator_n_layers_hidden": 2,
        "generator_n_units_hidden": 300,
        "generator_lr": 2e-3,

        "discriminator_n_iter": 5,
        "discriminator_n_layers_hidden": 2,
        "discriminator_n_units_hidden": 300,
        "discriminator_lr": 2e-3,
    }
    if duplicate_args: pategan_attack_args = pategan_synth_args

    with open(DATA_DIR + "meta.json") as f:
        meta_list = json.load(f)


    aux = pd.read_parquet(DATA_DIR + "base.parquet")
    aux['HHID'] = aux.index
    aux.index = range(aux.shape[0])
    columns = aux[np.take(aux.columns, range(15))].columns
    # columns = ['age', 'statefips']
    # columns = discrete_columns

    aux = convert_finite_ordered_to_numeric(aux, columns)

    # determine all 'candidate' households of minimum size for "set membership inference"
    hh_counts = aux['HHID'].value_counts()
    # candidate_households = hh_counts[hh_counts >= HH_size & hh_counts <= HH_max].index
    candidate_households = hh_counts[hh_counts >= HH_size].index
    print("num candidates: ", len(candidate_households))


    if save_data:

        # create training set and candidate members
        # -----------------------------
        target_hhids = pd.Series(candidate_households).sample(n=num_targets)
        targets = aux[aux.HHID.isin(target_hhids)]
        aux_sample_no_targets = aux[~aux.HHID.isin(target_hhids)][columns].sample(n=train_size)
        member_hhids = target_hhids.sample(n=num_members)
        members = aux[aux.HHID.isin(member_hhids)]
        membership = np.array([1 if t in member_hhids.values else 0 for t in target_hhids.values.tolist()])
        membership_df = pd.DataFrame({"hhid": target_hhids, "actual": membership})
        train = pd.concat([aux_sample_no_targets, members[columns]], ignore_index=True).sample(frac=1)  # sample() to shuffle targets in

        print("train shape: ", train.shape[0])
        # generate synthetic data
        pategan = Plugins().get("pategan", **pategan_synth_args)
        _, t_losses, g_losses, d_losses, enc_columns = pategan.fit(train, discrete_columns=discrete_columns, noise=training_noise, categories=meta)

        dump_artifact({i: enc_columns[i] for i in range(len(enc_columns))}, "../encoded_column_names")
        train_enc, gen_enc = pategan.get_gen_sample_encoded(train, train.shape[0])
        train_enc = np.insert(train_enc, 0, range(len(enc_columns)), axis=0)
        train_enc = np.append(train_enc, [train_enc[1:, :].sum(axis=0)], axis=0)
        train_enc = np.append(train_enc, [range(len(enc_columns))], axis=0)
        gen_enc = np.insert(gen_enc, 0, range(len(enc_columns)), axis=0)
        np.savetxt(DATA_DIR + "gen_sample", gen_enc, fmt='%1.2f', delimiter='\t')
        np.savetxt(DATA_DIR + "train_sample", train_enc, fmt='%1.2f', delimiter='\t')
        # assert False

        if verbose: graph_losses(t_losses, g_losses, d_losses)
        synth = pategan.generate(synth_size).dataframe()
        synth_decoded = revert_finite_ordered_to_numeric(synth, columns)
        synth_decoded.to_csv(DATA_DIR + 'pategan_synth_new_10.csv', index=True)
        train_decoded = revert_finite_ordered_to_numeric(train, columns)
        train_decoded.to_csv(DATA_DIR + 'pategan_train_new_10.csv', index=True)
        # assert False

        # compare distributions of new synth set to aux
        # for col in columns:
        #     print(col, synth_2[col].nunique(), aux[col].nunique(), len(meta[meta['name'] == col].representation.values[0]))

        # TODO: Verify that distributions of numeric/ordered/nonordered values are similar to aux

    #     dump_artifact(synth, f"pategan_new_synth_{epsilon}")
    #     dump_artifact(target_hhids, f"pategan_new_target_hhids_{epsilon}")
    #     dump_artifact(targets, f"pategan_new_targets_{epsilon}")
    #     dump_artifact(membership_df, f"pategan_new_membership_df_{epsilon}")
    #     dump_artifact(membership, f"pategan_new_membership_{epsilon}")
    #
    # if not save_data:  # so instead, load data
    #     synth = load_artifact(f"pategan_new_synth_{epsilon}")
    #     target_hhids = load_artifact(f"pategan_new_target_hhids_{epsilon}")
    #     targets = load_artifact(f"pategan_new_targets_{epsilon}")
    #     membership_df = load_artifact(f"pategan_new_membership_df_{epsilon}")
    #     membership = load_artifact(f"pategan_new_membership_{epsilon}")



        # Conduct Attack

        synth_scores_1 = np.zeros(targets.shape[0])
        synth_scores_2 = np.zeros(targets.shape[0])
        aux_scores_1 = np.zeros(targets.shape[0])
        aux_scores_2 = np.zeros(targets.shape[0])

        for j in range(n_runs):
            print(j, end=" ")

            #synth
            # TODO take out epsilon during attack fitting?
            # eps = 10000
            pategan = Plugins().get("pategan", **pategan_attack_args)
            _, t_losses, g_losses, d_losses, _ = pategan.fit(synth, discrete_columns=discrete_columns, noise=False, categories=meta)
            if verbose: graph_losses(t_losses, g_losses, d_losses)
            synth_scores_1 += pategan.score_target_1(synth, targets[columns])
            synth_scores_2 += pategan.score_target_2(synth, targets[columns])

            #aux
            aux_sample_no_targets = aux[~aux.HHID.isin(target_hhids)][columns].sample(n=aux_sample_size)
            aux_sample = aux_sample_no_targets if exclude_targets_to_aux_train else \
                pd.concat([aux_sample_no_targets, targets[columns]], ignore_index=True).sample(frac=1)
            # aux_sample = aux[columns].sample(n=aux_sample_size)

            pategan = Plugins().get("pategan", **pategan_attack_args)
            _, t_losses, g_losses, d_losses, _ = pategan.fit(aux_sample, discrete_columns=discrete_columns, noise=False, categories=meta)
            if verbose: graph_losses(t_losses, g_losses, d_losses)
            aux_scores_1 += pategan.score_target_1(aux_sample, targets[columns])
            aux_scores_2 += pategan.score_target_2(aux_sample, targets[columns])

        synth_scores_1 /= n_runs
        synth_scores_2 /= n_runs
        aux_scores_1 /= n_runs
        aux_scores_2 /= n_runs

        print(synth_scores_1)
        print(aux_scores_1)
        print(synth_scores_2)
        print(aux_scores_2)



        # Score Attack

        predictions = pd.DataFrame()
        predictions['hhid'] = targets['HHID']
        predictions['probability_1'] = synth_scores_1 / aux_scores_1
        predictions['probability_2'] = synth_scores_2 / aux_scores_2
        predictions['actual'] = predictions.apply(lambda x: membership_df[membership_df.hhid == x['hhid']].actual.values.tolist()[0], axis=1)
        grouped_predictions = predictions.groupby('hhid')

        final_scores_1 = []
        final_scores_2 = []
        for target_hhid in membership_df['hhid'].values.tolist():
            final_scores_1.append(grouped_predictions.get_group(target_hhid).probability_1.mean())
            final_scores_2.append(grouped_predictions.get_group(target_hhid).probability_2.mean())

        final_1 = np.array(final_scores_1)
        final_1 = activate_3(final_1, confidence=1)
        final_2 = np.array(final_scores_2)
        final_2 = activate_3(final_2, confidence=1)

        if verbose:
            bins = np.linspace(0, 1, 50)
            pyplot.hist(final_1, bins)
            pyplot.legend(loc='upper right')
            pyplot.show()

            bins = np.linspace(0, 1, 50)
            pyplot.hist(final_2, bins)
            pyplot.legend(loc='upper right')
            pyplot.show()

        #     final = final_1
        #     fpr, tpr, _ = metrics.roc_curve(membership, final)
        #     auc = metrics.auc(fpr, tpr)
        #     print("AUC 1:  {:.2f}".format(auc))
        #
        #     final = final_2
        #     fpr, tpr, _ = metrics.roc_curve(membership, final)
        #     auc = metrics.auc(fpr, tpr)
        #     print("AUC 2:  {:.2f}".format(auc))

        final = final_1
        weights = 2 * np.abs(0.5 - final)
        ma = membership_advantage(membership, final > 0.5, sample_weight=weights)
        print("MA 1:  {:.4f}".format(ma))

        final = final_2
        weights = 2 * np.abs(0.5 - final)
        ma = membership_advantage(membership, final > 0.5, sample_weight=weights)
        print("MA 2:  {:.4f}".format(ma))

def tailored_mst_get_ma(data_name, aux, synth, eps, targets, target_ids, membership):
    marginals_weights = load_artifact(f"FP_{data_name}_MST_eps{eps}_bins{C.n_baskets}")

    A = np.array([0.0]*targets.shape[0])
    for marginal, weight in marginals_weights:
        # if weight < 47: continue
        marginal_list = marginal.split(', ')
        D_synth = synth[marginal_list].value_counts(normalize=True)
        D_aux = aux[marginal_list].value_counts(normalize=True)

        default_val = 1e-10
        A += np.array([weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val)) for val in targets[marginal_list].values])
        # A += np.array([ P_synth.get(tuple(val), default=default_val) / P_base.get(tuple(val)) for val in targets[marginal_list].values])


    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if C.set_MI else targets.index.values,
        'A': pd.Series(A)
    })

    scores = []
    grouped_predictions = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped_predictions.get_group(hhid).A.mean())
    activated_scores = activate_3(np.array(scores))

    if C.verbose:
        bins = np.linspace(0, 1, 50)
        pyplot.hist(activated_scores, bins, color='r', alpha=.5)
        pyplot.legend(loc='upper right')
        pyplot.show()

    return membership_advantage(membership, activated_scores)


def tailored_privbayes_get_ma(data_name, aux, synth, eps, targets, target_ids, membership):
    conditionals_weights = load_artifact(f"FP_{data_name}_PrivBayes_eps{eps}_bins{C.n_baskets}")

    A = np.array([0.0] * targets.shape[0])
    for conditional, weight in conditionals_weights:
        # if weight < 20: continue
        conditional_list = conditional.split(', ')
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
            # ratio_conditionals.append(synth_conditional / aux_conditional)

        A += np.array(ratio_conditionals)

    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if C.set_MI else targets.index.values,
        'A': pd.Series(A)
    })

    scores = []
    grouped_predictions = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped_predictions.get_group(hhid).A.mean())
    activated_scores = activate_3(np.array(scores))

    if C.verbose:
        bins = np.linspace(0, 1, 50)
        pyplot.hist(activated_scores, bins)
        pyplot.legend(loc='upper right')
        pyplot.show()

    return membership_advantage(membership, activated_scores)


def tailored_rap_get_ma(data_name, columns, aux, synth, targets, target_ids, membership, cheat_queries=None):
    queries_weights = load_artifact(f"FP_{data_name}_RAP_eps100000_bins{C.n_baskets}_k{C.rap_k}") if cheat_queries is None else \
        {tuple(q.tolist()): C.rap_use_FP_threshold for q in cheat_queries}
    aux_encoded = pd.DataFrame(binarize_discrete_features_evenly(data_name, aux, columns)[0])
    targets_encoded = pd.DataFrame(binarize_discrete_features_evenly(data_name, targets, columns)[0])

    A = np.array([0.0]*targets.shape[0])
    # A = np.array([1.0]*targets.shape[0])
    # D_synth_total = np.array([0.0]*targets.shape[0])
    # D_aux_total = np.array([0.0]*targets.shape[0])
    for query, weight in queries_weights.items():
        # if weight < C.rap_use_FP_threshold: continue
        query_columns = list(query)
        D_synth = synth[query_columns].value_counts(normalize=True)
        D_aux = aux_encoded[query_columns].value_counts(normalize=True)

        default_val = 1e-6
        A += np.array([weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val)) for val in targets_encoded[query_columns].values])
        # A += np.array([D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val)) for val in targets_encoded[query_columns].values])
        # A *= moderate_sequentially(np.array([(D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val))) for val in targets_encoded[query_columns].values]))

        # s = np.array([np.log(D_synth.get(tuple(val), default=default_val)) for val in targets_encoded[query_columns].values])
        # D_synth_total += s
        # a = np.array([np.log(D_aux.get(tuple(val))) for val in targets_encoded[query_columns].values])
        # D_aux_total += a

    # s_min = D_synth_total.min()
    # a_min = D_aux_total.min()
    # minimum = abs(min(a_min, s_min))
    #
    # D_synth_total_exp = np.exp(D_synth_total / minimum)
    # D_aux_total_exp = np.exp(D_aux_total / minimum)
    # A = D_synth_total_exp / D_aux_total_exp

    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if C.set_MI else targets.index.values,
        'A': pd.Series(A)
    })

    scores = []
    grouped_predictions = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped_predictions.get_group(hhid).A.mean())
    activated_scores = activate_3(np.array(scores))

    if C.verbose:
        bins = np.linspace(0, 1, 50)
        pyplot.hist(activated_scores, bins, color='r', alpha=.5)
        pyplot.legend(loc='upper right')
        pyplot.show()

    return membership_advantage(membership, activated_scores)


def kde_get_ma(data_name, aux, synth, targets, target_ids, membership, sample_seed):
    encoded_synth, encoded_targets, encoded_aux_sample = encode_data_for_KDE(data_name, aux, synth, targets, target_ids, sample_seed)

    # Find density at each target in synth and in base
    density_synth = stats.gaussian_kde(encoded_synth.values.transpose(1, 0))
    density_aux = stats.gaussian_kde(encoded_aux_sample.values.transpose(1, 0))

    p_synth_evaluated = density_synth.evaluate(encoded_targets.to_numpy().transpose(1, 0))
    p_aux_evaluated = density_aux.evaluate(encoded_targets.to_numpy().transpose(1, 0))

    # Score
    A = p_synth_evaluated / (p_aux_evaluated + 1e-20)
    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if C.set_MI else targets.index.values,
        'A': pd.Series(A)
    })

    scores = []
    households = predictions.groupby("hhid")
    for hhid in target_ids.tolist():
        scores.append(households.get_group(hhid).A.mean())
    activated_scores = activate_3(np.array(scores))

    return membership_advantage(membership, activated_scores)


def run_mst(data_name, aux, meta, columns, train, eps, targets, target_ids, membership, kde_sample_seed):
    mst_gen = mst.MST(dataset=train[columns], metadata=meta, size=C.synth_size, epsilon=eps)
    mst_gen.run()
    synth = mst_gen.output

    if data_name == "snake":
        synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
    elif data_name == "cali":
        synth = synth.astype(int)

    kde_ma = kde_get_ma(data_name, aux, synth, targets, target_ids, membership, kde_sample_seed)
    tailored_ma = tailored_mst_get_ma(data_name, aux, synth, eps, targets, target_ids, membership)
    wd = wasserstein_distance(data_name, train, synth, columns)

    return kde_ma, tailored_ma, wd


def run_privbayes(data_name, aux, meta, columns, train, eps, targets, target_ids, membership, kde_sample_seed):
    privbayes_gen = privbayes.PRIVBAYES(dataset=train[columns], metadata=meta, size=C.synth_size, epsilon=eps)
    privbayes_gen.run()
    synth = privbayes_gen.output

    if data_name == "snake":
        synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
    elif data_name == "cali":
        synth = synth.astype(int)

    kde_ma = kde_get_ma(data_name, aux, synth, targets, target_ids, membership, kde_sample_seed)
    tailored_ma = tailored_privbayes_get_ma(data_name, aux, synth, eps, targets, target_ids, membership)

    wd = wasserstein_distance(data_name, train, synth, columns)

    return kde_ma, tailored_ma, wd


def run_rap(meta, data_name, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed):
    train_encoded, columns_domain = binarize_discrete_features_evenly(data_name, train, columns)
    synth, catg_synth, queries_used = get_rap_synth(data_name, columns, train_encoded, eps, columns_domain)
    dump_artifact(catg_synth, "temp_synth_catg")
    dump_artifact(synth, "temp_synth")
    dump_artifact(queries_used, "temp_synth_cheatqueries")
    # catg_synth = load_artifact("temp_synth_catg")

    synth_decoded = decode_rap_synth(data_name, columns, meta, catg_synth)
    kde_ma = kde_get_ma(data_name, aux, synth_decoded, targets, target_ids, membership, kde_sample_seed)
    # tailored_ma = tailored_rap_get_ma(data_name, columns, aux, synth, targets, target_ids, membership)
    wd = wasserstein_distance(data_name, train, synth, columns, encode_d2=False)
    # wd = wasserstein_distance(data_name, train, synth_decoded, columns)

    # SANITY CHECKS
    # print("CHECKING SANITY!!!!!")
    # sanity_ma_kde = kde_get_ma(data_name, aux, train, targets, target_ids, membership, kde_sample_seed)
    # sanity_ma_train = tailored_rap_get_ma(data_name, columns, aux, pd.DataFrame(train_encoded), targets, target_ids, membership)
    # print(f"\nkde: {sanity_ma_kde}, tailored: {sanity_ma_train}")
    sanity_ma_cheat_queries = tailored_rap_get_ma(data_name, columns, aux, synth, targets, target_ids, membership, cheat_queries=queries_used)
    # print(f"\nReal: {tailored_ma}, Sanity: {sanity_ma_cheat_queries}")

    # return kde_ma, tailored_ma, wd
    return kde_ma, sanity_ma_cheat_queries, wd
    # return kde_ma, sanity_ma_train, None



def MA_experiment(dataset_name):

    _, aux, columns, meta, _ = get_data(dataset_name)

    # determine_mst_marginals(dataset_name)
    # determine_privbayes_conditionals(dataset_name)
    # determine_rap_queries(dataset_name)

    MAs = {}
    for i, eps in enumerate(C.epsilons):

        MAs[eps] = {}
        KDE_MST_MAs = 0.0
        KDE_PRIV_MAs = 0.0
        KDE_RAP_MAs = 0.0
        TAILORED_MST_MAs = 0.0
        TAILORED_PRIV_MAs = 0.0
        TAILORED_RAP_MAs = 0.0

        for _ in tqdm(range(C.n_runs_MA)):
            target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(aux, columns)

            # MST -------------
            # kde_mst_ma, tailored_mst_ma, mst_wd = run_mst(dataset_name, aux, meta, columns, train, eps, targets, target_ids, membership, kde_sample_seed)
            # TAILORED_MST_MAs += tailored_mst_ma
            #
            # # PrivBayes -------------
            # kde_privbayes_ma, tailored_privbayes_ma, privbayes_wd = run_privbayes(dataset_name, aux, meta, columns, train, eps, targets, target_ids, membership, kde_sample_seed)
            # TAILORED_PRIV_MAs += tailored_privbayes_ma

            # RAP -------------
            kde_rap_ma, tailored_rap_ma, rap_wd = run_rap(meta, dataset_name, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed)
            TAILORED_RAP_MAs += tailored_rap_ma

            print()
            # print("KDE:", kde_mst_ma, kde_privbayes_ma, kde_rap_ma)
            # print("CUSTOM:", tailored_mst_ma, tailored_privbayes_ma, tailored_rap_ma)
            print("MA:", kde_rap_ma, tailored_rap_ma)
            # print("WD:", mst_wd, privbayes_wd, rap_wd)
            # print("MA:", tailored_rap_ma)
            # print("WD:", rap_wd)


            #
            # print("{}:  {:.3f}, {:.3f} | {:.3f}, {:.3f}  |||  {:.3f}, {:.3f} | {:.3f}, {:.3f}".format(j, \
            #                kde_mst_ma, tailored_mst_ma, \
            #                # kde_mst_ma, 0, \
            #                KDE_MST_MAs / (j+1), TAILORED_MST_MAs / (j+1), \
            #                kde_priv_ma, tailored_priv_ma, \
            #                # kde_priv_ma, 0, \
            #                KDE_PRIV_MAs / (j+1), TAILORED_PRIV_MAs / (j+1) \
            #       ))

        # MAs[eps] = {}
        MAs[eps]["mst"] = (KDE_MST_MAs / C.n_runs_MA, TAILORED_MST_MAs / C.n_runs_MA)
        MAs[eps]["privbayes"] = (KDE_PRIV_MAs / C.n_runs_MA, TAILORED_PRIV_MAs / C.n_runs_MA)
        MAs[eps]["rap"] = (KDE_PRIV_MAs / C.n_runs_MA, TAILORED_RAP_MAs / C.n_runs_MA)

    # dump_artifact(MAs, "kde_ORDINAL_custom_attack_MAs")

def RUNTIME_experiment():
    pass

def QUALITY_Inference_experiment():
    pass

def QUALITY_Wasserstein_experiment():
    pass

def FP_experiment():
    pass


MA_experiment("cali")
# determine_rap_queries("cali")
