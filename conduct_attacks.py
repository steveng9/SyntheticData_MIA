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


# from collections import Counter

from util import *



def attack_experiments(sdgs, cfg):

    _, full_aux, columns, meta, _ = get_data(cfg)

    results_filename = cfg.get_filename("MA_mst_priv", use_RAP_config=True, overlap=cfg.overlapping_aux)

    results = load_artifact(results_filename) or {}
    for eps in cfg.epsilons:
        if eps not in results:
            results[eps] = {}
        for sdg in sdgs.keys():
            if sdg not in results[eps]:
                results[eps][sdg] = {
                    "KDE_MA": [],
                    "KDE_AUC": [],
                    "KDE_time": [],
                    "custom_MA": [],
                    "custom_AUC": [],
                    "custom_MA_weighted": [],
                    "custom_AUC_weighted": [],
                    "custom_time": [],
                    "arbitrary_MA": [],
                    "distance": []
                }

    print(f"\n{cfg.data_name}, {cfg.set_MI}, EPS: ", end="")
    for eps in cfg.epsilons:
        print(eps)

        # use same sample for each sdg
        target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(cfg, full_aux, columns)

        aux = full_aux if cfg.overlapping_aux else full_aux[~full_aux.index.isin(train.index)]

        for sdg, run_fn in sdgs.items():

            kde_ma, kde_auc, kde_time, tailored_ma, tailored_auc, tailored_ma_w, tailored_auc_w, tailored_time, arbitrary_ma, distance = run_fn(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed)

            if C.save_results:
                if kde_ma is not None:
                    results[eps][sdg]["KDE_MA"].append(kde_ma)
                    results[eps][sdg]["KDE_AUC"].append(kde_auc)
                    results[eps][sdg]["KDE_time"].append(kde_time)
                if tailored_ma is not None:
                    results[eps][sdg]["custom_MA"].append(tailored_ma)
                    results[eps][sdg]["custom_AUC"].append(tailored_auc)
                    results[eps][sdg]["custom_time"].append(tailored_time)
                if tailored_ma_w is not None:
                    results[eps][sdg]["custom_MA_weighted"].append(tailored_ma_w)
                    results[eps][sdg]["custom_AUC_weighted"].append(tailored_auc_w)
                    results[eps][sdg]["custom_time"].append(tailored_time)
                if distance is not None:
                    results[eps][sdg]["distance"].append(distance)
                if arbitrary_ma is not None:
                    results[eps][sdg]["arbitrary_MA"].append(arbitrary_ma)
                # save off intermediate results
                dump_artifact(results, results_filename)

            if C.verbose:
                print(f"SDG: {sdg}, KDE: {'{0:.2f}'.format(kde_ma or 0)}, {'{0:.2f}'.format(kde_auc or 0)}, arbitrary: {'{0:.2f}'.format(arbitrary_ma or 0)}, MAMA-MIA: {'{0:.2f}'.format(max([tailored_ma or 0, tailored_auc or 0, tailored_ma_w or 0, tailored_auc_w or 0]))}, distance: {'{0:.2f}'.format(distance or 0)}")



## DOMIAS (using KDE) Proper
##-------------------------

def kde_get_ma(cfg, aux, synth, targets, target_ids, membership, sample_seed):
    encoded_synth, encoded_targets, encoded_aux_sample = encode_data_KDE(cfg, aux, synth, targets, target_ids, sample_seed)

    # Find density at each target in synth and in base
    try:
        start_time = time.process_time()
        density_synth = stats.gaussian_kde(encoded_synth.values.transpose(1, 0))
        density_aux = stats.gaussian_kde(encoded_aux_sample.values.transpose(1, 0))

        p_synth_evaluated = density_synth.evaluate(encoded_targets.to_numpy().transpose(1, 0))
        p_aux_evaluated = density_aux.evaluate(encoded_targets.to_numpy().transpose(1, 0))

        # Score
        A = p_synth_evaluated / (p_aux_evaluated + 1e-20)

        predictions, MA, AUC, ROC_scores = score_attack(cfg, A, [1]*targets.shape[0], targets, target_ids, membership)

        end_time = time.process_time()

        return MA, AUC, end_time - start_time, ROC_scores
    except np.linalg.LinAlgError:
        print("Error in calculating KDE.")
        return None, None, None


## MAMA-MIA
##-------------------------


def attack_mst(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed, fps, synth=None):
    mst_gen = mst.MST(dataset=train[columns], metadata=meta, size=cfg.synth_size, epsilon=eps)
    try:
        if synth is None:
            mst_gen.run()
            synth = mst_gen.output

            if cfg.data_name == "snake":
                synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
            else:
                synth = synth.astype(int)

        # kde_ma, kde_auc, kde_time, kde_roc = kde_get_ma(cfg, aux, synth, targets, target_ids, membership, kde_sample_seed)
        tailored_ma, tailored_auc, arbitrary_ma, tailored_time, tailored_roc = run_all_mst_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, fps)
        # wd = wasserstein_distance(cfg, train, synth, columns)

        return None, None, None, None, None, tailored_ma, tailored_auc, tailored_time, arbitrary_ma, None, None, tailored_roc
    except ValueError:
        print("Error in running MST.")
        return None, None, None, None, None, None, None, None, None, None, None, None


def attack_privbayes(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed, fps):

    # generate synthetic test data
    privbayes_gen = privbayes.PRIVBAYES(dataset=train[columns], metadata=meta, size=cfg.synth_size, epsilon=eps)

    # try:
    privbayes_gen.run()
    synth = privbayes_gen.output

    if cfg.data_name == "snake":
        synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
    elif cfg.data_name == "cali":
        synth = synth.astype(int)

    # conduct experiments
    kde_ma, kde_auc, kde_time, kde_roc = kde_get_ma(cfg, aux, synth, targets, target_ids, membership, kde_sample_seed)
    tailored_ma, tailored_auc, arbitrary_ma, tailored_time, tailored_roc = run_all_privbayes_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, fps)
    wd = wasserstein_distance(cfg, train, synth, columns)

    return kde_ma, kde_auc, kde_time, None, None, tailored_ma, tailored_auc, tailored_time, arbitrary_ma, wd, kde_roc, tailored_roc
    # except:
    #     print("Error in running PrivBayes.")
    #     return None, None, None, None, None, None, None, None, None, None


def attack_gsd(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed, fps):
    encoded_aux = encode_data_all_numeric(cfg, aux, minmax_encode_catg=False)
    encoded_train = encode_data_all_numeric(cfg, train, minmax_encode_catg=False)
    encoded_targets = encode_data_all_numeric(cfg, targets, minmax_encode_catg=False)

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
        # LOAD saved synthetic test data
        # encoded_synth, synth, train, encoded_targets, targets, target_ids, membership, query_ids = load_artifact(f"{cfg.data_name}_{'{0:.2f}'.format(eps)}_GSD_temp_synth_stuff")

        # GENERATE and SAVE synthetic test data
        encoded_synth, query_ids = algo.fit_dp(PRNGKey(seed), stat_module=stat_module, epsilon=eps, delta=delta)
        encoded_synth = encoded_synth.df
        synth = decode_data_from_numeric(cfg, encoded_synth, minmax_encode_catg=False)
        dump_artifact((encoded_synth, synth, train, encoded_targets, targets, target_ids, membership, query_ids),
                      f"{cfg.data_name}_{'{0:.2f}'.format(eps)}_GSD_temp_synth_stuff")

        if cfg.data_name == "snake":
            synth = synth.astype({'age': 'int', 'ownchild': 'int', 'hoursut': 'int'})
        elif cfg.data_name == "cali":
            synth = synth.astype(int)

        all_possible_queries = marginal_module2.queries
        kde_ma, kde_auc, kde_time, kde_roc = kde_get_ma(cfg, aux, synth, targets, target_ids, membership, kde_sample_seed)
        tailored_ma, tailored_auc, tailored_ma_w, tailored_auc_w, arbitrary_ma, tailored_time, tailored_roc = run_all_gsd_experiments(cfg, all_possible_queries, encoded_aux, encoded_synth, eps, encoded_targets, target_ids, membership, fps)
        wd = wasserstein_distance(cfg, train, synth, columns)

        return kde_ma, kde_auc, kde_time, tailored_ma, tailored_auc, tailored_ma_w, tailored_auc_w, tailored_time, arbitrary_ma, wd, kde_roc, tailored_roc
    except:
        print("Error in running GSD.")
        return None, None, None, None, None, None, None, None, None, None


def attack_rap(cfg, meta, aux, columns, train, eps, targets, target_ids, membership, kde_sample_seed, fps):

    # try:
    # LOAD saved synthetic test data
    # catg_synth, synth, queries_used, train, targets, target_ids, membership = load_artifact(f"{cfg.data_name}_{'{0:.2f}'.format(eps)}_RAP_temp_synth_stuff")

    # GENERATE and SAVE synthetic test data
    train_encoded, columns_domain = binarize_discrete_features_evenly(cfg, train, columns)
    synth, catg_synth, queries_used = get_rap_synth(cfg, columns, train_encoded, eps, columns_domain)
    dump_artifact((catg_synth, synth, queries_used, train, targets, target_ids, membership), f"{cfg.data_name}_{'{0:.2f}'.format(eps)}_RAP_temp_synth_stuff")

    # conduct experiments
    synth_decoded = decode_rap_synth(cfg, columns, meta, catg_synth)
    kde_ma, kde_auc, kde_time = kde_get_ma(cfg, aux, synth_decoded, targets, target_ids, membership, kde_sample_seed)
    tailored_ma, tailored_auc, tailored_ma_sparse, tailored_auc_sparse, arbitrary_ma, tailored_time = run_all_rap_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, fps)
    wd = wasserstein_distance(cfg, train, synth, columns, encode_d2=False)
    wd_2 = wasserstein_distance(cfg, train, synth_decoded, columns)

    assert wd_2 == wd, "Error in WD calculation!"

    # SANITY CHECKS
    # sanity_ma_verification_queries, sanity_auc, sanity_tailored_ma_sparse, sanity_tailored_auc_sparse, _, _ = run_all_rap_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, verification_queries=queries_used)
    # print(f"\nReal: {tailored_ma}, Sanity: {sanity_ma_verification_queries}")

    return kde_ma, kde_auc, kde_time, tailored_ma_sparse, tailored_auc_sparse, tailored_ma, tailored_auc, tailored_time, arbitrary_ma, wd, kde_roc, tailored_roc
    # except:
    #     print("Error in running RAP.")
    #     return None, None, None, None, None, None, None, None, None, None








def run_all_mst_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, fps):
    # marginals_weights = load_artifact(make_FP_filename(cfg, "MST", eps))
    marginals_weights = fps

    start_time = time.process_time()
    if cfg.data_name != "berka":
        scores, tailored_ma, tailored_auc, ROC_scores = custom_mst_attack(cfg, eps, aux, synth, targets, target_ids, membership, marginals_weights)
    else:
        scores, tailored_ma, tailored_auc, ROC_scores = custom_mst_attack_for_berka(cfg, eps, aux, synth, targets, target_ids,
                                                                          membership, marginals_weights)
    end_time = time.process_time()
    plot_output(scores)

    arbitrary_FP_ma = None
    if cfg.check_arbitrary_fps:
        arbitrary_marginals = generate_arbitrary_FPs(columns, len(columns)-1, 2, 2)
        if cfg.data_name != "berka":
            _, arbitrary_FP_ma, _arbitrary_tailored_auc = custom_mst_attack(cfg, eps, aux, synth, targets, target_ids, membership, arbitrary_marginals)
        else:
            _, arbitrary_FP_ma, _arbitrary_tailored_auc = custom_mst_attack_for_berka(cfg, eps, aux, synth, targets, target_ids,
                                                                            membership, arbitrary_marginals)
    return tailored_ma, tailored_auc, arbitrary_FP_ma, end_time - start_time, ROC_scores


def run_all_privbayes_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, fps):
    # conditionals_weights = load_artifact(make_FP_filename(cfg, "PrivBayes", eps))
    conditionals_weights = fps

    start_time = time.process_time()
    scores, tailored_ma, tailored_auc, ROC_scores = custom_privbayes_attack(cfg, eps, aux, synth, targets, target_ids, membership, conditionals_weights)
    end_time = time.process_time()
    plot_output(scores)

    arbitrary_FP_ma = None
    if cfg.check_arbitrary_fps:
        # only choose conditionals up to a certain # of parents
        max_conditional_size = 5
        # max_conditional_size = max([len(k) for k in conditionals_weights.keys()])
        arbitrary_conditionals = generate_arbitrary_FPs(columns, len(columns)-1, 1, max_conditional_size)
        _, arbitrary_FP_ma, _arbitrary_tailored_auc = custom_privbayes_attack(cfg, eps, aux, synth, targets, target_ids, membership, arbitrary_conditionals)

    return tailored_ma, tailored_auc, arbitrary_FP_ma, end_time - start_time, ROC_scores


def run_all_gsd_experiments(cfg, all_possible_queries, encoded_aux, encoded_synth, eps, encoded_targets, target_ids, membership, fps):
    # fp_weights = load_artifact(make_FP_filename(cfg, "GSD", eps))
    fp_weights = fps

    start_time = time.process_time()
    (scores, tailored_ma, tailored_auc, ROC_scores), (scores_w, tailored_ma_w, tailored_auc_w, ROC_scores_w) = custom_gsd_attack(cfg, eps, encoded_aux, encoded_synth, encoded_targets, target_ids, membership, fp_weights)
    end_time = time.process_time()
    plot_output(scores)

    arbitrary_FP_ma = None
    # if cfg.check_arbitrary_fps:
    #     assert False, "not yet implemented!"
        # arbitrary_conditionals = generate_arbitrary_FPs(columns, len(columns)-1, 1, max_conditional_size)
        # _, arbitrary_FP_ma, _arbitrary_tailored_auc = custom_gsd_attack(cfg, eps, aux, synth, targets, target_ids, membership, arbitrary_conditionals)

    return tailored_ma, tailored_auc, tailored_ma_w, tailored_auc_w, arbitrary_FP_ma, end_time - start_time, ROC_scores_w


def run_all_rap_experiments(cfg, columns, aux, synth, eps, targets, target_ids, membership, fps, verification_queries=None):
    # fp_filename = make_FP_filename(cfg, "RAP", eps, specify_epsilon=True, try_epsilons=[10, 100, 1000])
    # queries_weights = load_artifact(fp_filename) \
    #     if verification_queries is None else {tuple(q.tolist()): cfg.rap_use_FP_threshold for q in verification_queries}
    queries_weights = fps

    aux_encoded = pd.DataFrame(binarize_discrete_features_evenly(cfg, aux, columns)[0])
    targets_encoded = pd.DataFrame(binarize_discrete_features_evenly(cfg, targets, columns)[0])

    start_time = time.process_time()
    (scores, tailored_ma, tailored_auc), (scores_w, tailored_ma_sparse, tailored_auc_sparse) = custom_rap_attack(cfg, eps, aux_encoded, synth, targets, targets_encoded, target_ids, membership, queries_weights)
    end_time = time.process_time()

    assert (aux_encoded.columns == targets_encoded.columns).all()
    assert len(aux_encoded.columns) == synth.shape[1]

    arbitrary_FP_ma = None
    if cfg.check_arbitrary_fps:
        arbitrary_queries = generate_arbitrary_FPs(aux_encoded.columns, cfg.rap_top_q * cfg.rap_epochs, cfg.rap_k, cfg.rap_k)
        (_, arbitrary_FP_ma, _arbitrary_tailored_auc), (scores_w, tailored_ma_sparse, tailored_auc_sparse) = custom_rap_attack(cfg, eps, aux_encoded, synth, targets, targets_encoded, target_ids, membership, arbitrary_queries)

    plot_output(scores)

    return tailored_ma, tailored_auc, tailored_ma_sparse, tailored_auc_sparse, arbitrary_FP_ma, end_time - start_time







def custom_mst_attack(cfg, eps, aux, synth, targets, target_ids, membership, marginals_weights):
    A = np.array([0.0]*targets.shape[0])
    num_queries_used = np.array([0] * targets.shape[0])

    threshold = determine_weight_threshold(cfg, eps, marginals_weights)

    for marginal, weight in marginals_weights.items():
        if weight < threshold: continue

        marginal_list = list(marginal)
        D_synth = synth[marginal_list].value_counts(normalize=True)
        D_aux = aux[marginal_list].value_counts(normalize=True)

        default_val = 1e-10
        A += np.array([weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val), default=default_val) for val in targets[marginal_list].values])
        num_queries_used += 1

    return score_attack(cfg, A, num_queries_used, targets, target_ids, membership)


def custom_privbayes_attack(cfg, eps, aux, synth, targets, target_ids, membership, conditionals_weights):
    A = np.array([0.0] * targets.shape[0])
    num_queries_used = np.array([0] * targets.shape[0])

    threshold = determine_weight_threshold(cfg, eps, conditionals_weights)

    for conditional, weight in conditionals_weights.items():
        if weight < threshold: continue

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
            aux_conditional = max(D_aux.get(parent_vals or True, default=pd.Series(dtype=np.float64)).get(child_val, default=default_val), default_val)
            # aux_conditional = D_aux.get(parent_vals or True).get(child_val)
            # ratio_conditionals.append(weight * synth_conditional / aux_conditional)
            ratio_conditionals.append(synth_conditional / aux_conditional)

        A += np.array(ratio_conditionals)
        num_queries_used += 1

    return score_attack(cfg, A, num_queries_used, targets, target_ids, membership)


def custom_gsd_attack(cfg, eps, encoded_aux, encoded_synth, encoded_targets, target_ids, membership, fp_weights):
    A = np.array([0.0] * encoded_targets.shape[0])
    A_weighted = np.array([0.0] * encoded_targets.shape[0])
    num_queries_used = np.array([0] * encoded_targets.shape[0])
    num_queries_used_weighted = np.array([0] * encoded_targets.shape[0])

    threshold = determine_weight_threshold(cfg, eps, fp_weights)

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
        D_synth = pd.DataFrame(np.swapaxes(in_bin(encoded_synth.iloc[:, fp_feature_indeces].T.values),0,1)).value_counts(normalize=True)
        D_aux = pd.DataFrame(np.swapaxes(in_bin(encoded_aux.iloc[:, fp_feature_indeces].T.values),0,1)).value_counts(normalize=True)

        default_val = 1e-10
        target_vals = np.swapaxes(in_bin(encoded_targets.iloc[:, fp_feature_indeces].T.values), 0, 1)

        for w, val in enumerate(target_vals):
            if val.all():
                A[w] += D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val))
                A_weighted[w] += weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val))
                num_queries_used[w] += 1
                num_queries_used_weighted[w] += weight

    scores = score_attack(cfg, A, num_queries_used, encoded_targets, target_ids, membership)
    scores_weighted = score_attack(cfg, A_weighted, num_queries_used_weighted, encoded_targets, target_ids, membership)
    return scores, scores_weighted


def custom_rap_attack(cfg, eps, aux_encoded, synth, targets, targets_encoded, target_ids, membership, fp_weights):
    # assert False, "Implement custom RAP FP dynamic threshold!"
    A = np.array([0.0]*targets_encoded.shape[0])
    A_sparse = np.array([0.0]*targets_encoded.shape[0])
    num_queries_used = np.array([0] * targets_encoded.shape[0])
    num_queries_used_sparse = np.array([0] * targets_encoded.shape[0])

    # choose highest weight threshold of when to incorporate focal-point based on epsilon
    # threshold = max(fp_weights.values()) * max([t for e, t in cfg.fp_weight_thresholds.items() if eps >= e])
    for query, weight in fp_weights.items():
        # if weight < threshold: continue
        if weight < 2: continue

        query_columns = list(query)
        D_synth = synth[query_columns].value_counts(normalize=True)
        D_aux = aux_encoded[query_columns].value_counts(normalize=True)

        default_val = 1e-6
        target_vals = targets_encoded[query_columns].values

        for w, val in enumerate(target_vals):
            A[w] += weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val))
            num_queries_used[w] += weight

            # if val.all() or not val.any():
            if val.all():
                A_sparse[w] += weight * D_synth.get(tuple(val), default=default_val) / D_aux.get(tuple(val))
                num_queries_used_sparse[w] += weight

    return score_attack(cfg, A, num_queries_used, targets, target_ids, membership), score_attack(cfg, A_sparse, num_queries_used_sparse, targets, target_ids, membership)



def custom_mst_attack_for_berka(cfg, eps, aux, synth, targets, target_ids, membership, marginals_weights):
    A = np.array([0.0]*targets.shape[0])
    num_queries_used = np.array([0] * targets.shape[0])

    for marginal, weight in marginals_weights.items():
        marginal_list = list(marginal)
        D_synth = synth[marginal_list].value_counts(normalize=True)
        D_aux = aux[marginal_list].value_counts(normalize=True)

        for i, val in enumerate(targets[marginal_list].values):
            prop_synth = D_synth.get(tuple(val))
            if prop_synth == None:
                distances = np.linalg.norm(synth[marginal_list].values - val, axis=1)
                closest_idx = np.argsort(distances)[1]
                prop_synth = D_synth.get(tuple(synth[marginal_list].iloc[closest_idx].values))
            A[i] += weight * prop_synth / D_aux.get(tuple(val))
        num_queries_used += 1

    return score_attack(cfg, A, num_queries_used, targets, target_ids, membership)

def custom_mst_SUBMISSION_attack_for_berka(cfg, eps, aux, synth, targets, target_ids, marginals_weights):
    A = np.array([0.0]*targets.shape[0])
    num_queries_used = np.array([0] * targets.shape[0])

    for marginal, weight in marginals_weights.items():
        marginal_list = list(marginal)
        D_synth = synth[marginal_list].value_counts(normalize=True)
        D_aux = aux[marginal_list].value_counts(normalize=True)

        for i, val in enumerate(targets[marginal_list].values):
            prop_synth = D_synth.get(tuple(val))
            if prop_synth == None:
                distances = np.linalg.norm(synth[marginal_list].values - val, axis=1)
                closest_idx = np.argsort(distances)[1]
                prop_synth = D_synth.get(tuple(synth[marginal_list].iloc[closest_idx].values))
            A[i] += weight * prop_synth / D_aux.get(tuple(val))
        num_queries_used += 1

    predictions = pd.Series(A / np.maximum(np.array([1] * targets.shape[0]), num_queries_used))
    activated_predictions = activate_3(np.array(predictions))

    return activated_predictions