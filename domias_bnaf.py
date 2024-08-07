import time
from pathlib import Path

import pandas as pd

from util import *
from gen_tapas_shadowsets import gen_mst, gen_priv, gen_gsd

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
import torch
from domias.bnaf.density_estimation import compute_log_p_x, density_estimator_trainer


use_pregenerated_synthsets = False
encode_ordinal = True
encode_categorical = True
n_ensemble = 1
epochs = 50
early_stopping = 20
batch_dim = 400

# DIR = DATA_DIR + "Thesis/"
DIR = "/home/golobs/"
results_dir = "domias_bnaf/"
attack_completed_file = DIR + "experiment_artifacts/" + results_dir + "attack_completed_file.txt"
n_sizes = [100, 316, 1_000, 3_162, 10_000, 31_623]
epsilons = [round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 2)]
sdgs = ["mst", "priv", "gsd"]

def print_attack_status(location=results_dir, completed_file=attack_completed_file):
    attacks_completed = open(completed_file, "r").readlines()
    completed = []

    print("\nexperiment A")
    for sdg in sdgs:
        for eps in epsilons:
            if f"{sdg}, {fo(eps)}, {expA.n}, snake, True, False\n" not in attacks_completed and eps not in expA.exclude.get(sdg, []):
                print(f"\t{sdg}, e{fo(eps)}, n{expA.n}, snake", end="...")
                progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, eps, expA.n, "snake", True, False)) or {".": []}).values()])
                print(f"{progress} / {C.n_runs}")
            else:
                completed.append(f"{sdg}, {fo(eps)}, {expA.n}, snake, True, False")

        print()


    print("\nexperiment B")
    for sdg in sdgs:
        for n in n_sizes:
            if f"{sdg}, {fo(expB.eps)}, {n}, snake, True, False\n" not in attacks_completed and n not in expB.exclude.get(sdg, []):
                print(f"\t{sdg}, e{fo(expB.eps)}, n{n}, snake", end="...")
                progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, expB.eps, n, "snake", True, False)) or {".": []}).values()])
                print(f"{progress} / {C.n_runs}")
            else:
                completed.append(f"{sdg}, {fo(expB.eps)}, {n}, snake, True, False")
        print()


    print("\nexperiment D")
    for sdg in sdgs:
        for eps in epsilons:
            for data in ["snake", "cali"]:
                if f"{sdg}, {fo(eps)}, {expD.n}, {data}, True, False\n" not in attacks_completed and eps not in expD.exclude.get(sdg, []):
                    print(f"\t{sdg}, e{fo(eps)}, n{expD.n}, {data}", end="...")
                    progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, eps, expD.n, data, True, False)) or {".": []}).values()])
                    print(f"{progress} / {C.n_runs}")
                    # print("NOT LAUNCHED")
                else:
                    completed.append(f"{sdg}, {fo(eps)}, {expD.n}, {data}, True, False")
        print()


    print("\nnon-overlapping")
    for sdg in sdgs:
        for eps in epsilons:
            for data in ["snake", "cali"]:
                if f"{sdg}, {fo(eps)}, {expD.n}, {data}, False, False\n" not in attacks_completed and eps not in expD.exclude.get(sdg, []):
                    print(f"\t{sdg}, e{fo(eps)}, n{expD.n}, {data}", end="...")
                    progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, eps, expD.n, data, False, False)) or {".": []}).values()])
                    print(f"{progress} / {C.n_runs}")
                    # print("NOT LAUNCHED")
                else:
                    completed.append(f"{sdg}, {fo(eps)}, {expD.n}, {data}, False, False")
        print()


    print("\nset MI")
    for sdg in sdgs:
        for eps in epsilons:
            for data in ["snake", "cali"]:
                if f"{sdg}, {fo(eps)}, {expD.n}, {data}, True, True\n" not in attacks_completed and eps not in expD.exclude.get(sdg, []):
                    print(f"\t{sdg}, e{fo(eps)}, n{expD.n}, {data}", end="...")
                    progress = max([len(l) for l in (load_artifact(attack_results_filename(location, sdg, eps, expD.n, data, True, True)) or {".": []}).values()])
                    print(f"{progress} / {C.n_runs}")
                    # print("NOT LAUNCHED")
                else:
                    completed.append(f"{sdg}, {fo(eps)}, {expD.n}, {data}, True, True")
        print()

    print("\ncompleted:")
    for c in completed:
        print(c)













if not Path(attack_completed_file).exists():
    with open(attack_completed_file, "w") as f:
        f.writelines("sdg, epsilon, N, data, overlap, setMI\n")


task = sys.argv[1]
if task == "domias_status":
    print_attack_status(location=results_dir, completed_file=attack_completed_file)
    sys.exit()

sdg = sys.argv[2]
epsilon = float(sys.argv[3])
n_size = int(sys.argv[4])
data = sys.argv[5]

cfg = Config(data, train_size=n_size, set_MI=False, overlapping_aux=True)
_, aux, columns, meta, _ = get_data(cfg)


ordered_columns_idx = [0, 5, 7, 11, 12]
# ordered_columns_idx = [0, 3, 5, 7, 8, 11, 12, 13]
ordered_columns = ['age', 'ownchild', 'gradeatn', 'hoursut', 'faminc']
# ordered_columns = ['age', 'female', 'ownchild', 'gradeatn', 'cow1', 'hoursut', 'faminc', 'mind16']
categorical_columns_idx = [3, 8, 13]
categorical_columns = ['female', 'cow1', 'mind16']
# categorical_columns_idx = [column_idx for column_idx in range(15) if column_idx not in ordered_columns_idx]
# categorical_columns = [column_name for column_name in meta.name.values.tolist() if column_name not in ordered_columns]

meta2 = pd.DataFrame(meta)
ord_enc = OrdinalEncoder(categories=[meta2.representation.values.tolist()[i] for i in ordered_columns_idx])
if encode_ordinal and data == "snake":
    ord_enc.fit(aux[ordered_columns])
oh_enc = OneHotEncoder(sparse_output=False, categories=[meta2.representation.values.tolist()[i] for i in categorical_columns_idx])
if encode_categorical and data == "snake":
    oh_enc.fit(aux[categorical_columns])

def encode_data_for_bnaf(unencoded):
    if data == "cali":
        return unencoded
    if data == "snake":
        data_encoded = pd.DataFrame()
        if encode_categorical:
            data_encoded = pd.DataFrame(oh_enc.transform(unencoded[categorical_columns]), columns=oh_enc.get_feature_names_out(categorical_columns))
        if encode_ordinal:
            data_encoded[ordered_columns] = ord_enc.transform(unencoded[ordered_columns])
        return data_encoded

def attack_results_filename(location, sdg, epsilon, n, data, overlap, set_MI):
    return f"{location}results_{sdg}_e{fo(epsilon)}_n{n}_{data}_o{overlap}_set{set_MI}"


# aux_encoded = encode_data_for_bnaf(aux)



results_filename = attack_results_filename(results_dir, sdg, epsilon, n_size, data, True, False)
results = load_artifact(results_filename) or {
    "BNAF_AUC": [],
    "BNAF_time": [],
}

for i in range(C.n_runs):

    runtime = 0

    if use_pregenerated_synthsets and data == "snake":
        synth = pd.read_parquet(DIR + f"experiment_artifacts/shadowsets/{sdg}/s{i}.parquet")
        label_matrix = load_artifact(DIR + f"experiment_artifacts/shadowsets/label_matrix_singleMI")

        #TODO are these in the correct order? Do they need to be sorted?
        target_ids = sorted(label_matrix.columns)
        membership = label_matrix.reindex(sorted(label_matrix.columns), axis=1).to_numpy()[i]
        targets = aux[aux.index.isin(target_ids)]

    else:
        gen_fns = {"mst": gen_mst, "priv": gen_priv, "gsd": gen_gsd}
        target_ids, targets, membership, train, kde_sample_seed = sample_experimental_data(cfg, aux, columns)
        synth = gen_fns[sdg](cfg, train, meta, n_size, epsilon)

    targets_encoded = encode_data_for_bnaf(targets)
    targets_encoded_np = targets_encoded.to_numpy()
    synth_encoded = encode_data_for_bnaf(synth)
    synth_encoded_np = synth_encoded.to_numpy()

    print()
    print()
    print()
    print(f"RUN: {i}")

    target_results_synth = np.array([0.0] * targets.shape[0])
    target_results_base = np.array([0.0] * targets.shape[0])

    for j in range(n_ensemble):
        aux_sample = aux[aux.merge(targets.drop_duplicates(), how='left', indicator=True)["_merge"] == "left_only"].sample(n=n_size)
        aux_sample_encoded_np = encode_data_for_bnaf(aux_sample).to_numpy()

        #### build BNAF model
        ##---------------------------------------
        split = .5

        start = time.process_time()

        _, p_G_model = density_estimator_trainer(
            synth_encoded_np,
            synth_encoded_np[: int(split * synth.shape[0])],
            synth_encoded_np[int(split * synth.shape[0]):],
            epochs=epochs,
            early_stopping=early_stopping,
            batch_dim=batch_dim,
        )

        _, p_R_model = density_estimator_trainer(
            aux_sample_encoded_np,
            aux_sample_encoded_np[: int(split * aux_sample_encoded_np.shape[0])],
            aux_sample_encoded_np[int(split * aux_sample_encoded_np.shape[0]):],
            epochs=epochs,
            early_stopping=early_stopping,
            batch_dim=batch_dim,
        )

        p_G_evaluated = np.exp(
            compute_log_p_x(p_G_model, torch.as_tensor(targets_encoded).float().to('cpu'))
            .cpu()
            .detach()
            .numpy()
        )

        p_R_evaluated = np.exp(
            compute_log_p_x(p_R_model, torch.as_tensor(targets_encoded).float().to('cpu'))
            .cpu()
            .detach()
            .numpy()
        )

        p_G_evaluated = np.nan_to_num(p_G_evaluated, nan=1e26)
        p_G_evaluated[np.isinf(p_G_evaluated)] = 1e26
        p_R_evaluated = np.nan_to_num(p_R_evaluated, nan=1e26)
        p_R_evaluated[np.isinf(p_R_evaluated)] = 1e26

        end = time.process_time()

        target_results_synth += p_G_evaluated
        target_results_base += p_R_evaluated

        runtime += (end - start)

    p_rel = (target_results_synth / n_ensemble) / (target_results_base / n_ensemble)
    p_rel[np.isinf(p_rel)] = 1e26

    predictions, MA, AUC = score_attack(cfg, p_rel, [1] * targets.shape[0], targets, target_ids, membership, activation_fn=activate_3)

    if AUC is not None:
        results["BNAF_AUC"].append(AUC)
        results["BNAF_time"].append(runtime)

    # save off intermediate results
    dump_artifact(results, results_filename)



print(f"completed DOMIAS BNAF attack for {sdg} e{epsilon}, n{n_size}, {data}")
with open(attack_completed_file, "a") as f:
    f.writelines(f"{sdg}, {fo(epsilon)}, {n_size}, {data}, True, False\n")
