import json
import pandas as pd
import numpy as np
import pickle
import time
import sys
from types import SimpleNamespace
from sklearn.metrics import roc_auc_score, confusion_matrix
import tapas.datasets
from tapas.datasets.data_description import DataDescription
import tapas.generators
import tapas.threat_models
import tapas.attacks
import tapas.report
from tapas.datasets.dataset import _parse_csv, validate_header, TabularDataset
import tqdm










# DIR = "/Users/golobs/Documents/GradSchool/"
DIR = "/home/azureuser/"
# shadowset_directory = DIR + "shadowsets/"
shadowset_directory = "/datadrive/shadowsets/"

meta_filepath = DIR + "SNAKE/meta.json"
aux_filepath = DIR + "SNAKE/base.parquet"

# n_sizes = {100: 10, 316: 18, 1_000: 32, 3_162: 56, 10_000: 100, 31_623: 178}
N_SIZES = [100, 316, 1_000, 3_162, 10_000, 31_623]
# n_sizes = [100, 316]
t_sizes = [10, 18, 32, 56, 100, 178]
# t_sizes = [10, 18]
EPSILONS = [round(10 ** x, 2) for x in np.arange(-1, 3.1, 1 / 2)]
# epsilons = [.1, 1]
SDGs = ["mst", "priv", "gsd"]
# sdgs = ["mst", "priv"]

expA = SimpleNamespace(
    s=500,
    r=30,
    n=10_000,
    t=100,
    exclude={},
)

expB = SimpleNamespace(
    s=500,
    r=30,
    eps=10,
    exclude={"gsd": [3_162, 31_623]}
)

expC = SimpleNamespace(
    s=4_000,
    r=30,
    n=1000,
    t=32,
    eps=10,
    exclude={},
)





def main():

    task = sys.argv[1].upper()
    if task == "A":
        epsilons = EPSILONS
        if sys.argv[2] != ".":
            epsilons = [float(sys.argv[2])]
        for eps in epsilons:
            tapas_attack(task, '{0:.2f}'.format(eps), expA.n, expA.s, expA.r, [sdg for sdg in SDGs if eps in expA.exclude.get(sdg, [])])

    elif task == "B":
        n_sizes = N_SIZES
        if sys.argv[2] != ".":
            n_sizes = [int(sys.argv[2])]
        for n in n_sizes:
            tapas_attack(task, expB.eps, n, expB.s, expB.r, [sdg for sdg in SDGs if n in expB.exclude.get(sdg, [])])

    elif task == "C":
        tapas_attack(task, expC.eps, expC.n, expC.s, expC.r, list(expC.exclude.keys()))

    else:
        assert False, f"Invalid parameter: {task}"



def tapas_attack(task, eps, n, s, r, sdgs_excluded):

    tapas_data, aux, description, columns = load_data()
    sdgs = SDGs
    if sys.argv[3] != ".":
        sdgs = [sys.argv[3]]

    subexps = {"A": f"e{eps}/", "B": f"n{n}/", "C": ""}
    single_label_matrix = load_artifact(shadowset_directory + f"exp{task}/{subexps[task]}label_matrix_singleMI")
    # set_label_matrix = load_artifact(... # todo

    target_ids = single_label_matrix.columns
    targets = aux[aux.index.isin(target_ids)]
    tapas_targets = TabularDataset(targets[columns], description)
    tapas_data = tapas_data.drop_records(target_ids, in_place=False)

    for sdg in sdgs:
        if sdg in sdgs_excluded:
            print(f"\tskipping {sdg}, e{eps}, n{n}...")
            continue
        print(f"\t{sdg}, e{eps}, n{n}...")
        shadowset_directory_ = shadowset_directory + f"exp{task}/{subexps[task]}{sdg}/"
        auc, aucs, runtime = tapas_attack_with_shadowsets_and_targets(tapas_data, tapas_targets, shadowset_directory_, n, s, r)
        print("\tauc: ", auc)

        with open(shadowset_directory + f"exp{task}/{sdg}_e{eps}_n{n}_results.txt", 'w') as f:
            f.write(f"AUC: {auc}\n")
            f.write(f"runtime: {runtime}\n\n")
            f.write(f"all AUCs\n")
            f.writelines(f"{x}\n" for x in aucs)




def tapas_attack_with_shadowsets_and_targets(data, targets, shadowset_directory_, n, s, r):

    start = time.process_time()
    data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(
        data,
        auxiliary_split=0.5,
        #aux_data=data,snake_targets_wo_hhid.csv
        #test_data=target_record,
        num_training_records=n,
    )

    sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(
        None,
        num_synthetic_records=n,
    )

    threat_model = tapas.threat_models.TargetedMIA(
        attacker_knowledge_data=data_knowledge,
        target_record=targets,
        attacker_knowledge_generator=sdg_knowledge,
        generate_pairs=True,
        replace_target=True,
        iterator_tracker=tqdm.tqdm
    )

    attacker = tapas.attacks.GroundhogRQAttack(targets=targets, use_pregenerated=shadowset_directory_, schema=data.description)

    print("Training the attack...")
    attacker.train(threat_model, num_samples=s)

    print("Testing the attack...")
    attack_summary = threat_model.test(attacker, num_samples=r, ignore_memory=False, shadowset_indeces=(s, s + r))
    end = time.process_time()

    # metrics = attack_summary.get_metrics()
    average_auc, all_aucs = custom_metric(attack_summary)
    return average_auc, all_aucs, end - start





def load_data():
    print("Loading dataset...")
    with open(meta_filepath) as f:
        schema = json.load(f)
    aux = pd.read_parquet(aux_filepath)

    aux['HHID'] = aux.index
    aux.index = range(aux.shape[0])
    columns = aux[np.take(aux.columns, range(15))].columns.tolist()
    # numeric_columns = ['age', 'ownchild', 'hoursut']
    # catg_columns = [col for col in columns if col not in numeric_columns]

    description = DataDescription(schema, label="snake")
    return TabularDataset(aux[columns], description), aux, description, columns


def custom_metric(summary):
    # predictions = np.swapaxes(summary.predictions, 0, 1)
    scores = np.swapaxes(summary.scores, 0, 1)
    average_auc = 0
    all_aucs = []
    # for label, prediction, score in zip(summary.labels, predictions, scores):
    for label, score in zip(summary.labels, scores):
        auc = .5 if len(np.unique(label)) < 2 else roc_auc_score(label, score)
        average_auc += auc
        all_aucs.append(auc)

    return average_auc / summary.labels.shape[0], all_aucs


def dump_artifact(artifact, name):
    pickle_file = open(name, 'wb')
    pickle.dump(artifact, pickle_file)
    pickle_file.close()


def load_artifact(name):
    try:
        pickle_file = open(name, 'rb')
        artifact = pickle.load(pickle_file)
        pickle_file.close()
        return artifact
    except:
        return None


main()
