import sys
import numpy as np
import json
import math
from types import SimpleNamespace
import pickle


from pandas.api.types import is_numeric_dtype
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder

import warnings
warnings.filterwarnings("ignore")
# sys.path.append('reprosyn-main/src/reprosyn/methods/mbi/')

# sys.path.append('relaxed-adaptive-projection/relaxed_adaptive_projection/')
import pandas as pd


## Specify directory to store results here
DATA_DIR = "/Users/golobs/Documents/GradSchool/"



# constants
C = SimpleNamespace(
    verbose=True,
    save_results=True,
    n_bins=20,
    n_runs=30,

    # shadow modelling
    shadow_epsilons=list(reversed([round(10 ** x, 2) for x in np.arange(-1, 3.1)])),
    # shadow_epsilons=[1000, 100, 10],
    n_shadow_runs=50,
    shadow_train_size=1_000,
    shadow_synth_size=1_000,

    # KDE
    use_categorical_features=True,
    samples_append_targets=False,
    rap_bucket_numeric=True,
)


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


def california_data(cfg):

    columns = [str(x) for x in range(9)]
    meta = [{"name": col, "representation": list(range(C.n_bins))} for col in columns] # TODO is this range correct?
    aux_original = pd.DataFrame(StandardScaler().fit_transform(fetch_california_housing(as_frame=True).frame.sample(frac=1)), columns=columns)

    fit_continuous_features_equaldepth(aux_original, "cali")
    aux = discretize_continuous_features_equaldepth(aux_original, "cali")
    fit_discrete_features_evenly("cali", aux, pd.DataFrame(meta), columns)
    fit_data_all_numeric("cali", aux, meta, columns, [])

    aux["HHID"] = np.hstack([[i]*cfg.household_min_size for i in range(math.ceil(aux.shape[0] / cfg.household_min_size))])[:aux.shape[0]]
    meta = [{'name': str(col), 'type': 'finite/ordered', 'representation': range(C.n_bins)} for col in columns]
    cfg.numeric_columns = columns
    cfg.categorical_columns = []
    return None, aux, columns, meta, "cali"


def snake_data(cfg):
    with open(DATA_DIR + "SNAKE/meta.json") as f:
        meta = json.load(f)
    # meta = pd.read_json(DATA_DIR + "SNAKE/meta.json")

    aux = pd.read_parquet(DATA_DIR + "SNAKE/base.parquet")
    aux['HHID'] = aux.index
    aux.index = range(aux.shape[0])
    columns = aux[np.take(aux.columns, range(15))].columns.tolist()
    numeric_columns = ['age', 'ownchild', 'hoursut']
    catg_columns = [col for col in columns if col not in numeric_columns]

    fit_discrete_features_evenly("snake", aux, pd.DataFrame(meta), columns)
    fit_data_all_numeric("snake", aux, meta, numeric_columns, catg_columns)

    cfg.numeric_columns = numeric_columns
    cfg.categorical_columns = catg_columns
    return None, aux, columns, meta, "snake"



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


# fit encoders to convert discrete data into one hot encoded form
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


# convert discrete data into one hot encoded form
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

# decode one hot encoded form into numerical data form
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


# fit encoders to convert data into all numeric values
def fit_data_all_numeric(data_name, aux, meta, numerical_columns, catg_columns):
    meta = pd.DataFrame(meta)
    catg_encoder = OrdinalEncoder(categories=[meta[meta["name"] == col].representation.values[0] for col in catg_columns]).fit(aux[catg_columns])
    catg_minmax_scaler = MinMaxScaler().fit(catg_encoder.transform(aux[catg_columns])) if catg_columns else None
    scalars = {col: MinMaxScaler().fit(aux[[col]]) for col in numerical_columns}
    dump_artifact((catg_encoder, catg_minmax_scaler, catg_columns), f"numeric_{data_name}_catg_encoder")
    dump_artifact((scalars, numerical_columns), f"numeric_{data_name}_numerical_encoder_bins{C.n_bins}")


# convert data into all numeric values.
def encode_data_all_numeric(cfg, data, minmax_encode_catg=True, keep_indeces=True):
    numeric_encoders, numeric_columns = load_artifact(f"numeric_{cfg.data_name}_numerical_encoder_bins{C.n_bins}")
    encoded_data = pd.DataFrame()
    for col in numeric_columns:
        encoded_data[[col]] = numeric_encoders[col].transform(data[[col]])
    if C.use_categorical_features:
        catg_encoder, catg_minmax_scaler, catg_columns = load_artifact(f"numeric_{cfg.data_name}_catg_encoder")
        if len(catg_columns) > 0:
            if minmax_encode_catg and catg_minmax_scaler:
                encoded_data[catg_columns] = catg_minmax_scaler.transform(catg_encoder.transform(data[catg_columns]))
            else:
                encoded_data[catg_columns] = catg_encoder.transform(data[catg_columns])

    encoded_data = encoded_data[data.drop('HHID', axis=1, errors='ignore').columns]

    if keep_indeces:
        if cfg.set_MI:
            encoded_data["HHID"] = data["HHID"].values
        else:
            encoded_data.index = data.index
    return encoded_data


# revert from GSD numeric form into original form
def decode_data_from_numeric(cfg, data, minmax_encode_catg=True):
    numeric_encoders, numeric_columns = load_artifact(f"numeric_{cfg.data_name}_numerical_encoder_bins{C.n_bins}")
    decoded_data = pd.DataFrame()
    for col in numeric_columns:
        decoded_data[[col]] = numeric_encoders[col].inverse_transform(data[[col]])
    if C.use_categorical_features:
        catg_encoder, catg_minmax_scaler, catg_columns = load_artifact(f"numeric_{cfg.data_name}_catg_encoder")
        if len(catg_columns) > 0:
            if minmax_encode_catg:
                decoded_data[catg_columns] = catg_encoder.inverse_transform(catg_minmax_scaler.inverse_transform(data[catg_columns]))
            else:
                decoded_data[catg_columns] = catg_encoder.inverse_transform(data[catg_columns])
    return decoded_data


# convert and sample data into numeric form for KDE measurement
def encode_data_KDE(cfg, aux, synth, targets, target_ids, sample_seed):
    target_exclusion_list = aux.HHID.isin(target_ids) if cfg.set_MI else aux.index.isin(target_ids)
    aux_sample = aux[~target_exclusion_list].sample(n=synth.shape[0], random_state=sample_seed)  # TODO: see if density is n-independent (try huge n for aux_sample)
    if C.samples_append_targets: aux_sample = pd.concat([aux_sample, targets]).sample(frac=1)  # shuffle
    encoded_synth = encode_data_all_numeric(cfg, synth, keep_indeces=False)
    encoded_targets = encode_data_all_numeric(cfg, targets, keep_indeces=False)
    encoded_aux_sample = encode_data_all_numeric(cfg, aux_sample, keep_indeces=False)
    return encoded_synth, encoded_targets, encoded_aux_sample
