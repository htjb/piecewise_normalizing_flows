import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from collections import Counter
from sklearn.cluster import KMeans

def load_data(path):

    data_train = pd.read_csv(filepath_or_buffer=join(path, "1000_train.csv"), index_col=False)
    data_test = pd.read_csv(filepath_or_buffer=join(path, "1000_test.csv"), index_col=False)

    return data_train, data_test

def load_data_no_discrete(path):
    """
    Loads the positive class examples from the first 10 percent of the dataset.
    """
    data_train, data_test = load_data(path)

    # Gets rid of any background noise examples i.e. class label 0.
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    # Because the data set is messed up!
    data_test = data_test.drop(data_test.columns[-1], axis=1)

    return data_train, data_test

def load_data_no_discrete_normalised(path):

    data_train, data_test = load_data_no_discrete(path)
    mu = data_train.mean()
    s = data_train.std()
    data_train = (data_train - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_test

def load_data_no_discrete_normalised_as_array(path):

    data_train, data_test = load_data_no_discrete_normalised(path)
    data_train, data_test = data_train.values, data_test.values

    i = 0
    # Remove any features that have too many re-occurring real values.
    features_to_remove = []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

    N = data_train.shape[0]
    N_validate = int(N*0.1)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test

data_train, data_validate, data_test = load_data_no_discrete_normalised_as_array("raw_physical_data/hepmass/")

from margarine.clustered import clusterMAF
from margarine.maf import MAF
import math
from sklearn.cluster import KMeans

try:
    flow = MAF.load('physical_benchmarks/hep_maf.pkl')
except FileNotFoundError:
    flow = MAF(data_train)
    flow.train(10000, early_stop=True)
    flow.save('physical_benchmarks/hep__maf.pkl')

lps = flow.log_prob(data_test.astype(np.float32))
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))

try:
    flow = clusterMAF.load('physical_benchmarks/hep_clustermaf.pkl')
except:
    _ = clusterMAF(data_train)
    nn = math.ceil(17424/_.cluster_number/2904)
    print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

    kmeans = KMeans(_.cluster_number, random_state=0)
    labels = kmeans.fit(data_train).predict(data_train)

    flow = clusterMAF(data_train, cluster_number=_.cluster_number, cluster_labels=labels, number_networks=nn)
    flow.train(10000, early_stop=True)
    flow.save('physical_benchmarks/hep_clustermaf.pkl')

lps = flow.log_prob(data_test.astype(np.float32))
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))
