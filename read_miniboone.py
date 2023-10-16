import numpy as np


def load_data(root_path):
    # NOTE: To remember how the pre-processing was done.
    # data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
    # print data.head()
    # data = data.as_matrix()
    # # Remove some random outliers
    # indices = (data[:, 0] < -100)
    # data = data[~indices]
    #
    # i = 0
    # # Remove any features that have too many re-occuring real values.
    # features_to_remove = []
    # for feature in data.T:
    #     c = Counter(feature)
    #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
    #     if max_count > 5:
    #         features_to_remove.append(i)
    #     i += 1
    # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
    # np.save("~/data/miniboone/data.npy", data)

    data = np.load(root_path)
    N_test = int(0.1*data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1*data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(root_path):

    data_train, data_validate, data_test = load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_validate, data_test

data_train, dv, data_test = load_data_normalised('raw_physical_data/miniboone/data.npy')

from margarine.clustered import clusterMAF
from margarine.maf import MAF
import math
from sklearn.cluster import KMeans

try:
    flow = MAF.load('miniboone_maf.pkl')
except FileNotFoundError:
    flow = MAF(data_train)
    flow.train(10000, early_stop=True)
    flow.save('miniboone_maf.pkl')

lps = flow.log_prob(data_test.astype(np.float32))
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))

try:
    flow = clusterMAF.load('miniboone_clustermaf.pkl')
except:
    _ = clusterMAF(data_train)
    nn = math.ceil(17424/_.cluster_number/2904)
    print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

    kmeans = KMeans(_.cluster_number, random_state=0)
    labels = kmeans.fit(data_train).predict(data_train)

    flow = clusterMAF(data_train, cluster_number=_.cluster_number, cluster_labels=labels, number_networks=nn)
    flow.train(10000, early_stop=True)
    flow.save('miniboone_clustermaf.pkl')

lps = flow.log_prob(data_test.astype(np.float32))
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))
