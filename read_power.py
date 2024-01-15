import numpy as np


def load_data():
    return np.load('raw_physical_data/power/data.npy')


def load_data_split_with_noise():

    rng = np.random.RandomState(42)

    data = load_data()
    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    ############################
    # Add noise
    ############################
    # global_intensity_noise = 0.1*rng.rand(N, 1)
    voltage_noise = 0.01*rng.rand(N, 1)
    # grp_noise = 0.001*rng.rand(N, 1)
    gap_noise = 0.001*rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
    # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise

    N_test = int(0.1*data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1*data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised():

    data_train, data_validate, data_test = load_data_split_with_noise()
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu)/s
    data_validate = (data_validate - mu)/s
    data_test = (data_test - mu)/s

    return data_train, data_validate, data_test

np.random.seed(42)

data_train, dv, data_test = load_data_normalised()
data_test = data_test[np.random.choice(len(data_test), 10000)]
data_train = data_train[np.random.choice(len(data_train), 25000)]
print('data shape: ', data_train.shape)


from margarine.clustered import clusterMAF
from margarine.maf import MAF
import math
from sklearn.cluster import KMeans

try:
    flow = MAF.load('physical_benchmarks/power_maf.pkl')
except FileNotFoundError:
    flow = MAF(data_train)
    flow.train(10000, early_stop=True)
    flow.save('physical_benchmarks/power_maf.pkl')

lps = flow.log_prob(data_test.astype(np.float32))
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask]))*2)

try:
    flow = clusterMAF.load('physical_benchmarks/power_clustermaf.pkl')
except:
    _ = clusterMAF(data_train)
    nn = math.ceil(17424/_.cluster_number/2904)
    print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

    kmeans = KMeans(_.cluster_number, random_state=0)
    labels = kmeans.fit(data_train).predict(data_train)

    flow = clusterMAF(data_train, cluster_number=_.cluster_number, cluster_labels=labels, number_networks=nn)
    flow.train(10000, early_stop=True)
    flow.save('physical_benchmarks/power_clustermaf.pkl')

print('cluster number: ', flow.cluster_number)
lps = flow.log_prob(data_test.astype(np.float32))
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask]))*2)