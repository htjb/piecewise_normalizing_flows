import numpy as np
from margarine.maf import MAF
from margarine.clustered import clusterMAF
import math
from sklearn.cluster import KMeans

np.random.seed(1420)

data_train = np.load('raw_physical_data/gas/data_train.npy')
data_test = np.load('raw_physical_data/gas/data_test.npy')

data_test = data_test[np.random.choice(len(data_test), 10000)]
data_train = data_train[np.random.choice(len(data_train), 25000)]
print('data shape: ', data_train.shape)

"""from anesthetic import MCMCSamples
import matplotlib.pyplot as plt	

data = MCMCSamples(data=data_train)
data.plot_2d(data.columns[:8])
plt.show()"""


try:
	flow = MAF.load('physical_benchmarks/gas_maf.pkl')
except FileNotFoundError:
	flow = MAF(data_train)
	flow.train(10000, early_stop=True)
	flow.save('physical_benchmarks/gas_maf.pkl')

lps = flow.log_prob(data_test.astype(np.float32))
lps = np.array(lps)
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask]))*2)

try:
	flow = clusterMAF.load('physical_benchmarks/gas_clustermaf.pkl')
except FileNotFoundError:
	_ = clusterMAF(data_train)
	nn = math.ceil(17424/_.cluster_number/2904)
	print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

	kmeans = KMeans(_.cluster_number, random_state=0)
	labels = kmeans.fit(data_train).predict(data_train)

	flow = clusterMAF(data_train, cluster_number=_.cluster_number, 
			cluster_labels=labels, number_networks=nn)
	flow.train(10000, early_stop=True)
	flow.save('physical_benchmarks/gas_clustermaf.pkl')

	
lps = flow.log_prob(data_test.astype(np.float32))
lps = np.array(lps)
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask]))*2)
