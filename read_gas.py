import numpy as np
from datasets.gas import GAS
from margarine.maf import MAF
from margarine.clustered import clusterMAF
import math
from sklearn.cluster import KMeans

gas = GAS()

data_train = gas.trn.x
data_test = gas.tst.x

try:
	flow = MAF.load('gas_maf.pkl')
except FileNotFoundError:
	flow = MAF(data_train)
	flow.train(10000, early_stop=True)
	flow.save('gas_maf.pkl')

lps = flow.log_prob(data_test.astype(np.float32))
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))
print('-'*20 + 'global maf done' + '_'*20)

try:
	flow = clusterMAF.load('gas_clustermaf.plkl')
except FileNotFoundError:
	_ = clusterMAF(data_train)
	nn = math.ciel(17424/_.cluster_number/2904)
	print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

	kmeans = KMeans(_.cluster_number, random_state=0)
	labels = kmeans.fit(data_train).predict(data_train)

	flow = clusterMAF(data_train, cluster_number=_.cluster_number, 
			cluster_labels=labels, number_networks=nn)
	flow.train(10000, early_stop=True)
	flow.save('gas_clustermaf.pkl')

lps = flow.log_prob(data_test.astype(np.float32))
mask = np.isfinite(lps)
print(np.mean(lps[mask]))
print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))
