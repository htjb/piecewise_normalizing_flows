import numpy as np
from mpi4py import MPI
from margarine.maf import MAF
from margarine.clustered import clusterMAF
import math
from sklearn.cluster import KMeans, MeanShift, Birch, MiniBatchKMeans
from sklearn.metrics import silhouette_score

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank, size)

data_train = np.load('raw_physical_data/gas/data_train.npy')
data_test = np.load('raw_physical_data/gas/data_test.npy')

if rank == 0:

	try:
		flow = MAF.load('physical_benchmarks/gas_maf.pkl')
	except FileNotFoundError:
		flow = MAF(data_train)
		flow.train(10000, early_stop=True)
		flow.save('physical_benchmarks/gas_maf.pkl')
	print('Log:flow loaded')

comm.Barrier()

if rank != 0:
	flow = MAF.load('physical_benchmarks/gas_maf.pkl')

print('Log:flow loaded now doing log_prob')

perrank = len(data_test)//size

print(rank*perrank, (rank+1)*perrank)

lps = []
for i in range(rank*perrank, (rank+1)*perrank):
	try:
		lps.append(flow.log_prob(data_test[i].astype(np.float32)))
	except:
		print('Log:No more data')
lps = np.array(lps)
comm.Barrier()

lps = comm.allgather(lps)[0]

if rank == 0:

	mask = np.isfinite(lps)
	print(np.mean(lps[mask]))
	print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))
	print('Log:-'*20 + 'global maf done' + '_'*20)
	
	try:
		flow = clusterMAF.load('physical_benchmarks/gas_clustermaf.pkl')
	except FileNotFoundError:
		kmax, k = 20, 2
		losses = []
		s_not_improv = 0
		while k < kmax and s_not_improv < 2:
			print(k)
			dt = data_train.copy()[np.random.choice(len(data_train), 25000)]
			sc = MiniBatchKMeans(k, random_state=0)
			labels = sc.fit_predict(dt)
			losses.append(-silhouette_score(dt, labels))
			print(losses[-1])
			if len(losses) > 2:
				if losses[-1] > losses[-2]:
					s_not_improv += 1
			k += 1

		losses = np.array(losses)
		minimum_index = np.argmin(losses)
		cluster_number = minimum_index + 2

		clu = MiniBatchKMeans(cluster_number, random_state=0)
		cluster_labels = clu.fit_predict(data_train)
		n_clusters = len(np.unique(cluster_labels))
		nn = int((17424/n_clusters/2904)//1 + 1)
		print('Log:number clusters: ', n_clusters, ' number_networks: ', nn)

		flow = clusterMAF(data_train, cluster_number=n_clusters, 
				cluster_labels=cluster_labels, number_networks=nn)
		flow.train(10000, early_stop=True)
		flow.save('physical_benchmarks/gas_clustermaf.pkl')

comm.Barrier()

if rank != 0:
	flow = clusterMAF.load('physical_benchmarks/gas_clustermaf.pkl')

print('Log:clustered flow loaded now doing log_prob')

print(rank*perrank, (rank+1)*perrank)
lps = []
for i in range((rank)*perrank, (rank+1)*perrank):
	try:
		lps.append(flow.log_prob(data_test[i].astype(np.float32)))
	except:
		print('Log:No more data')
lps = np.array(lps)

comm.Barrier()

lps = comm.allgather(lps)[0]

if rank == 0:
	mask = np.isfinite(lps)
	print(np.mean(lps[mask]))
	print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))
