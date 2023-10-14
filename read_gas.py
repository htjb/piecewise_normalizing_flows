from mpi4py import MPI
import numpy as np
from datasets.gas import GAS
from margarine.maf import MAF
from margarine.clustered import clusterMAF
import math
from sklearn.cluster import KMeans

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	gas = GAS()

	data_train = gas.trn.x
	data_test = gas.tst.x

	try:
		flow = MAF.load('gas_maf.pkl')
	except FileNotFoundError:
		flow = MAF(data_train)
		flow.train(10000, early_stop=True)
		flow.save('gas_maf.pkl')

comm.bcast(data_train, root=0)
comm.bcast(data_test, root=0)
comm.Barrier()

if rank != 0:
	flow = MAF.load('gas_maf.pkl')

perrank = len(data_test)//size

lps = []
for i in range(rank*perrank, (rank+1)*perrank):
	try:
		lps.append(flow.log_prob(data_test[i].astype(np.float32)))
	except:
		print('No more data')
lps = np.array(lps)
comm.Barrier()

lps = comm.allgather(lps)

if rank == 0:

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

comm.Barrier()

if rank != 0:
	flow = clusterMAF.load('gas_clustermaf.pkl')

lps = []
for i in range((rank)*perrank, (rank+1)*perrank):
	try:
		lps.append(flow.log_prob(data_test[i].astype(np.float32)))
	except:
		print('No more data')
lps = np.array(lps)

comm.Barrier()

lps = comm.allgather(lps)

if rank == 0:
	mask = np.isfinite(lps)
	print(np.mean(lps[mask]))
	print(np.std(lps[mask]) / np.sqrt(len(lps[mask])))
