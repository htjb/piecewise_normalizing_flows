import numpy as np
import pickle
from margarine.maf import MAF
from margarine.clustered import clusterMAF
from target_dists_stimper import TwoMoons, CircularGaussianMixture, RingMixture
from sklearn.cluster import KMeans
import time

nsample= 10000
pEpochs = 10000
epochs = 10000

cgm = CircularGaussianMixture()
tm = TwoMoons()
rm = RingMixture()

base_dir = 'timing/'

try:
    cgm_maf_epochs = list(np.loadtxt(base_dir + 'cgm_maf_epochs.txt'))
    cgm_clustermaf_epochs = list(np.loadtxt(base_dir + 'cgm_clustermaf_epochs.txt'))
    tm_maf_epochs = list(np.loadtxt(base_dir + 'tm_maf_epochs.txt'))
    tm_clustermaf_epochs = list(np.loadtxt(base_dir + 'tm_clustermaf_epochs.txt'))
    rm_maf_epochs = list(np.loadtxt(base_dir + 'rm_maf_epochs.txt'))
    rm_clustermaf_epochs = list(np.loadtxt(base_dir + 'rm_clustermaf_epochs.txt'))

    cgm_maf_time = list(np.loadtxt(base_dir + 'cgm_maf_time.txt'))
    cgm_clustermaf_time = list(np.loadtxt(base_dir + 'cgm_clustermaf_time.txt'))
    tm_maf_time = list(np.loadtxt(base_dir + 'tm_maf_time.txt'))
    tm_clustermaf_time = list(np.loadtxt(base_dir + 'tm_clustermaf_time.txt'))
    rm_maf_time = list(np.loadtxt(base_dir + 'rm_maf_time.txt'))
    rm_clustermaf_time = list(np.loadtxt(base_dir + 'rm_clustermaf_time.txt'))
    
    cgm_maf_cost = list(np.loadtxt(base_dir + 'cgm_maf_cost.txt'))
    cgm_clustermaf_cost = list(np.loadtxt(base_dir + 'cgm_clustermaf_cost.txt'))
    tm_maf_cost = list(np.loadtxt(base_dir + 'tm_maf_cost.txt'))
    tm_clustermaf_cost = list(np.loadtxt(base_dir + 'tm_clustermaf_cost.txt'))
    rm_maf_cost = list(np.loadtxt(base_dir + 'rm_maf_cost.txt'))
    rm_clustermaf_cost = list(np.loadtxt(base_dir + 'rm_clustermaf_cost.txt'))
except:
    cgm_maf_epochs, tm_maf_epochs, rm_maf_epochs = [], [], []
    cgm_clustermaf_epochs, tm_clustermaf_epochs, rm_clustermaf_epochs = [], [], []
    cgm_maf_time, tm_maf_time, rm_maf_time = [], [], []
    cgm_clustermaf_time, tm_clustermaf_time, rm_clustermaf_time = [], [], []
    #cgm_cluster_number, tm_cluster_number, rm_cluster_number = [], [], []
    cgm_maf_cost, tm_maf_cost, rm_maf_cost = [], [], []
    cgm_clustermaf_cost, tm_clustermaf_cost, rm_clustermaf_cost = [], [], []

for d in range(len(cgm_maf_cost), 3):

    s = cgm.sample(nsample).numpy()

    # noraml maf for circle of gaussian
    sAFlow = MAF(s)
    start = time.time()
    sAFlow.train(epochs, early_stop=True)
    e = time.time()
    cgm_maf_time.append(e-start)
    cgm_maf_epochs.append(len(sAFlow.loss_history))
    hyps = 0
    for j, made in enumerate(sAFlow.mades):
        hyps += made.count_params()
    cgm_maf_cost.append(len(sAFlow.loss_history)*len(sAFlow.theta)*hyps)
    
    _ = clusterMAF(s)
    #cgm_cluster_number.append(_.cluster_number)
    nn = int((17424/_.cluster_number/2904)//1 + 1)
    print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

    kmeans = KMeans(_.cluster_number, random_state=0)
    labels = kmeans.fit(s).predict(s)
    sAFlow = clusterMAF(s, cluster_labels=labels, 
                        cluster_number=_.cluster_number, number_networks=nn)
    start = time.time()
    sAFlow.train(pEpochs, early_stop=True)
    e = time.time()
    cgm_clustermaf_time.append(e-start)
    cgm_clustermaf_epochs.append(np.sum([len(sAFlow.flow[i].loss_history) for i in range(len(sAFlow.flow))]))
    cost = []
    for i in range(len(sAFlow.flow)):
        hyps = 0
        for j, made in enumerate(sAFlow.flow[i].mades):
            hyps += made.count_params()
        cost.append(len(sAFlow.flow[i].loss_history)*len(sAFlow.flow[i].theta)*hyps)
    cgm_clustermaf_cost.append(np.sum(cost))

    s = tm.sample(nsample).numpy()

    # noraml maf for circle of gaussian
    sAFlow = MAF(s)
    start = time.time()
    sAFlow.train(epochs, early_stop=True)
    e = time.time()
    tm_maf_time.append(e-start)
    tm_maf_epochs.append(len(sAFlow.loss_history))
    hyps = 0
    for j, made in enumerate(sAFlow.mades):
        hyps += made.count_params()
    tm_maf_cost.append(len(sAFlow.loss_history)*len(sAFlow.theta)*hyps)

    _ = clusterMAF(s)
    #tm_cluster_number.append(_.cluster_number)
    nn = int((17424/_.cluster_number/2904)//1 + 1)
    print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

    kmeans = KMeans(_.cluster_number, random_state=0)
    labels = kmeans.fit(s).predict(s)
    sAFlow = clusterMAF(s, cluster_labels=labels, 
                        cluster_number=_.cluster_number, number_networks=nn)
    start = time.time()
    sAFlow.train(pEpochs, early_stop=True)
    e = time.time()
    tm_clustermaf_time.append(e-start)
    tm_clustermaf_epochs.append(np.sum([len(sAFlow.flow[i].loss_history) for i in range(len(sAFlow.flow))]))
    cost = []
    for i in range(len(sAFlow.flow)):
        hyps = 0
        for j, made in enumerate(sAFlow.flow[i].mades):
            hyps += made.count_params()
        cost.append(len(sAFlow.flow[i].loss_history)*len(sAFlow.flow[i].theta)*hyps)
    tm_clustermaf_cost.append(np.sum(cost))

    s = rm.sample(nsample).numpy()

    # noraml maf for circle of gaussian
    sAFlow = MAF(s)
    start = time.time()
    sAFlow.train(epochs, early_stop=True)
    e = time.time()
    rm_maf_time.append(e-start)
    rm_maf_epochs.append(len(sAFlow.loss_history))
    hyps = 0
    for j, made in enumerate(sAFlow.mades):
        hyps += made.count_params()
    rm_maf_cost.append(len(sAFlow.loss_history)*len(sAFlow.theta)*hyps)

    _ = clusterMAF(s)
    #rm_cluster_number.append(_.cluster_number)
    nn = int((17424/_.cluster_number/2904)//1 + 1)
    print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

    kmeans = KMeans(_.cluster_number, random_state=0)
    labels = kmeans.fit(s).predict(s)
    sAFlow = clusterMAF(s, cluster_labels=labels, 
                        cluster_number=_.cluster_number, number_networks=nn)
    start = time.time()
    sAFlow.train(pEpochs, early_stop=True)
    e = time.time()
    rm_clustermaf_time.append(e-start)
    rm_clustermaf_epochs.append(np.sum([len(sAFlow.flow[i].loss_history) for i in range(len(sAFlow.flow))]))
    cost = []
    for i in range(len(sAFlow.flow)):
        hyps = 0
        for j, made in enumerate(sAFlow.flow[i].mades):
            hyps += made.count_params()
        cost.append(len(sAFlow.flow[i].loss_history)*len(sAFlow.flow[i].theta)*hyps)
    rm_clustermaf_cost.append(np.sum(cost))

    np.savetxt(base_dir + 'cgm_maf_epochs.txt', cgm_maf_epochs)
    np.savetxt(base_dir + 'cgm_clustermaf_epochs.txt', cgm_clustermaf_epochs)
    np.savetxt(base_dir + 'tm_maf_epochs.txt', tm_maf_epochs)
    np.savetxt(base_dir + 'tm_clustermaf_epochs.txt', tm_clustermaf_epochs)
    np.savetxt(base_dir + 'rm_maf_epochs.txt', rm_maf_epochs)
    np.savetxt(base_dir + 'rm_clustermaf_epochs.txt', rm_clustermaf_epochs)

    np.savetxt(base_dir + 'cgm_maf_time.txt', cgm_maf_time)
    np.savetxt(base_dir + 'cgm_clustermaf_time.txt', cgm_clustermaf_time)
    np.savetxt(base_dir + 'tm_maf_time.txt', tm_maf_time)
    np.savetxt(base_dir + 'tm_clustermaf_time.txt', tm_clustermaf_time)
    np.savetxt(base_dir + 'rm_maf_time.txt', rm_maf_time)
    np.savetxt(base_dir + 'rm_clustermaf_time.txt', rm_clustermaf_time)

    np.savetxt(base_dir + 'cgm_maf_cost.txt', cgm_maf_cost)
    np.savetxt(base_dir + 'cgm_clustermaf_cost.txt', cgm_clustermaf_cost)
    np.savetxt(base_dir + 'tm_maf_cost.txt', tm_maf_cost)
    np.savetxt(base_dir + 'tm_clustermaf_cost.txt', tm_clustermaf_cost)
    np.savetxt(base_dir + 'rm_maf_cost.txt', rm_maf_cost)
    np.savetxt(base_dir + 'rm_clustermaf_cost.txt', rm_clustermaf_cost)

    


print('cgm maf epochs: ', np.mean(cgm_maf_epochs), ' +/- ', np.std(cgm_maf_epochs)/np.sqrt(5))
print('cgm clustermaf epochs: ', np.mean(cgm_clustermaf_epochs),  ' +/- ', np.std(cgm_clustermaf_epochs)/np.sqrt(5))

print('tm maf epochs: ', np.mean(tm_maf_epochs), ' +/- ', np.std(tm_maf_epochs)/np.sqrt(5))
print('tm clustermaf epochs: ', np.mean(tm_clustermaf_epochs),  ' +/- ', np.std(tm_clustermaf_epochs)/np.sqrt(5))

print('rm maf epochs: ', np.mean(rm_maf_epochs), ' +/- ', np.std(rm_maf_epochs)/np.sqrt(5))
print('rm clustermaf epochs: ', np.mean(rm_clustermaf_epochs),  ' +/- ', np.std(rm_clustermaf_epochs)/np.sqrt(5))

print('cgm maf time: ', np.mean(cgm_maf_time), ' +/- ', np.std(cgm_maf_time)/np.sqrt(5))
print('cgm clustermaf time: ', np.mean(cgm_clustermaf_time),  ' +/- ', np.std(cgm_clustermaf_time)/np.sqrt(5))

print('tm maf time: ', np.mean(tm_maf_time), ' +/- ', np.std(tm_maf_time)/np.sqrt(5))
print('tm clustermaf time: ', np.mean(tm_clustermaf_time),  ' +/- ', np.std(tm_clustermaf_time)/np.sqrt(5))

print('rm maf time: ', np.mean(rm_maf_time), ' +/- ', np.std(rm_maf_time)/np.sqrt(5))
print('rm clustermaf time: ', np.mean(rm_clustermaf_time),  ' +/- ', np.std(rm_clustermaf_time)/np.sqrt(5))

print('cgm maf cost: ', np.mean(cgm_maf_cost), ' +/- ', np.std(cgm_maf_cost)/np.sqrt(5))
print('cgm clustermaf cost: ', np.mean(cgm_clustermaf_cost),  ' +/- ', np.std(cgm_clustermaf_cost)/np.sqrt(5))

print('tm maf cost: ', np.mean(tm_maf_cost), ' +/- ', np.std(tm_maf_cost)/np.sqrt(5))
print('tm clustermaf cost: ', np.mean(tm_clustermaf_cost),  ' +/- ', np.std(tm_clustermaf_cost)/np.sqrt(5))

print('rm maf cost: ', np.mean(rm_maf_cost), ' +/- ', np.std(rm_maf_cost)/np.sqrt(5))
print('rm clustermaf cost: ', np.mean(rm_clustermaf_cost),  ' +/- ', np.std(rm_clustermaf_cost)/np.sqrt(5))

print('cgm clustermaf cost: ', np.mean(cgm_clustermaf_cost/cgm_maf_cost), ' +/- ', 
      np.std(cgm_clustermaf_cost/cgm_maf_cost)/np.sqrt(5))
print('tm clustermaf cost: ', np.mean(tm_clustermaf_cost/tm_maf_cost), ' +/- ',
        np.std(tm_clustermaf_cost/tm_maf_cost)/np.sqrt(5))
print('rm clustermaf cost: ', np.mean(rm_clustermaf_cost/rm_maf_cost), ' +/- ',
       np.std(rm_clustermaf_cost/rm_maf_cost)/np.sqrt(5))

"""cgm_cluster_number = np.array(cgm_cluster_number)
tm_cluster_number = np.array(tm_cluster_number)
rm_cluster_number = np.array(rm_cluster_number)

cgm_clsutermaf_time = np.array(cgm_clustermaf_time)
tm_clustermaf_time = np.array(tm_clustermaf_time)
rm_clustermaf_time = np.array(rm_clustermaf_time)

print('cgm time per cluster: ', np.mean(cgm_clustermaf_time/cgm_cluster_number), ' +/- ',  np.std(cgm_clustermaf_time/cgm_cluster_number)/np.sqrt(5))
print('tm time per cluster: ', np.mean(tm_clustermaf_time/tm_cluster_number), ' +/- ',  np.std(tm_clustermaf_time/tm_cluster_number)/np.sqrt(5))
print('rm time per cluster: ', np.mean(rm_clustermaf_time/rm_cluster_number), ' +/- ',  np.std(rm_clustermaf_time/rm_cluster_number)/np.sqrt(5))
"""