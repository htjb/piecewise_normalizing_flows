import numpy as np
import matplotlib.pyplot as plt
from target_dists_stimper import RingMixture
from margarine.maf import MAF
from margarine.clustered import clusterMAF
import torch
from scipy.special import logsumexp
from matplotlib import rc
import matplotlib as mpl
from sklearn.cluster import (KMeans,
                                MeanShift, SpectralClustering,
                                AgglomerativeClustering,
                                Birch, MiniBatchKMeans)
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import normflows as nf
import larsflow as lf
import torch
import pickle
import os

# figure formatting
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', 
     '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [
#    r'\usepackage{amsmath}',
#    r'\usepackage{amssymb}']
#rc('font', family='serif')
#rc('font', serif='cm')
rc('savefig', pad_inches=0.05)

def create_model(p, base='gauss'):

    """this function and the next are taken from Vincent Stimpers work
    on realNVPs with resampled bases
    https://github.com/VincentStimper/resampled-base-flows"""

    # Set up model

    # Define flows
    K = 8
    torch.manual_seed(10)

    latent_size = 2
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        param_map = nf.nets.MLP([latent_size // 2, 17, 17, latent_size], init_zeros=True)
        flows += [nf.flows.AffineCouplingBlock(param_map)]
        flows += [nf.flows.Permute(latent_size, mode='swap')]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set prior and q0
    if base == 'resampled':
        a = nf.nets.MLP([latent_size, 128, 128, 1], output_fn="sigmoid")
        q0 = lf.distributions.ResampledGaussian(latent_size, a, 100, 0.1, trainable=False)
    elif base == 'gaussian_mixture':
        n_modes = 10
        q0 = nf.distributions.GaussianMixture(n_modes, latent_size, trainable=True,
                                              loc=(np.random.rand(n_modes, latent_size) - 0.5) * 5,
                                              scale=0.5 * np.ones((n_modes, latent_size)))
    elif base == 'gauss':
        q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
    else:
        raise NotImplementedError('This base distribution is not implemented.')

    # Construct flow model
    model = lf.NormalizingFlow(q0=q0, flows=flows, p=p)

    # Move model on GPU if available
    return model.to(device)

def train(model, max_iter=20000, num_samples=2 ** 10, lr=1e-3, weight_decay=1e-3, 
          q0_weight_decay=1e-4):
    """
    train() has been modified to include early stopping
    This is the train for the realNVP from stimper et al.
    """
    # Do mixed precision training
    optimizer = torch.optim.Adam(model.parameters(),  lr=lr, weight_decay=weight_decay)
    model.train()

    x = model.p.sample(num_samples)
    w = np.ones(len(x))
    x_train, x_test, w_train, w_test = train_test_split(x, w, test_size=0.2)

    train_loss = []
    test_loss = []
    c = 0
    for it in tqdm(range(max_iter)):

        loss = model.forward_kld(x_train)
        train_loss.append(loss)
        test_loss.append(model.forward_kld(x_test).detach())

        loss.backward()
        optimizer.step()

        # Clear gradients
        nf.utils.clear_grad(model)

        c += 1
        if it == 0:
            minimum_loss = test_loss[-1]
            minimum_epoch = it
            minimum_model = None
        else:
            if test_loss[-1] < minimum_loss:
                minimum_loss = test_loss[-1]
                minimum_epoch = it
                minimum_model = model
                c = 0
        #print(i, minimum_epoch, minimum_loss.numpy(), test_loss[-1].numpy())
        if minimum_model:
            if c == round((max_iter/100)*2):
                print('Early stopped. Epochs used = ' + str(it))
                return minimum_model

def get_cluster_number(cluster_algorithm, samples):

    ks = np.arange(2, 21)
    losses = []
    for k in ks:
        cluster_algorithm = cluster_algorithm(k, random_state=0)
        labels = cluster_algorithm.fit(samples).predict(samples)
        losses.append(-silhouette_score(s, labels))
    losses = np.array(losses)
    minimum_index = np.argmin(losses)
    cluster_number = ks[minimum_index]

    clu = cluster_algorithm(cluster_number, random_state=0)
    cluster_labels = cluster_algorithm.fit(samples).predict(samples)
    return cluster_number, cluster_labels

def mask_arr(arr):
    return arr[np.isfinite(arr)], np.isfinite(arr)

def calc_kl(samples, Flow, base):
    """Calculate KL divergences for the MAFs"""
    
    target_logprob = base.log_prob(torch.from_numpy(samples)).numpy()
    logprob = Flow.log_prob(samples)
    logprob, mask = mask_arr(logprob)
    target_logprob = target_logprob[mask]
    logprob -= logsumexp(logprob)
    target_logprob -= logsumexp(target_logprob)
    delta_logprob = target_logprob - logprob
    kldiv = np.mean(delta_logprob)

    kl_error = np.std(delta_logprob)/np.sqrt(len(delta_logprob))
    return kldiv, kl_error

device = torch.device('cpu')

nsample= 10000
kl_nsample= 10000
pEpochs = 20000
epochs = 20000

base = 'clustering/'
if not os.path.exists(base):
    os.mkdir(base)

cluster_algorithms = ['maf', 'kmeans', 'minbatchkmeans', 'mean_shift', 'spectral_clustering',
                      'agglomerative_clustering', 'birch']

# repeating for N times to get errors on KLs
kls_repeats, errorkl_repeats = [], []
for d in range(10):
    kls, kl_errors = [], []
    fig, axes = plt.subplots(3, 3, figsize=(5, 5))

    # generate samples with Stimpers RingMixture model
    rm = RingMixture()
    s = rm.sample(nsample).numpy()
    axes[0, 0].hist2d(s[:, 0], s[:, 1], bins=80, cmap='Blues')
    axes[0, 0].set_title('Target')

    try:
        sAFlow = MAF.load(base + "rm_single_maf_" + str(d) + ".pkl")
    except:
        sAFlow = MAF(s)
        sAFlow.train(epochs, early_stop=True)
        sAFlow.save(base + "rm_single_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kl, kl_error = calc_kl(s, sAFlow, rm)
    kls.append(kl)
    kl_errors.append(kl_error)
    axes[0, 1].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')
    axes[0, 1].set_title('MAF')

    try:
        rmrealnvp = pickle.load(open(base + "rm_realnvp_resampled_base_" + str(d) + ".pkl","rb"))
        print('successfully loaded rm realnvp')
    except:
        rmrealnvp = create_model(rm, 'resampled')
        rmrealnvp = train(rmrealnvp)
        file = open(base + "rm_realnvp_resampled_base_" + str(d) + ".pkl","wb")
        pickle.dump(rmrealnvp,file)
    samples = rmrealnvp.sample(kl_nsample)[0]#.detach().numpy()
    samples = samples.detach().numpy()
    axes[0, 2].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')
    axes[0, 2].set_title('RealNVP')

    # kmeans flow for ring model
    try:
        sAFlow = clusterMAF.load(base + "rm_kmeans_maf_" + str(d) + ".pkl")
    except:
        _ = clusterMAF(s)
        nn = int((17424/_.cluster_number/2904)//1 + 1)
        print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

        kmeans = KMeans(_.cluster_number, random_state=0)
        labels = kmeans.fit(s).predict(s)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=_.cluster_number, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "rm_kmeans_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kl, kl_error = calc_kl(s, sAFlow, rm)
    kls.append(kl)
    kl_errors.append(kl_error)
    axes[1, 0].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')
    axes[1, 0].set_title('K-Means')
    

    # minibatch kmeans flow for ring model
    try:
        SAFlow = clusterMAF.load(base + "rm_minibatch_kmeans_maf_" + str(d) + ".pkl")
    except:
        ks = np.arange(2, 21)
        losses = []
        for k in ks:
            sc = MiniBatchKMeans(k, random_state=0)
            labels = sc.fit_predict(s)
            losses.append(-silhouette_score(s, labels))
        losses = np.array(losses)
        minimum_index = np.argmin(losses)
        cluster_number = ks[minimum_index]

        clu = MiniBatchKMeans(cluster_number, random_state=0)
        cluster_labels = clu.fit_predict(s)
        n_clusters = len(np.unique(cluster_labels))
        nn = int((17424/n_clusters/2904)//1 + 1)

        print('number clusters: ', n_clusters, ' number_networks: ', nn)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=n_clusters, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "rm_minibatch_kmeans_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kl, kl_error = calc_kl(s, sAFlow, rm)
    kls.append(kl)
    kl_errors.append(kl_error)
    axes[1, 1].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')
    axes[1, 1].set_title('MiniBatchKMeans')

    
    # mean_shift flow for ring model
    try:
        sAFlow = clusterMAF.load(base + "rm_mean_shift_maf_" + str(d) + ".pkl")
    except:

        mean_shift = MeanShift(bandwidth=0.5, bin_seeding=True)
        labels = mean_shift.fit(s).predict(s)
        n_clusters = len(np.unique(labels))
        nn = int((17424/n_clusters/2904)//1 + 1)
        print('number clusters: ', n_clusters, ' number_networks: ', nn)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=n_clusters, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "rm_mean_shift_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kl, kl_error = calc_kl(s, sAFlow, rm)
    kls.append(kl)
    kl_errors.append(kl_error)
    axes[1, 2].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')
    axes[1, 2].set_title('MeanShift')

    # spectral_clustering flow for ring model
    try:
        sAFlow = clusterMAF.load(base + "rm_spectral_clustering_maf_" + str(d) + ".pkl")
    except:
        ks = np.arange(2, 21)
        losses = []
        for k in ks:
            sc = SpectralClustering(k, random_state=0)
            labels = sc.fit_predict(s)
            losses.append(-silhouette_score(s, labels))
        losses = np.array(losses)
        minimum_index = np.argmin(losses)
        cluster_number = ks[minimum_index]

        clu = SpectralClustering(cluster_number, random_state=0)
        cluster_labels = clu.fit_predict(s)
        n_clusters = len(np.unique(cluster_labels))
        nn = int((17424/n_clusters/2904)//1 + 1)

        print('number clusters: ', n_clusters, ' number_networks: ', nn)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=n_clusters, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "rm_spectral_clustering_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kl, kl_error = calc_kl(s, sAFlow, rm)
    kls.append(kl)
    kl_errors.append(kl_error)
    axes[2, 0].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')
    axes[2, 0].set_title('Spectral \n Clustering')
    

    # agglomerative_clustering flow for ring model
    try:
        sAFlow = clusterMAF.load(base + "rm_agglomerative_clustering_maf_" + str(d) + ".pkl")
    except:
        ks = np.arange(2, 21)
        losses = []
        for k in ks:
            ac = AgglomerativeClustering(k)
            labels = ac.fit_predict(s)
            losses.append(-silhouette_score(s, labels))
        losses = np.array(losses)
        minimum_index = np.argmin(losses)
        cluster_number = ks[minimum_index]

        clu = AgglomerativeClustering(cluster_number)
        cluster_labels = clu.fit_predict(s)
        n_clusters = len(np.unique(cluster_labels))
        nn = int((17424/n_clusters/2904)//1 + 1)

        print('number clusters: ', n_clusters, ' number_networks: ', nn)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=n_clusters, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "rm_agglomerative_clustering_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kl, kl_error = calc_kl(s, sAFlow, rm)
    kls.append(kl)
    kl_errors.append(kl_error)
    axes[2, 1].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')
    axes[2, 1].set_title('Agglomerative \n Clustering')


    # Birch flow for ring model
    try:
        sAFlow = clusterMAF.load(base + "rm_birch_clustering_maf_" + str(d) + ".pkl")
    except:
        ks = np.arange(2, 21)
        losses = []
        for k in ks:
            ac = Birch(n_clusters=k)
            labels = ac.fit_predict(s)
            losses.append(-silhouette_score(s, labels))
        losses = np.array(losses)
        minimum_index = np.argmin(losses)
        cluster_number = ks[minimum_index]

        clu = Birch(n_clusters=cluster_number)
        cluster_labels = clu.fit_predict(s)
        n_clusters = len(np.unique(cluster_labels))
        nn = int((17424/n_clusters/2904)//1 + 1)

        print('number clusters: ', n_clusters, ' number_networks: ', nn)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=n_clusters, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "rm_birch_clustering_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kl, kl_error = calc_kl(s, sAFlow, rm)
    kls.append(kl)
    kl_errors.append(kl_error)
    axes[2, 2].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')
    axes[2, 2].set_title('Birch')

    for i in range(len(axes)):
        for j in range(axes.shape[-1]):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    plt.savefig(base + 'clustering_' + str(d) + '.pdf')
    #plt.show()
    plt.close()

    kls_repeats.append(kls)
    errorkl_repeats.append(kl_errors)

kls_repeats = np.array(kls_repeats)
errorkl_repeats = np.array(errorkl_repeats)
kls = np.mean(kls_repeats, axis=0)
kl_errors = np.sqrt(np.sum(errorkl_repeats**2, axis=0))

for i in range(kls.shape[0]):
    print(cluster_algorithms[i], kls[i], kl_errors[i])
