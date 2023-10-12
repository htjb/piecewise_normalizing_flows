import numpy as np
import matplotlib.pyplot as plt
from target_dists_stimper import TwoMoons, CircularGaussianMixture, RingMixture
from margarine.maf import MAF
from margarine.clustered import clusterMAF
import torch
from scipy.special import logsumexp
from tensorflow import keras
import normflows as nf
import larsflow as lf
from tqdm import tqdm
from matplotlib import rc
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import pickle
from sklearn.cluster import KMeans
import os


# figure formatting
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color',
    ['ff7f00', '984ea3', '999999', '377eb8', 
     '4daf4a','f781bf', 'a65628', 'e41a1c', 'dede00'])
mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [
#    r'\usepackage{amsmath}',
#    r'\usepackage{amssymb}']
rc('font', family='serif')
rc('font', serif='cm')
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
    delta_logprob = logprob - target_logprob
    kldiv = np.mean(delta_logprob)

    kl_error = np.std(delta_logprob)/np.sqrt(len(delta_logprob))
    return kldiv, kl_error

device = torch.device('cpu')

# for the MAFs and piecewise MAFs
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=25,
    decay_rate=0.9)

nsample= 10000
kl_nsample= 10000
pEpochs = 20000
epochs = 20000

base = 'benchmark_duplicates/'
if not os.path.exists(base):
    os.mkdir(base)

# repeating for N times to get errors on KLs
tm_maf_kls, tm_nvp_kls, tm_clus_kls = [], [], []
rm_maf_kls, rm_nvp_kls, rm_clus_kls = [], [], []
cgm_maf_kls, cgm_nvp_kls, cgm_clus_kls = [], [], []
for d in range(10):
    fig, axes = plt.subplots(3, 4, figsize=(6.3, 5))

    # generate samples with Stimpers RingMixture model
    rm = RingMixture()
    s = rm.sample(nsample).numpy()
    axes[2, 0].hist2d(s[:, 0], s[:, 1], bins=80, cmap='Blues')

     # noraml maf for ring model
    try:
        sAFlow = MAF.load(base + "rm_single_maf_" + str(d) + ".pkl")
    except:
        sAFlow = MAF(s)
        sAFlow.train(epochs, early_stop=True)
        sAFlow.save(base + "rm_single_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kldiv, kl_error = calc_kl(samples, sAFlow, rm)
    rm_maf_kls.append([kldiv, kl_error])
    axes[2, 1].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')

    # cluster flow for ring model
    try:
        sAFlow = clusterMAF.load(base + "rm_cluster_maf_" + str(d) + ".pkl")
    except:
        _ = clusterMAF(s)
        nn = int((17424/_.cluster_number/2904)//1 + 1)
        print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

        kmeans = KMeans(_.cluster_number, random_state=0)
        labels = kmeans.fit(s).predict(s)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=_.cluster_number, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "rm_cluster_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kl, kl_error = calc_kl(samples, sAFlow, rm)
    rm_clus_kls.append([kl, kl_error])
    axes[2, 3].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')

    # realNVP with reasampled base for ring model
    try:
        model = pickle.load(open(base + "rm_realnvp_resampled_base_" + str(d) + ".pkl","rb"))
    except:
        model = create_model(rm, 'resampled')
        model = train(model)
        file = open(base + "rm_realnvp_resampled_base_" + str(d) + ".pkl","wb")
        pickle.dump(model,file)
    samples = model.sample(kl_nsample)[0]#.detach().numpy()
    logprob = model.log_prob(samples).detach().numpy()
    target_logprob = rm.log_prob(samples).detach().numpy()
    logprob, mask = mask_arr(logprob)
    target_logprob = target_logprob[mask]
    logprob -= logsumexp(logprob)
    target_logprob -= logsumexp(target_logprob)
    delta_logprob = logprob - target_logprob
    kldiv = np.mean(delta_logprob)
    kl_error = np.std(delta_logprob)/np.sqrt(len(delta_logprob))
    rm_nvp_kls.append([kldiv, kl_error])
    samples = samples.detach().numpy()
    axes[2, 2].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')

    tm = TwoMoons()

    s = tm.sample(nsample).numpy()
    axes[0, 0].hist2d(s[:, 0], s[:, 1], bins=80, cmap='Blues')

    # noraml maf for two moons
    try:
        flow = MAF.load(base + "tm_single_maf_" + str(d) + ".pkl")
    except:
        flow = MAF(s)
        flow.train(epochs, early_stop=True)
        flow.save(base + "tm_single_maf_" + str(d) + ".pkl")
    samples = flow.sample(kl_nsample).numpy()
    kldiv, kl_error = calc_kl(samples, flow, tm)
    tm_maf_kls.append([kldiv, kl_error])
    axes[0, 1].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')

    # cluster flow for two moons
    try:
        sAFlow = clusterMAF.load(base + "tm_cluster_maf_" + str(d) + ".pkl")
    except:
        _ = clusterMAF(s)
        nn = int((17424/_.cluster_number/2904)//1 + 1)
        print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

        kmeans = KMeans(_.cluster_number, random_state=0)
        labels = kmeans.fit(s).predict(s)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=_.cluster_number, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "tm_cluster_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kldiv, kl_error = calc_kl(samples, sAFlow, tm)
    tm_clus_kls.append([kldiv, kl_error])
    axes[0, 3].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')

    # realNVP with reasampled base for two moons
    try:
        model = pickle.load(open(base + "tm_realnvp_resampled_base_" + str(d) + ".pkl","rb"))
    except:
        model = create_model(tm, 'resampled')
        model = train(model)
        file = open(base + "tm_realnvp_resampled_base_" + str(d) + ".pkl","wb")
        pickle.dump(model,file)
    samples = model.sample(kl_nsample)[0]#.detach().numpy()
    logprob = model.log_prob(samples).detach().numpy()
    target_logprob = tm.log_prob(samples).detach().numpy()
    logprob, mask = mask_arr(logprob)
    target_logprob = target_logprob[mask]
    logprob -= logsumexp(logprob)
    target_logprob -= logsumexp(target_logprob)
    delta_logprob = logprob - target_logprob
    kldiv = np.mean(delta_logprob)
    kl_error = np.std(delta_logprob)/np.sqrt(len(delta_logprob))
    tm_nvp_kls.append([kldiv, kl_error])
    samples = samples.detach().numpy()
    axes[0, 2].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')


    titles = ['Target', 'MAF\nGaussian Base\ne.g. Papamakarios\net al. 2017', 
              'Real NVP\nResampled Base\nStimper et al. 2022', 
              'Piecewise MAF\nGaussian Base\nThis work',]
    types = ['Two Moons', 'Circle of Gaussians', 'Two Rings']
    for i in range(len(axes)):
        for j in range(axes.shape[-1]):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_ylabel(types[i])

    for i in range(axes.shape[-1]):
        axes[0, i].set_title(titles[i])

    cgm = CircularGaussianMixture()

    s = cgm.sample(nsample).numpy()

    axes[1, 0].hist2d(s[:, 0], s[:, 1], bins=80, cmap='Blues')

    # noraml maf for circle of gaussians
    try:
        sAFlow = MAF.load(base + 'cgm_single_maf_' + str(d) + '.pkl')
    except:
        sAFlow = MAF(s)
        sAFlow.train(epochs, early_stop=True)
        sAFlow.save(base + 'cgm_single_maf_' + str(d) + '.pkl')
    samples = sAFlow.sample(kl_nsample).numpy()
    kldiv, kl_error = calc_kl(samples, sAFlow, cgm)
    cgm_maf_kls.append([kldiv, kl_error])
    axes[1, 1].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')

    # cluster flow for circle of gaussians
    try:
        sAFlow = MAF.load(base + "cgm_cluster_maf_" + str(d) + ".pkl")
    except:
        _ = clusterMAF(s)
        nn = int((17424/_.cluster_number/2904)//1 + 1)
        print('number clusters: ', _.cluster_number, ' number_networks: ', nn)

        kmeans = KMeans(_.cluster_number, random_state=0)
        labels = kmeans.fit(s).predict(s)
        sAFlow = clusterMAF(s, cluster_labels=labels, 
                            cluster_number=_.cluster_number, number_networks=nn)
        sAFlow.train(pEpochs, early_stop=True)
        sAFlow.save(base + "cgm_cluster_maf_" + str(d) + ".pkl")
    samples = sAFlow.sample(kl_nsample).numpy()
    kldiv, kl_error = calc_kl(samples, sAFlow, cgm)
    cgm_clus_kls.append([kldiv, kl_error])
    axes[1, 3].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')

    # realNVP with reasampled base for circle of gaussians
    try:
        model = pickle.load(open(base + "cgm_realnvp_resampled_base_" + str(d) + ".pkl","rb"))
    except:
        model = create_model(cgm, 'resampled')
        model = train(model)
        file = open(base + "cgm_realnvp_resampled_base_" + str(d) + ".pkl","wb")
        pickle.dump(model,file)
    samples = model.sample(kl_nsample)[0]#.detach().numpy()
    logprob = model.log_prob(samples).detach().numpy()
    target_logprob = cgm.log_prob(samples).detach().numpy()
    logprob, mask = mask_arr(logprob)
    target_logprob = target_logprob[mask]
    logprob -= logsumexp(logprob)
    target_logprob -= logsumexp(target_logprob)
    delta_logprob = logprob - target_logprob
    kldiv = np.mean(delta_logprob)
    kl_error = np.std(delta_logprob)/np.sqrt(len(delta_logprob))
    cgm_nvp_kls.append([kldiv, kl_error])
    samples = samples.detach().numpy()
    axes[1, 2].hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='Blues')

    plt.tight_layout()
    plt.savefig(base + 'benchmarks_with_resampled_base_test_' + str(d) + '.pdf')
    #plt.show()
    plt.close()

tm_maf_kls, tm_clus_kls, tm_nvp_kls = np.array(tm_maf_kls).T, \
    np.array(tm_clus_kls).T, np.array(tm_nvp_kls).T
rm_maf_kls, rm_clus_kls, rm_nvp_kls = np.array(rm_maf_kls).T, \
    np.array(rm_clus_kls).T, np.array(rm_nvp_kls).T
cgm_maf_kls, cgm_clus_kls, cgm_nvp_kls = np.array(cgm_maf_kls).T, \
    np.array(cgm_clus_kls).T, np.array(cgm_nvp_kls).T


tm_maf_mean = np.mean(tm_maf_kls[:, 0])
tm_maf_error = np.sqrt(np.sum(tm_maf_kls[:, 1]**2))
tm_nvp_mean = np.mean(tm_nvp_kls[:, 0])
tm_nvp_error = np.sqrt(np.sum(tm_nvp_kls[:, 1]**2))
tm_clus_mean = np.mean(tm_clus_kls[:, 0])
tm_clus_error = np.sqrt(np.sum(tm_clus_kls[:, 1]**2))

rm_maf_mean = np.mean(rm_maf_kls[:, 0])
rm_maf_error = np.sqrt(np.sum(rm_maf_kls[:, 1]**2))
rm_nvp_mean = np.mean(rm_nvp_kls[:, 0])
rm_nvp_error = np.sqrt(np.sum(rm_nvp_kls[:, 1]**2))
rm_clus_mean = np.mean(rm_clus_kls[:, 0])
rm_clus_error = np.sqrt(np.sum(rm_clus_kls[:, 1]**2))

cgm_maf_mean = np.mean(cgm_maf_kls[:, 0])
cgm_maf_error = np.sqrt(np.sum(cgm_maf_kls[:, 1]**2))
cgm_nvp_mean = np.mean(cgm_nvp_kls[:, 0])
cgm_nvp_error = np.sqrt(np.sum(cgm_nvp_kls[:, 1]**2))
cgm_clus_mean = np.mean(cgm_clus_kls[:, 0])
cgm_clus_error = np.sqrt(np.sum(cgm_clus_kls[:, 1]**2))


print('tm maf kl: {:.5f} +/- {:.5f}'.format(tm_maf_mean, tm_maf_error))
print('tm nvp kl: {:.5f} +/- {:.5f}'.format(tm_nvp_mean, tm_nvp_error))
print('tm clus kl: {:.5f} +/- {:.5f}'.format(tm_clus_mean, tm_clus_error))

print('rm maf kl: {:.5f} +/- {:.5f}'.format(rm_maf_mean, rm_maf_error))
print('rm nvp kl: {:.5f} +/- {:.5f}'.format(rm_nvp_mean, rm_nvp_error))
print('rm clus kl: {:.5f} +/- {:.5f}'.format(rm_clus_mean, rm_clus_error))

print('cgm maf kl: {:.5f} +/- {:.5f}'.format(cgm_maf_mean, cgm_maf_error))
print('cgm nvp kl: {:.5f} +/- {:.5f}'.format(cgm_nvp_mean, cgm_nvp_error))
print('cgm clus kl: {:.5f} +/- {:.5f}'.format(cgm_clus_mean, cgm_clus_error))
