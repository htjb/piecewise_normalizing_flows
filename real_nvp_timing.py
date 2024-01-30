import numpy as np
import matplotlib.pyplot as plt
from target_dists_stimper import TwoMoons, CircularGaussianMixture, RingMixture
import torch
from scipy.special import logsumexp
from tensorflow import keras
import normflows as nf
import larsflow as lf
from tqdm import tqdm
from matplotlib import rc
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import os
import time


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
                return minimum_model, it
            
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

tm = TwoMoons()
times, epochs, cost = [], [], []
for d in range(5):
    s = time.time()
    model = create_model(tm, 'resampled')
    model, it = train(model)
    e = time.time()
    times.append(e-s)
    epochs.append(it)
    cost.append(it*nsample)

print('Two Moons')
print('Mean time: ', np.mean(times))
print('Std time: ', np.std(times)/np.sqrt(len(times)))
print('Mean epochs: ', np.mean(epochs))
print('Std epochs: ', np.std(epochs)/np.sqrt(len(epochs)))
print('Mean cost: ', np.mean(cost))
print('Std cost: ', np.std(cost)/np.sqrt(len(cost)))

cgm = CircularGaussianMixture()
times, epochs, cost = [], [], []
for d in range(5):
    s = time.time()
    model = create_model(cgm, 'resampled')
    model, it = train(model)
    e = time.time()
    times.append(e-s)
    epochs.append(it)
    cost.append(it*nsample)

print('Circle Gaussian Mixture')
print('Mean time: ', np.mean(times))
print('Std time: ', np.std(times)/np.sqrt(len(times)))
print('Mean epochs: ', np.mean(epochs))
print('Std epochs: ', np.std(epochs)/np.sqrt(len(epochs)))
print('Mean cost: ', np.mean(cost))
print('Std cost: ', np.std(cost)/np.sqrt(len(cost)))

rm = RingMixture()
times, epochs, cost = [], [], []
for d in range(5):
    s = time.time()
    model = create_model(rm, 'resampled')
    model, it = train(model)
    e = time.time()
    times.append(e-s)
    epochs.append(it)
    cost.append(it*nsample)

print('Ring Mixture')
print('Mean time: ', np.mean(times))
print('Std time: ', np.std(times)/np.sqrt(len(times)))
print('Mean epochs: ', np.mean(epochs))
print('Std epochs: ', np.std(epochs)/np.sqrt(len(epochs)))
print('Mean cost: ', np.mean(cost))
print('Std cost: ', np.std(cost)/np.sqrt(len(cost)))
    