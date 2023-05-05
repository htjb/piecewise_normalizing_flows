import numpy as np
from anesthetic import MCMCSamples
import matplotlib.pyplot as plt
from margarine.maf import MAF

# generate a set of multi-modal samples
nsamples = 5000
x = np.hstack([np.random.normal(0, 1, int(3*nsamples/4)), 
               np.random.normal(8, 0.5, int(1*nsamples/4))])
y = np.hstack([np.random.normal(6, 1, int(2*nsamples/4)), 
               np.random.normal(-5, 0.5, int(2*nsamples/4))])
data = np.vstack([x, y]).T

samples = MCMCSamples(data=data)
names = [i for i in range(data.shape[-1])]

fig, axes = plt.subplots(1, 3, figsize=(6.3, 3))

# plot original samples
axes[0].hist2d(data[:, 0], data[:, 1], bins=80)

# try and load the example maf only flow else generate and plot samples
try:
    bij = MAF.load("figure1_normal_maf.pkl")
except:
    bij = MAF(samples[names].values, 
              samples.get_weights().astype('float64'))
    bij.train(10000, early_stop=True)
    bij.save("figure1_normal_maf.pkl")
bij_samples = bij.sample(nsamples)
axes[1].hist2d(bij_samples[:, 0], bij_samples[:, 1], bins=80)

# try and load the example cluster maf only flow else generate and plot samples
try:
    bij = MAF.load("figure1_cluster.pkl")
except:
    bij = MAF(samples[names].values, 
              samples.get_weights().astype('float64'), clustering=True)
    bij.train(10000, early_stop=True)
    bij.save("figure1_cluster.pkl")
bij_samples = bij.sample(nsamples)
axes[2].hist2d(bij_samples[:, 0], bij_samples[:, 1], bins=80)

# formatting
title = ['Target.', 'MAF\nGaussian Base\ne.g. Papamakarios et al. 2017', 
         'Piecewise MAF\nGaussian Base\nThis work']
for i in range(len(axes)):
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    axes[i].set_title(title[i])

plt.tight_layout()
plt.savefig('Figures/figure1.pdf')
plt.show()