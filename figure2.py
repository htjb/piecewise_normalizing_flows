import numpy as np
import matplotlib.pyplot as plt
from target_dists_stimper import CircularGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from matplotlib import rc
import matplotlib as mpl

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

# generate example samples
nsample = 1000
cgm = CircularGaussianMixture()
s = cgm.sample(nsample).numpy()

# calculate the required number of clusters
ks = np.arange(2, 13)
losses = []
labels = []
for k in ks:
    kmeans = KMeans(k, random_state=0)
    label = kmeans.fit(s).predict(s)
    losses.append(-silhouette_score(s, label))
    labels.append(label)
losses = np.array(losses)
print(ks, losses)
cluster_number = ks[np.where(losses == losses.min())[0][0]]
print(cluster_number)

# plot a few examples of the clustering and the corresponding score.
fig = plt.figure(constrained_layout=True, figsize=(6.3, 4))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.8, 0.8])

axes = []
for i in range(2):
    ax = []
    for j in range(3):
        ax.append(fig.add_subplot(gs[i, j]))
    axes.append(ax)
axes = np.array(axes)

cmap = 'inferno'

axes[0, 0].scatter(s[:, 0], s[:, 1], c=labels[0], s=1, cmap=cmap, rasterized=True)
axes[0, 1].scatter(s[:, 0], s[:, 1], c=labels[2], s=1, cmap=cmap, rasterized=True)
axes[0, 2].scatter(s[:, 0], s[:, 1], c=labels[4], s=1, cmap=cmap, rasterized=True)
axes[1, 0].scatter(s[:, 0], s[:, 1], c=labels[6], s=1, cmap=cmap, rasterized=True)
axes[1, 1].scatter(s[:, 0], s[:, 1], c=labels[8], s=1, cmap=cmap, rasterized=True)

axes[1, 2].scatter(s[:, 0], s[:, 1], c=labels[10], s=1, cmap=cmap)

axes[0, 0].set_title(r'$k=2$,' +'\n' + r'$s=$' + '{:.3f}'.format(-losses[0]), fontsize=10)
axes[0, 1].set_title(r'$k=4$,' +'\n' + r'$s=$' + '{:.3f}'.format(-losses[2]), fontsize=10)
axes[0, 2].set_title(r'$k=6$,' +'\n' + r'$s=$' + '{:.3f}'.format(-losses[4]), fontsize=10)
axes[1, 0].set_title(r'$k=8$,' +'\n' + r'$s=$' + '{:.3f}'.format(-losses[6]), fontsize=10)
axes[1, 1].set_title(r'$k=10$,' +'\n' + r'$s=$' + '{:.3f}'.format(-losses[8]), fontsize=10)
axes[1, 2].set_title(r'$k=12$,' +'\n' + r'$s=$' + '{:.3f}'.format(-losses[10]), fontsize=10)

for i in range(2):
    for j in range(3):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

ax = fig.add_subplot(gs[:, 3:])
ax.plot(ks, -losses, 'o-', c='k')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Silhouette score')
ax.set_xticks(ks[::2])

#plt.tight_layout()
plt.savefig('figure2.pdf', dpi=300)
plt.show()