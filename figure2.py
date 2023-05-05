import numpy as np
import matplotlib.pyplot as plt
from target_dists_stimper import CircularGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# generate example samples
nsample = 10000
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
fig, axes = plt.subplots(2, 3, figsize=(6.3, 4))

cmap = 'inferno'

axes[0, 0].scatter(s[:, 0], s[:, 1], c=labels[0], s=1, cmap=cmap, rasterized=True)
axes[0, 1].scatter(s[:, 0], s[:, 1], c=labels[2], s=1, cmap=cmap, rasterized=True)
axes[0, 2].scatter(s[:, 0], s[:, 1], c=labels[4], s=1, cmap=cmap, rasterized=True)
axes[1, 0].scatter(s[:, 0], s[:, 1], c=labels[6], s=1, cmap=cmap, rasterized=True)
axes[1, 1].scatter(s[:, 0], s[:, 1], c=labels[8], s=1, cmap=cmap, rasterized=True)

axes[1, 2].scatter(s[:, 0], s[:, 1], c=labels[10], s=1, cmap=cmap)

axes[0, 0].set_title(r'$k=2, s=$' + '{:.3f}'.format(-losses[0]))
axes[0, 1].set_title(r'$k=4, s=$' + '{:.3f}'.format(-losses[2]))
axes[0, 2].set_title(r'$k=6, s=$' + '{:.3f}'.format(-losses[4]))
axes[1, 0].set_title(r'$k=8, s=$' + '{:.3f}'.format(-losses[6]))
axes[1, 1].set_title(r'$k=10, s=$' + '{:.3f}'.format(-losses[8]))
axes[1, 2].set_title(r'$k=12, s=$' + '{:.3f}'.format(-losses[10]))

for i in range(2):
    for j in range(3):
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.savefig('figure2.pdf', dpi=300)
plt.show()