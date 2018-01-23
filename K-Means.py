"""
ffzs
2018.1.23
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(100)
n_samples = 1500

# X, y = make_blobs(n_samples=1500, n_features=2, centers=3, cluster_std=[1.0, 2.0, 3.0], shuffle=False)
X, y = make_blobs(n_samples=1500, n_features=2, centers=[[10, 3], [-10, -3], [3, -10]], cluster_std=[1.0, 2.0, 3.0])

import visdom
viz = visdom.Visdom()
viz.scatter(Y=y+1, X=X, opts=dict(title='original'))

# k = 4
for k in range(2, 6):
    y_pred = KMeans(n_clusters=k).fit_predict(X)
    viz.scatter(Y=y_pred+1, X=X, opts=dict(title='k = {}'.format(k)))