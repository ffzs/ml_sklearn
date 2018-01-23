"""
ffzs
2018.1.23
"""
import plotly
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

centers = [[3, 3], [-3, -3], [3, -3]]
X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=1,
                            random_state=20)

# X = StandardScaler().fit_transform(X)

import plotly.graph_objs as go
trace = go.Scatter(
    x=X[:, 0], y=X[:, 1], mode='markers',
    marker=dict(size='10', color=labels_true, colorscale='Jet', showscale=False, line=go.Line(color='black', width=2),
    ),
)
plotly.offline.plot([trace], 'db.html')

import visdom
#
viz = visdom.Visdom()
viz.scatter(X=X, Y=labels_true+1)

# for i in range(2,12):
#     eps = i/10.0
# eps 为范围， min_samples 为成为核心对象最小个数
eps = 0.9
db = DBSCAN(eps=eps, min_samples=15).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# 标记核心对象
core_samples_mask[db.core_sample_indices_] = True
# -1代表噪点，其他为聚类类型
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# 输出算法各类参数
print('Estimated number of clusters: %d' % n_clusters_)
# 同质性：每个群集只包含单个类的成员。
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# 完整性：给定类的所有成员都分配给同一个群集。
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# v = 2 * (同质性 * 完整性) / (同质性 + 完整性)
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# 聚类算法的准确率
print("Adjusted Rand Index: %0.3f"
  % metrics.adjusted_rand_score(labels_true, labels))
# 调整后互信度
print("Adjusted Mutual Information: %0.3f"
  % metrics.adjusted_mutual_info_score(labels_true, labels))
# 轮廓系数 越接近1 越合理
print("Silhouette Coefficient: %0.3f"
  % metrics.silhouette_score(X, labels))

#size ,symbol 个性化准备
t_label = np.arange(-1, n_clusters_)
t_size = [7]+[12]*n_clusters_
t_symbol = ["x"]+["circle-open-dot"]*n_clusters_
size, symbol = [], []
for p in range(len(labels)):
    for q in range(len(t_label)):
        if labels[p] == t_label[q]:
            size.append(t_size[q])
            symbol.append(t_symbol[q])

trace = go.Scatter(
    x=X[:, 0], y=X[:, 1], mode='markers',
    marker=dict(size=size, symbol=symbol, color=labels, colorscale='Jet', showscale=False, line=go.Line(color='black', width=2),
    ),
)
layout = go.Layout(showlegend=False,
                   title='Estimated number of clusters: %d' % n_clusters_,
                   xaxis=dict(showgrid=False, zeroline=False),
                   yaxis=dict(showgrid=False, zeroline=False),
                   height=600, width=1000)
fig = go.Figure(data=[trace], layout=layout)
plotly.offline.plot(fig, 'db_1.html')

viz = visdom.Visdom()
viz.scatter(X=X, Y=labels+2,
        opts=dict(
            colormap='Jet',
            markersymbol="x",
            markersize=12,
            title='eps:{} ; n_clusters:{} ' .format(eps,n_clusters_)
        ))