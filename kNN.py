"""
ffzs
2018.1.22
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
# 加载数据
iris = load_iris()
X = iris.data[:, :2]
y = iris.target
print(iris.feature_names)

# for i in range(5,20):
knn = KNeighborsClassifier(n_neighbors=9, weights='uniform')
knn.fit(X, y)
print("准确率", knn.score(X, y))


import plotly.graph_objs as go
import plotly
import numpy as np

# 2维绘图
h = .02
cmap_light =[[0, 'rgba(255, 192, 203,0.7)'], [0.5, 'rgba(0, 229, 238, 0.7)'], [1, 'rgba(124, 252, 0, 0.7)']]
cmap_bold = [[0, '#FF0000'], [0.5, '#0000FF'], [1, '#00FF00']]
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_ = np.arange(x_min, x_max, h)
y_ = np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(x_, y_)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
trace1 = go.Heatmap(x=x_, y=y_, z=Z,
                    showscale=False,
                    colorscale=cmap_light)
trace2 = go.Scatter(
    x=X[:, 0], y=X[:, 1], mode='markers',
    marker=dict(size='10', color=y, colorscale=cmap_bold, showscale=False,
        line=dict(color='black', width=1)
    ),
)
data = [trace1,trace2]
plotly.offline.plot(data,'s1.html')

# 3维绘图

# Z = knn.predict(X)

# trace = go.Scatter3d(
#     x=X[:, 2], y=X[:, 1], z=X[:, 0], mode='markers',
#     marker=dict(
#         size='10',
#         color=Z,
#         colorscale='Jet',
#         showscale=False,
#         line=Line(color='black',width=2),
#     ),
# )
#
# plotly.offline.plot([trace], 's3d.html')
