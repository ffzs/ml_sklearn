"""
ffzs
2018.1.22
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
# 加载数据
iris = load_iris()
X = iris.data[:, :3]
y = iris.target
print(iris.feature_names)
print(X[:3])
print(y[:3])

# for i in range(5,20):
knn = KNeighborsClassifier(n_neighbors=9, weights='uniform')
knn.fit(X, y)
print("准确率", knn.score(X, y))


import plotly.graph_objs as go
from plotly.graph_objs import *
import plotly

Z = knn.predict(X)

trace = go.Scatter3d(
    x=X[:, 2], y=X[:, 1], z=X[:, 0], mode='markers',
    marker=dict(
        size='10',
        color=Z,
        colorscale='Jet',
        showscale=False,
        line=Line(color='black',width=2),
    ),
)

plotly.offline.plot([trace], 's3d.html')

# trace1 = go.Scatter(
#     x=X[:, 0], y=X[:, 1],z=X[:, 2], mode='markers',
#     marker=dict(
#         size='10',
#         color=z,
#         colorscale='Jet',
#         showscale=False
#     ),
# )
# trace2 = go.Scatter(
#     x=X[:, 0], y=X[:, 1],z=X[:, 2], mode='markers',
#     marker=dict(
#         size='10',
#         color=y,
#         colorscale='Jet',
#         showscale=False
#     ),
# )
# fig = plotly.tools.make_subplots(rows=1, cols=2)
#
# fig.append_trace(trace1, 1, 1)
# fig.append_trace(trace2, 1, 2)
# fig['layout'].update(height=600, width=1200)
# plotly.offline.plot(fig,'s1.html')