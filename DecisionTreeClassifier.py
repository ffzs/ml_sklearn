"""
ffzs
2018.1.22
blog: http://blog.csdn.net/tonydz0523
"""
import pandas as pd

data = pd.read_csv('data/titanic_data.csv', encoding='utf-8')
del data['PassengerId']

# 将性别内容转换为数字表示 男：1  女：0
def trans(x):
    # x = 1 if x == 'male' else 0
    return 1 if x == 'male' else 0
data['Sex'] = data['Sex'].apply(trans)

print(data.isnull().any())
data.fillna(data.Age.median(), inplace=True)

# 可视化要安装plotly
import plotly
import plotly.figure_factory as ff

data['Survived'] = data['Survived'].astype('str')
fig = ff.create_scatterplotmatrix(data, diag='histogram',index='Survived',colormap=[ '#32CD32', '#00F5FF'],
                                  height=800, width=800)
plotly.offline.plot(fig, filename='p2.html')

from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
X = data.iloc[:, 1:4]
y = data.iloc[:, 0]
dtc = DTC(criterion='entropy')  # 基于信息熵
dtc.fit(X, y)
print('准确率：', dtc.score(X, y))

with open('data/tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=X.columns, out_file=f)

import pydot
(graph,) = pydot.graph_from_dot_file('data/tree.dot')
graph.write_png('data/tree.png')
graph.write_pdf('data/tree.pdf')

# test
import random
for _ in range(10):
    i = random.randint(0, len(data))
    pred = dtc.predict(X)[i]
    sign = '✗' if y[i] != pred else '✓'
    print(y[i], pred, sign)