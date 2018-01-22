# coding:utf-8

import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz

data = pd.read_csv('data/titanic_data.csv', encoding='utf-8')
del data['PassengerId']
# 将性别内容转换为数字表示 男：1  女：0
def trans(x):
    x = 1 if x == 'male' else 0
    return x
data['Sex'] = data['Sex'].apply(trans)
print(data.isnull().any())
#



print(data.head())