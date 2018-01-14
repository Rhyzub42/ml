
## 一、数据驱动

### 1.相关性:皮尔逊系数


```python
from numpy.random import randn
import numpy as np
from scipy.stats.stats import pearsonr
```

#### 1.随机数据:相关性也很随机


```python
x = randn(8)
y = randn(8)
pearsonr(x,y)[0]
```




    0.17812517311493814



#### 2.正相关：大于0，小于1.一般大于0.5就很相关了


```python
data1 = [0.8, 2.2, 2.8, 3.5, 4.8]
x = np.array(data1)
y = np.arange(5) 
pearsonr(x,y)[0]
```




    0.9886905691829575




```python
x = np.arange(15) 
y = np.arange(15) 
pearsonr(x,y)[0]
```




    1.0



#### 3.负相关：大于-1，小于1


```python
data1 = [0.8, 2.2, 2.8, 3.5, 4.8]
x = np.array(data1)
y = -np.arange(5) 
pearsonr(x,y)[0]
```




    -0.9886905691829575




```python
x = np.arange(15) 
y = -np.arange(15) 
pearsonr(x,y)[0]
```




    -1.0



### 2.迭代删除（增加）


```python
## 导入工具包
import pandas
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 
le = LabelEncoder()
le.fit(iris['Species'])   
y = le.transform(iris['Species']) # 对花的类别进行编号处理
lm = linear_model.LogisticRegression() # 选择逻辑回归
features = ['PetalLengthCm','PetalWidthCm','SepalLengthCm','SepalWidthCm'] # 特征变量


selected_features = []
rest_features = features[:]
best_acc = 0
while len(rest_features)>0:
    temp_best_i = ''
    temp_best_acc = 0
    for feature_i in rest_features:
        temp_features = selected_features + [feature_i,]
        X = iris[temp_features]  # 每次选择一个特征进行训练
        scores = cross_val_score(lm,X,y,cv=5 , scoring='accuracy')
        acc = np.mean(scores)
        if acc > temp_best_acc:
            temp_best_acc = acc
            temp_best_i = feature_i
    print("select",temp_best_i,"acc:",temp_best_acc) # 打印选择的特征和评分
    if temp_best_acc > best_acc:                     # 判断是否大于最好的评分
        best_acc = temp_best_acc                   
        selected_features += [temp_best_i,]          
        rest_features.remove(temp_best_i)
    else: 
        break
print("best feature set: ",selected_features,"acc: ",best_acc)
```

    ('select', 'PetalWidthCm', 'acc:', 0.85333333333333328)
    ('select', 'SepalWidthCm', 'acc:', 0.94000000000000006)
    ('select', 'PetalLengthCm', 'acc:', 0.95333333333333337)
    ('select', 'SepalLengthCm', 'acc:', 0.96000000000000019)
    ('best feature set: ', ['PetalWidthCm', 'SepalWidthCm', 'PetalLengthCm', 'SepalLengthCm'], 'acc: ', 0.96000000000000019)
    

### 3.基于模型

### 3.1.单变量特征选定


```python
# 通过卡方检验选定数据特征
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# 导入数据
iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 

# 将数据分为输入数据和输出结果
arrary = iris.values
# print(arrary)
X =arrary[:,0:4]

le = LabelEncoder()
le.fit(iris['Species'])   
Y = le.transform(iris['Species']) # 对花的类别进行编号处理
# print Y
# 特征选定
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
print(features)
```

    [  10.818    3.594  116.17    67.245]
    [[ 5.1  3.5  1.4  0.2]
     [ 4.9  3.   1.4  0.2]
     [ 4.7  3.2  1.3  0.2]
     [ 4.6  3.1  1.5  0.2]
     [ 5.   3.6  1.4  0.2]
     [ 5.4  3.9  1.7  0.4]
     [ 4.6  3.4  1.4  0.3]
     [ 5.   3.4  1.5  0.2]
     [ 4.4  2.9  1.4  0.2]
     [ 4.9  3.1  1.5  0.1]
     [ 5.4  3.7  1.5  0.2]
     [ 4.8  3.4  1.6  0.2]
     [ 4.8  3.   1.4  0.1]
     [ 4.3  3.   1.1  0.1]
     [ 5.8  4.   1.2  0.2]
     [ 5.7  4.4  1.5  0.4]
     [ 5.4  3.9  1.3  0.4]
     [ 5.1  3.5  1.4  0.3]
     [ 5.7  3.8  1.7  0.3]
     [ 5.1  3.8  1.5  0.3]
     [ 5.4  3.4  1.7  0.2]
     [ 5.1  3.7  1.5  0.4]
     [ 4.6  3.6  1.   0.2]
     [ 5.1  3.3  1.7  0.5]
     [ 4.8  3.4  1.9  0.2]
     [ 5.   3.   1.6  0.2]
     [ 5.   3.4  1.6  0.4]
     [ 5.2  3.5  1.5  0.2]
     [ 5.2  3.4  1.4  0.2]
     [ 4.7  3.2  1.6  0.2]
     [ 4.8  3.1  1.6  0.2]
     [ 5.4  3.4  1.5  0.4]
     [ 5.2  4.1  1.5  0.1]
     [ 5.5  4.2  1.4  0.2]
     [ 4.9  3.1  1.5  0.1]
     [ 5.   3.2  1.2  0.2]
     [ 5.5  3.5  1.3  0.2]
     [ 4.9  3.1  1.5  0.1]
     [ 4.4  3.   1.3  0.2]
     [ 5.1  3.4  1.5  0.2]
     [ 5.   3.5  1.3  0.3]
     [ 4.5  2.3  1.3  0.3]
     [ 4.4  3.2  1.3  0.2]
     [ 5.   3.5  1.6  0.6]
     [ 5.1  3.8  1.9  0.4]
     [ 4.8  3.   1.4  0.3]
     [ 5.1  3.8  1.6  0.2]
     [ 4.6  3.2  1.4  0.2]
     [ 5.3  3.7  1.5  0.2]
     [ 5.   3.3  1.4  0.2]
     [ 7.   3.2  4.7  1.4]
     [ 6.4  3.2  4.5  1.5]
     [ 6.9  3.1  4.9  1.5]
     [ 5.5  2.3  4.   1.3]
     [ 6.5  2.8  4.6  1.5]
     [ 5.7  2.8  4.5  1.3]
     [ 6.3  3.3  4.7  1.6]
     [ 4.9  2.4  3.3  1. ]
     [ 6.6  2.9  4.6  1.3]
     [ 5.2  2.7  3.9  1.4]
     [ 5.   2.   3.5  1. ]
     [ 5.9  3.   4.2  1.5]
     [ 6.   2.2  4.   1. ]
     [ 6.1  2.9  4.7  1.4]
     [ 5.6  2.9  3.6  1.3]
     [ 6.7  3.1  4.4  1.4]
     [ 5.6  3.   4.5  1.5]
     [ 5.8  2.7  4.1  1. ]
     [ 6.2  2.2  4.5  1.5]
     [ 5.6  2.5  3.9  1.1]
     [ 5.9  3.2  4.8  1.8]
     [ 6.1  2.8  4.   1.3]
     [ 6.3  2.5  4.9  1.5]
     [ 6.1  2.8  4.7  1.2]
     [ 6.4  2.9  4.3  1.3]
     [ 6.6  3.   4.4  1.4]
     [ 6.8  2.8  4.8  1.4]
     [ 6.7  3.   5.   1.7]
     [ 6.   2.9  4.5  1.5]
     [ 5.7  2.6  3.5  1. ]
     [ 5.5  2.4  3.8  1.1]
     [ 5.5  2.4  3.7  1. ]
     [ 5.8  2.7  3.9  1.2]
     [ 6.   2.7  5.1  1.6]
     [ 5.4  3.   4.5  1.5]
     [ 6.   3.4  4.5  1.6]
     [ 6.7  3.1  4.7  1.5]
     [ 6.3  2.3  4.4  1.3]
     [ 5.6  3.   4.1  1.3]
     [ 5.5  2.5  4.   1.3]
     [ 5.5  2.6  4.4  1.2]
     [ 6.1  3.   4.6  1.4]
     [ 5.8  2.6  4.   1.2]
     [ 5.   2.3  3.3  1. ]
     [ 5.6  2.7  4.2  1.3]
     [ 5.7  3.   4.2  1.2]
     [ 5.7  2.9  4.2  1.3]
     [ 6.2  2.9  4.3  1.3]
     [ 5.1  2.5  3.   1.1]
     [ 5.7  2.8  4.1  1.3]
     [ 6.3  3.3  6.   2.5]
     [ 5.8  2.7  5.1  1.9]
     [ 7.1  3.   5.9  2.1]
     [ 6.3  2.9  5.6  1.8]
     [ 6.5  3.   5.8  2.2]
     [ 7.6  3.   6.6  2.1]
     [ 4.9  2.5  4.5  1.7]
     [ 7.3  2.9  6.3  1.8]
     [ 6.7  2.5  5.8  1.8]
     [ 7.2  3.6  6.1  2.5]
     [ 6.5  3.2  5.1  2. ]
     [ 6.4  2.7  5.3  1.9]
     [ 6.8  3.   5.5  2.1]
     [ 5.7  2.5  5.   2. ]
     [ 5.8  2.8  5.1  2.4]
     [ 6.4  3.2  5.3  2.3]
     [ 6.5  3.   5.5  1.8]
     [ 7.7  3.8  6.7  2.2]
     [ 7.7  2.6  6.9  2.3]
     [ 6.   2.2  5.   1.5]
     [ 6.9  3.2  5.7  2.3]
     [ 5.6  2.8  4.9  2. ]
     [ 7.7  2.8  6.7  2. ]
     [ 6.3  2.7  4.9  1.8]
     [ 6.7  3.3  5.7  2.1]
     [ 7.2  3.2  6.   1.8]
     [ 6.2  2.8  4.8  1.8]
     [ 6.1  3.   4.9  1.8]
     [ 6.4  2.8  5.6  2.1]
     [ 7.2  3.   5.8  1.6]
     [ 7.4  2.8  6.1  1.9]
     [ 7.9  3.8  6.4  2. ]
     [ 6.4  2.8  5.6  2.2]
     [ 6.3  2.8  5.1  1.5]
     [ 6.1  2.6  5.6  1.4]
     [ 7.7  3.   6.1  2.3]
     [ 6.3  3.4  5.6  2.4]
     [ 6.4  3.1  5.5  1.8]
     [ 6.   3.   4.8  1.8]
     [ 6.9  3.1  5.4  2.1]
     [ 6.7  3.1  5.6  2.4]
     [ 6.9  3.1  5.1  2.3]
     [ 5.8  2.7  5.1  1.9]
     [ 6.8  3.2  5.9  2.3]
     [ 6.7  3.3  5.7  2.5]
     [ 6.7  3.   5.2  2.3]
     [ 6.3  2.5  5.   1.9]
     [ 6.5  3.   5.2  2. ]
     [ 6.2  3.4  5.4  2.3]
     [ 5.9  3.   5.1  1.8]]
    

#### 结论：3 4 1 2

### 3.2.递归特征消除


```python
# 通过递归消除来选定特征
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# 导入数据
iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 

# 将数据分为输入数据和输出结果
arrary = iris.values
# print(arrary)
X =arrary[:,0:4]
# print X

le = LabelEncoder()
le.fit(iris['Species'])   
Y = le.transform(iris['Species']) # 对花的类别进行编号处理

# 特征选定
model = LogisticRegression()
rfe = RFE(model, 2)
fit = rfe.fit(X, Y)
print("特征个数：")
print(fit.n_features_)
print("被选定的特征：")
print(fit.support_)
print("特征排名：")
print(fit.ranking_)
```

    特征个数：
    2
    被选定的特征：
    [False  True False  True]
    特征排名：
    [3 1 2 1]
    

### 3.5.主要成分分析


```python
# 通过主要成分分析选定数据特征
from pandas import read_csv
from sklearn.decomposition import PCA

# 导入数据
iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 

# 将数据分为输入数据和输出结果
arrary = iris.values
# print(arrary)
X =arrary[:,0:4]
# print X
le = LabelEncoder()
le.fit(iris['Species'])   
Y = le.transform(iris['Species']) # 对花的类别进行编号处理

# 特征选定
pca = PCA(n_components=2)
fit = pca.fit(X)
print("解释方差：%s" % fit.explained_variance_ratio_)
print(fit.components_)

```

    解释方差：[ 0.92461621  0.05301557]
    [[ 0.36158968 -0.08226889  0.85657211  0.35884393]
     [ 0.65653988  0.72971237 -0.1757674  -0.07470647]]
    

#### 结论： 4 3 2 1

### 3.6.特征重要性


```python
# 通过决策树计算特征的重要性
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier

# 导入数据
iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 

# 将数据分为输入数据和输出结果
arrary = iris.values
X =np.array(arrary[:,0:4])
le = LabelEncoder()
le.fit(iris['Species'])   
Y = np.array(le.transform(iris['Species'])) # 对花的类别进行编号处理
# 特征选定
model = ExtraTreesClassifier()
fit = model.fit(X, Y)
print(fit.feature_importances_)
```

    [ 0.03809246  0.05882966  0.40479618  0.49828169]
    

#### 结论： 4 3 2 1

### 3.7.随机森林特征选择法 —— Gini Importance


```python
import pandas
import numpy as np
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder

iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
le = LabelEncoder()
le.fit(iris['Species'])
rf = ensemble.RandomForestClassifier()
features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
y = np.array(le.transform(iris['Species']))
X = np.array(iris[features])
#Gini importance
rf.fit(X,y)
print(rf.feature_importances_)
```

    [ 0.08703922  0.03382421  0.43010863  0.44902794]
    

#### 结论： 3 4 2 1

### 3.8.随机森林特征选择法 —— Mean Decrease Accuracy


```python
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=10,test_size=0.1)
scores = np.zeros((10,4))
count = 0
for train_idx, test_idx in rs.split(X):
    X_train , X_test = X[train_idx] , X[test_idx]
    y_train , y_test = y[train_idx] , y[test_idx]
    r = rf.fit(X_train,y_train)
    acc = accuracy_score(y_test,rf.predict(X_test))
    for i in range(len(features)):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = accuracy_score(y_test,rf.predict(X_t))
        scores[count,i] = ((acc-shuff_acc)/acc)
    count += 1
print(np.mean(scores,axis=0))
```

    [ 0.          0.          0.30047619  0.27362637]
    

#### 结论： 3 4 1 2

## 二、领域专家

### 其实绝大多数的情况咨询一些领域的专家或学习该领域的一些知识能够帮助你更好的进行特征选择。

## 三、参考及git


```python
git: https://coding.net/u/RuoYun/p/Python-of-machine-learning/git/tree/master
```

### 参考：https://read.douban.com/reader/column/6939417/chapter/35756931/
