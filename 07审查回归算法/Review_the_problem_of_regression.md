
## 一、线性算法

### 1.线性回归

### &ensp;&ensp;&ensp;&ensp;1.1 原理 ：y = ax +b的升级版。

### &ensp;&ensp;&ensp;&ensp;1.2 适用场景 ：普遍适用，简单粗暴的算法

### 2.岭回归

### &ensp;&ensp;&ensp;&ensp;2.1 原理 ：改良的最小二乘估计法

### &ensp;&ensp;&ensp;&ensp;2.2 适用场景 ：病态数据拟合（损失部分信息，降低精度，换取更好的回归系数）

### 3.套索回归

### &ensp;&ensp;&ensp;&ensp;3.1 原理 ：和岭回归类似。惩罚函数是绝对值。

### 4.弹性网络回归

### &ensp;&ensp;&ensp;&ensp;4.1 原理 ：岭回归+套索回归。

## 二、非线性算法

### 1.K近邻（KNN）

### &ensp;&ensp;&ensp;&ensp;1.1 原理 ：按照距离预测结果

### 2.分类与决策树（CART）

### 3.支持向量机（SVM）

## 三、测试


```python
from matplotlib import pyplot
from pandas import read_csv
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# 导入数据
boston = load_boston() 

# 将数据分为输入数据和输出结果
X = boston.data
Y = boston.target


num_folds = 10
seed = 7
kfold = KFold(n_splits=n_splits, random_state=seed)
models = {}

# 线性回归
models['LR'] =  LinearRegression()
# 岭回归
models['RD'] = Ridge()
# 套索
models['LO'] = Lasso()
# 弹性网络
models['EN'] = ElasticNet()
# K近邻
models['KNN'] = KNeighborsRegressor()
# 分类与决策树
models['DT'] = DecisionTreeRegressor()
# 支持向量机
models['SVM'] = SVR()

scoring = 'neg_mean_squared_error'

results = []
for name in models:
    result = cross_val_score(models[name], X, Y, cv=kfold, scoring=scoring)
    results.append(result)
    msg = '%s: %.3f (%.3f)' % (name, result.mean(), result.std())
    print(msg)

# 图表显示
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()
```

    KNN: -107.287 (79.840)
    EN: -31.163 (22.706)
    LO: -34.468 (27.886)
    RD: -34.135 (45.951)
    SVM: -91.048 (71.102)
    LR: -34.763 (45.614)
    DT: -41.356 (31.795)
    


![png](output_17_1.png)


## 四、git与参考

### git ：https://coding.net/u/RuoYun/p/Python-of-machine-learning/git/tree/master

### 参考：https://read.douban.com/column/6939417/
