
## 一、分离训练数据集和评估数据集


```python
# 通过卡方检验选定数据特征
import pandas
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 导入数据
iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 

# 将数据分为输入数据和输出结果
arrary = iris.values
X =arrary[:,0:4]

le = LabelEncoder()
le.fit(iris['Species'])   
Y = le.transform(iris['Species']) # 对花的类别进行编号处理
# 分为测试集和训练集
test_size = 0.33
seed = 6
X_train, X_test, Y_traing, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# 选择模型
model = LogisticRegression()
# 进行训练
model.fit(X_train, Y_traing)
# 用测试集查看训练结果
result = model.score(X_test, Y_test)
print("算法评估结果：%.3f%%" % (result * 100))
```

    算法评估结果：98.000%
    

## 二、K折交叉验证分离


```python
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
# 导入数据
iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 

# 将数据分为输入数据和输出结果
arrary = iris.values
X =arrary[:,0:4]

le = LabelEncoder()
le.fit(iris['Species'])   
Y = le.transform(iris['Species']) # 对花的类别进行编号处理

# K折交叉
n_splits=10 
seed=6
kfold=KFold(n_splits=n_splits,random_state=seed,shuffle=False)
# 选择模型
model = LogisticRegression()
# 查看训练结果
result=cross_val_score(model, X,Y,cv=kfold)
print("结果：%.3f,%.3f"%(result.mean()*100, result.std()*100)) 
```

    结果：88.000,14.847
    

## 三、弃一交叉验证分离


```python
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# 导入数据
iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 

# 将数据分为输入数据和输出结果
arrary = iris.values
X =arrary[:,0:4]

le = LabelEncoder()
le.fit(iris['Species'])   
Y = le.transform(iris['Species']) # 对花的类别进行编号处理
# 选择弃一交叉验证
loocv = LeaveOneOut()
# 选择模型
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv=loocv)
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))
```

    算法评估结果：95.333% (21.092%)
    

## 四、重复随机评估、训练集分离


```python
from pandas import read_csv
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
## 导入数据
iris =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
iris.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'] 

# 将数据分为输入数据和输出结果
arrary = iris.values
X =arrary[:,0:4]

le = LabelEncoder()
le.fit(iris['Species'])   
Y = le.transform(iris['Species']) # 对花的类别进行编号处理

# 重复10次分离，起到K交叉验证的作用
n_splits = 10
test_size = 0.33
seed = 6
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
# 选择模型
model = LogisticRegression()
# 查看结果
result = cross_val_score(model, X, Y, cv=kfold)
print("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std() * 100))
```

    算法评估结果：94.600% (2.973%)
    

## 五、总结

#### 5.1一般情况下，都会用：K折交叉验证来分离数据集。

#### 5.2 数据量比较大，或者算法效率较低的情况下会考虑：分离训练数据集和评估数据

#### 5.3 平衡评估算法，模型训练的速度以及数据集的大小:弃一交叉验证和重复随机评估、训练集分离

## 六、参考及git

### git: https://coding.net/u/RuoYun/p/Python-of-machine-learning/git

### 参考：https://read.douban.com/column/6939417/
