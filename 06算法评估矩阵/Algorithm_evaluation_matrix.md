
## 一、算法评估矩阵

#### 目的：合理有效的评估算法，寻找最适合的算法和参数

## 二、分类算法矩阵

### 2.1 分类正确率：分对了多少


```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas

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
# 进行训练
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
result = cross_val_score(model, X, Y, cv=kfold)
print("算法评估结果准确度：%.3f (%.3f)" % (result.mean(), result.std()))
```

    算法评估结果准确度：0.880 (0.148)
    

### 2.2 对数损失函数


```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# 导入数据
pima =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',header=None)

# 将数据分为输入数据和输出结果
array = pima.values
X = array[:, 0:8]
Y = array[:, 8]
# 进行训练
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
scoring = 'neg_log_loss'
result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('Logloss %.3f (%.3f)' % (result.mean(), result.std()))
```

    Logloss -0.493 (0.047)
    

### 2.3 AUC图


```python
from pandas import read_csv
import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


# 导入数据
pima =pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',header=None)

# 将数据分为输入数据和输出结果
array = pima.values
X = array[:, 0:8]
Y = array[:, 8]
# 进行训练
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = LogisticRegression()
scoring = 'roc_auc'
result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('AUC %.3f (%.3f)' % (result.mean(), result.std()))
```

    AUC 0.824 (0.041)
    

### 2.4 混淆矩阵

## &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;预&nbsp;测 
## &nbsp;&nbsp; &nbsp;  &nbsp;&nbsp;1  &nbsp; 0
## 实  1   X  Y
## 际  0   Z  A


```python
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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
# 进行训练
test_size = 0.33
seed = 4
X_train, X_test, Y_traing, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_traing)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)
classes = ['0', '1', '2']
dataframe = pd.DataFrame(data=matrix,index=classes,columns=classes)
print(dataframe)

```

        0   1   2
    0  23   0   0
    1   0  10   2
    2   0   1  14
    

### 2.5 分类报告

### precision：精确率（真正/真正+假正）-- 找的准
### recall       ：召回率（真正/真正+假负）--找的全
### f1-score   ：精确值和召回率的调和均值,也就是2F1=1P+1R 
### support：样本数目


```python
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 将数据分为输入数据和输出结果
arrary = iris.values
# print(arrary)
X =arrary[:,0:4]

le = LabelEncoder()
le.fit(iris['Species'])   
Y = le.transform(iris['Species']) # 对花的类别进行编号处理
# 进行训练
test_size = 0.33
seed = 4
X_train, X_test, Y_traing, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_traing)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)
```

                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00        23
              1       0.91      0.83      0.87        12
              2       0.88      0.93      0.90        15
    
    avg / total       0.94      0.94      0.94        50
    
    

## 三、回归算法矩阵

### 3.1平均绝对误差：平均绝对误差能更好地反映预测值误差的实际情况


```python
from pandas import read_csv
import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston 
# 导入数据
boston = load_boston() 

# 将数据分为输入数据和输出结果
X = boston.data
Y = boston.target

n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('MAE: %.3f (%.3f)' % (result.mean(), result.std()))
```

    MAE: -4.010 (2.086)
    

### 3.2均方误差：可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。


```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# 导入数据
boston = load_boston() 

# 将数据分为输入数据和输出结果
X = boston.data
Y = boston.target

n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('MSE: %.3f (%.3f)' % (result.mean(), result.std()))

```

    MSE: -34.763 (45.614)
    

### 3.1决定系数：说明列入模型的所有解释变量对因变量的联合的影响程度


```python
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# 导入数据
boston = load_boston() 

# 将数据分为输入数据和输出结果
X = boston.data
Y = boston.target

n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits, random_state=seed)
model = LinearRegression()
scoring = 'r2'
result = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print('MSE: %.3f (%.3f)' % (result.mean(), result.std()))

```

    MSE: 0.200 (0.599)
    
