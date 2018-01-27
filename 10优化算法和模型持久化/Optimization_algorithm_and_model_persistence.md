
## 一、机器学习算法的参数

### &nbsp;&nbsp;&nbsp; &nbsp;1.影响准确度          ------优化
### &nbsp;&nbsp;&nbsp; &nbsp;2.防止过拟合          ------优化
### &nbsp;&nbsp;&nbsp; &nbsp;3.其他参数

## 二、网格搜索优化参数（参数少，3个以内）

### &nbsp;&nbsp;&nbsp; &nbsp;1.原理

### &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;它是通过对遍历已定义参数的列表来评估算法的参数，从而找到最有的参数

### &nbsp;&nbsp;&nbsp; &nbsp;2.示例


```python
from pandas import read_csv
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
# 导入数据
pima =pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',header=None)

# 将数据分为输入数据和输出结果
array = pima.values
X = array[:, 0:8]
Y = array[:, 8]

# 算法实例化
model = Ridge()
# 设置要遍历的参数
param_grid = {'alpha': [1, 0.1, 0.01, 0.001, 0]}
# 通过网格搜索查询最优参数
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
# 搜索结果
print('最高得分：%.3f' % grid.best_score_)
print('最优参数：%s' % grid.best_estimator_.alpha)
```

    最高得分：0.280
    最优参数：1
    

## 三、随机搜索优化参数（参数多，大于3个）

### &nbsp;&nbsp;&nbsp; &nbsp;1.原理

### &nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;随机搜索通过对固定次数的迭代，对算法的参数采用随机采样分布的方式，搜索合适的参数

### &nbsp;&nbsp;&nbsp; &nbsp;2.示例


```python
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
# 导入数据
pima =pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',header=None)

# 将数据分为输入数据和输出结果
array = pima.values
X = array[:, 0:8]
Y = array[:, 8]

# 算法实例化
model = Ridge()
# 设置要遍历的参数
param_grid = {'alpha': uniform()}
# 通过网格搜索查询最优参数
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, random_state=7)
grid.fit(X, Y)
# 搜索结果
print('最高得分：%.3f' % grid.best_score_)
print('最优参数：%s' % grid.best_estimator_.alpha)
```

    最高得分：0.280
    最优参数：0.977989511997
    

## 四、通过pickle来序列化和反序列化机器学习的模型


```python
# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

```

    0.755905511811
    

## 五、通过Joblib来序列化和发序列化机器学习的模型


```python
# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)

```

    0.755905511811
    

## 六、生成模型的技巧

### &nbsp;&nbsp;&nbsp; &nbsp;1.python版本保持一致

### &nbsp;&nbsp;&nbsp; &nbsp;2.序列化和反序列化版本一致

### &nbsp;&nbsp;&nbsp; &nbsp;3.记录算法调参过程
