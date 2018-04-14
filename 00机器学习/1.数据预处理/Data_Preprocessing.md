
## 一、导入标准库


```python
# Importing the libraries 导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## 二、导入数据


```python
# Importing the dataset 导入数据
dataset = pd.read_csv('./Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
dataset
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Age</th>
      <th>Salary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>France</td>
      <td>44.0</td>
      <td>72000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Spain</td>
      <td>27.0</td>
      <td>48000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Germany</td>
      <td>30.0</td>
      <td>54000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Spain</td>
      <td>38.0</td>
      <td>61000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Germany</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>France</td>
      <td>35.0</td>
      <td>58000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>NaN</td>
      <td>52000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>7</th>
      <td>France</td>
      <td>48.0</td>
      <td>79000.0</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>50.0</td>
      <td>83000.0</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>France</td>
      <td>37.0</td>
      <td>67000.0</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
X
```




    array([['France', 44.0, 72000.0],
           ['Spain', 27.0, 48000.0],
           ['Germany', 30.0, 54000.0],
           ['Spain', 38.0, 61000.0],
           ['Germany', 40.0, nan],
           ['France', 35.0, 58000.0],
           ['Spain', nan, 52000.0],
           ['France', 48.0, 79000.0],
           ['Germany', 50.0, 83000.0],
           ['France', 37.0, 67000.0]], dtype=object)




```python
y
```




    array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'], dtype=object)



## 三、处理缺失数据


```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) # 将na用平均值代替
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X
```




    array([['France', 44.0, 72000.0],
           ['Spain', 27.0, 48000.0],
           ['Germany', 30.0, 54000.0],
           ['Spain', 38.0, 61000.0],
           ['Germany', 40.0, 63777.77777777778],
           ['France', 35.0, 58000.0],
           ['Spain', 38.77777777777778, 52000.0],
           ['France', 48.0, 79000.0],
           ['Germany', 50.0, 83000.0],
           ['France', 37.0, 67000.0]], dtype=object)



## 四、数据转换


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # 将国家变成数字
X
```




    array([[0, 44.0, 72000.0],
           [2, 27.0, 48000.0],
           [1, 30.0, 54000.0],
           [2, 38.0, 61000.0],
           [1, 40.0, 63777.77777777778],
           [0, 35.0, 58000.0],
           [2, 38.77777777777778, 52000.0],
           [0, 48.0, 79000.0],
           [1, 50.0, 83000.0],
           [0, 37.0, 67000.0]], dtype=object)




```python
onehotencoder = OneHotEncoder(categorical_features = [0]) # 将国家数字变成虚拟编码
X = onehotencoder.fit_transform(X).toarray()
X
```




    array([[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              4.40000000e+01,   7.20000000e+04],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              2.70000000e+01,   4.80000000e+04],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              3.00000000e+01,   5.40000000e+04],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              3.80000000e+01,   6.10000000e+04],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              4.00000000e+01,   6.37777778e+04],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              3.50000000e+01,   5.80000000e+04],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              3.87777778e+01,   5.20000000e+04],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              4.80000000e+01,   7.90000000e+04],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              5.00000000e+01,   8.30000000e+04],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              3.70000000e+01,   6.70000000e+04]])




```python
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) # 将是否购买变成数字
y
```




    array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])



## 五、区分训练集和测试集


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

```

## 六、特征缩放


```python
X_test
```




    array([[  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              3.00000000e+01,   5.40000000e+04],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              5.00000000e+01,   8.30000000e+04]])




```python
from sklearn.preprocessing import StandardScaler  # 导入标准化库
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_test
```




    array([[-1.        ,  2.64575131, -0.77459667, -1.45882927, -0.90166297],
           [-1.        ,  2.64575131, -0.77459667,  1.98496442,  2.13981082]])




```python
y
```




    array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])


