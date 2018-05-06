
## 一、导入标准库


```python
# python3
# Importing the libraries 导入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 使图像能够调整
%matplotlib notebook 
#中文字体显示  
plt.rc('font', family='SimHei', size=8)
```

## 二、导入数据


```python
dataset = pd.read_csv('./Churn_Modelling.csv') # 通过已知的顾客信息来预测顾客是否留在这个银行
dataset.head()
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
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = dataset.iloc[:, 3:13].values # 影响顾客是否去留的因素
y = dataset.iloc[:, 13].values     # 顾客去留
X,y
```




    (array([[619, 'France', 'Female', ..., 1, 1, 101348.88],
            [608, 'Spain', 'Female', ..., 0, 1, 112542.58],
            [502, 'France', 'Female', ..., 1, 0, 113931.57],
            ..., 
            [709, 'France', 'Female', ..., 0, 1, 42085.58],
            [772, 'Germany', 'Male', ..., 1, 0, 92888.52],
            [792, 'France', 'Female', ..., 1, 0, 38190.78]], dtype=object),
     array([1, 0, 1, ..., 1, 1, 0], dtype=int64))



## 三、虚拟变量处理


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1= LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # 将国家变为数字

```


```python
labelencoder_X_2= LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # 将性别变为数字
onehotencoder = OneHotEncoder(categorical_features = [1])# 将国家变为虚拟变量（避免三个值有前后顺序）
X = onehotencoder.fit_transform(X).toarray()    
X = X[:,1:] # 删去第0列，避免虚拟变量陷阱
X
```




    array([[  0.00000000e+00,   0.00000000e+00,   6.19000000e+02, ...,
              1.00000000e+00,   1.00000000e+00,   1.01348880e+05],
           [  0.00000000e+00,   1.00000000e+00,   6.08000000e+02, ...,
              0.00000000e+00,   1.00000000e+00,   1.12542580e+05],
           [  0.00000000e+00,   0.00000000e+00,   5.02000000e+02, ...,
              1.00000000e+00,   0.00000000e+00,   1.13931570e+05],
           ..., 
           [  0.00000000e+00,   0.00000000e+00,   7.09000000e+02, ...,
              0.00000000e+00,   1.00000000e+00,   4.20855800e+04],
           [  1.00000000e+00,   0.00000000e+00,   7.72000000e+02, ...,
              1.00000000e+00,   0.00000000e+00,   9.28885200e+04],
           [  0.00000000e+00,   0.00000000e+00,   7.92000000e+02, ...,
              1.00000000e+00,   0.00000000e+00,   3.81907800e+04]])



## 四、区分训练集和测试集


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
```

## 五、创建神经网络


```python
# 导入标准库
import keras
from keras.models import Sequential
from keras.layers import Dense
```

## 六、ANN


```python
classifier = Sequential()
# 隐藏层 RELU（线性整流函数），输出层用S函数 

# 添加第一个隐藏层
classifier.add(Dense(units=6 ,activation='relu' ,kernel_initializer='uniform' ,input_dim = 11))  # 输入11个变量，输出1、所以隐藏层设置输入+输出/2,所以设置为6

# 添加第二个隐藏层
classifier.add(Dense(units=6 ,activation='relu' ,kernel_initializer='uniform' ))  # 输入11个变量，输出1、所以隐藏层设置输入+输出/2,所以设置为6

# 输出层
classifier.add(Dense(units=1 ,activation='sigmoid' ,kernel_initializer='uniform' ))   # 输出大于1个时，kernel_initializer 为softmax

# 编译
classifier.compile(optimizer= 'adam',loss= 'binary_crossentropy' , metrics=['accuracy']) # 分类结果2个binary_crossentropy,3个及其以上categorial_crossentropy

# 拟合
classifier.fit(X_train,y_train, batch_size= 10, epochs= 100)
```

    Epoch 1/100
    8000/8000 [==============================] - 60s 8ms/step - loss: 0.5448 - acc: 0.7927
    Epoch 2/100
    8000/8000 [==============================] - 54s 7ms/step - loss: 0.5227 - acc: 0.7960
    Epoch 3/100
    8000/8000 [==============================] - 54s 7ms/step - loss: 0.5118 - acc: 0.7960
    Epoch 4/100
    8000/8000 [==============================] - 38s 5ms/step - loss: 0.5085 - acc: 0.7960
    Epoch 5/100
    8000/8000 [==============================] - 36s 4ms/step - loss: 0.5058 - acc: 0.7960
    Epoch 6/100
    8000/8000 [==============================] - 50s 6ms/step - loss: 0.5034 - acc: 0.7960:  - ETA: 12s - loss: 0.5041 - ETA: 11s -  - ETA: 10s - loss: 0. - ETA: 8s - loss: 0.5051 - - ETA: 7s - loss: 0.5060 - acc: 0.7 - ETA: 7s - ETA: - ETA - ETA: 0s - loss: 0.5036 - 
    Epoch 7/100
    8000/8000 [==============================] - 55s 7ms/step - loss: 0.5047 - acc: 0.7960
    Epoch 8/100
    8000/8000 [==============================] - 57s 7ms/step - loss: 0.5023 - acc: 0.7960A: 9s - l - ETA: 7s - loss: 0.5052 - acc - ETA: 7s - loss: 0.5043 - ac - ETA: 6s - loss: 0.505 - ETA: 5s - l - ETA: 2s - loss
    Epoch 9/100
    8000/8000 [==============================] - 43s 5ms/step - loss: 0.5009 - acc: 0.7960: 0s - loss: 0.5026
    Epoch 10/100
    8000/8000 [==============================] - 44s 6ms/step - loss: 0.5015 - acc: 0.7960- 
    Epoch 11/100
    8000/8000 [==============================] - 48s 6ms/step - loss: 0.5005 - acc: 0.7960TA: 1s - loss: 0.4
    Epoch 12/100
    8000/8000 [==============================] - 53s 7ms/step - loss: 0.5002 - acc: 0.7960: 20s - loss: 0.5038 - - ETA: 19s  - ETA: 4s - loss: 0.5004 - acc: 0.795 - ETA: 4s - loss: 0.5008 - acc: 0.795 - ETA: 4s - loss: 0.5008 - - ETA: 3s - loss: 0.5026 - acc: 0 - ETA: 3s - loss: 0.5027 - ac - ETA: 3s - loss: 0.5029 - acc: 0.7 - ETA: 2s - loss: - ETA: 1s - loss: 0.502
    Epoch 13/100
    8000/8000 [==============================] - 52s 6ms/step - loss: 0.5017 - acc: 0.7960: 11s - loss: 0.5058 - ETA: 10s - loss: 0.5054 - acc: 0 - ETA: 9s - loss: 0.5054 - acc: 0 - ETA: 9s - loss: 0.5053 - acc:  - ETA: 6s - loss: 0.50 - ETA: 5s - l
    Epoch 14/100
    8000/8000 [==============================] - 47s 6ms/step - loss: 0.5008 - acc: 0.7960
    Epoch 15/100
    8000/8000 [==============================] - 38s 5ms/step - loss: 0.5015 - acc: 0.7960
    Epoch 16/100
    8000/8000 [==============================] - 26s 3ms/step - loss: 0.5004 - acc: 0.7960
    Epoch 17/100
    8000/8000 [==============================] - 25s 3ms/step - loss: 0.5001 - acc: 0.7960
    Epoch 18/100
    8000/8000 [==============================] - 24s 3ms/step - loss: 0.5002 - acc: 0.7960
    Epoch 19/100
    8000/8000 [==============================] - 26s 3ms/step - loss: 0.5006 - acc: 0.7960
    Epoch 20/100
    8000/8000 [==============================] - 24s 3ms/step - loss: 0.5000 - acc: 0.7960
    Epoch 21/100
    8000/8000 [==============================] - 25s 3ms/step - loss: 0.4999 - acc: 0.7960
    Epoch 22/100
    8000/8000 [==============================] - 26s 3ms/step - loss: 0.5005 - acc: 0.7960
    Epoch 23/100
    8000/8000 [==============================] - 25s 3ms/step - loss: 0.5001 - acc: 0.7960
    Epoch 24/100
    8000/8000 [==============================] - 27s 3ms/step - loss: 0.5001 - acc: 0.7960: 0s - loss: 0.4988 - a
    Epoch 25/100
    8000/8000 [==============================] - 24s 3ms/step - loss: 0.4994 - acc: 0.7960:
    Epoch 26/100
    8000/8000 [==============================] - 22s 3ms/step - loss: 0.5005 - acc: 0.7960
    Epoch 27/100
    8000/8000 [==============================] - 20s 3ms/step - loss: 0.5006 - acc: 0.7960
    Epoch 28/100
    8000/8000 [==============================] - 20s 2ms/step - loss: 0.5001 - acc: 0.7960
    Epoch 29/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.4997 - acc: 0.7960
    Epoch 30/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.4998 - acc: 0.7960
    Epoch 31/100
    8000/8000 [==============================] - 20s 2ms/step - loss: 0.4997 - acc: 0.7960
    Epoch 32/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.5009 - acc: 0.7960
    Epoch 33/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.4999 - acc: 0.7960
    Epoch 34/100
    8000/8000 [==============================] - 20s 2ms/step - loss: 0.4998 - acc: 0.7960
    Epoch 35/100
    8000/8000 [==============================] - 22s 3ms/step - loss: 0.5002 - acc: 0.7960
    Epoch 36/100
    8000/8000 [==============================] - 23s 3ms/step - loss: 0.5002 - acc: 0.7960
    Epoch 37/100
    8000/8000 [==============================] - 25s 3ms/step - loss: 0.5016 - acc: 0.7960
    Epoch 38/100
    8000/8000 [==============================] - 24s 3ms/step - loss: 0.5006 - acc: 0.7960
    Epoch 39/100
    8000/8000 [==============================] - 21s 3ms/step - loss: 0.5011 - acc: 0.7960
    Epoch 40/100
    8000/8000 [==============================] - 20s 2ms/step - loss: 0.5000 - acc: 0.7960
    Epoch 41/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.5005 - acc: 0.7960
    Epoch 42/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.4999 - acc: 0.7960
    Epoch 43/100
    8000/8000 [==============================] - 20s 3ms/step - loss: 0.4994 - acc: 0.7960
    Epoch 44/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.4990 - acc: 0.7960
    Epoch 45/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.5001 - acc: 0.7960
    Epoch 46/100
    8000/8000 [==============================] - 22s 3ms/step - loss: 0.4995 - acc: 0.7960
    Epoch 47/100
    8000/8000 [==============================] - 19s 2ms/step - loss: 0.4995 - acc: 0.7960
    Epoch 48/100
    8000/8000 [==============================] - 20s 2ms/step - loss: 0.4992 - acc: 0.7960
    Epoch 49/100
    8000/8000 [==============================] - 26s 3ms/step - loss: 0.4998 - acc: 0.7960: 0s - loss: 0.4981 - acc: 0.79 - ETA: 0s - loss: 0.4979 - acc: 0.796 - ETA: 0s - loss: 0.4981 -
    Epoch 50/100
    8000/8000 [==============================] - 34s 4ms/step - loss: 0.4991 - acc: 0.7960
    Epoch 51/100
    8000/8000 [==============================] - 54s 7ms/step - loss: 0.5002 - acc: 0.7960
    Epoch 52/100
    8000/8000 [==============================] - 54s 7ms/step - loss: 0.5000 - acc: 0.7960
    Epoch 53/100
    8000/8000 [==============================] - 53s 7ms/step - loss: 0.4990 - acc: 0.7960TA: 7s - loss: 0.4976 - acc: 0.797 - ETA: 7s - loss: 0.4976 - a
    Epoch 54/100
    8000/8000 [==============================] - 49s 6ms/step - loss: 0.4994 - acc: 0.7960
    Epoch 55/100
    8000/8000 [==============================] - 58s 7ms/step - loss: 0.4987 - acc: 0.7960: 1s - loss: 0.49
    Epoch 56/100
    8000/8000 [==============================] - 54s 7ms/step - loss: 0.4993 - acc: 0.7960
    Epoch 57/100
    8000/8000 [==============================] - 46s 6ms/step - loss: 0.4999 - acc: 0.7960
    Epoch 58/100
    8000/8000 [==============================] - ETA: 0s - loss: 0.4995 - acc: 0.795 - 53s 7ms/step - loss: 0.4992 - acc: 0.7960
    Epoch 59/100
    8000/8000 [==============================] - 47s 6ms/step - loss: 0.5000 - acc: 0.7960
    Epoch 60/100
    8000/8000 [==============================] - 46s 6ms/step - loss: 0.5004 - acc: 0.7960
    Epoch 61/100
    8000/8000 [==============================] - 44s 6ms/step - loss: 0.4989 - acc: 0.7960: 6 - ET
    Epoch 62/100
    8000/8000 [==============================] - 51s 6ms/step - loss: 0.4993 - acc: 0.7960: 9s - loss: 0.4955 - acc:  - 
    Epoch 63/100
    8000/8000 [==============================] - 49s 6ms/step - loss: 0.5003 - acc: 0.7960
    Epoch 64/100
    8000/8000 [==============================] - 46s 6ms/step - loss: 0.4995 - acc: 0.7960
    Epoch 65/100
    8000/8000 [==============================] - 50s 6ms/step - loss: 0.5001 - acc: 0.7960: 9s  - ETA: 7s -
    Epoch 66/100
    8000/8000 [==============================] - 51s 6ms/step - loss: 0.4994 - acc: 0.7960
    Epoch 67/100
    8000/8000 [==============================] - 47s 6ms/step - loss: 0.4987 - acc: 0.7960
    Epoch 68/100
    8000/8000 [==============================] - 53s 7ms/step - loss: 0.4995 - acc: 0.7960: 1s - loss: 0.4
    Epoch 69/100
    8000/8000 [==============================] - 65s 8ms/step - loss: 0.4998 - acc: 0.7960: - ETA: 1s - loss: 0.5009 - acc: 0.795 - ETA: 0s - loss: 0.5009 - acc: 0. - ETA: 0s - loss: 0.4999 - acc: 0.79 - ETA: 0s - loss: 0.5002 - acc: 
    Epoch 70/100
    8000/8000 [==============================] - 57s 7ms/step - loss: 0.4985 - acc: 0.7960: 10s - loss: 0.5001 - ETA: 10s -
    Epoch 71/100
    8000/8000 [==============================] - 54s 7ms/step - loss: 0.4995 - acc: 0.7960: 3s - loss: 0.5006 - - ETA: 2s - loss:  - ETA: 1s - loss: 0
    Epoch 72/100
    8000/8000 [==============================] - 55s 7ms/step - loss: 0.4989 - acc: 0.7960
    Epoch 73/100
    8000/8000 [==============================] - ETA: 0s - loss: 0.4991 - acc: 0.7959  E - ETA: 10s - loss: 0.5004 - acc: 0. - ETA: 10s - loss:  - ETA: 9s - loss: 0.5019 - a - ETA: 8s - loss: 0.5035 - a - 56s 7ms/step - loss: 0.4989 - acc: 0.7960
    Epoch 74/100
    8000/8000 [==============================] - 50s 6ms/step - loss: 0.5010 - acc: 0.7960: 0s - loss: 0.5010 - acc: 0.79
    Epoch 75/100
    8000/8000 [==============================] - ETA: 0s - loss: 0.4992 - acc: 0.7962- ETA: 4s - loss: 0.5021 - acc - ETA: 4s - loss: 0.5014 - ac - ETA - ETA: 1s - lo - 52s 7ms/step - loss: 0.4995 - acc: 0.7960
    Epoch 76/100
    8000/8000 [==============================] - 56s 7ms/step - loss: 0.4992 - acc: 0.7960: 7s - loss: 0.4997 - acc: 
    Epoch 77/100
    8000/8000 [==============================] - 39s 5ms/step - loss: 0.5010 - acc: 0.7960
    Epoch 78/100
    8000/8000 [==============================] - 53s 7ms/step - loss: 0.4990 - acc: 0.7960: 1s - loss: 0.4
    Epoch 79/100
    8000/8000 [==============================] - 60s 8ms/step - loss: 0.5000 - acc: 0.7960: 3s - - ETA: 1s - loss: 
    Epoch 80/100
    8000/8000 [==============================] - ETA: 0s - loss: 0.4999 - acc: 0.7957- ETA:  - 67s 8ms/step - loss: 0.4995 - acc: 0.7960
    Epoch 81/100
    8000/8000 [==============================] - 73s 9ms/step - loss: 0.4996 - acc: 0.7960TA: 0s - loss: 0.4986 - acc
    Epoch 82/100
    8000/8000 [==============================] - 72s 9ms/step - loss: 0.4990 - acc: 0.7960: 5s - loss: 0.4 - ETA: 4s - loss: 0 - ETA: 2s - 
    Epoch 83/100
    8000/8000 [==============================] - 77s 10ms/step - loss: 0.4984 - acc: 0.7960 7s - loss: 0.4975 -  - ETA: 5s - los -
    Epoch 84/100
    8000/8000 [==============================] - 69s 9ms/step - loss: 0.4987 - acc: 0.7960: 18s - loss: 0.501
    Epoch 85/100
    8000/8000 [==============================] - 64s 8ms/step - loss: 0.4985 - acc: 0.7960
    Epoch 86/100
    8000/8000 [==============================] - 68s 9ms/step - loss: 0.4995 - acc: 0.7960
    Epoch 87/100
    8000/8000 [==============================] - 70s 9ms/step - loss: 0.4993 - acc: 0.7960
    Epoch 88/100
    8000/8000 [==============================] - 66s 8ms/step - loss: 0.4985 - acc: 0.7960: 6s - loss: 0.4965 - acc: 0.797 - ETA: 
    Epoch 89/100
    8000/8000 [==============================] - 75s 9ms/step - loss: 0.4983 - acc: 0.7960: 9s - loss: 0.4954 - - ETA: 8s - loss: 0.4946 - acc: 0.7 - ETA: 8s - loss: 0.4946 - ETA: 3s - loss:  - ETA: 1s - loss: 0.4978 - acc: 0 - ETA: 1s - loss: 0.4981 - acc: 0. - ETA: 0s - loss: 0.4979 - ac
    Epoch 90/100
    8000/8000 [==============================] - 78s 10ms/step - loss: 0.4986 - acc: 0.7960
    Epoch 91/100
    8000/8000 [==============================] - 74s 9ms/step - loss: 0.5003 - acc: 0.7960: 0s - loss: 0.5003 - acc: 
    Epoch 92/100
    8000/8000 [==============================] - 72s 9ms/step - loss: 0.4997 - acc: 0.7960
    Epoch 93/100
    8000/8000 [==============================] - 56s 7ms/step - loss: 0.4994 - acc: 0.7960: 16s - loss: 0.5027 - acc: 0.79 - ETA: 16 - ETA: 14s  - ETA: 10s - loss:  - ETA: 9s - loss: 0.5027  - ETA: 7s - loss: 0.5029 - acc:  - ETA: 7s - lo
    Epoch 94/100
    8000/8000 [==============================] - 51s 6ms/step - loss: 0.4989 - acc: 0.7960
    Epoch 95/100
    8000/8000 [==============================] - 45s 6ms/step - loss: 0.4998 - acc: 0.7960
    Epoch 96/100
    8000/8000 [==============================] - 36s 5ms/step - loss: 0.4990 - acc: 0.7960
    Epoch 97/100
    8000/8000 [==============================] - 35s 4ms/step - loss: 0.4994 - acc: 0.7960: 1s - loss: 0.50
    Epoch 98/100
    8000/8000 [==============================] - 43s 5ms/step - loss: 0.4992 - acc: 0.7960: 5s - loss: 0.4940 - acc: 0.799 - ETA: 5s - loss: - ETA: 4s - loss: 0.4965 - acc: 0.79 - ETA: 4s - loss: 0.4964  - ETA
    Epoch 99/100
    8000/8000 [==============================] - 56s 7ms/step - loss: 0.4990 - acc: 0.7960
    Epoch 100/100
    8000/8000 [==============================] - 56s 7ms/step - loss: 0.4989 - acc: 0.7960
    




    <keras.callbacks.History at 0xda305b0>



## 七、预测测试集的结果


```python
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
```

## 八、混淆矩阵


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm
```




    array([[1595,    0],
           [ 405,    0]], dtype=int64)



## 九、项目地址

### https://coding.net/u/RuoYun/p/Python-of-machine-learning/git/blob/master/00%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/5.%E8%BF%9B%E9%98%B6%E7%AE%97%E6%B3%95/2.%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/1.%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/0.txt?public=true
