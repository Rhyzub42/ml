
## 一、数据导入


```python
import pandas
from pandas import set_option
#括号里面直接指定了数据的来源，当然你也可以按照老师视频中所讲授的来操作
iris = pandas.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
iris.columns=['sepal_length','sepal_width','petal_length','petal_width','species']
```

## 二、查看前10个数据


```python
print (iris.head(10))
```

       sepal_length  sepal_width  petal_length  petal_width      species
    0           4.9          3.0           1.4          0.2  Iris-setosa
    1           4.7          3.2           1.3          0.2  Iris-setosa
    2           4.6          3.1           1.5          0.2  Iris-setosa
    3           5.0          3.6           1.4          0.2  Iris-setosa
    4           5.4          3.9           1.7          0.4  Iris-setosa
    5           4.6          3.4           1.4          0.3  Iris-setosa
    6           5.0          3.4           1.5          0.2  Iris-setosa
    7           4.4          2.9           1.4          0.2  Iris-setosa
    8           4.9          3.1           1.5          0.1  Iris-setosa
    9           5.4          3.7           1.5          0.2  Iris-setosa
    

#### 数据集中主要有sepal_length、sepal_width、petal_length、petal_width。这四个属性决定最终花的种类

## 三、查看数据的维度


```python
print(iris.shape)
```

    (149, 5)
    

## 四、查看数据的属性和类型


```python
print(iris.dtypes)
```

    sepal_length    float64
    sepal_width     float64
    petal_length    float64
    petal_width     float64
    species          object
    dtype: object
    

## 五、描述性统计


```python
set_option('display.width', 100)
# 设置数据的精确度
set_option('precision', 4)
print(iris.describe())
```

           sepal_length  sepal_width  petal_length  petal_width
    count      149.0000     149.0000      149.0000     149.0000
    mean         5.8483       3.0510        3.7745       1.2054
    std          0.8286       0.4335        1.7597       0.7613
    min          4.3000       2.0000        1.0000       0.1000
    25%          5.1000       2.8000        1.6000       0.3000
    50%          5.8000       3.0000        4.4000       1.3000
    75%          6.4000       3.3000        5.1000       1.8000
    max          7.9000       4.4000        6.9000       2.5000
    

## 六、数据分组分布（适用于分类算法）


```python
print(iris.groupby('species').size())
```

    species
    Iris-setosa        49
    Iris-versicolor    50
    Iris-virginica     50
    dtype: int64
    

#### 样本中每种花的样本数量差距不大

## 七、数据属性的相关性


```python
set_option('display.width', 100)
# 设置数据的精确度
set_option('precision', 2) 
print(iris.corr(method='pearson')) # 皮尔逊相关系数判断相关性:1 表示变量完全正相关， 0 表示无关，-1 表示完全负相关。
```

                  sepal_length  sepal_width  petal_length  petal_width
    sepal_length          1.00        -0.10          0.87         0.82
    sepal_width          -0.10         1.00         -0.42        -0.35
    petal_length          0.87        -0.42          1.00         0.96
    petal_width           0.82        -0.35          0.96         1.00
    

#### 在机器学习中，有些算法（linear，逻辑回归算法等），当数据的关联性比较高时，算法的性能会降低。这为后续的训练模型构建起到帮助

## 八、数据的分布分析


```python
print(iris.skew())  # skew()函数的结果，显示了数据分布的左偏或右偏。当数据接近0是，表示数据的偏差非常小。
```

    sepal_length    0.30
    sepal_width     0.35
    petal_length   -0.29
    petal_width    -0.12
    dtype: float64
    

## 九、总结

### 审查数字：通常描述性分析给出的数据是不充分的。我们应该停下来观察，思考一下我们的数据。找到数据的内在联系和给我们带来了什么。
### 问为什么：审查数字后多问几个为什么。你是如何看到和为什么看到这些数字的特殊性。思考一下这些数字和问题如何关联到一起。这些数字和我们的问题有什么关系等。
### 写下想法：写下自己通过观察得到的想法。通过便签，记事本等将观察到的数据各维度的关联关系，我们的想法都记录下来。数字代表什么，我们将采用什么样的技术继续挖掘数据等。你写下的这些想法将会对你的新的尝试带了极大的价值。



## 十、参考及git
### git: https://coding.net/u/RuoYun/p/Python-of-machine-learning/git/tree/master
### 参考：https://read.douban.com/reader/column/6939417/chapter/35749876/

