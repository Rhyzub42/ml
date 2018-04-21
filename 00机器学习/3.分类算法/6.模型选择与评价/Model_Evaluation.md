
## 一、伪阳性和伪阴性

### 1. 伪阳性----I型错误，伪阴性---II型错误。

### 2. II型错误要比I型错误严重的多

## 二、混淆矩阵


```python
### y 预测        0       1
```


```python
### y实际 0      35     5
```


```python
###      1      10     50
```

### 5位I型错误（伪阳），10为II型错误（伪阴）

###  准确率： （35+50）/100 = 85%

###  错误率： （5+10）/100 = 15%

## 三、混淆矩阵悖论（不能单靠混淆矩阵判断）

## 四、累计准确曲线CAP curve

###  ![avatar](./1.png)

### 蓝色：随机曲线 、黑色;完美曲线（水晶球曲线）、红色：较好模型的曲线

## 五、累计准确曲线分析CAP curve Ansy

###  ![avatar](./2.png)

###  ![avatar](./3.png)

## 六、项目地址

### https://coding.net/u/RuoYun/p/Python-of-machine-learning/git/tree/master/00%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/3.%E5%88%86%E7%B1%BB%E7%AE%97%E6%B3%95/6.%E6%A8%A1%E5%9E%8B%E9%80%89%E6%8B%A9%E4%B8%8E%E8%AF%84%E4%BB%B7?public=true
