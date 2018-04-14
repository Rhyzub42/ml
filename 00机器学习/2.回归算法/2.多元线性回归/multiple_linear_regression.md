
## 零、模型

### 0.1、模型介绍

### y = b0 + b1X1+B2X2+BnXn

### 0.2、限定条件

#### 1.线性、2.同方差性、3.多元正太分布、4.误差独立、5.无多重共线性

## 0.3 模型的建立方法

### 1.全部选取  ：反向淘汰的第一步、必须全部选取的时候、先验知识

### 2.反向淘汰  ：自变量对于P值的影响， 计算每个自变量的P值，进行与自定义SL值比较。

### 3.顺向选择  ：每个变量是否能够进入模型，

### 4.双向淘汰 ： 选择两个显著性值，同时进行反向淘汰与顺向选择

### 5.信息量比较：对所有可能的模型进行打分

## 一、导入标准库


```python
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
# 根据各项开支预测利润
dataset = pd.read_csv('./50_Startups.csv')
X = dataset.iloc[:, :-1].values  # 选取自变量
y = dataset.iloc[:, 4].values    # 选取因变量
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
      <th>R&amp;D Spend</th>
      <th>Administration</th>
      <th>Marketing Spend</th>
      <th>State</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165349.20</td>
      <td>136897.80</td>
      <td>471784.10</td>
      <td>New York</td>
      <td>192261.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>162597.70</td>
      <td>151377.59</td>
      <td>443898.53</td>
      <td>California</td>
      <td>191792.06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>153441.51</td>
      <td>101145.55</td>
      <td>407934.54</td>
      <td>Florida</td>
      <td>191050.39</td>
    </tr>
    <tr>
      <th>3</th>
      <td>144372.41</td>
      <td>118671.85</td>
      <td>383199.62</td>
      <td>New York</td>
      <td>182901.99</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142107.34</td>
      <td>91391.77</td>
      <td>366168.42</td>
      <td>Florida</td>
      <td>166187.94</td>
    </tr>
    <tr>
      <th>5</th>
      <td>131876.90</td>
      <td>99814.71</td>
      <td>362861.36</td>
      <td>New York</td>
      <td>156991.12</td>
    </tr>
    <tr>
      <th>6</th>
      <td>134615.46</td>
      <td>147198.87</td>
      <td>127716.82</td>
      <td>California</td>
      <td>156122.51</td>
    </tr>
    <tr>
      <th>7</th>
      <td>130298.13</td>
      <td>145530.06</td>
      <td>323876.68</td>
      <td>Florida</td>
      <td>155752.60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>120542.52</td>
      <td>148718.95</td>
      <td>311613.29</td>
      <td>New York</td>
      <td>152211.77</td>
    </tr>
    <tr>
      <th>9</th>
      <td>123334.88</td>
      <td>108679.17</td>
      <td>304981.62</td>
      <td>California</td>
      <td>149759.96</td>
    </tr>
    <tr>
      <th>10</th>
      <td>101913.08</td>
      <td>110594.11</td>
      <td>229160.95</td>
      <td>Florida</td>
      <td>146121.95</td>
    </tr>
    <tr>
      <th>11</th>
      <td>100671.96</td>
      <td>91790.61</td>
      <td>249744.55</td>
      <td>California</td>
      <td>144259.40</td>
    </tr>
    <tr>
      <th>12</th>
      <td>93863.75</td>
      <td>127320.38</td>
      <td>249839.44</td>
      <td>Florida</td>
      <td>141585.52</td>
    </tr>
    <tr>
      <th>13</th>
      <td>91992.39</td>
      <td>135495.07</td>
      <td>252664.93</td>
      <td>California</td>
      <td>134307.35</td>
    </tr>
    <tr>
      <th>14</th>
      <td>119943.24</td>
      <td>156547.42</td>
      <td>256512.92</td>
      <td>Florida</td>
      <td>132602.65</td>
    </tr>
    <tr>
      <th>15</th>
      <td>114523.61</td>
      <td>122616.84</td>
      <td>261776.23</td>
      <td>New York</td>
      <td>129917.04</td>
    </tr>
    <tr>
      <th>16</th>
      <td>78013.11</td>
      <td>121597.55</td>
      <td>264346.06</td>
      <td>California</td>
      <td>126992.93</td>
    </tr>
    <tr>
      <th>17</th>
      <td>94657.16</td>
      <td>145077.58</td>
      <td>282574.31</td>
      <td>New York</td>
      <td>125370.37</td>
    </tr>
    <tr>
      <th>18</th>
      <td>91749.16</td>
      <td>114175.79</td>
      <td>294919.57</td>
      <td>Florida</td>
      <td>124266.90</td>
    </tr>
    <tr>
      <th>19</th>
      <td>86419.70</td>
      <td>153514.11</td>
      <td>0.00</td>
      <td>New York</td>
      <td>122776.86</td>
    </tr>
    <tr>
      <th>20</th>
      <td>76253.86</td>
      <td>113867.30</td>
      <td>298664.47</td>
      <td>California</td>
      <td>118474.03</td>
    </tr>
    <tr>
      <th>21</th>
      <td>78389.47</td>
      <td>153773.43</td>
      <td>299737.29</td>
      <td>New York</td>
      <td>111313.02</td>
    </tr>
    <tr>
      <th>22</th>
      <td>73994.56</td>
      <td>122782.75</td>
      <td>303319.26</td>
      <td>Florida</td>
      <td>110352.25</td>
    </tr>
    <tr>
      <th>23</th>
      <td>67532.53</td>
      <td>105751.03</td>
      <td>304768.73</td>
      <td>Florida</td>
      <td>108733.99</td>
    </tr>
    <tr>
      <th>24</th>
      <td>77044.01</td>
      <td>99281.34</td>
      <td>140574.81</td>
      <td>New York</td>
      <td>108552.04</td>
    </tr>
    <tr>
      <th>25</th>
      <td>64664.71</td>
      <td>139553.16</td>
      <td>137962.62</td>
      <td>California</td>
      <td>107404.34</td>
    </tr>
    <tr>
      <th>26</th>
      <td>75328.87</td>
      <td>144135.98</td>
      <td>134050.07</td>
      <td>Florida</td>
      <td>105733.54</td>
    </tr>
    <tr>
      <th>27</th>
      <td>72107.60</td>
      <td>127864.55</td>
      <td>353183.81</td>
      <td>New York</td>
      <td>105008.31</td>
    </tr>
    <tr>
      <th>28</th>
      <td>66051.52</td>
      <td>182645.56</td>
      <td>118148.20</td>
      <td>Florida</td>
      <td>103282.38</td>
    </tr>
    <tr>
      <th>29</th>
      <td>65605.48</td>
      <td>153032.06</td>
      <td>107138.38</td>
      <td>New York</td>
      <td>101004.64</td>
    </tr>
    <tr>
      <th>30</th>
      <td>61994.48</td>
      <td>115641.28</td>
      <td>91131.24</td>
      <td>Florida</td>
      <td>99937.59</td>
    </tr>
    <tr>
      <th>31</th>
      <td>61136.38</td>
      <td>152701.92</td>
      <td>88218.23</td>
      <td>New York</td>
      <td>97483.56</td>
    </tr>
    <tr>
      <th>32</th>
      <td>63408.86</td>
      <td>129219.61</td>
      <td>46085.25</td>
      <td>California</td>
      <td>97427.84</td>
    </tr>
    <tr>
      <th>33</th>
      <td>55493.95</td>
      <td>103057.49</td>
      <td>214634.81</td>
      <td>Florida</td>
      <td>96778.92</td>
    </tr>
    <tr>
      <th>34</th>
      <td>46426.07</td>
      <td>157693.92</td>
      <td>210797.67</td>
      <td>California</td>
      <td>96712.80</td>
    </tr>
    <tr>
      <th>35</th>
      <td>46014.02</td>
      <td>85047.44</td>
      <td>205517.64</td>
      <td>New York</td>
      <td>96479.51</td>
    </tr>
    <tr>
      <th>36</th>
      <td>28663.76</td>
      <td>127056.21</td>
      <td>201126.82</td>
      <td>Florida</td>
      <td>90708.19</td>
    </tr>
    <tr>
      <th>37</th>
      <td>44069.95</td>
      <td>51283.14</td>
      <td>197029.42</td>
      <td>California</td>
      <td>89949.14</td>
    </tr>
    <tr>
      <th>38</th>
      <td>20229.59</td>
      <td>65947.93</td>
      <td>185265.10</td>
      <td>New York</td>
      <td>81229.06</td>
    </tr>
    <tr>
      <th>39</th>
      <td>38558.51</td>
      <td>82982.09</td>
      <td>174999.30</td>
      <td>California</td>
      <td>81005.76</td>
    </tr>
    <tr>
      <th>40</th>
      <td>28754.33</td>
      <td>118546.05</td>
      <td>172795.67</td>
      <td>California</td>
      <td>78239.91</td>
    </tr>
    <tr>
      <th>41</th>
      <td>27892.92</td>
      <td>84710.77</td>
      <td>164470.71</td>
      <td>Florida</td>
      <td>77798.83</td>
    </tr>
    <tr>
      <th>42</th>
      <td>23640.93</td>
      <td>96189.63</td>
      <td>148001.11</td>
      <td>California</td>
      <td>71498.49</td>
    </tr>
    <tr>
      <th>43</th>
      <td>15505.73</td>
      <td>127382.30</td>
      <td>35534.17</td>
      <td>New York</td>
      <td>69758.98</td>
    </tr>
    <tr>
      <th>44</th>
      <td>22177.74</td>
      <td>154806.14</td>
      <td>28334.72</td>
      <td>California</td>
      <td>65200.33</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1000.23</td>
      <td>124153.04</td>
      <td>1903.93</td>
      <td>New York</td>
      <td>64926.08</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1315.46</td>
      <td>115816.21</td>
      <td>297114.46</td>
      <td>Florida</td>
      <td>49490.75</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.00</td>
      <td>135426.92</td>
      <td>0.00</td>
      <td>California</td>
      <td>42559.73</td>
    </tr>
    <tr>
      <th>48</th>
      <td>542.05</td>
      <td>51743.15</td>
      <td>0.00</td>
      <td>New York</td>
      <td>35673.41</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.00</td>
      <td>116983.80</td>
      <td>45173.06</td>
      <td>California</td>
      <td>14681.40</td>
    </tr>
  </tbody>
</table>
</div>



## 三、虚拟变量的处理


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) # 将地区变为数字
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()    # 将地区变为虚拟变量
X
```




    array([[  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              1.65349200e+05,   1.36897800e+05,   4.71784100e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              1.62597700e+05,   1.51377590e+05,   4.43898530e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              1.53441510e+05,   1.01145550e+05,   4.07934540e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              1.44372410e+05,   1.18671850e+05,   3.83199620e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              1.42107340e+05,   9.13917700e+04,   3.66168420e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              1.31876900e+05,   9.98147100e+04,   3.62861360e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              1.34615460e+05,   1.47198870e+05,   1.27716820e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              1.30298130e+05,   1.45530060e+05,   3.23876680e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              1.20542520e+05,   1.48718950e+05,   3.11613290e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              1.23334880e+05,   1.08679170e+05,   3.04981620e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              1.01913080e+05,   1.10594110e+05,   2.29160950e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              1.00671960e+05,   9.17906100e+04,   2.49744550e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              9.38637500e+04,   1.27320380e+05,   2.49839440e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              9.19923900e+04,   1.35495070e+05,   2.52664930e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              1.19943240e+05,   1.56547420e+05,   2.56512920e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              1.14523610e+05,   1.22616840e+05,   2.61776230e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              7.80131100e+04,   1.21597550e+05,   2.64346060e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              9.46571600e+04,   1.45077580e+05,   2.82574310e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              9.17491600e+04,   1.14175790e+05,   2.94919570e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              8.64197000e+04,   1.53514110e+05,   0.00000000e+00],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              7.62538600e+04,   1.13867300e+05,   2.98664470e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              7.83894700e+04,   1.53773430e+05,   2.99737290e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              7.39945600e+04,   1.22782750e+05,   3.03319260e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              6.75325300e+04,   1.05751030e+05,   3.04768730e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              7.70440100e+04,   9.92813400e+04,   1.40574810e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              6.46647100e+04,   1.39553160e+05,   1.37962620e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              7.53288700e+04,   1.44135980e+05,   1.34050070e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              7.21076000e+04,   1.27864550e+05,   3.53183810e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              6.60515200e+04,   1.82645560e+05,   1.18148200e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              6.56054800e+04,   1.53032060e+05,   1.07138380e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              6.19944800e+04,   1.15641280e+05,   9.11312400e+04],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              6.11363800e+04,   1.52701920e+05,   8.82182300e+04],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              6.34088600e+04,   1.29219610e+05,   4.60852500e+04],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              5.54939500e+04,   1.03057490e+05,   2.14634810e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              4.64260700e+04,   1.57693920e+05,   2.10797670e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              4.60140200e+04,   8.50474400e+04,   2.05517640e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              2.86637600e+04,   1.27056210e+05,   2.01126820e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              4.40699500e+04,   5.12831400e+04,   1.97029420e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              2.02295900e+04,   6.59479300e+04,   1.85265100e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              3.85585100e+04,   8.29820900e+04,   1.74999300e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              2.87543300e+04,   1.18546050e+05,   1.72795670e+05],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              2.78929200e+04,   8.47107700e+04,   1.64470710e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              2.36409300e+04,   9.61896300e+04,   1.48001110e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              1.55057300e+04,   1.27382300e+05,   3.55341700e+04],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              2.21777400e+04,   1.54806140e+05,   2.83347200e+04],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              1.00023000e+03,   1.24153040e+05,   1.90393000e+03],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
              1.31546000e+03,   1.15816210e+05,   2.97114460e+05],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   1.35426920e+05,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
              5.42050000e+02,   5.17431500e+04,   0.00000000e+00],
           [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              0.00000000e+00,   1.16983800e+05,   4.51730600e+04]])



### 虚拟变量只需要N-1个变量即可拟合，所以去掉其中一个变量


```python
X = X[:,1:]
X
```




    array([[  0.00000000e+00,   1.00000000e+00,   1.65349200e+05,
              1.36897800e+05,   4.71784100e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.62597700e+05,
              1.51377590e+05,   4.43898530e+05],
           [  1.00000000e+00,   0.00000000e+00,   1.53441510e+05,
              1.01145550e+05,   4.07934540e+05],
           [  0.00000000e+00,   1.00000000e+00,   1.44372410e+05,
              1.18671850e+05,   3.83199620e+05],
           [  1.00000000e+00,   0.00000000e+00,   1.42107340e+05,
              9.13917700e+04,   3.66168420e+05],
           [  0.00000000e+00,   1.00000000e+00,   1.31876900e+05,
              9.98147100e+04,   3.62861360e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.34615460e+05,
              1.47198870e+05,   1.27716820e+05],
           [  1.00000000e+00,   0.00000000e+00,   1.30298130e+05,
              1.45530060e+05,   3.23876680e+05],
           [  0.00000000e+00,   1.00000000e+00,   1.20542520e+05,
              1.48718950e+05,   3.11613290e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.23334880e+05,
              1.08679170e+05,   3.04981620e+05],
           [  1.00000000e+00,   0.00000000e+00,   1.01913080e+05,
              1.10594110e+05,   2.29160950e+05],
           [  0.00000000e+00,   0.00000000e+00,   1.00671960e+05,
              9.17906100e+04,   2.49744550e+05],
           [  1.00000000e+00,   0.00000000e+00,   9.38637500e+04,
              1.27320380e+05,   2.49839440e+05],
           [  0.00000000e+00,   0.00000000e+00,   9.19923900e+04,
              1.35495070e+05,   2.52664930e+05],
           [  1.00000000e+00,   0.00000000e+00,   1.19943240e+05,
              1.56547420e+05,   2.56512920e+05],
           [  0.00000000e+00,   1.00000000e+00,   1.14523610e+05,
              1.22616840e+05,   2.61776230e+05],
           [  0.00000000e+00,   0.00000000e+00,   7.80131100e+04,
              1.21597550e+05,   2.64346060e+05],
           [  0.00000000e+00,   1.00000000e+00,   9.46571600e+04,
              1.45077580e+05,   2.82574310e+05],
           [  1.00000000e+00,   0.00000000e+00,   9.17491600e+04,
              1.14175790e+05,   2.94919570e+05],
           [  0.00000000e+00,   1.00000000e+00,   8.64197000e+04,
              1.53514110e+05,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   7.62538600e+04,
              1.13867300e+05,   2.98664470e+05],
           [  0.00000000e+00,   1.00000000e+00,   7.83894700e+04,
              1.53773430e+05,   2.99737290e+05],
           [  1.00000000e+00,   0.00000000e+00,   7.39945600e+04,
              1.22782750e+05,   3.03319260e+05],
           [  1.00000000e+00,   0.00000000e+00,   6.75325300e+04,
              1.05751030e+05,   3.04768730e+05],
           [  0.00000000e+00,   1.00000000e+00,   7.70440100e+04,
              9.92813400e+04,   1.40574810e+05],
           [  0.00000000e+00,   0.00000000e+00,   6.46647100e+04,
              1.39553160e+05,   1.37962620e+05],
           [  1.00000000e+00,   0.00000000e+00,   7.53288700e+04,
              1.44135980e+05,   1.34050070e+05],
           [  0.00000000e+00,   1.00000000e+00,   7.21076000e+04,
              1.27864550e+05,   3.53183810e+05],
           [  1.00000000e+00,   0.00000000e+00,   6.60515200e+04,
              1.82645560e+05,   1.18148200e+05],
           [  0.00000000e+00,   1.00000000e+00,   6.56054800e+04,
              1.53032060e+05,   1.07138380e+05],
           [  1.00000000e+00,   0.00000000e+00,   6.19944800e+04,
              1.15641280e+05,   9.11312400e+04],
           [  0.00000000e+00,   1.00000000e+00,   6.11363800e+04,
              1.52701920e+05,   8.82182300e+04],
           [  0.00000000e+00,   0.00000000e+00,   6.34088600e+04,
              1.29219610e+05,   4.60852500e+04],
           [  1.00000000e+00,   0.00000000e+00,   5.54939500e+04,
              1.03057490e+05,   2.14634810e+05],
           [  0.00000000e+00,   0.00000000e+00,   4.64260700e+04,
              1.57693920e+05,   2.10797670e+05],
           [  0.00000000e+00,   1.00000000e+00,   4.60140200e+04,
              8.50474400e+04,   2.05517640e+05],
           [  1.00000000e+00,   0.00000000e+00,   2.86637600e+04,
              1.27056210e+05,   2.01126820e+05],
           [  0.00000000e+00,   0.00000000e+00,   4.40699500e+04,
              5.12831400e+04,   1.97029420e+05],
           [  0.00000000e+00,   1.00000000e+00,   2.02295900e+04,
              6.59479300e+04,   1.85265100e+05],
           [  0.00000000e+00,   0.00000000e+00,   3.85585100e+04,
              8.29820900e+04,   1.74999300e+05],
           [  0.00000000e+00,   0.00000000e+00,   2.87543300e+04,
              1.18546050e+05,   1.72795670e+05],
           [  1.00000000e+00,   0.00000000e+00,   2.78929200e+04,
              8.47107700e+04,   1.64470710e+05],
           [  0.00000000e+00,   0.00000000e+00,   2.36409300e+04,
              9.61896300e+04,   1.48001110e+05],
           [  0.00000000e+00,   1.00000000e+00,   1.55057300e+04,
              1.27382300e+05,   3.55341700e+04],
           [  0.00000000e+00,   0.00000000e+00,   2.21777400e+04,
              1.54806140e+05,   2.83347200e+04],
           [  0.00000000e+00,   1.00000000e+00,   1.00023000e+03,
              1.24153040e+05,   1.90393000e+03],
           [  1.00000000e+00,   0.00000000e+00,   1.31546000e+03,
              1.15816210e+05,   2.97114460e+05],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              1.35426920e+05,   0.00000000e+00],
           [  0.00000000e+00,   1.00000000e+00,   5.42050000e+02,
              5.17431500e+04,   0.00000000e+00],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
              1.16983800e+05,   4.51730600e+04]])



## 四、区分训练集和测试集


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

```

## 五、用线性回归训练


```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
y_pred = regressor.predict(X_test)

```

## 六、反向淘汰选择模型变量

### 设定当 P>|t|大于0.05即被淘汰 


```python
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((40,1)),values = X_train,axis = 1) # 增加新的一列
X_opt = X_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.950</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.943</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   129.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 14 Apr 2018</td> <th>  Prob (F-statistic):</th> <td>3.91e-21</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:08:24</td>     <th>  Log-Likelihood:    </th> <td> -421.10</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    40</td>      <th>  AIC:               </th> <td>   854.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    34</td>      <th>  BIC:               </th> <td>   864.3</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 4.255e+04</td> <td> 8358.538</td> <td>    5.091</td> <td> 0.000</td> <td> 2.56e+04</td> <td> 5.95e+04</td>
</tr>
<tr>
  <th>x1</th>    <td> -959.2842</td> <td> 4038.108</td> <td>   -0.238</td> <td> 0.814</td> <td>-9165.706</td> <td> 7247.138</td>
</tr>
<tr>
  <th>x2</th>    <td>  699.3691</td> <td> 3661.563</td> <td>    0.191</td> <td> 0.850</td> <td>-6741.822</td> <td> 8140.560</td>
</tr>
<tr>
  <th>x3</th>    <td>    0.7735</td> <td>    0.055</td> <td>   14.025</td> <td> 0.000</td> <td>    0.661</td> <td>    0.886</td>
</tr>
<tr>
  <th>x4</th>    <td>    0.0329</td> <td>    0.066</td> <td>    0.495</td> <td> 0.624</td> <td>   -0.102</td> <td>    0.168</td>
</tr>
<tr>
  <th>x5</th>    <td>    0.0366</td> <td>    0.019</td> <td>    1.884</td> <td> 0.068</td> <td>   -0.003</td> <td>    0.076</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>15.823</td> <th>  Durbin-Watson:     </th> <td>   2.468</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  23.231</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.094</td> <th>  Prob(JB):          </th> <td>9.03e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.025</td> <th>  Cond. No.          </th> <td>1.49e+06</td>
</tr>
</table>




```python
 X_opt = X_train [:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.950</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.944</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   166.7</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 14 Apr 2018</td> <th>  Prob (F-statistic):</th> <td>2.87e-22</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:08:48</td>     <th>  Log-Likelihood:    </th> <td> -421.12</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    40</td>      <th>  AIC:               </th> <td>   852.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    35</td>      <th>  BIC:               </th> <td>   860.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 4.292e+04</td> <td> 8020.397</td> <td>    5.352</td> <td> 0.000</td> <td> 2.66e+04</td> <td> 5.92e+04</td>
</tr>
<tr>
  <th>x1</th>    <td>-1272.1608</td> <td> 3639.780</td> <td>   -0.350</td> <td> 0.729</td> <td>-8661.308</td> <td> 6116.986</td>
</tr>
<tr>
  <th>x2</th>    <td>    0.7754</td> <td>    0.053</td> <td>   14.498</td> <td> 0.000</td> <td>    0.667</td> <td>    0.884</td>
</tr>
<tr>
  <th>x3</th>    <td>    0.0319</td> <td>    0.065</td> <td>    0.488</td> <td> 0.629</td> <td>   -0.101</td> <td>    0.165</td>
</tr>
<tr>
  <th>x4</th>    <td>    0.0363</td> <td>    0.019</td> <td>    1.902</td> <td> 0.065</td> <td>   -0.002</td> <td>    0.075</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>16.074</td> <th>  Durbin-Watson:     </th> <td>   2.467</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  24.553</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.086</td> <th>  Prob(JB):          </th> <td>4.66e-06</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.164</td> <th>  Cond. No.          </th> <td>1.43e+06</td>
</tr>
</table>




```python
X_opt = X_train [:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.950</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.946</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   227.8</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 14 Apr 2018</td> <th>  Prob (F-statistic):</th> <td>1.85e-23</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:08:50</td>     <th>  Log-Likelihood:    </th> <td> -421.19</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    40</td>      <th>  AIC:               </th> <td>   850.4</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    36</td>      <th>  BIC:               </th> <td>   857.1</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 4.299e+04</td> <td> 7919.773</td> <td>    5.428</td> <td> 0.000</td> <td> 2.69e+04</td> <td> 5.91e+04</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.7788</td> <td>    0.052</td> <td>   15.003</td> <td> 0.000</td> <td>    0.674</td> <td>    0.884</td>
</tr>
<tr>
  <th>x2</th>    <td>    0.0294</td> <td>    0.064</td> <td>    0.458</td> <td> 0.650</td> <td>   -0.101</td> <td>    0.160</td>
</tr>
<tr>
  <th>x3</th>    <td>    0.0347</td> <td>    0.018</td> <td>    1.896</td> <td> 0.066</td> <td>   -0.002</td> <td>    0.072</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>15.557</td> <th>  Durbin-Watson:     </th> <td>   2.481</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  22.539</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.081</td> <th>  Prob(JB):          </th> <td>1.28e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.974</td> <th>  Cond. No.          </th> <td>1.43e+06</td>
</tr>
</table>




```python
X_opt = X_train [:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.950</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.947</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   349.0</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 14 Apr 2018</td> <th>  Prob (F-statistic):</th> <td>9.65e-25</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:08:53</td>     <th>  Log-Likelihood:    </th> <td> -421.30</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    40</td>      <th>  AIC:               </th> <td>   848.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    37</td>      <th>  BIC:               </th> <td>   853.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 4.635e+04</td> <td> 2971.236</td> <td>   15.598</td> <td> 0.000</td> <td> 4.03e+04</td> <td> 5.24e+04</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.7886</td> <td>    0.047</td> <td>   16.846</td> <td> 0.000</td> <td>    0.694</td> <td>    0.883</td>
</tr>
<tr>
  <th>x2</th>    <td>    0.0326</td> <td>    0.018</td> <td>    1.860</td> <td> 0.071</td> <td>   -0.003</td> <td>    0.068</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>14.666</td> <th>  Durbin-Watson:     </th> <td>   2.518</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.001</td> <th>  Jarque-Bera (JB):  </th> <td>  20.582</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-1.030</td> <th>  Prob(JB):          </th> <td>3.39e-05</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.847</td> <th>  Cond. No.          </th> <td>4.97e+05</td>
</tr>
</table>




```python
X_opt = X_train [:, [0, 3]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.945</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.944</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   652.4</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 14 Apr 2018</td> <th>  Prob (F-statistic):</th> <td>1.56e-25</td>
</tr>
<tr>
  <th>Time:</th>                 <td>23:08:55</td>     <th>  Log-Likelihood:    </th> <td> -423.09</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    40</td>      <th>  AIC:               </th> <td>   850.2</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    38</td>      <th>  BIC:               </th> <td>   853.6</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 4.842e+04</td> <td> 2842.717</td> <td>   17.032</td> <td> 0.000</td> <td> 4.27e+04</td> <td> 5.42e+04</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.8516</td> <td>    0.033</td> <td>   25.542</td> <td> 0.000</td> <td>    0.784</td> <td>    0.919</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>13.132</td> <th>  Durbin-Watson:     </th> <td>   2.325</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.001</td> <th>  Jarque-Bera (JB):  </th> <td>  16.254</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.991</td> <th>  Prob(JB):          </th> <td>0.000295</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.413</td> <th>  Cond. No.          </th> <td>1.57e+05</td>
</tr>
</table>



## 七、项目地址

### https://coding.net/u/RuoYun/p/Python-of-machine-learning/git/tree/master/00%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/2.%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%95/2.%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92?public=true
