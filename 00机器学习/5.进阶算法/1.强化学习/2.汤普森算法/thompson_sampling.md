
## 一、UCB和汤普森算法比较

## ![avatar](./1.png)

## 二、导入标准库


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

## 三、导入数据


```python
dataset = pd.read_csv('Ads_CTR_Optimisation.csv') # 数据表示虚拟环境，模拟我将投放哪些广告
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
      <th>Ad 1</th>
      <th>Ad 2</th>
      <th>Ad 3</th>
      <th>Ad 4</th>
      <th>Ad 5</th>
      <th>Ad 6</th>
      <th>Ad 7</th>
      <th>Ad 8</th>
      <th>Ad 9</th>
      <th>Ad 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9970</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9971</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9972</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9973</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9974</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9975</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9976</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9977</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9978</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9979</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9980</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9981</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9982</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9983</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9984</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9985</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9986</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9987</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9988</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9989</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9990</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9991</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9992</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9994</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 10 columns</p>
</div>



## 四、汤普森算法在多臂老虎机中的实现过程

## ![avatar](./2.png)

## 五、实现


```python
import random
N = 10000                    # 1000个用户
d = 10                       # 10个广告
ads_selected = []           # 广告选择
numbers_of_rewards_1 = [0] * d # 广告i奖励1的总和
numbers_of_rewards_0 = [0] * d # 广告i奖励0的总和
total_reward = 0            # 总奖励
for n in range(0, N):        # 每个用户循环
    ad = 0
    max_random = 0
    for i in range(0, d):   # 第i个广告
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1) # beta分布
        if random_beta > max_random:  # 寻找最大随机量，最大随机量对应的广告为我们选择投放的广告
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]  # 获得奖励
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
print(total_reward)
```

    2625
    

## 六、画图展示


```python
# 画图
plt.hist(ads_selected)
plt.title(u'广告选择直方图')
plt.xlabel(u'广告')
plt.ylabel(u'每个广告的点击数')
plt.show()
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAJYCAYAAACadoJwAAAgAElEQVR4Xu3dC7h1a1kX/H/ISSVUDAUl5fwBohh8AgYKklqiIpliKWiRhUCSogEpCeYBCNTEKAUEjY1Y8qGlpFh5TERlm3kAvg6gEngAhA6CIIfvuvnGysm7137fNddzrznmM+dvXNfL3rxrzHs843c/e+/5f59x+FOxESBAgAABAgQIECBAYEcCf2pHx3EYAgQIECBAgAABAgQIRAAxCQgQIECAAAECBAgQ2JmAALIzagciQIAAAQIECBAgQEAAMQcIECBAgAABAgQIENiZgACyM2oHIkCAAAECBAgQIEBAADEHCBAgQIAAAQIECBDYmYAAsjNqByJAgAABAgQIECBAQAAxBwgQIECAAAECBAgQ2JmAALIzagciQIAAAQIECBAgQEAAMQcIECBAgAABAgQIENiZgACyM2oHIkCAAAECBAgQIEBAADEHCBAgQIAAAQIECBDYmYAAsjNqByJAgAABAgQIECBAQAAxBwgQIECAAAECBAgQ2JmAALIzagciQIAAAQIECBAgQEAAMQcIECBAgAABAgQIENiZgACyM2oHIkCAAAECBAgQIEBAADEHCBAgQIAAAQIECBDYmYAAsjNqByJAgAABAgQIECBAQAAxBwgQIECAAAECBAgQ2JmAALIzagciQIAAAQIECBAgQEAAMQcIECBAgAABAgQIENiZgACyM2oHIkCAAAECBAgQIEBAADEHCBAgQIAAAQIECBDYmYAAsjNqByJAgAABAgQIECBAQAAxBwgQIECAAAECBAgQ2JmAALIzagciQIAAAQIECBAgQEAAMQcIECBAgAABAgQIENiZgACyM2oHIkCAAAECBAgQIEBAADEHCBAgQIAAAQIECBDYmYAAsjNqByJAgAABAgQIECBAQAAxBwgQIECAAAECBAgQ2JmAALIzagciQIAAAQIECBAgQEAAMQcIECBAgAABAgQIENiZgACyM2oHIkCAAAECBAgQIEBAADEHCBAgQIAAAQIECBDYmYAAsjNqByJAgAABAgQIECBAQAAxBwgQIECAAAECBAgQ2JmAALIzagciQIAAAQIECBAgQEAAMQcIECBAgAABAgQIENiZgACyM2oHIkCAAAECBAgQIEBAADEHCBAgQIAAAQIECBDYmYAAsjNqByJAgAAB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PSL3P1/AcGsFpFZH3nkBtZUkQIAAgT0REED2pBGGQYAAAQIECBAgQOAYBASQY+iycyRAgAABAgQIECCwJwICyJ40wjAIECBAgAABAgQIHIOAAHIMXXaOBAgQIECAAAECBPZEQADZk0YYBgECBAgQIECAAIFjEBBAjqHLzpEAAQIECBAgQIDAnggIIHvSCMMgQIAAAQIECBAgcAwCAsgxdNk5EiBAgAABAgQIENgTAQFkTxphGAQIECBAgAABAgSOQUAAOYYuO0cCBAgQIECAAAECeyIggOxJIwyDAAECBAgQIECAwDEICCDH0GXnSIAAAQIECBAgQGBPBASQPWmEYRAgQIAAAQIECBA4BgEB5Bi67BwJECBAgAABAgQI7ImAALInjTAMAgQIECBAgAABAscgIIAcQ5edIwECBAgQIECAAIE9ERBA9qQRhkGAAAECBAgQIEDgGAQEkGPosnMkQIAAAQIECBAgsCcCAsieNMIwCBAgQIAAAQIECByDgAByDF12jgQIECBAgAABAgT2REAA2ZNGGAYBAgQIECBAgACBYxAQQI6hy86RAAECBAgQIECAwJ4I/H8hYTDfnoxMfQAAAABJRU5ErkJggg==" width="640">


### 看来还是汤普森更加NB。随机生成的总奖励1282（见上一个专题），置信区间上界算法总奖励2358（见上一个专题），汤普森的总奖励2625

## 七、项目地址

### https://coding.net/u/RuoYun/p/Python-of-machine-learning/git/tree/master/00%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/5.%E8%BF%9B%E9%98%B6%E7%AE%97%E6%B3%95/1.%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/2.%E6%B1%A4%E6%99%AE%E6%A3%AE%E7%AE%97%E6%B3%95?public=true
