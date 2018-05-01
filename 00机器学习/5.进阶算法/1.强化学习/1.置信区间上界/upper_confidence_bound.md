
## 零、算法原理

## ![avatar](./UCB_Algorithm_Slide.png)

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

# 二、导入数据


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



# 三、每个用户随机抽选广告得到的点击数


```python
import random
N = 10000  # 1000个用户
d = 10     # 10个广告
ads_selected = [] # 广告选择
total_reward = 0 
for n in range(0, N):      # 每个用户循环
    ad = random.randrange(d) # 随机选择广告
    ads_selected.append(ad)  # 将选择的广告加入list中
    reward = dataset.values[n, ad] # 取出数据集中n行ad列查看是否命中，命中值为1，未命中值为0（1即为奖励）
    total_reward = total_reward + reward  # 每轮奖励累计相加

print(total_reward)
# 画图
plt.hist(ads_selected)
plt.title(u'广告选择直方图')
plt.xlabel(u'广告')
plt.ylabel(u'每个广告的点击数')
plt.show()
```

    1282
    


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAJYCAYAAACadoJwAAAgAElEQVR4Xu3dCZhsa1ke7CfIpCIqiIISPYw/IIo5/AIGlMEhEQGJUZxAIzFBIBJFAydKBOMACGrE4AAIGkA08uOIijGOUVE5zgJ/BkAJOMARYiIIAifXm6yG2vv07qrqWl31fqvvdV2bs+la9a133e/q7nr2t4a/FQsBAgQIECBAgAABAgT2JPC39rQdmyFAgAABAgQIECBAgEAEEAcBAQIECBAgQIAAAQJ7ExBA9kZtQwQIECBAgAABAgQICCCOAQIECBAgQIAAAQIE9iYggOyN2oYIECBAgAABAgQIEBBAHAMECBAgQIAAAQIECOxNQADZG7UNESBAgAABAgQIECAggDgGCBAgQIAAAQIECBDYm4AAsjdqGyJAgAABAgQIECBAQABxDBAgQIAAAQIECBAgsDcBAWRv1DZEgAABAgQIECBAgIAA4hggQIAAAQIECBAgQGBvAgLI3qhtiAABAgQIECBAgAABAcQxQIAAAQIECBAgQIDA3gQEkL1R2xABAgQIECBAgAABAgKIY4AAAQIECBAgQIAAgb0JCCB7o7YhAgQIECBAgAABAgQEEMcAAQIECBAgQIAAAQJ7ExBA9kZtQwQIECBAgAABAgQICCCOAQIECBAgQIAAAQIE9iYggOyN2oYIECBAgAABAgQIEBBAHAMECBAgQIAAAQIECOxNQADZG7UNESBAgAABAgQIECAggDgGCBAgQIAAAQIECBDYm4AAsjdqGyJAgAABAgQIECBAQABxDBAgQIAAAQIECBAgsDcBAWRv1DZEgAABAgQIECBAgIAA4hggQIAAAQIECBAgQGBvAgLI3qhtiAABAgQIECBAgAABAcQxQIAAAQIECBAgQIDA3gQEkL1R2xABAgQIECBAgAABAgKIY4AAAQIECBAgQIAAgb0JCCB7o7YhAgQIECBAgAABAgQEEMcAAQIECBAgQIAAAQJ7ExBA9kZtQwQIECBAgAABAgQICCCOAQIECBAgQIAAAQIE9iYggOyN2oYIECBAgAABAgQIEBBAHAMECBAgQIAAAQIECOxNQADZG7UNESBAgAABAgQIECAggDgGCBAgQIAAAQIECBDYm4AAsjdqGyJAgAABAgQIECBAQABxDBAgQIAAAQIECBAgsDcBAWRv1DZEgAABAgQIECBAgIAA4hggQIAAAQIECBAgQGBvAgLI3qhtiAABAgQIECBAgAABAcQxQIAAAQIECBAgQIDA3gQEkL1R2xABAgS2FrhOkr/Z+l3eQIAAAQIEGgsIII2bozQCBM61wP2TfFeSj0ryFxtI3CzJVUnePq37D5NckeRuSd65wfsvXuX6Sf46yQcl+fokT0/y+yeMc5skX5Hk65K8Lsl1k3xokres1LT69rcleeuaut4rySOS/HaS/3SKfai3PCTJLyb54w3e/z5T3Rus+u5V3pHkf23zBusSIEDgvAsIIOf9CLD/BAh0FagP/vWh+duS/MsNivzBJLdMctck70ryRUmelaQ+xK9b7pPk45J8w7TizZP81ySfn+Q3pjrukeRXThjo706v3y7J/5/kjmsCy/2SvHhlvBslqVBSf46W600f7h+b5FtWvl6/u+q12s8KOJda3jvJq6bw8lnrEJI8e3LbYNV3r/KyJB+7zRusS4AAgfMuIICc9yPA/hMg0FngW5P8XpLnrBRZgaJmJ+rD99EMwk2S/Pckj5xCR63+j6a/X/uiD+41M7H6Ib9erg/nL5gCTIWehyb5N0k+ctrWf05SIeWV0wf/NyepP7Vca/rzMUl+cxrjDVNwuO30tQojr53Wf26SmmmoGZrVpcZ7/y2b8bVJnrDmPTWDUrM3/2+SK9es+4wkFYQ+c2W9BySpU+FelOTqi97/uCSfmuTuW9ZtdQIECJxrAQHkXLffzhMg0ETgm5L8iy1r+eYkXzm9p2YHvnzD99fpUTXDsbpUiKhw8fwk9aG+Tlm683QKVv2eqA/l/2P6AF4fxv91kqq5lvpgX8FjdalTvo6CT4Womj2pwFFhoWZaarbkTRe950OS1OlMdQrZ0Qf9WyX5nSR/Z5qROXpL1VRBqtb9n9MXa+bl6GurQ1e9n5Tkp47xqRorjP2X6bUKKlXHUQCpoPeKKdxVEDmqq67L+askFUA+JcknbGhvNQIECBBIIoA4DAgQIHB4gcdPsw51+tS6pcLC0alJNdNQH7xrlqSuv6jZhaPl85I8bbqG4+IP7n92zEbqg3R9GK/X6lqPCgkVLOpUsNpOhYAKAxcvNZvxEdPMx08k+fgpXPzhtGLN2PyHJPdK8popgBy3/aNxaxbk6HdTzSzUmB++EjRqvaMAsFpLXSNympmIlyT5+9NANetT4ewogDwpSZ3+dfHymCRPmQJIhZvaNwsBAgQIbCgggGwIZTUCBAicoUDNZHzZMTMTm2yyZhfqwu86FaguGj9ajjsF67jxKkDUB/qatajTuj54uqbhz6d/8a9A8LPT9SB1KtbR9Rf191rnaKlTsOpi8b89zRjU1/+fJA+fTg2rkFQh5seS/ECS303yR8dcw1HXdNS1GyctFWgqMK0uNftR+3B0wX2dtnbDY071Wn1PhaP6c3Th/moAeeB02lUFqQpjNdNSr9cMTgWdmq2pGRABZJOj1DoECBBYERBAHA4ECBA4vECFj7rQvE7/2Wa5cZL6c/QBevW99a/49S/4tz5mwDr1qO6sVX9qVuNO0zpPTfKTSb59CiUfmOSmSf5guhvX66fZjTqtqWYGat2LA0jddatCTY1R15DUBen/KskPT6cqfeN0oXy9r7Z38alndWpWXUD/I5eAqPfUtSV1StRJy/OmFx+8BehRAPkn0ylpT5xqrTt8/UySz5lmeOo0tloEkC1wrUqAAIEjAQHEsUCAAIHDC1QA+eokdTH5Nss/T1Ifko8LICeNUwHkUdNdnyr01LUN/366W1R9qD5aavz6AF93earTp+q1ow/2F4//95L89DQzUNej1EXnNWvw89MF7DWLUhej14xCnTZWMzZ1564KNavLG6dQdVL9P5qkZihOWurUrU87YYW65uX2F72+OgNSM0E1w3Pv6W5dNSvz5OmWxEe33RVAtjlarUuAAIFJQABxKBAgQODwAvWBvWYUarbhUEsFhLqOYvWuUhVuKjTUM0kuFUA+cboovU5TquVLk3zndOH60exMzYTUqVd1J60/ma5hqdv81oXcFy8VQL54zQxIzeqsCyB1x6tfSvLdx2yjZoZqlubi07iOAkjNRtVF83Wr4ApL5fDy6TbFte3/mKSuHanrY5yCdagj1nYJEBhWQAAZtnUKJ0BgQQJfM512dItT7lNdRH3cXZ6OG64+dNc1FBcvxwWQlyb5oSR1x61LBZC6C9b3TTMjdXrV0TUgNcNSH/TrWon6XXOD6da8R9eQ1IXbxz1XpALISc/3qNPCalblpABSp4jV7EuFiNVnjRztc1nVduohhavLagCpZ4J8//T+X0/yC0m+Z6qt3lch5DIB5JRHrLcRIHCuBQSQc91+O0+AQBOB+lf6uoh7k7tgHVdyfZivD+X136PrEy5e78OmD9H3nGYG1gWQqufXpgvJ69kgqwGknoxeF5FXOKmlTumqmZKLL0I/2sbRQwkrYNU4Jy1zzIDU9S//brqg/rinlP/WdE1HPSn+uACy+hyQ2re6JXHNUn36NCNSF8/X4hSsJt9AyiBAYCwBAWSsfqmWAIFlCtTpQnWKz8X/Ir/p3tZzKOrZHSd9wK9/rX/1dBF1nWp1UgCpO1/V7ETNFBxdJF53vaqQU09mr1mFupPV6hPaV++CVR/aaxbi6M5addF4nbJUddYpWPV6/alrLFbvpFU17XoNSN3Vqp5OXhfX18Xsxy0V0uq0qn+7JoDU6VbVm6q9HkRYJkcPVBRANj06rUeAAIGLBAQQhwQBAgQOK1AXQlf4eFiSehL3aZZ69kZd77BJAKnrMCpIXLzU1+pWuXVBfH3Yrudh1LhHt/atO1nVv/jX7W6vmmZr/tvKIKsBpE5VqvceBZB6dkmdglWnRdXvnfr/NU6NVxd2ry4VQOri9+NOnar16pbDdZrXpU7BqtPA6onwH5WkZm4uXt5vuvvXg6Y7c62+XnfuutnKc0AqRNV+1LqfPYWqustX3RWsFg8iPM3R6j0ECJx7AQHk3B8CAAgQOLBAfdivO0LVw/wung3YtLQ6raquUdhkqbtV1S1lj5a6C1Z98K47aVUAqetJ6gN2nSpVpyrVrXrr2Rz1rJCj6zfq+Rp1x66661Vdr1Hhoy5CryeJf+g0y7FaS4WBGvtSAemWSd532kad9lUf7Osai+OWeghgPZiwLnavgPDfpxpr3Tqlqk4P+wdJfnzlzTV23cnrA6ZZkbqFb22zZoRWl7p4vmaKauyLl3oq+idfNO4jklQgOboAfxN/6xAgQODcCwgg5/4QAECAwAEFLk9SF3p/13Rb3NOWcnQL3PrX+vpAftxSMxq/PN3Rqm5Re7TUMy/qA3nNKNSMRT1l/b7T9SR17UbNNtQH9/p6nd5US92294UrpzjV3+uC7/8yPaPj4u3XAwgrzNwqyauOKa4u9q7t12xLBZpNlppFqTtZfeF0O9+vTfJVSeoBjM+/aICqu67bqOtgyqcuqq8Lzi9eagaqPLZZ6mnxd9nmDdYlQIDAeRcQQM77EWD/CRA4tEA9BLBOFaoH8J3lUh/Ya+aiLsquO1PNudSteGtGpGZWasbk4uVohuYOSV4x54ZXxjq6WLzuWHXcUqe6lfGfnrD9uvNVGa1ehH5SuTVT8xlJKkhaCBAgQGBDAQFkQyirESBAgMDiBerhgxXUTgopi0ewgwQIEDhrAQHkrIWNT4AAAQIECBAgQIDAuwUEEAcDAQIECBAgQIAAAQJ7ExBA9kZtQwQIECBAgAABAgQICCCOAQIECBAgQIAAAQIE9iYggOyN2oYIECBAgAABAgQIEBBAHAMECBAgQIAAAQIECOxNQADZG/VGG7pxknqgWD2BuB7IZSFAgAABAgQIEOglcP0klyV5SZKrepU2RjUCSK8+fd4xT/DtVaFqCBAgQIAAAQIESuDzk3w/iu0FBJDtzc7yHX83ya8873nPy+1vXw/ttRAgQIAAAQIECHQSeMUrXpEHP/jBVdLdk/xqp9pGqUUA6dWpy5NceeWVV+byy+uvFgIECBAgQIAAgU4Cv/Vbv5U73/nOVVL9z291qm2UWgSQXp0SQHr1QzUECBAgQIAAgQsEBJDdDwgBZHfDOUcQQObUNBYBAgQIECBAYGYBAWR3UAFkd8M5RxBA5tQ0FgECBAgQIEBgZgEBZHdQAWR3wzlHEEDm1DQWAQIECBAgQGBmAQFkd1ABZHfDOUcQQObUNBYBAgQIECBAYGYBAWR3UAFkd8M5RxBA5tQ0FgECBAgQIEBgZgEBZHdQAWR3wzlHEEDm1DQWAQIECBAgQGBmAQFkd1ABZHfDOUcQQObUNBYBAgQIECBAYGYBAWR3UAFkd8M5RxBA5tQ0FgECBAgQIEBgZgEBZHdQAWR3wzlHEEDm1DQWAQIECBAgQGBmAQFkd1ABZHfDOUcQQObUNBYBAgQIECBAYGYBAWR3UAFkd8M5RxBA5tQ0FgECBAgQIEBgZgEBZHdQAWR3wzlHEEDm1DQWAQIECBAgQGBmAQFkd1ABZHfDOUcQQObUNBYBAgQIECBAYGYBAWR3UAFkd8M5RxBA5tQ0FgECBAgQIEBgZgEBZHdQAWR3wzlHEEDm1DQWAQIECBAgQGBmAQFkd1ABZHfDOUcQQObUNBYBAgQIECBAYGYBAWR3UAFkd8M5RxBA5tQ0FgECBAgQIEBgZgEBZHdQAWR3wzlHEEDm1DQWAQIECBAgQGBmAQFkd1ABZHfDOUcQQObUNBYBAgQIENhA4LIrXrzBWuOu8ponfdq4xTesXADZvSkCyO6Gc44ggMypaSwCBAgQILCBgACyAZJV3i0ggOx+MAgguxvOOYIAMqemsQgQIECAwAYCAsgGSFYRQGY8BgSQGTFnGEoAmQHREAQIECBAYBsBAWQbLeuaAdn9GBBAdjeccwQBZE5NYxEgQIAAgQ0EBJANkKxiBmTGY0AAmRFzhqEEkBkQDUGAAAECBLYREEC20bKuGZDdjwEBZHfDOUcQQObUNBYBAgQIENhAQADZAMkqZkBmPAYEkBkxZxhKAJkB0RAECBAgQGAbAQFkGy3rmgHZ/RgQQHY3nHMEAWROTWMRIECAAIENBASQDZCsYgZkxmNAAJkRc4ahBJAZEA1BgAABAgS2ERBAttGyrhmQ3Y8BAWR3wzlHEEDm1DQWAQIECBDYQEAA2QDJKmZAZjwGBJAZMWcYSgCZAdEQyxJY+geD6tZrnvRpy2qavSEwmMDSf874GTPvAWkGZHdPAWR3wzlHEEDm1DTWIgSW/sFAAFnEYWonBhdY+s8ZAWTeA1QA2d1TANndcM4RBJA5NY21CIGlfzAQQBZxmNqJwQWW/nNGAJn3ABVAdvcUQHY3nHMEAWROTWMtQmDpHwwEkEUcpnZicIGl/5wRQOY9QAWQ3T0FkN0N5xxBAJlT01iLEFj6BwMBZBGHqZ0YXGDpP2cEkHkPUAFkd08BZHfDOUcQQObUNNYiBJb+wUAAWcRhaicGF1j6zxkBZN4DVADZ3VMA2d1wzhEEkDk1jbUIgaV/MBBAFnGY2onBBZb+c0YAmfcAFUB29xRAdjeccwQBZE5NYy1CYOkfDASQRRymdmJwgaX/nBFA5j1ABZDdPQWQ3Q3nHEEAmVPTWIsQWPoHAwFkEYepnRhcYOk/ZwSQeQ9QAWR3TwFkd8M5RxBA5tQ01iIElv7BQABZxGFqJwYXWPrPGQFk3gNUANndUwDZ3XDOEQSQOTWNtQiBpX8wEEAWcZjaicEFlv5zRgCZ9wAVQHb3FEB2N5xzBAFkTk1jLUJg6R8MBJBFHKZ2YnCBpf+cEUDmPUAFkN09BZDdDeccQQCZU9NYixBY+gcDAWQRh6mdGFxg6T9nBJB5D1ABZHdPAWR3wzlHEEDm1DTWIgSW/sFAAFnEYWonBhdY+s8ZAWTeA1QA2d1TANndcM4RBJA5NY21CIGlfzAQQBZxmNqJwQWW/nNGAJn3ABVAdvcUQHY3nHMEAWROTWMtQmDpHwwW0aQ1O+HDz3no8tj7uPSfM74H5z0+BZDdPQWQ3Q3nHEEAmVPTWIsQWPoHg0U0SQA5D21c9D4u/eeMADLv4SuA7O4pgOxuOOcIAsicmsZahMDSPxgsokkCyHlo46L3cek/ZwSQeQ9fAWR3TwFkd8M5RxBA5tQ01iIElv7BYBFNEkDOQxsXvY9L/zkjgMx7+Aogu3ue5wDyQUl+M8m9k7xmorxjkuckuXWSZyV5TJKrd3xtmy4JINtoWfdcCCz9g8F5aKIPP+ehy2Pv49J/zvgenPf4FEB29zyvAaTCx08kuWuSW0wB5HpJXpnkJUmekuRpSV44BZLTvrZthwSQbcWsv3iBpX8wWHwD6wfskz7tPOymfRxYYOk/Z3wPzntwCiC7e57XAPKzSX4sybetBJAHJnl2kpsneUuSOyV5epJ7JDnta9t2SADZVmyD9f1i2QCp8SpL719j+tlK8+FnNsqDDeT78GD0s2zY9+AsjO8eRADZ3fO8BpCa9Xj1dHrV0QzI46cZkftOrGVzVZIbJTnta9t2SADZVmyD9Zf+i3Ppv1iW3r8NDuHhV1n6MTp8gzbYAd+HGyA1XsX34LzNEUB29zyvAeRIrq7vOAog35zk+kkeucL6hiS3TfK4U772phNadLMk9Wd1uV2S51955ZW5/PLKIpY5BJb+i3Ppv1iW3r85jvHuYyz9GO3uP0d9vg/nUDzcGL4H57UXQHb3FEDeE0CenOQ6SR69wvraJHdL8qhTvva6E1r0hGlm5RqrCCC7H9irIyz9F+fSf7EsvX/zHu09R1v6MdpTfd6qfB/O67nv0XwPzisugOzuKYC8J4A8NkndBeshK6xvTnKbJA895Ws1g3KpxQzI7sfvRiMs/Rfn0n+xLL1/Gx3Eg6+09GN08PZsVL7vw42Y2q7ke3De1gggu3sKIO8JIPdJ8ozpFrwlW6dmvTzJDZLc85SvvXPLFrkGZEuwTVZf+i/Opf9iWXr/NjmGR19n6cfo6P3ZpH7fh5so9V3H9+C8vRFAdvcUQN4TQK6d5PVJaiakngXyzCQ3TXL/JKd9bdsOCSDbim2w/tJ/cS79F8vS+7fBITz8Kks/Rodv0AY74PtwA6TGq/genLc5AsjungLIewJIaT4gyQuSvDXJu5Lca5oF2eW1bbokgGyjteG6S//FufRfLEvv34aH8dCrLf0YHbo5Gxbv+3BDqKar+R6ctzECyO6e5z2AHCdYsx53TvLS6Ta8q+uc9rVNOyWAbCq1xXpL/8W59F8sS+/fFofysKsu/RgdtjFbFO77cAushqv6Hpy3KQLI7p4CyO6Gc44ggMypOY3lF+cZoBqSwBYCS//w42fMFgeDVQ8isPTvwX2jCiC7iwsguxvOOYIAMqemAHIGmoYksL3A0j/8CCDbHxPesV+BpX8P7lczEUB2FxdAdjeccwQBZE5NAeQMNA1JYHuBpX/4EUC2Pya8Y78CS/8e3K+mADKHtwAyh+J8Ywgg81m+eyQfDs4A1ZAEthBY+ocfP2O2OBisehCBpX8P7hvVDMju4gLI7oZzjiCAzKlpBuQMNA1JYHuBpX/4EUC2Pya8Y78CS/8e3K+mGZA5vAWQORTnG0MAmc/SDMgZWBqSwGkElv7hRwA5zVHhPfsUWPr34D4ta1tmQHYXF0B2N5xzBAFkTk0zIGegaUgC2wss/cOPALL9MeEd+xVY+vfgfjUFkDm8BZA5FOcbQwCZz9IMyBlYGpLAaQSW/uFHADnNUeE9+xRY+vfgPi3NgMyjLYDM4zjXKALIXJIr4/hwcAaohiSwhcDSP/z4GbPFwWDVgwgs/Xtw36hOwdpdXADZ3XDOEQSQOTWnsXw4OANUQxLYQmDpH378jNniYLDqQQSW/j24b1QBZHdxAWR3wzlHEEDm1BRAzkDTkAS2F1j6hx8BZPtjwjv2K7D078H9aroGZA5vAWQOxfnGEEDms3z3SD4cnAGqIQlsIbD0Dz9+xmxxMFj1IAJL/x7cN6oZkN3FBZDdDeccYe8BxC/OOdtnLAIEjhNY+ocfP0cd990Flv49uG9/AWR3cQFkd8M5RxBA5tQ0FgECLQSW/uFHAGlxmCniBIGlfw/uu/kCyO7iAsjuhnOOIIDMqWksAgQIECBAIALIvAeBALK7pwCyu+GcIwggc2oaiwABAgQIEDgXAvsMWQLI7oeUALK74ZwjCCBzahqLAAECBAgQOBcCAshYbRZAevVLAOnVD9UQIECAAAECAwgIIAM0aaVEAaRXvwSQXv1QDQECBAgQIDCAgAAyQJMEkLZNEkDatkZhBAgQIECAQFcBAaRrZ46vywxIr34JIL36oRoCBAgQIEBgAAEBZIAmmQFp2yQBpG1rFEaAAAECBAh0FRBAunbGDMgInRFARuiSGgkQIECAAIFWAgJIq3asLcYpWGuJ9rqCALJXbhsjQIAAAQIEliAggIzVRQGkV78EkF79UA0BAgQIECAwgIAAMkCTVkoUQHr1SwDp1Q/VECBAgAABAgMICCADNEkAadskAaRtaxRGgAABAgQIdBUQQLp25vi6zID06pcA0qsfqiFAgAABAgQGEBBABmiSGZC2TRJA2rZGYQQIECBAgEBXAQGka2fMgIzQGQFkhC6pkQABAgQIEGglIIC0asfaYpyCtZZorysIIHvltjECBAgQIEBgCQICyFhdFEB69UsA6dUP1RAgQIAAAQIDCAggAzRppUQBpFe/BJBe/VANAQIECBAgMICAADJAkwSQtk0SQNq2RmEECBAgQIBAVwEBpGtnjq/LDEivfgkgvfqhGgIECBAgQGAAAQFkgCaZAWnbJAGkbWsURoAAAQIECHQVEEC6dsYMyAidEUBG6JIaCRAgQIAAgVYCAkirdqwtxilYa4n2uoIAslduGyNAgAABAgSWICCAjNVFAaRXvwSQXv1QDQECBAgQIDCAgAAyQJNWShRAevVLAOnVD9UQIECAAAECAwgIIAM0SQBp2yQBpG1rFEaAAAECBAh0FRBAunbm+LrMgPTqlwDSqx+qIUCAAAECBAYQEEAGaJIZkLZNEkDatkZhBAgQIECAQFcBAaRrZ8yAjNAZAWSELqmRAAECBAgQaCUggLRqx9pinIK1lmivKwgge+W2MQIECBAgQGAJAgLIWF0UQHr1SwDp1Q/VECBAgAABAgMICCADNGmlRAGkV78EkF79UA0BAgQIECAwgIAAMkCTBJC2TRJA2rZGYQQIECBAgEBXAQGka2eOr8sMSK9+CSC9+qEaAgQIECBAYAABAWSAJpkBadskAaRtaxRGgAABAgQIdBUQQLp2xgzICJ0RQEbokhoJECBAgACBVgICSKt2rC3GKVhrifa6ggCyV24bI0CAAAECBJYgIICM1UUBpFe/BJBe/VANAQIECBAgMICAADJAk1ZKFEB69UsA6dUP1RAgQIAAAQIDCAggAzRJAGnbJAGkbWsURoAAAQIECHQVEEC6dub4usyA9OqXANKrH6ohQIAAAQIEBhAQQAZokhmQtk0SQNq2RmEECBAgQIBAVwEBpGtnzICM0BkBZIQuqZEAAQIECBBoJSCAtGrH2mKcgrWWaK8rCCB75bYxAgQIECBAYAkCAshYXRRAevVLAOnVD9UQIECAAAECAwgIIAM0aaVEAaRXvwSQXv1QDQECBAgQIDCAgAAyQJMEkLZNEkDatkZhBAgQIECAQFcBAaRrZ46vywxIr34JIL36oRoCBAgQIEBgAAEBZIAmmQFp2yQBpG1rFEaAAAECBAh0FRBAunbGDMgInRFARuiSGgkQIECAAIFWAgJIq3asLcYpWGuJ9rqCABow09UAACAASURBVLJXbhsjQIAAAQIEliAggIzVRQGkV78EkF79UA0BAgQIECAwgIAAMkCTVkoUQHr1SwDp1Q/VECBAgAABAgMICCADNEkAadskAaRtaxRGgAABAgQIdBUQQLp25vi6zID06pcA0qsfqiFAgAABAgQGEBBABmiSGZATm/TFSR6f5MZJfiPJQ5O8Kskdkzwnya2TPCvJY5JcPY100mvbHBECyDZa1iVAgAABAgQIJBFAxjoMzIBc2K9bJfn5JA9M8sYpiNwmyScneWWSlyR5SpKnJXnhFEiud8Jr2x4NAsi2YtYnQIAAAQIEzr2AADLWISCAXNivz0zyoOlPvXL3JD+U5BFJnp3k5knekuROSZ6e5B5TWLnUa9seDQLItmLWJ0CAAAECBM69gAAy1iEggFzYrzsk+aUkn5Tk1Um+I8k7plOw7prkvtPq5XZVkhtNsySXem3bo0EA2VbM+gQIECBAgMC5FxBAxjoEBJBr9uu7kjxs+nKFkAoXVyS5fpJHrqz+hiS3TfK4E1570wmHw82S1J/V5XZJnn/llVfm8ssri5z9ctkVLz77jdgCAQIECBAgQOAMBQSQM8Q9g6EFkAtR75Lkh5N8xnRdR11o/ilJfi7JdZI8emX11ya5W5JHnfDa607o2ROm2ZNrrCKAnMGRbkgCBAgQIEBgsQICyFitFUAu7Ne3JnlXkq+Yvnx0qtWTp7tgPWRl9TcnqQvU6y5ZdRes416rWZJLLWZAxvpeUS0BAgQIECDQVEAAadqYS5QlgFwI823TdR1HYeKGSf48yVcnefh0C956xy2SvDzJDZLcM8kzLvHaO7c8HFwDsiWY1QkQIECAAAECAshYx4AAcmG/6i5Y3zdd1/FnSeqZILecZjrqdKrHTrfefWaSmya5f5JrJ3n9JV7b9mgQQLYVsz4BAgQIECBw7gUEkLEOAQHkwn6VR11UXsGjTpH6gyT/OMlvJ3lAkhckeet0mta9plmQGuGk17Y5IgSQbbSsS4AAAQIECBDwIMLhjgEBZLuW1azHnZO8dLoN7+q7T3pt060IIJtKWY8AAQIECBAgMAmYARnrUBBAevVLAOnVD9UQIECAAAECAwgIIAM0aaVEAaRXvwSQXv1QDQECBAgQIDCAgAAyQJMEkLZNEkDatkZhBAgQIECAQFcBAaRrZ46vywxIr34JIL36oRoCBAgQIEBgAAEBZIAmmQFp2yQBpG1rFEaAAAECBAh0FRBAunbGDMgInRFARuiSGgkQIECAAIFWAgJIq3asLcYpWGuJ9rqCALJXbhsjQIAAAQIEliAggIzVRQGkV78EkF79UA0BAgQIECAwgIAAMkCTVkoUQHr1SwDp1Q/VECBAgAABAgMICCADNEkAadskAaRtaxRGgAABAgQIdBUQQLp25vi6zID06pcA0qsfqiFAgAABAgQGEBBABmiSGZC2TRJA2rZGYQQIECBAgEBXAQGka2fMgIzQGQFkhC6pkQABAgQIEGglIIC0asfaYpyCtZZorysIIHvltjECBAgQIEBgCQICyFhdFEB69UsA6dUP1RAgQIAAAQIDCAggAzRppUQBpFe/BJBe/VANAQIECBAgMICAADJAkwSQtk0SQNq2RmEECBAgQIBAVwEBpGtnjq/LDEivfgkgvfqhGgIECBAgQGAAAQFkgCaZAWnbJAGkbWsURoAAAQIECHQVEEC6dsYMyAidEUBG6JIaCRAgQIAAgVYCAkirdqwtxilYa4n2uoIAslduGyNAgAABAgSWICCAjNVFAaRXvwSQXv1QDQECBAgQIDCAgAAyQJNWShRAevVLAOnVD9UQIECAAAECAwgIIAM0SQBp2yQBpG1rFEaAAAECBAh0FRBAunbm+LrMgPTqlwDSqx+qIUCAAAECBAYQEEAGaJIZkLZNEkDatkZhBAgQIECAQFcBAaRrZ8yAjNAZAWSELqmRAAECBAgQaCUggLRqx9pinIK1lmivKwgge+W2MQIECBAgQGAJAgLIWF0UQHr1SwDp1Q/VECBAgAABAgMICCADNGmlRAGkV78EkF79UA0BAgQIECAwgIAAMkCTBJC2TRJA2rZGYQQIECBAgEBXAQGka2eOr8sMSK9+CSC9+qEaAgQIECBAYAABAWSAJpkBadskAaRtaxRGgAABAgQIdBUQQLp25nzMgNSMztVjteCCagWQgZundAIECBAgQOAwAgLIYdxPu9WRTsG6fpJfT3KnE3b2uUmek+TnTgty4PcJIAdugM0TIECAAAEC4wkIIGP1bKQAUrLvSvK/krw+yX9O8qtJfjLJ7yV5VJLHJfnoJH86VhveXa0AMmjjlE2AAAECBAgcTkAAOZz9abY8WgB5dZLbJ/mwJLdIco8kn5fkfya5cZL7JfmD00A0eY8A0qQRyiBAgAABAgTGERBAxulVVTpCAPnuJG9M8rNJvifJLSfiy5I8MMmXJvnjJB+S5O5J3jRWCy6oVgAZuHlKJ0CAAAECBA4jIIAcxv20Wx0hgHxBkrsk+cQkt07ya9PsR51mVaHk+5K8cgoi/yDJfU6L0eB9AkiDJiiBAAECBAgQGEtAABmrXyMEkJtMMyB3SPKpSa6X5EuS/HKSCifvWJkRqQvQa5bkeWO14d3VCiCDNk7ZBAgQIECAwOEEBJDD2Z9myyMEkMcn+ewkP5Xkhkm+d7rg/EnTDMgPJHl2kv8vyYOni9R/5zQYDd4jgDRoghIIECBAgACBsQQEkLH6NUIAKdG63uPvJfk7Sep2vN+Q5L8keV2SpyT56img1IzIyIsAMnL31E6AAAECBAgcREAAOQj7qTc6QgCpi8rfluS605+6De/bp2s+vnW6KP37pwvQvzDJL5xa4/BvFEAO3wMVECBAgAABAoMJCCBjNWyEAPK3pztc3S7Js5L8+HQK1jOnC9O/MclnTX/qdKyaJRl1EUBG7Zy6CRAgQIAAgYMJCCAHoz/VhkcIIA9I8rQkdTve2yR5rySvSvJ10wMJn5HkI5LU7EfNhLxgCimnAjnwmwSQAzfA5gkQIECAAIHxBASQsXo2QgD5uCR/Pt2C9+OTfH2S/5Tkvkl+f5odOVL/9CTXni5IH6sT/7daAWTErqmZAAECBAgQOKiAAHJQ/q03PkIAOW6n3m96+vnWO9z8DQJI8wYpjwABAgQIEOgnIID068lJFY0aQP7VdCesd43FvbZaAWQtkRUIECBAgAABAhcKCCBjHREjBZCPmk65KuHXTLfmHUt7fbUCyHojaxAgQIAAAQIELhAQQMY6IEYKIHX73Q9M8jfTReh3SfIj0y15S/1aSd47yV3HasEF1QogAzdP6QQIECBAgMBhBASQw7ifdqsjBZB6HkgFkFrqLlh3SPLAKZDUfjw9ycOTvOi0GA3eJ4A0aIISCBAgQIAAgbEEBJCx+jVSAPmLJDdaCSC3vIj61UluMRb/NaoVQAZvoPIJECBAgACB/QsIIPs332WLSwogNStycSjZxeYQ7xVADqFumwQIECBAgMDQAgLIWO0bIYDU083fnuRBSf79xFunXtX1H0dL7Uc9A+SGY/GbARm8X8onQIAAAQIEGggIIA2asEUJIwSQRyb56+lp6F86XWz+xCRXXBRAvjHJB2+x7x1XNQPSsStqIkCAAAECBFoLCCCt23ON4kYIIEdFX3wRumtAZjjWLrvixTOMYggCBAgQIECAwOEEBJDD2Z9myyMFkHUXobsG5BRHgAByCjRvIUCAAAECBFoJCCCt2rG2mJECyF+uXONRYeN2SZ6T5G1Jaj/+oWtA1vb7GisIINubeQcBAgQIECDQS0AA6dWPddWMEkCqznr6+a2SvGN6Dshtk/yLlQcRvk+Sr1u3w81fdw1I8wYpjwABAgQIEOgnIID068lJFY0SQC7eh9cm+Ygk7xqLe221AshaIisQIECAAAECBC4UEEDGOiJGDSAfl+TXxqLeqFoBZCMmKxEgQIAAAQIE3iMggIx1NIwaQI5T/sAkt07ym2O14IJqBZCBm6d0AgQIECBA4DACAshh3E+71RECyLWS/FySe52wk/8oyVOn9eqBhaMuAsionVM3AQIECBAgcDABAeRg9Kfa8AgBpHbsr5K877SH9TT0ejDh1Umul+SfJfkPSZ6Q5EdPpdDnTQJIn16ohAABAgQIEBhEQAAZpFFTmaMEkNWHEL4lyeOmJ6I/Psn7jUV+YrUCyIKaaVcIECBAgACB/QgIIPtxnmsrowSQ1YcQroaRNyS5SZKbTTMg9doVc+EcYBwB5ADoNkmAAAECBAiMLSCAjNW/EQPIahj58+nZIPWMkJ9O8k1JfnesFlxQrQAycPOUToAAAQIECBxGQAA5jPtptzpCAHliki9L8pTpieePTHKjaYcrgHzwNAtSsyGjLwLI6B1UPwECBAgQILB3AQFk7+Q7bXCEAPKC6aLzox399GMCSO3H/ZL8zTQTshPKAd8sgBwQ36YJECBAgACBMQUEkLH6NkIAuVh09RqQmgG5aZI/SHL9JN+a5NvHasEF1QogAzdP6QQIECBAgMBhBASQw7ifdqsjBpA3J/mo6S5YVyb5oCS3T/LK6da8p7W4+H1PTnKHJPefXrhjkudMDzt8VpLHrGzvpNe2qUcA2UbLugQIECBAgACBJALIWIfBiAHkXdMH/6q9ngXyYUk+OcnzZgwgH53kV5LcKcmrpueNVMB5yXQtytOSvHAKJPUskku9tu3RIIBsK2Z9AgQIECBA4NwLCCBjHQIjBpAbTNd6VBCpD/8fk+SHkrw1yb9O8r07tqCevP6rSX4myddMYz0wybOT3DxJPYekgsnTk9wjyUmvbVuKALKtmPUJECBAgACBcy8ggIx1CIwYQI4TvnaSL5huyfvVO7bgEdMsx5cmeeN0Ufu/THLXJPedxi63q6aL4ethiJd6bdtSBJBtxaxPgAABAgQInHsBAWSsQ2ApAWQu9ZpdeXWSP03yoiSfkOR9k/zydJF73QL4aKnb/t52eip7XQB/3Gt1wfyllnp4Yv1ZXW6X5PlXXnllLr+8ssjZL5dd8eKz34gtECBAgAABAgTOUEAAOUPcMxh6lABynWl2423TLXnrv/WnTruqU7HmWmoW5buSfPg0+1EzK78/3WmrLkB/9MqGXpvkbkkelaTqO+61151Q2BOS1OzJNRYBZK52GocAAQIECBA4DwICyFhdHiWA3CrJHyb54yTXnT7w16xD/Xlnkv85zVzULMQuT0L/qiSflOQ+K238wSQPmi5yf8jK1+tuXLdJ8tAkdRes41476eGIZkDG+l5RLQECBAgQINBUQABp2phLlDVSAPmJ6Xa7x+3KByS5Z5J/Mj2Q8LRdqBmQhyf5uJUBXpqkHoZY14Tcevr6LZK8PEmdslXbfcYlXqtwtM3iGpBttKxLgAABAgQIEHAb3uGOge4BpE6BqlOs6gP/j0/P5ThCfq8kn52kZiJ+cvpizZJ85A5duPF0293HJqnA8xlJ6nkgda1HPXOkvl6nYj1zOi2rnhFSNb7+Eq9tW4oAsq2Y9QkQIECAAIFzL2AGZKxDoHsAqRmJr03yo0k+ZSWA1ExH3SL3r6aAUKGglodN4WCX60LunuSp0612/yTJl03h5wHTTMjRdSf3mmZBarsnvbbNESGAbKNlXQIECBAgQICAGZDhjoHuAeRDpmdt3G963kY9n+MpSW4yPY9j37dwummSOyep07LqNryry0mvbXpgCCCbSlmPAAECBAgQIDAJmAEZ61DoHkBWNeuC88+d7hxVF37/3FjUG1UrgGzEZCUCBAgQIECAwHsEBJCxjoaRAkg94byefF5BpE6DunipazH+Y5LvHqsFF1QrgAzcPKUTIECAAAEChxEQQA7jftqtjhRA6pa2X7myo09fefhf7ce/TfKJSX79tBgN3ieANGiCEggQIECAAIGxBASQsfrVPYA8MMmXJ/nh6dSrD1zhraeMn/T/x+rE/61WABmxa2omQIAAAQIEDioggByUf+uNdw8glyWpu03dIck/S/I+K3v4F0ludML/3xqjwRsEkAZNUAIBAgQIECAwloAAMla/ugeQVc03JvnLlS98RJI/uuj/P3t6GOFYXXhPtQLIqJ1TNwECBAgQIHAwAQHkYPSn2vBIAaQeMFjP97jUMz7qIvS6OP1Vp5Lo8SYBpEcfVEGAAAECBAgMJCCADNSsJCMFkLFkT1etAHI6N+8iQIAAAQIEzrGAADJW80cMINdJ8iNJPm0s6o2qFUA2YrISAQIECBAgQOA9AgLIWEfDKAHkvZJ8VZKvm3jrNKtbTs8Fedv0tQom9ectY7XggmoFkIGbp3QCBAgQIEDgMAICyGHcT7vVUQJI7d+rk9ziogBS14NcfdHO1615Vy9WP63NId4ngBxC3TYJECBAgACBoQUEkLHaN3oAuSrJXaZrWa6cnqNRsyMXh5JRuiKAjNIpdRIgQIAAAQJtBASQNq3YqJCRAsjRaVe1Y0d/r6ej32Ta04ufC7IRQLOVBJBmDVEOAQIECBAg0F9AAOnfo9UKRwgg/zTJO5I8Kcljp9mOx03XgAggOx5vl13x4h1H8HYCBAgQIECAwGEFBJDD+m+79RECyK8keXuSuyV56bSD9RDCughdANm24xetL4DsCOjtBAgQIECAwMEFBJCDt2CrAkYIIEc7dNxF6ALIVu2+5soCyI6A3k6AAAECBAgcXEAAOXgLtipghABSt+B958p1H7WDrgHZqs2XXlkAmQnSMAQIECBAgMDBBASQg9GfasMjBJAvT/LxST5mOu1qNYDU7XYfOV0X8p1JviTJc08l0eNNLkLv0QdVECBAgAABAgMJCCADNWv64N694psm+dIpXLxouhD9ZVMYqf/WgwjreSDXSnLtJHftvkMn1CeADNw8pRMgQIAAAQKHERBADuN+2q2OMANytG83T/JNSR6R5HeT1IXoS1sEkKV11P4QIECAAAECZy4ggJw58awbGCmAHO34dZM8MclXzCrRYzABpEcfVEGAAAECBAgMJCCADNSsQU7BGkt0t2oFkN38vJsAAQIECBA4hwICyFhNH2UG5AZJ6ol59xyLd+tqBZCtybyBAAECBAgQOO8CAshYR8AoAeR9k1yV5Ppj8W5drQCyNZk3ECBAgAABAuddQAAZ6wgYJYBcb3rq+Q3H4t26WgFkazJvIECAAAECBM67gAAy1hEggPTqlwDSqx+qIUCAAAECBAYQEEAGaNJKiaMEkDr16s+TmAGZ+fjyJPSZQQ1HgAABAgQI7F1AANk7+U4b7B5A6tkfXzvt4WcJIDv1+tg3CyDzmxqRAAECBAgQ2K+AALJf71231j2APCjJ5yR5dpLvF0B2bfc13y+AzG9qRAIECBAgQGC/AgLIfr133Vr3AHK0fy5C37XTl3i/AHJGsIYlQIAAAQIE9iYggOyNepYNCSCzMM42iIvQZ6M0EAECBAgQIHBeBASQsTo9WgC5UZLrJrlWkquTvDXJu8YiP7FaAWRBzbQrBAgQIECAwH4EBJD9OM+1lZECSIWNCh0XL3+W5PeSvCjJc6dQMpfPvscRQPYtbnsECBAgQIDA8AICyFgtHCWAlOptkrxlJWDULMgNklyW5N5JHpHk7Uk+PsmrxmrDu6sVQAZtnLIJECBAgACBwwkIIIezP82WRwog6/bv/ZM8KsnXrVux8esCSOPmKI0AAQIECBDoKSCA9OzLpaoaMYBcO8k7xmLeuFoBZGMqKxIgQIAAAQIE/q+AADLWkTBiAPmRJD+X5GljUW9UrQCyEZOVCBAgQIAAAQLvERBAxjoaRgsgt07yyiRfmOT5Y1FvVK0AshGTlQgQIECAAAECAsiox8BoAeRnkrxvkruPCr6mbgFkoY21WwQIECBAgMDZCZgBOTvbsxh5pADymCSPTfJ3kvzxWWA0GFMAadAEJRAgQIAAAQJjCQggY/VrlABSt9j9+iSfkuRlYxFvVa0AshWXlQkQIECAAAECLkIf7RjoHkA+OclnJbnn9N/3mZ79cdzTz+vuWNdJ8uLRmrBSrwAycPOUToAAAQIECBxGwAzIYdxPu9XuAeSnplmPz0zyw9Of2yX564t2uPbjulMAqQcWjroIIKN2Tt0ECBAgQIDAwQQEkIPRn2rD3QNIPe38i5J883Tb3a851V6O8yYBZJxeqZQAAQIECBBoIiCANGnEhmV0DyBHu1GzGjUD8stJHr7hvo24mgAyYtfUTIAAAQIECBxUQAA5KP/WGx8lgNSO3STJryX5niRP3HpPx3iDADJGn1RJgAABAgQINBIQQBo1Y4NSRgogtTsfm+Q/JfmEJL++wf6NtooAMlrH1EuAAAECBAgcXEAAOXgLtipgtABSO/eMJHVK1r232tMxVhZAxuiTKgkQIECAAIFGAgJIo2ZsUMqIAeQjpjte/dckV2+wjyOtIoCM1C21EiBAgAABAi0EBJAWbdi4iBECSN1e94ok//qivXpTkg+cvlbPAHlQku/feM97riiA9OyLqggQIECAAIHGAgJI4+YcU9ooAeRVSW6e5AFJ/iLJm5P8hyRHzwSpBxO+elpnrA5cWK0AMnL31E6AAAECBAgcREAAOQj7qTc6QgCpnasAcssk/yPJLya5YZK7TaGjno5+/SQ3TvL7SR6W5DdOLXLYNwogh/W3dQIECBAgQGBAAQFkrKaNFkBqluMWE/Hq3+tLr0nyHUnukqSenD7iIoCM2DU1EyBAgAABAgcVEEAOyr/1xkcNIB80zXLUrMjNkvxZkv+W5LOm07Ket7VEjzcIID36oAoCBAgQIEBgIAEBZKBmJRkxgPyLJE9JUheef9z0dPRnJfnH02laY3XgwmoFkJG7p3YCBAgQIEDgIAICyEHYT73R0QLI65LU3a++IMkPTxekvzTJI6cZkKPTs04NcuA3CiAHboDNEyBAgAABAuMJCCBj9WyEAHKtJPXMjzrdqq77qP/W8z/q7x8zXZhe6kcXqo/VATMgI/dL7QQIECBAgEADAQGkQRO2KKF7APmoJP8uyftPweOqJE+dbsP7hCSfN/396La8FU5GXsyAjNw9tRMgQIAAAQIHERBADsJ+6o12DyB1i91HJfmqJE+eZj0+IMlNkjw6yY9NDyP84CRvT3KPU0v0eKMA0qMPqiBAgAABAgQGEhBABmrWQBeh3ybJC5O8LMkXT6dg/XGSDx+Le221AshaIisQIECAAAECBC4UEEDGOiK6z4CsatZpWC+ZZj5+NclPJfnUsbjXViuArCWyAgECBAgQIEBAABn5GBgpgJTz9ZK8bWTwNbULIAturl0jQIAAAQIEzkbADMjZuJ7VqKMFkLNy6DKuANKlE+ogQIAAAQIEhhEQQIZp1f8pVADp1S8BpFc/VEOAAAECBAgMICCADNCklRIFkF79EkB69UM1BAgQIECAwAACAsgATRJA2jZJAGnbGoURIECAAAECXQUEkK6dOb4uMyC9+iWA9OqHaggQIECAAIEBBASQAZpkBqRtkwSQtq1RGAECBAgQINBVQADp2hkzICN0RgAZoUtqJECAAAECBFoJCCCt2rG2GKdgrSXa6woCyF65bYwAAQIECBBYgoAAMlYXBZBe/RJAevVDNQQIECBAgMAAAgLIAE1aKVEAOblfP53kB5J8b5I7JnlOklsneVaSxyS5enr7Sa9tc0QIINtoWZcAAQIECBAgkEQAGeswEEAu3a/PT/K8JF+U5AVJXpnkJUmekuRpSV44BZLrnfDatkeDALKtmPUJECBAgACBcy8ggIx1CAggx/frRklenuTNSZ40/ffZSW6e5C1J7pTk6UnukeSBSS712rZHgwCyrZj1CRAgQIAAgXMvIICMdQgIIMf3q061+usk753kF5J8RJK7JrnvtHq5XZWkgsrjT3jtpKPhZknqz+pyuyTPv/LKK3P55ZVFzn657IoXn/1GbIEAAQIECBAgcIYCAsgZ4p7B0ALINVHvneT7knxkkm+fAshHJbl+kkeurP6GJLdN8rgTXnvTCT17whRerrGKAHIGR7ohCRAgQIAAgcUKCCBjtVYAubBfFTJ+L8mXJ6mpgbr4vGZAbp/kOkkevbL6a5PcLcmjTnjtdSccDmZAxvpeUS0BAgQIECDQVEAAadqYS5QlgFwI8w1JLktSF6DXchRAPmS6C9ZDVlav60Nuk+ShJ7xWsyTbLK4B2UbLugQIECBAgAABd8Ea7hgQQC5s2auT3CTJO6Yvv8/099ckue50C9566RbTReo3SHLPJM+4xGvv3PKIEEC2BLM6AQIECBAgQMAMyFjHgAByYb/qLlfXXvnSU5O8dJoJqbtiPXa69e4zk9w0yf2n9V9/ide2PRoEkG3FrE+AAAECBAicewEBZKxDQAA5uV9Hp2DVfx8wPQ/krUneleRe0yxIjXDSa9scEQLINlrWJUCAAAECBAg4BWu4Y0AA2a5lNetx52lWpG7Du7qc9NqmWxFANpWyHgECBAgQIEBgEjADMtahIID06pcA0qsfqiFAgAABAgQGEBBABmjSSokCSK9+CSC9+qEaAgQIECBAYAABAWSAJgkgbZskgLRtjcIIECBAgACBrgICSNfOHF+XGZBe/RJAevVDNQQIECBAgMAAAgLIAE0yA9K2SQJI29YojAABAgQIEOgqIIB07YwZkBE6I4CM0CU1EiBAgAABAq0EBJBW7VhbjFOw1hLtdQUBZK/cNkaAAAECBAgsQUAAGauLAkivfgkgvfqhGgIECBAgQGAAAQFkgCatlCiA9OqXANKrH6ohQIAAAQIEBhAQQAZokgDStkkCSNvWKIwAAQIECBDoKiCAdO3M8XWZAenVLwGkVz9UQ4AAAQIECAwgIIAM0CQzIG2bJIC0bY3CCBAgQIAAga4CAkjXzpgBGaEzAsgIXVIjAQIECBAg0EpAAGnVjrXFOAVrLdFeVxBA9sptYwQIECBAgMASBASQsboogPTqlwDSqx+qIUCAx0lvDAAAGlZJREFUAAECBAYQEEAGaNJKiQJIr34JIL36oRoCBAgQIEBgAAEBZIAmCSBtmySAtG2NwggQIECAAIGuAgJI184cX5cZkF79EkB69UM1BAgQIECAwAACAsgATTID0rZJAkjb1iiMAAECBAgQ6CoggHTtjBmQETojgIzQJTUSIECAAAECrQQEkFbtWFuMU7DWEu11BQFkr9w2RoAAAQIECCxBQAAZq4sCSK9+CSC9+qEaAgQIECBAYAABAWSAJq2UKID06pcA0qsfqiFAgAABAgQGEBBABmiSANK2SQJI29YojAABAgQIEOgqIIB07czxdZkB6dUvAaRXP1RDgAABAgQIDCAggAzQJDMgbZskgLRtjcIIECBAgACBrgICSNfOmAEZoTMCyAhdUiMBAgQIECDQSkAAadWOtcU4BWst0V5XEED2ym1jBAgQIECAwBIEBJCxuiiA9OqXANKrH6ohQIAAAQIEBhAQQAZo0kqJAkivfgkgvfqhGgIECBAgQGAAAQFkgCYJIG2bJIC0bY3CCBAgQIAAga4CAkjXzhxflxmQXv0SQHr1QzUECBAgQIDAAAICyABNMgPStkkCSNvWKIwAAQIECBDoKiCAdO2MGZAROiOAjNAlNRIgQIAAAQKtBASQVu1YW4xTsNYS7XUFAWSv3DZGgAABAgQILEFAABmriwJIr34JIL36oRoCBAgQIEBgAAEBZIAmrZQogPTqlwDSqx+qIUCAAAECBAYQEEAGaJIA0rZJAkjb1iiMAAECBAgQ6CoggHTtzPF1mQHp1S8BpFc/VEOAAAECBAgMICCADNAkMyBtmySAtG2NwggQIECAAIGuAgJI186YARmhMwLICF1SIwECBAgQINBKQABp1Y61xTgFay3RXlcQQPbKbWMECBAgQIDAEgQEkLG6KID06pcA0qsfqiFAgAABAgQGEBBABmjSSokCSK9+CSC9+qEaAgQIECBAYAABAWSAJgkgbZskgLRtjcIIECBAgACBrgICSNfOHF+XGZBe/RJAevVDNQQIECBAgMAAAgLIAE0yA9K2SQJI29YojAABAgQIEOgqIIB07YwZkBE6I4CM0CU1EiBAgAABAq0EBJBW7VhbjFOw1hLtdQUBZK/cNkaAAAECBAgsQUAAGauLAkivfgkgvfqhGgIECBAgQGAAAQFkgCatlCiA9OqXANKrH6ohQIAAAQIEBhAQQAZokgDStkkCSNvWKIwAAQIECBDoKiCAdO3M8XWZAenVLwGkVz9UQ4AAAQIECAwgIIAM0CQzIG2bJIC0bY3CCBAgQIAAga4CAkjXzpgBGaEzAsgIXVIjAQIECBAg0EpAAGnVjrXFOAVrLdFeVxBA9sptYwQIECBAgMASBASQsboogPTqlwDSqx+qIUCAAAECBAYQEEAGaNJKiQJIr34JIL36oRoCBAgQIEBgAAEBZIAmCSBtmySAtG2NwggQIECAAIGuAgJI184cX5cZkF79EkB69UM1BAgQIECAwAACAsgATTID0rZJAkjb1iiMAAECBAgQ6CoggHTtjBmQETojgIzQJTUSIECAAAECrQQEkFbtWFuMU7DWEu11BQFkr9w2RoAAAQIECCxBQAAZq4sCSK9+CSC9+qEaAgQIECBAYAABAWSAJq2UKID06pcA0qsfqiFAgAABAgQGEBBABmiSANK2SQJI29YojAABAgQIEOgqIIB07czxdZkB6dUvAaRXP1RDgAABAgQIDCAggAzQJDMgbZskgLRtjcIIECBAgACBrgICSNfOmAEZoTMCyAhdUiMBAgQIECDQSkAAadWOtcU4BWst0V5XEED2ym1jBAgQIECAwBIEBJCxuiiAXLNfn57kW5N8eJI/SPK5SV6R5I5JnpPk1kmeleQxSa6e3n7Sa9scEQLINlrWJUCAAAECBAgkEUDGOgwEkAv7daskv5nkS5L8YpJvT/JhSe6T5JVJXpLkKUmeluSFUyC53gmvbXs0CCDbilmfAAECBAgQOPcCAshYh4AAcmG/7pfkQ5M8Y/ryvZO8OMnnJXl2kpsneUuSOyV5epJ7JHngCa9tezQIINuKWZ8AAQIECBA49wICyFiHgABycr9qJuThSV6U5K5J7jutXm5XJblRksef8NpJo98sSf1ZXW6X5PlXXnllLr+8ssjZL5ddUfnKQoAAAQIECBAYV0AAGat3Asil+3XdJH+Y5Fum6z6un+SRK6u/IcltkzwuyaVee9MJh8MTpvByjVUEkLG+iVRLgAABAgQIHFZAADms/7ZbF0AuLfbEJJ+a5GOTfH2S6yR59Mrqr01ytySPOuG1153QEDMg2x6t1idAgAABAgQIHCMggIx1WAggx/erLjr/kSlgvDzJY6e7YD1kZfU3J7lNkoee8FrNkmyzuAZkGy3rEiBAgAABAgTcBWu4Y0AAuWbLbpHkpUm+Mslzp5crkNSF6XUL3lpqnQomN0hyzxNee+eWR4QAsiWY1QkQIECAAAECZkDGOgYEkAv79d5JXpbkVy463eptSep0qpoJqWeBPDPJTZPcP8m1k7z+Eq9tezQIINuKWZ8AAQIECBA49wICyFiHgAByYb/qIYR16tXFS814fHSSFyR5a5J3JbnXNAtS6z7ghNe2OSIEkG20rEuAAAECBAgQcArWcMeAALJdy2rW487TKVp1G97V5aTXNt2KALKplPUIECBAgAABApOAGZCxDgUBpFe/BJBe/VANAQIECBAgMICAADJAk1ZKFEB69UsA6dUP1RAgQIAAAQIDCAggAzRJAGnbJAGkbWsURoAAAQIECHQVEEC6dub4usyA9OqXANKrH6ohQIAAAQIEBhAQQAZokhmQtk0SQNq2RmEECBAgQIBAVwEBpGtnzICM0BkBZIQuqZEAAQIECBBoJSCAtGrH2mKcgrWWaK8rCCB75bYxAgQIECBAYAkCAshYXRRAevVLAOnVD9UQIECAAAECAwgIIAM0aaVEAaRXvwSQXv1QDQECBAgQIDCAgAAyQJMEkLZNEkDatkZhBAgQIECAQFcBAaRrZ46vywxIr34JIL36oRoCBAgQIEBgAAEBZIAmmQFp2yQBpG1rFEaAAAECBAh0FRBAunbGDMgInRFARuiSGgkQIECAAIFWAgJIq3asLcYpWGuJ9rqCALJXbhsjQIAAAQIEliAggIzVRQGkV78EkF79UA0BAgQIECAwgIAAMkCTVkoUQHr1SwDp1Q/VECBAgAABAgMICCADNEkAadskAaRtaxRGgAABAgQIdBUQQLp25vi6zID06pcA0qsfqiFAgAABAgQGEBBABmiSGZC2TRJA2rZGYQQIECBAgEBXAQGka2fMgIzQGQFkhC6pkQABAgQIEGglIIC0asfaYpyCtZZorysIIHvltjECBAgQIEBgCQICyFhdFEB69UsA6dUP1RAgQIAAAQIDCAggAzRppUQBpFe/BJBe/VANAQIECBAgMICAADJAkwSQtk0SQNq2RmEECBAgQIBAVwEBpGtnjq/LDEivfgkgvfqhGgIECBAgQGAAAQFkgCaZAWnbJAGkbWsURoAAAQIECHQVEEC6dsYMyAidEUBG6JIaCRAgQIAAgVYCAkirdqwtxilYa4n2uoIAslduGyNAgAABAgSWICCAjNVFAaRXvwSQXv1QDQECBAgQIDCAgAAyQJNWShRAevVLAOnVD9UQIECAAAECAwgIIAM0SQBp2yQBpG1rFEaAAAECBAh0FRBAunbm+LrMgPTqlwDSqx+qIUCAAAECBAYQEEAGaJIZkLZNEkDatkZhBAgQIECAQFcBAaRrZ8yAjNAZAWSELqmRAAECBAgQaCUggLRqx9pinIK1lmivKwgge+W2MQIECBAgQGAJAgLIWF0UQHr1SwDp1Q/VECBAgAABAgMICCADNGmlRAGkV78EkF79UA0BAgQIECAwgIAAMkCTBJC2TRJA2rZGYQQIECBAgEBXAQGka2eOr8sMSK9+CSC9+qEaAgQIECBAYAABAWSAJpkBadskAaRtaxRGgAABAgQIdBUQQLp2xgzICJ0RQEbokhoJECBAgACBVgICSKt2rC3GKVhrifa6ggCyV24bI0CAAAECBJYgIICM1UUBpFe/BJBe/VANAQIECBAgMICAADJAk1ZKFEB69UsA6dUP1RAgQIAAAQIDCAggAzRJAGnbJAGkbWsURoAAAQIECHQVEEC6dub4usyA9OqXANKrH6ohQIAAAQIEBhAQQAZokhmQtk0SQNq2RmEECBAgQIBAVwEBpGtnzICM0BkBZIQuqZEAAQIECBBoJSCAtGrH2mKcgrWWaK8rCCB75bYxAgQIECBAYAkCAshYXRRAevVLAOnVD9UQIECAAAECAwgIIAM0aaVEAaRXvwSQXv1QDQECBAgQIDCAgAAyQJMEkLZNEkDatkZhBAgQIECAQFcBAaRrZ46vywxIr34JIL36oRoCBAgQIEBgAAEBZIAmmQFp2yQBpG1rFEaAAAECBAh0FRBAunbGDMgInRFARuiSGgkQIECAAIFWAgJIq3asLcYpWGuJ9rqCALJXbhsjQIAAAQIEliAggIzVRQGkV78EkF79UA0BAgQIECAwgIAAMkCTVkoUQHr1SwDp1Q/VECBAgAABAgMICCADNEkAadskAaRtaxRGgAABAgQIdBUQQLp25vi6zID06pcA0qsfqiFAgAABAgQGEBBABmiSGZC2TRJA2rZGYQQIECBAgEBXAQGka2fMgIzQGQFkhC6pkQABAgQIEGglIIC0asfaYpyCtZZorysIIHvltjECBAgQIEBgCQICyFhdFEB69UsA6dUP1RAgQIAAAQIDCAggAzRppUQBpFe/BJBe/VANAQIECBAgMICAADJAkwSQtk0SQNq2RmEECBAgQIBAVwEBpGtnjq/LDEivfgkgvfqhGgIECBAgQGAAAQFkgCaZAWnbJAGkbWsURoAAAQIECHQVEEC6dsYMyAidEUBG6JIaCRAgQIAAgVYCAkirdqwtxilYa4n2uoIAslduGyNAgAABAgSWICCAjNVFAWS+ft0xyXOS3DrJs5I8JsnVWw4vgGwJZnUCBAgQIECAgAAy1jEggMzTr+sleWWSlyR5SpKnJXnhFEi22YIAso2WdQkQIECAAAECSQSQsQ4DAWSefj0wybOT3DzJW5LcKcnTk9xjy+EFkC3BrE6AAAECBAgQEEDGOgYEkHn69fgkd01y32m4cr0qyY1OGP5mSerP6vLRNWvyvOc9L7e//e3nqWzNKJ/2tF/ey3ZshAABAgQIECBwVgIvftTHn9XQ1xj3Fa94RR784AfX1++e5Ff3tuEFbUgAmaeZ35zk+kkeuTLcG5LcNsmbLrGJJySp4GIhQIAAAQIECBAYT+Dzk3z/eGUfvmIBZJ4ePDnJdZI8emW41ya5W5LXXWITx82AvH+Smvr4nSR/PU9pJ45yuyTPT1LfQHUNi2X5Anq+/B4ft4f6fv76rud6fv4E9rfH9Y/Ol03X/tYZL5YtBQSQLcEusfpjk9RdsB6y8vqbk9wmSc2EdF3+zzUnSe6c5Le6FqmuWQX0fFbOYQbT92FaNVuhej4b5TAD6fkwrVKoADLPMXCfJM+YbsFbI94iycuT3CDJO+fZxJmM4ofVmbC2HlTPW7fnzIrT9zOjbTuwnrdtzZkVpudnRmvguQUEkHlEr53k9UlqJqSeBfLMJDdNcv95hj+zUfywOjPatgPredvWnGlh+n6mvC0H1/OWbTnTovT8THkNPqeAADKf5gOSvCDJW5O8K8m9plmQ+bYw/0h+WM1v2n1EPe/eobOpT9/PxrXzqHreuTtnU5uen42rUc9AQACZF7VmPep6ipdOt+Gdd/T5R6sL4R+W5LuT/Mn8wxuxoYCeN2zKHkrS9z0gN9uEnjdryB7K0fM9INvEPAICyDyORiFAgAABAgQIECBAYAMBAWQDJKsQIECAAAECBAgQIDCPgAAyj6NRCBAgQIAAAQIECBDYQEAA2QDJKgQIECBAgAABAgQIzCMggMzjaBQCBAgQIECAAAECBDYQEEA2QLIKgYUIfHqSb03y4Un+IMnnJnnFQvbNbqwX+OkkP5Dke9evao2FCDw5yR0GeCbVQrgPuhtfnOTxSW6c5DeSPDTJqw5akY0TOEFAADm/h8cdp4cm3jrJs5I8JsnV55dj8Xt+qyS/meRLkvxikm9P8mFJ7r74PbeDJfD5SZ6X5IsEkHNzQHx0kl9JcicfRBff8/r5/vNJHpjkjVMQuU2ST1j8ntvBYQUEkGFbt1Ph10vyyiQvSfKUJE9L8sIpkOw0sDe3Fbhfkg9N8oypwnsneXGS92lbscLmErjR9FDUNyd5kgAyF2vrca6V5FeT/EySr2ldqeLmEPjMJA+a/tR49Q9LPzT9zJ9jfGMQmF1AAJmddIgB619Jnp3k5kneMv0L2dOT3GOI6hU5h0DNhDx86v0c4xmjr8Bzkvx1kvdO8gsCSN9GzVjZI6Z/XPrS6V/E6/S7t884vqF6CdRpdr+U5JOSvDrJdyR5R5Iv7FWmagi8R0AAOZ9HQ50netck9512v46Dq5LUv5Rali9w3SR/mORbknzn8nf3XO9hzXR9X5KPnE67E0CWfzjcYPoQ+qdJXjSdhvO+Se6Z5K3L3/1zu4ffleRh095XCKnf8W84txp2vL2AANK+RWdS4DcnuX6SR66MXj+obpvkTWeyRYN2Enhikk9N8rFJ/qZTYWqZVaC+x38vyZdPp9vVxecCyKzELQf7giT1YbRuNlHXA1w7ye9PN6A4OgWzZeGKOrXAXZL8cJLPmE6vrms6PyVJfd21nadm9cazFBBAzlK379h1Z5TrJHn0SomvTXK3JK/rW7bKZhC4T5IfmXr98hnGM0RfgW9Ictl0AXpVKYD07dWclX3VdCpOfa8fLT+Y5I+mm43MuS1j9RCouxu+K8lXTOUcndVQx8Dv9ChRFQQuFBBAzucR8dgkdResh6zsfl2gWnfNMGW73GPiFklemuQrkzx3ubtpzyaBOg3jJtO54PWluuFAnRdeQaSuEbAsU6BmQOr6ro9b2b36vn9Bkm9b5i6f+72qvtYp1Ee/02+Y5M+ni9GvPPc6AFoKCCAt23LmRdW/itRUfN2Ct5b6YFr/Gl7nDr/zzLduA4cQqAuQXzbdlnN15uuvTNEfoh172WbdZKJOvzlanjoF0AogdWqOZZkC9RyIev5D/UPTT0yn5dSsd51iWzPdluUJ1F2w6lqvxyX5syT1TJBbTv+o6DTb5fV7EXskgCyijVvvRH0oef30C6rukPPMJDf1sKqtHUd6Qz2EsE69unip8PmakXZEracWcArWqemGe2PdhrUCZz0D5E+SfFmSHx9uLxS8qUB9lqvwUcHjZtODZv9xkt/edADrEdi3gACyb/E+23vANCVfd0Wpc0fvNc2C9KlQJQQIECBAgAABAosTEEAW19KtdqhmPe48nZZRt+G1ECBAgAABAgQIEDhTAQHkTHkNToAAAQIECBAgQIDAqoAA4nggQIAAAQIECBAgQGBvAgLI3qhtiAABAgQIECBAgAABAcQxQIAAAQIECBAgQIDA3gQEkL1R2xABAgQIECBAgAABAgKIY4AAAQIECBAgQIAAgb0JCCB7o7YhAgQILFrgY5O8d5JfWvRe2jkCBAgQ2FlAANmZ0AAECBAgkOSfJ/lnSW6f5B0niPzLJM9O8mdJvjHJLZN8DkECBAgQOD8CAsj56bU9JUCAwJwC10uy+jvkA5I8N8k/TfIn04beK8l1krx5+v+fnOR7ktwxyV8meUKSWyd58Ephtf7fzFmosQgQIECgl4AA0qsfqiFAgMAoAq9P8oFJ3nlCwRVAKqhcN8nVSX4tyYeuzJBUaLl2kjeujPHKJH9/FAR1EiBAgMD2AgLI9mbeQYAAAQLJf03yyCQv2RDjYUm+YjpF6yi0HDcDsuFwViNAgACBUQUEkFE7p24CBAgcVuD3k9T1HD+xQRk1y1GnVt0kyR+vrH+pAFIzJm/fYFyrECBAgMCAAgLIgE1TMgECBBoIVAD56iQ/tkEtn5nk36y5OH11mAort03yVxuMbRUCBAgQGExAABmsYcolQIBAE4E/THJFkh9vUo8yCBAgQGAQAQFkkEYpkwABAs0E/ijJQ5P8xy3qeup0q97/dYn31EXt90vyU1uMaVUCBAgQGExAABmsYcolQIBAA4G6RqNCxEcnqbtWbbrUcz/qLlj/6BJv+O9JHpLk5zcd0HoECBAgMJ6AADJez1RMgACBQwvcZ7r2o2Ystnlmx9cnubkAcuj22T4BAgQOKyCAHNbf1gkQIDCiwM8muSrJZ29ZfM2AfGWSt1zifTdM8qlb3Np3y81bnQABAgQ6CAggHbqgBgIECIwjcM8kP5fkrkletmXZT5luxXvSKVhfsuGtfbfctNUJECBAoIuAANKlE+ogQIDAOAJ1i9z/fAbl1gxIzY684wzGNiQBAgQINBEQQJo0QhkECBAgQIAAAQIEzoOAAHIeumwfCRAgQIAAAQIECDQREECaNEIZBAgQIECAAAECBM6DgAByHrpsHwkQIECAAAECBAg0ERBAmjRCGQQIECBAgAABAgTOg4AAch66bB8JECBAgAABAgQINBEQQJo0QhkECBAgQIAAAQIEzoOAAHIeumwfCRAgQIAAAQIECDQREECaNEIZBAgQIECAAAECBM6DgAByHrpsHwkQIECAAAECBAg0ERBAmjRCGQQIECBAgAABAgTOg4AAch66bB8JECBAgAABAgQINBEQQJo0QhkECBAgQIAAAQIEzoOAAHIeumwfCRAgQIAAAQIECDQREECaNEIZBAgQIECAAAECBM6DgAByHrpsHwkQIECAAAECBAg0ERBAmjRCGQQIECBAgAABAgTOg4AAch66bB8JECBAgAABAgQINBEQQJo0QhkECBAgQIAAAQIEzoOAAHIeumwfCRAgQIAAAQIECDQREECaNEIZBAgQIECAAAECBM6DgAByHrpsHwkQIECAAAECBAg0Efjf98OE3/2FwOoAAAAASUVORK5CYII=" width="640">


###  图中我们可以看到，由于是随机的，10000个人选择10种广告，每个人平均会在1000次左右

## 四、置信区间上界算法


```python
import math
N = 10000  # 1000个用户
d = 10     # 10个广告
ads_selected = [] # 广告选择
numbers_of_selections = [0] * d # 多项选择
sums_of_rewards = [0] * d  # 奖励总和
total_reward = 0
for n in range(0, N): # 第n个用户
    ad = 0 # 广告初始化
    max_upper_bound = 0  # 最大上界初始化
    for i in range(0, d): # 第i个广告
        if (numbers_of_selections[i] > 0):
            average_reward = float(sums_of_rewards[i]) / float(numbers_of_selections[i]) # 平均奖励，这里如果python2记得用float
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i]) # 置信区间
            upper_bound = average_reward + delta_i # 置信区间上界
#             print(average_reward)
#             print(delta_i )
        else:
            upper_bound = 10000
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
#     print(ad)
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
# print(ads_selected)
print(total_reward)
# print(numbers_of_selections)
# print(sums_of_rewards)
```

    2358
    


```python
# 画图
plt.hist(ads_selected)
plt.title(u'广告选择直方图')
plt.xlabel(u'广告')
plt.ylabel(u'每个广告的点击数')
plt.show()
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAJYCAYAAACadoJwAAAgAElEQVR4Xu3dDdxtZV3n/88oTypZYhYoKSgwYCgFo+CIgZbNiIrUpJWCD0zlA8mkGZiRUFpqaBYOUwFBjSA18VdLTSmz1FRKj5UPwPQgKoEpEkwlKAr8X7+6dm7vc5/77LWv395rXWt91ut1gs5e929d6/27juzvudbDf8BNAQUUUEABBRRQQAEFFFiTwH9Y03E8jAIKKKCAAgoooIACCiiAAcRJoIACCiiggAIKKKCAAmsTMICsjdoDKaCAAgoooIACCiiggAHEOaCAAgoooIACCiiggAJrEzCArI3aAymggAIKKKCAAgoooIABxDmggAIKKKCAAgoooIACaxMwgKyN2gMpoIACCiiggAIKKKCAAcQ5oIACCiiggAIKKKCAAmsTMICsjdoDKaCAAgoooIACCiiggAHEOaCAAgoooIACCiiggAJrEzCArI3aAymggAIKKKCAAgoooIABxDmggAIKKKCAAgoooIACaxMwgKyN2gMpoIACCiiggAIKKKCAAcQ5oIACCiiggAIKKKCAAmsTMICsjdoDKaCAAgoooIACCiiggAHEOaCAAgoooIACCiiggAJrEzCArI3aAymggAIKKKCAAgoooIABxDmggAIKKKCAAgoooIACaxMwgKyN2gMpoIACCiiggAIKKKCAAcQ5oIACCiiggAIKKKCAAmsTMICsjdoDKaCAAgoooIACCiiggAHEOaCAAgoooIACCiiggAJrEzCArI3aAymggAIKKKCAAgoooIABxDmggAIKKKCAAgoooIACaxMwgKyN2gMpoIACCiiggAIKKKCAAcQ5oIACCiiggAIKKKCAAmsTMICsjdoDKaCAAgoooIACCiiggAHEOaCAAgoooIACCiiggAJrEzCArI3aAymggAIKKKCAAgoooIABxDmggAIKKKCAAgoooIACaxMwgKyN2gMpoIACCiiggAIKKKCAAcQ5oIACCiiggAIKKKCAAmsTMICsjdoDKaCAAgoooIACCiiggAHEOaCAAgoooIACCiiggAJrEzCArI3aAymggAIKKKCAAgoooIABxDmggAIKKKCAAgoooIACaxMwgKyN2gMpoIACCiiggAIKKKCAAcQ5oIACCiiggAIKKKCAAmsTMICsjdoDKaCAAgoooIACCiiggAHEOaCAAgoooIACCiiggAJrEzCArI3aAymggAIKKKCAAgoooIABxDmggAIKKKCAAgoooIACaxMwgKyN2gMpoIACCiiggAIKKKCAAcQ5oIACCiiggAIKKKCAAmsTMICsjdoDKaCAAgoooIACCiiggAHEOaCAAgoooIACCiiggAJrEzCArI3aAymggAKdBXYFvtz5p/wBBRRQQAEFBixgABlwcxyaAgpMWuCJwK8CDwH+cQGJfYAbgdvKvv8NeDFwFHD7Aj+/cZc9gC8C3wi8HDgX+OgWdQ4Efhx4GXAdsBtwX+CWuTHN//iXgFt3Mq67As8D/gL40yXOIX7kJODdwKcX+Pm7l3EvsOu/7/IV4F+6/ID7KqCAAlMXMIBMfQZ4/gooMFSB+OIfX5p/GfjJBQb528ADgSOBO4BnARcA8SV+Z9tjgEcAP1d23Bf4W+BpwJ+XcRwNvG+LQv+5fH4w8H+BQ3cSWJ4AvG2u3l5AhJL4Ndt2L1/uTwd+ce73479d8VmcZwScHW13Az5RwsuTd4YAXFjcFtj133f5EPCwLj/gvgoooMDUBQwgU58Bnr8CCgxZ4LXAR4CL5gYZgSJWJ+LL92wF4T7A3wOnlNARuz+z/PsuG764x8rE/Jf8+Di+nF9aAkyEnpOBXwK+tRzrr4EIKVeXL/43A/ErtruUX98GfLDUuKEEh4PK70UYubbs/3ogVhpihWZ+i3pf37EZPwOctZOfiRWUWL35T8C2nex7HhBB6Pvm9jseiEvh3gjcueHnzwAeBzyy47jdXQEFFJi0gAFk0u335BVQYCACvwD8RMexvAZ4UfmZWB14wYI/H5dHxQrH/BYhIsLFJUB8qY9Llo4ol2DFfyfiS/n/K1/A48v4zwIx5tjii30Ej/ktLvmaBZ8IUbF6EoEjwkKstMRqyU0bfuabgbicKS4hm33RfxDwl8C3lxWZ2Y/EmCJIxb7/XH4zVl5mvzdfOsb7XcDbN/GJMUYY+5vyWQSVGMcsgETQu6qEuwgis3HFfTlfACKAfDfwHQvau5sCCiigAGAAcRoooIAC/QucWVYd4vKpnW0RFmaXJsVKQ3zxjlWSuP8iVhdm21OBc8o9HBu/uH92k4PEF+n4Mh6fxb0eERIiWMSlYHGcCAERBjZusZrxgLLy8VbgUSVcfLzsGCs2fwgcC3yyBJDNjj+rG6sgs/82xcpC1Lz/XNCI/WYBYH4scY/IMisRlwP/tRSKVZ8IZ7MA8kogLv/auJ0GnF0CSISbODc3BRRQQIEFBQwgC0K5mwIKKLBCgVjJ+LFNViYWOWSsLsSN33EpUNw0Pts2uwRrs3oRIOILfaxaxGVd31Tuafhc+Rv/CATvLPeDxKVYs/sv4t9jn9kWl2DFzeLfUlYM4vf/I/DccmlYhKQIMb8H/BbwV8CnNrmHI+7piHs3ttoi0ERgmt9i9SPOYXbDfVy2ds9NLvWa/5kIR/FrduP+fAA5oVx2FUEqwlistMTnsYITQSdWa2IFxACyyCx1HwUUUGBOwADidFBAAQX6F4jwETeax+U/XbZ7A/Fr9gV6/mfjb/Hjb/AP2KRgXHoUT9aKX7GqcVjZ59XA7wOvK6HkXsDewMfK07iuL6sbcVlTrAzEvhsDSDx1K0JN1Ih7SOKG9J8G3lQuVfr5cqN8/Fwcb+OlZ3FpVtxA/+YdQMTPxL0lcUnUVtvF5cMTO4DOAsgPl0vSXlHGGk/4+gPgB8oKT1zGFpsBpAOuuyqggAIzAQOIc0EBBRToXyACyE8BcTN5l+1/APElebMAslWdCCCnlqc+ReiJexv+T3laVHypnm1RP77Ax1Oe4vKp+Gz2xX5j/f8CvKOsDMT9KHHTeawa/HG5gT1WUeJm9FhRiMvGYsUmntwVoWZ++3wJVVuN/3eBWKHYaotLtx6/xQ5xz8shGz6fXwGJlaBY4Xl0eVpXrMq8qjySePbYXQNIl9nqvgoooEARMIA4FRRQQIH+BeILe6woxGpDX1sEhLiPYv6pUhFuIjTEO0l2FEC+s9yUHpcpxfZ84FfKjeuz1ZlYCYlLr+JJWp8p97DEY37jRu6NWwSQH9rJCkis6uwsgMQTr94D/Nomx4iVoVil2XgZ1yyAxGpU3DQfjwqOsBQOV5bHFMex/wiIe0fi/hgvweprxnpcBRRoVsAA0mzrHLgCCoxI4KXlsqP9lzynuIl6s6c8bVYuvnTHPRQbt80CyBXA7wDxxK0dBZB4CtZvlpWRuLxqdg9IrLDEF/24VyL+W7NneTTv7B6SuHF7s/eKRADZ6v0ecVlYrKpsFUDiErFYfYkQMf+ukdk5h1UcJ15SOL/NB5B4J8gbys//GfAnwK+XscXPRQjZzwCy5Iz1xxRQYNICBpBJt9+TV0CBgQjE39LHTdyLPAVrsyHHl/n4Uh7/nN2fsHG/+5Uv0ceUlYGdBZAYzwfKjeTxbpD5ABJvRo+byCOcxBaXdMVKycab0GfHmL2UMAJW1Nlqy1gBiftf/ne5oX6zt5R/uNzTEW+K3yyAzL8HJM4tHkkcq1RPKisicfN8bF6CNZA/QA5DAQXaEjCAtNUvR6uAAuMUiMuF4hKfjX8jv+jZxnso4t0dW33Bj7+tv6bcRB2XWm0VQOLJV7E6ESsFs5vE46lXEXLizeyxqhBPspp/Q/v8U7DiS3usQsyerBU3jcclSzHOuAQrPo9fcY/F/JO0Yky194DEU63i7eRxc33czL7ZFiEtLqv6nzsJIHG5VfQmxh4vIgyT2QsVDSCLzk73U0ABBTYIGECcEgoooEC/AnEjdISPZwPxJu5ltnj3RtzvsEgAifswIkhs3OL34lG5cUN8fNmO92FE3dmjfeNJVvE3/vG42xvLas3fzRWZDyBxqVL87CyAxLtL4hKsuCwq/rsT/3/UiXpxY/f8FgEkbn7f7NKp2C8eORyXee3oEqy4DCzeCP8QIFZuNm5fV57+9ZTyZK75z+PJXfvMvQckQlScR+z7/SVUxVO+4qlgsfkiwmVmqz+jgAKTFzCATH4KCKCAAj0LxJf9eCJUvMxv42rAokOLy6riHoVFtnhaVTxSdrbFU7Dii3c8SSsCSNxPEl+w41KpuFQpHtUb7+aId4XM7t+I92vEE7viqVdxv0aEj7gJPd4kft+yyjE/lggDUXtHAemBwD3KMeKyr/hiH/dYbLbFSwDjxYRxs3sEhL8vY4x945KquDzse4C3zP1w1I4neX1DWRWJR/jGMWNFaH6Lm+djpShqb9zireiP3VD3eUAEktkN+Iv4u48CCigweQEDyOSngAAKKNCjwOFA3Oj9q+WxuMsOZfYI3Pjb+vhCvtkWKxrvLU+0ikfUzrZ450V8IY8VhVixiLesH1fuJ4l7N2K1Ib64x+/H5U2xxWN7L5u7xCn+PW74/pvyjo6Nx48XEEaYeRDwiU0GFzd7x/FjtSUCzSJbrKLEk6yeUR7n+zPAS4B4AeMlGwrEuOO+jbgPJnzipvq44XzjFitQ4dFli7fFP7zLD7ivAgooMHUBA8jUZ4Dnr4ACfQvESwDjUqF4Ad8qt/jCHisXcVN2PJkqc4tH8caKSKysxIrJxm22QvNg4KrMA8/Vmt0sHk+s2myLS93C+B+2OH48+SqM5m9C32q4sVLzvUAESTcFFFBAgQUFDCALQrmbAgoooMDoBeLlgxHUtgopo0fwBBVQQIFVCxhAVi1sfQUUUEABBRRQQAEFFPh3AQOIk0EBBRRQQAEFFFBAAQXWJmAAWRu1B1JAAQUUUEABBRRQQAEDiHNAAQUUUEABBRRQQAEF1iZgAFkbtQdSQAEFFFBAAQUUUEABA4hzQAEFFFBAAQUUUEABBdYmYABZG/VCB7o3EC8UizcQxwu53BRQQAEFFFBAAQWGJbAHsB9wOXDjsIbWxmgMIMPq01M3eYPvsEboaBRQQAEFFFBAAQVC4GnAG6ToLmAA6W62yp/4z8D7Lr74Yg45JF7a66aAAgoooIACCigwJIGrrrqKE088MYb0SOD9QxpbK2MxgAyrU4cD27Zt28bhh8e/uimggAIKKKCAAgoMSeDDH/4wRxxxRAwp/s+HhzS2VsZiABlWpwwgw+qHo1FAAQUUUEABBb5GwABSPyEMIPWGmRUMIJma1lJAAQUUUEABBZIFDCD1oAaQesPMCgaQTE1rKaCAAgoooIACyQIGkHpQA0i9YWYFA0imprUUUEABBRRQQIFkAQNIPagBpN4ws4IBJFPTWgoooIACCiigQLKAAaQe1ABSb5hZwQCSqWktBRRQQAEFFFAgWcAAUg9qAKk3zKxgAMnUtJYCCiiggAIKKJAsYACpBzWA1BtmVjCAZGpaSwEFFFBAAQUUSBYwgNSDGkDqDTMrGEAyNa2lgAIKKKCAAgokCxhA6kENIPWGmRUMIJma1lJAAQUUUEABBZIFDCD1oAaQesPMCgaQTE1rKaCAAgoooIACyQIGkHpQA0i9YWYFA0imprUUUEABBRRQQIFkAQNIPagBpN4ws4IBJFPTWgoooIACCiigQLKAAaQe1ABSb5hZwQCSqWktBRRQQAEFFFAgWcAAUg9qAKk3zKxgAMnUtJYCCiiggAIKKJAsYACpBzWA1BtmVjCAZGpaSwEFFFBAAQUUSBYwgNSDGkDqDTMrGEAyNa2lgAIKKKCAAgokCxhA6kENIPWGmRUMIJma1lJAAQUUUEABBZIFDCD1oAaQesPMCgaQTE1rKaCAAgoooIACyQIGkHpQA0i9YWYFA0imprUUaERgvxe/rZGRLjfMT77y8cv9oD+lgAIKDFDAAFLfFANIvWFmBQNIpqa1FGhEwADSSKMcpgIKKAAYQOqngQGk3jCzggEkU9NaCjQiYABppFEOUwEFFDCApMwBA0gKY1oRA0gapYUUaEfAANJOrxypAgoo4ApI/RwwgNQbZlYwgGRqWkuBRgQMII00ymEqoIACroCkzAEDSApjWhEDSBqlhRRoR8AA0k6vHKkCCijgCkj9HDCA1BtmVjCAZGpaS4FGBAwgjTTKYSqggAKugKTMAQNICmNaEQNIGqWFFGhHwADSTq8cqQIKKOAKSP0cMIDUG2ZWMIBkalpLgUYEDCCNNMphKqCAAq6ApMwBA0gKY1oRA0gapYUUaEfAANJOrxypAgoo4ApI/RwwgNQbZlYwgGRqWkuBRgQMII00ymEqoIACroCkzAEDSApjWhEDSBqlhRRoR8AA0k6vHKkCCijgCkj9HDCA1BtmVjCAZGpaS4FGBAwgjTTKYSqggAKugKTMAQNICmNaEQNIGqWFFGhHwADSTq8cqQIKKOAKSP0cMIDUG2ZWMIBkalpLgUYEDCCNNMphKqCAAq6ApMwBA0gKY1oRA0gapYUUaEfAANJOrxypAgoo4ApI/RwwgNQbZlYwgGRqWkuBRgQMII00ymEqoIACroCkzAEDSApjWhEDSBqlhRRoR8AA0k6vHKkCCijgCkj9HDCA1BtmVjCAZGpaS4FGBAwgjTTKYSqggAKugKTMgSkGkGcCF22i9yzgQ+WzA4ALgNOAO8u+hy75WZdGGUC6aLmvAiMRMICMpJGehgIKTELAFZD6Nk8xgOwG3H2Obk/gL4CjgXcAlwNnA+cAl5XQsTtw9RKfde2QAaSrmPsrMAIBA8gImugpKKDAZAQMIPWtnmIA2aj2EmA/4PeBC4F9gVuAw4BzSzA5YcnPunbIANJVzP0VGIGAAWQETfQUFFBgMgIGkPpWTz2A7AF8CjgSeEb553GFNWxuBPYCzlzys606tA8Qv+a3g4FLtm3bxuGHRxZxU0CBKQgYQKbQZc9RAQXGImAAqe/k1APIyUCsbhwPvAaIQHLKHOsNwEHAGUt+dtMWLTqrBJvtdjGA1E9sKyjQkoABpKVuOVYFFJi6gAGkfgZMPYD8ORBBIC6/ehWwK/DCOdZrgaOAU5f87LotWuQKSP38tYICoxAwgIyijZ6EAgpMRMAAUt/oKQeQeNJVBJBvBr4MnA7Ek65OmmO9GTgQiJWSZT6LFZQum/eAdNFyXwVGImAAGUkjPQ0FFJiEgAGkvs1TDiBx83ncc/H0wvgY4Dwggkls+wNXAvGUrGOW/Oz2ji0ygHQEc3cFxiBgABlDFz0HBRSYioABpL7TUw4g7wF+ozzdKiR3Aa4vKyHxnpDzgb2BJ1Z81rVDBpCuYu6vwAgEDCAjaKKnoIACkxEwgNS3eqoB5G5AXF4Vj9qN93vMtrgZ/VLgVuAO4NiyChKfL/tZly4ZQLpoua8CIxEwgIykkZ6GAgpMQsAAUt/mqQaQreRi1eMI4IryGN75fZf9bNFOGUAWlXI/BUYkYAAZUTM9FQUUGL2AAaS+xQaQesPMCgaQTE1rKdCIgAGkkUY5TAUUUAAwgNRPAwNIvWFmBQNIpqa1FGhEwADSSKMcpgIKKGAASZkDBpAUxrQiBpA0Sgsp0I6AAaSdXjlSBRRQwBWQ+jlgAKk3zKxgAMnUtJYCjQgYQBpplMNUQAEFXAFJmQMGkBTGtCIGkDRKCynQjoABpJ1eOVIFFFDAFZD6OWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQVcAUmZAwaQFMa0IgaQNEoLKdCOgAGknV45UgUUUMAVkPo5YACpN8ysYADJ1LSWAo0IGEAaaZTDVEABBVwBSZkDBpAUxrQiBpA0Sgsp0I6AAaSdXjlSBRRQwBWQ+jlgAKk3zKxgAMnUtJYCjQgYQBpplMNUQAEFXAFJmQMGkBTGtCIGkDRKCynQjoABpJ1eOVIFFFDAFZD6OWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQVcAUmZAwaQFMa0IgaQNEoLKdCOgAGknV45UgUUUMAVkPo5YACpN8ysYADJ1LSWAo0IGEAaaZTDVEABBVwBSZkDBpAUxrQiBpA0Sgsp0I6AAaSdXjlSBRRQwBWQ+jlgAKk3zKxgAMnUtJYCjQgYQBpplMNUQAEFXAFJmQMGkBTGtCIGkDRKCynQjoABpJ1eOVIFFFDAFZD6OWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQVcAUmZAwaQFMa0IgaQNEoLKdCOgAGknV45UgUUUMAVkPo5YACpN8ysYADJ1LSWAo0IGEAaaZTDVEABBVwBSZkDBpAUxrQiBpA0Sgsp0I6AAaSdXjlSBRRQwBWQ+jlgAKk3zKxgAMnUtJYCjQgYQBpplMNUQAEFXAFJmQMGkBTGtCIGkDRKCynQjoABpJ1eOVIFFFDAFZD6OWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQVcAUmZAwaQFMa0IgaQNEoLKdCOgAGknV45UgUUUMAVkPo5YACpN8ysYADJ1LSWAo0IGEAaaZTDVEABBVwBSZkDBpAUxrQiBpA0Sgsp0I6AAaSdXjlSBRRQwBWQ+jlgAKk3zKxgAMnUtJYCjQgYQBpplMNUQAEFXAFJmQMGkBTGtCIGkDRKCynQjoABpJ1eOVIFFFDAFZD6OWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQVcAUmZAwaQFMa0IgaQNEoLKdCOgAGknV45UgUUUMAVkPo5YACpN8ysYADJ1LSWAo0IGEAaaZTDVEABBVwBSZkDBpAUxrQiBpA0Sgsp0I6AAaSdXjlSBRRQwBWQ+jlgAKk3zKxgAMnUtJYCjQgYQBpplMNUQAEFXAFJmQMGkBTGtCIGkDRKCynQjoABpJ1eOVIFFFDAFZD6OWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQVcAUmZAwaQFMa0IgaQNEoLKdCOgAGknV45UgUUUMAVkPo5YACpN8ysYADJ1LSWAo0IGEAaaZTDVEABBVwBSZkDBpAUxrQiBpA0Sgsp0I6AAaSdXjlSBRRQwBWQ+jlgAKk3zKxgAMnUtJYCjQgYQBpplMNUQAEFXAFJmQMGkBTGtCIGkDRKCynQjoABpJ1eOVIFFFDAFZD6OWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQVcAUmZAwaQFMa0IgaQNEoLKdCOgAGknV45UgUUUMAVkPo5YACpN8ysYADJ1LSWAo0IGEAaaZTDVEABBVwBSZkDBpAUxrQiBpA0Sgsp0I6AAaSdXjlSBRRQwBWQ+jlgAKk3zKxgAMnUtJYCjQgYQBpplMNUQAEFXAFJmQMGkBTGtCIGkDRKCynQjoABpJ1eOVIFFFDAFZD6OWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQVcAUmZAwaQFMa0IgaQNEoLKdCOgAGknV45UgUUUMAVkPo5MPUA8irgwcATC+WhwEXAAcAFwGnAnZWfdemSAaSLlvsqMBIBA8hIGulpKKDAJAQMIPVtnnIAeSjwPuAw4BPA7sDVwOXA2cA5wGUlkCz7WdcOGUC6irm/AiMQMICMoImeggIKTEbAAFLf6qkGkLsA7wf+AHhpYTwBuBDYF7ilBJNzgaOBZT/r2iEDSFcx91dgBAIGkBE00VNQQIHJCBhA6ls91QDyvLLK8Xzg88A7gJ8EjgSOK6xhcyOwF3Dmkp9t1aF9gPg1vx0MXLJt2zYOPzyyiJsCCkxBwAAyhS57jgooMBYBA0h9J6cYQPYErgH+AXgj8B3APYD3AnsAp8yx3gAcBJyx5Gc3bdGis0qw2W4XA0j9xLaCAi0JGEBa6pZjVUCBqQsYQOpnwBQDyNOBXwXuX1Y/dgE+Cuxd7vd44RzrtcBRwKnArkDXz67bokWugNTPXysoMAoBA8go2uhJKKDARAQMIPWNnmIAeQnwXcBj5vh+G3gKcDFw0tzv3wwcCJwMxBOyun4WKyhdNu8B6aLlvgqMRMAAMpJGehoKKDAJAQNIfZunGEBiBeS5wCPm+K4ALgXinpB4BG9s+wNXAnHJ1jHAeUt8dnvHFhlAOoK5uwJjEDCAjKGLnoMCCkxFwABS3+kpBpB7l8fung68FfheIN4HEvd6bAPi9+NdIOeXy7LiHSFxmdb1S3zWtUMGkK5i7q/ACAQMICNooqeggAKTETCA1Ld6igEk1B4JvLo8avczwI8BbwGOLyshtwJ3AMeWVZD4mWU/69IlA0gXLfdVYCQCBpCRNNLTUECBSQgYQOrbPNUAspVc3Ix+BBCXZcVjeOe3ZT9btFMGkEWl3E+BEQkYQEbUTE9FAQVGL2AAqW+xAaTeMLOCASRT01oKNCJgAGmkUQ5TAQUUAAwg9dPAAFJvmFnBAJKpaS0FGhEwgDTSKIepgAIKGEBS5oABJIUxrYgBJI3SQgq0I2AAaadXjlQBBRRwBaR+DhhA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAFXQFLmgAEkhTGtiAEkjdJCCrQjYABpp1eOVAEFFHAFpH4OGEDqDTMrGEAyNa2lQCMCBpBGGuUwFVBAAVdAUuaAASSFMa2IASSN0kIKtCNgAGmnV45UAQUUcAWkfg4YQOoNMysYQDI1raVAIwIGkEYa5TAVUEABV0BS5oABJIUxrYgBJI3SQgq0I2AAaadXjlQBBRRwBaR+DhhA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAFXQFLmgAEkhTGtiAEkjdJCCrQjYABpp1eOVAEFFHAFpH4OGEDqDTMrGEAyNa2lQCMCBpBGGuUwFVBAAVdAUuaAASSFMa2IASSN0kIKtCNgAGmnV45UAQUUcAWkfg4YQOoNMysYQDI1raVAIwIGkEYa5TAVUEABV0BS5oABJIUxrYgBJI3SQgq0I2AAaadXjlQBBRRwBaR+DhhA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAFXQFLmgAEkhTGtiAEkjdJCCrQjYPyisksAACAASURBVABpp1eOVAEFFHAFpH4OGEDqDTMrGEAyNa2lQCMCBpBGGuUwFVBAAVdAUuaAASSFMa2IASSN0kIKtCNgAGmnV45UAQUUcAWkfg4YQOoNMysYQDI1raVAIwIGkEYa5TAVUEABV0BS5oABJIUxrYgBJI3SQgq0I2AAaadXjlQBBRRwBaR+DhhA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAFXQFLmgAEkhTGtiAEkjdJCCrQjYABpp1eOVAEFFHAFpH4OGEDqDTMrGEAyNa2lQCMCBpBGGuUwFVBAAVdAUuaAASSFMa2IASSN0kIKtCNgAGmnV45UAQUUcAWkfg4YQOoNMysYQDI1raVAIwIGkEYa5TAVUEABV0BS5oABJIUxrYgBJI3SQgq0I2AAaadXjlQBBRRwBaR+DhhA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAFXQFLmgAEkhTGtiAEkjdJCCrQjYABpp1eOVAEFFHAFpH4OGEDqDTMrGEAyNa2lQCMCBpBGGuUwFVBAAVdAUuaAASSFMa2IASSN0kIKtCNgAGmnV45UAQUUcAWkfg4YQOoNMysYQDI1raVAIwIGkEYa5TAVUEABV0BS5oABJIUxrYgBJI3SQgq0I2AAaadXjlQBBRRwBaR+DhhA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAFXQFLmgAEkhTGtiAEkjdJCCrQjYABpp1eOVAEFFHAFpH4OGEDqDTMrGEAyNa2lQCMCBpBGGuUwFVBAAVdAUuaAASSFMa2IASSN0kIKtCNgAGmnV45UAQUUcAWkfg4YQOoNMysYQDI1raVAIwIGkEYa5TAVUEABV0BS5oABJIUxrYgBJI3SQgq0I2AAaadXjlQBBRRwBaR+DhhA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAFXQFLmgAEkhTGtiAEkjdJCCrQjYABpp1eOVAEFFHAFpH4OGEDqDTMrGEAyNa2lQCMCBpBGGuUwFVBAAVdAUuaAASSFMa2IASSN0kIKtCNgAGmnV45UAQUUcAWkfg4YQOoNMysYQDI1raVAIwIGkEYa5TAVUEABV0BS5oABJIUxrYgBJI3SQgq0I2AAaadXjlQBBRRwBaR+DhhA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAFXQFLmgAEkhTGtiAEkjdJCCrQjYABpp1eOVAEFFHAFpH4OGEDqDTMrGEAyNa2lQCMCBpBGGuUwFVBAAVdAUubAVAPIOcDz5wT/DjgAOBS4qPz7BcBpwJ1lv2U/69IoA0gXLfdVYCQCBpCRNNLTUECBSQi4AlLf5qkGkPcDLwfin7HdDtwGXA1cDpwNREi5rASS3Zf8rGuHDCBdxdxfgREIGEBG0ERPQQEFJiNgAKlv9RQDyC7AjcD9gH+ZIzwBuBDYF7gFOAw4FzgaWPazrh0ygHQVc38FRiBgABlBEz0FBRSYjIABpL7VUwwg3w68B/hsCSHvBn4EeBZwJHBcYQ2bCCp7AWcu+dlWHdoHiF/z28HAJdu2bePwwyOLuCmgwBQEDCBT6LLnqIACYxEwgNR3cooB5GnAC8o9IJ8HXgvEqsjHgT2AU+ZYbwAOAs5Y8rObtmjRWSXYbLeLAaR+YltBgZYEDCAtdcuxKqDA1AUMIPUzYIoBZKPa/YFryj0fccP5C+d2uBY4CjgV2HWJz67bokWugNTPXysoMAoBA8go2uhJKKDARAQMIPWNNoD826rHrcCLy1OwTppjvRk4EDh5yc9iBaXL5j0gXbTcV4GRCBhARtJIT0MBBSYhYACpb/MUA0g84eovgDcUvkcD7wSeWFZB4nG8se0PXAnsCRwDnFcez9vls3i6VpfNANJFy30VGImAAWQkjfQ0FFBgEgIGkPo2TzGAnAj8LPDDwF2B15XH8cb/fz1wenn07vnA3iWYxD0iy3zWtUMGkK5i7q/ACAQMICNooqeggAKTETCA1Ld6igEk1F4BPLe8/+Ni4CXAF4DjgUvLJVl3AMeWVZD4mWU/69IlA0gXLfdVYCQCBpCRNNLTUECBSQgYQOrbPNUAspVcrHocAVxRHsM7v++yny3aKQPIolLup8CIBAwgI2qmp6KAAqMXMIDUt9gAUm+YWcEAkqlpLQUaETCANNIoh6mAAgoABpD6aWAAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQUMIClzwACSwphWxACSRmkhBdoRMIC00ytHqoACCrgCUj8HDCD1hpkVDCCZmtZSoBEBA0gjjXKYCiiggCsgKXPAAJLCmFbEAJJGaSEF2hEwgLTTK0eqgAIKuAJSPwcMIPWGmRUMIJma1lKgEQEDSCONcpgKKKCAKyApc2BsASTO584UmX6KGED6cfeoCvQqYADpld+DK6CAAp0EXAHpxLXpzi0FkD2APwMO2+K0X1/eYv6ueppeKhhAemH3oAr0K2AA6dffoyuggAJdBAwgXbQ237elABJnEG8n/xfgeuCvgfcDvw98BDgVOAN4KPAP9TS9VDCA9MLuQRXoV8AA0q+/R1dAAQW6CBhAumiNI4BcAxwC3A/YHzgaeCrwz8C9gScAH6tn6a2CAaQ3eg+sQH8CBpD+7D2yAgoo0FXAANJVbPv9W1gB+TXg88A7gV8HHlhOYz/gBOD5wKeBbwYeCdxUz9JbBQNIb/QeWIH+BAwg/dl7ZAUUUKCrgAGkq1ibAeTpwMOB7wQOAD5QVj/iMqsIJb8JXF2CyPcAj6ln6a2CAaQ3eg+sQH8CBpD+7D2yAgoo0FXAANJVrM0Acp+yAvJg4HHA7sBzgPcCEU6+MrciclFZJbm4nqaXCgaQXtg9qAL9ChhA+vX36AoooEAXAQNIF63N923hEqwzge8H3g7cE/iNcsP5K8sKyG8BFwL/H3BiuUn9L+tpeqlgAOmF3YMq0K+AAaRff4+ugAIKdBEwgHTRajeAxMjjfo//Anw7EI/j/Tngb4DrgLOBnyoBJVZEWt4MIC13z7ErsKSAAWRJOH9MAQUU6EHAAFKP3sIKSNxU/iVgt/IrHsN7W7nn47XlpvQ3lBvQnwH8ST1LbxUMIL3Re2AF+hMwgPRn75EVUECBrgIGkK5i2+/fQgD5lvKEq4OBC4C3lEuwzi83pv888OTyKy7HilWSVjcDSKudc9wKVAgYQCrw/FEFFFBgzQIGkHrwFgLI8cA5QDyO90DgrsAngJeVFxKeBzwAiNWPWAm5tISUep31VzCArN/cIyrQu4ABpPcWOAAFFFBgYQEDyMJUO9yxhQDyCOBz5RG8jwJeDvwpcBzw0bI6MjvBJwG7lBvS63XWX8EAsn5zj6hA7wIGkN5b4AAUUECBhQUMIAtTNR1ANhv815W3n9cLDKuCAWRY/XA0CqxFwACyFmYPooACCqQIGEDqGVtYAdnsLH+6PAnrjnqCQVUwgAyqHQ5GgfUIGEDW4+xRFFBAgQwBA0i9YksB5CHlkqs460+WR/PWCwyrggFkWP1wNAqsRcAAshZmD6KAAgqkCBhA6hlbCiDx+N17AV8uN6E/HHhzeSRvSNwFuBtwZD1LbxUMIL3Re2AF+hMwgPRn75EVUECBrgIGkK5i2+/fUgCJ94FEAIktnoL1YOCEEkjiPM4Fngu8sZ6ltwoGkN7oPbAC/QkYQPqz98gKKKBAVwEDSFextgPIPwJ7zQWQB244nWuA/etJeq1gAOmV34Mr0I+AAaQfd4+qgAIKLCNgAFlG7Wt/pqUVkJ0FkFgV2RhK6oXWW8EAsl5vj6bAIAQMIINog4NQQAEFFhIwgCzEtOVOLQSQeLv5bcBTgP9TziYuvYr7P2ZbnEe8A+Se9SS9VjCA9MrvwRXoR8AA0o+7R1VAAQWWETCALKP2tT/TQgA5BfhieRv688vN5q8AXrwhgPw88E31JL1WMID0yu/BFehHwADSj7tHVUABBZYRMIAso9ZeAJmNeONN6N4DUt9/KyigwAAEDCADaIJDUEABBRYUMIAsCLXFbi2sgMyG7z0g9f22ggIKDFDAADLApjgkBRRQYAcCBpD6qdFSAPmnuXs84obzg4GLgC8BcR7/zXtA6ieEFRRQYP0CBpD1m3tEBRRQYFkBA8iycl/9uVYCSIwz3n7+IOAr5T0gBwE/MfciwrsDL6sn6bWC94D0yu/BFehHwADSj7tHVUABBZYRMIAso/a1P9NKANl4ptcCDwDuqCcYVAUDyKDa4WAUWI+AAWQ9zh5FAQUUyBAwgNQrthpAHgF8oP70B1fBADK4ljggBVYvYABZvbFHUEABBbIEDCD1kq0GkM3O/F7AAcAH61l6q2AA6Y3eAyvQn4ABpD97j6yAAgp0FTCAdBXbfv8WAshdgHcBx25xus8EXl32ixcWtroZQFrtnONWoELAAFKB548qoIACaxYwgNSDtxBA4iy/ANyjnG68DT1eTHgnsDvwo8AfAmcBv1tP0msFA0iv/B5cgX4EDCD9uHtUBRRQYBkBA8gyal/7M60EkPmXEN4CnFHeiH4m8HX1DIOpYAAZTCsciALrEzCArM/aIymggAK1AgaQWsF/e39GC9v8Swjnw8gNwH2AfcoKSHz24hZOaAdjNIA03DyHrsCyAgaQZeX8OQUUUGD9AgaQevMWA8h8GPlceTdIvCPkHcAvAH9Vz9JbBQNIb/QeWIH+BAwg/dl7ZAUUUKCrgAGkq9j2+7cQQF4B/Bhwdnnj+SnAXuVUIoB8U1kFidWQ1jcDSOsddPwKLCFgAFkCzR9RQAEFehIwgNTDtxBALi03nc/O9kmbBJA4jycAXy4rIfUy/VQwgPTj7lEV6FXAANIrvwdXQAEFOgkYQDpxbbpzCwFk48Dn7wGJFZC9gY8BewCvBV5Xz9JbBQNIb/QeWIH+BAwg/dl7ZAUUUKCrgAGkq9j2+7cYQG4GHlKegrUN+EbgEODq8mjeepX+KhhA+rP3yAr0JmAA6Y3eAyuggAKdBQwgncm2+4EWA8gdJWjE2ONdIPcDHgtcbACpnxBWUECB9QsYQNZv7hEVUECBZQUMIMvKffXnWgwge5Z7PSKIxIsIvw34HeBW4GeB36hn6a2CKyC90XtgBfoTMID0Z++RFVBAga4CBpCuYtvv32IA2eysdwGeXh7J+1P1LL1VMID0Ru+BFehPwADSn71HVkABBboKGEC6io03gNRLDKOCAWQYfXAUCqxVwACyVm4PpoACClQJGECq+P71h1tZAdm1rG58qTySN/4Zv+Kyq7gUayybAWQsnfQ8FOggYADpgOWuCiigQM8CBpD6BrQSQB4EfBz4NLAbEIEkHrsbv24H/hm4BoiXFHZ9E3q8Qf23yr0jhwIXAQcAFwCnzd3YvuxnXbpkAOmi5b4KjETAADKSRnoaCigwCQEDSH2bWwogby2P293srL8BOAb44fJCwkVlnlaenvUsIF54GI/yvby8df0c4LISSOJm92U+W3Qcs/0MIF3F3F+BEQgYQEbQRE9BAQUmI2AAqW/10ANI3Fwel1jtD7wFePDcKd8V+H4g3gvy++X3Y5XkWxdk2Qu4svz8K8s/LwT2BW4BDgPOBY4GTgCW+WzBofz7bgaQrmLur8AIBAwgI2iip6CAApMRMIDUt3roASSebPUzwO8C3z0XQGKl46XAF4BXlVWK0Hg2cP6C94XEpVZfBO4G/AnwAOBI4LjCGjY3AhFUzlzys606tA8Qv+a3g4FLtm3bxuGHRxZxU0CBKQgYQKbQZc9RAQXGImAAqe/k0APIN5cViCeUVYj3l8uj7lNWKd62JMGjgd8sqyWvKwEk3q4e95TEfSSz7QbgIOCMJT+7aYvxnVWCzXa7GECW7Ko/pkCjAgaQRhvnsBVQYJICBpD6tg89gMyfYYSDHyxf2k8G3rXk6UedjwAvACLAxIsLYwXkkHJz+wvn6l4LHAWcuuRn120xRldAlmygP6bA2AQMIGPrqOejgAJjFjCA1He3pQASQSFuBo8AEY/f3bjF/SJ/BPzaTlh+DtgPiBvQY5sFkFhtiSddnTT383F/yYFABJ5lPosVlC6b94B00XJfBUYiYAAZSSM9DQUUmISAAaS+zS0FkPgy/6K5U44bxGeXS8V5/E/gO4E/2wlLPK43LuH6Stnv7uXfP1ke8RuP4I0tbnyPm9T3LE/YOq88nrfLZ/GI4C6bAaSLlvsqMBIBA8hIGulpKKDAJAQMIPVtHnoAiadPxaVSbyqXXt1r7pTj/oqt/v8d6cRTrmK1ZLa9GriirIRE4Di93NQeN7PvDTyx7H/9Ep917ZABpKuY+yswAgEDyAia6CkooMBkBAwg9a0eegCJS6WOLU+/+lEgVitm2z+WJ1Tt6P9fVGd2CVb88/jyPpDZG9bj2BFKYlv2s0XHEfsZQLpoua8CIxEwgIykkZ6GAgpMQsAAUt/moQeQ+TP8PPBPc78Rj8391Ib/P97VEY/ordli1eOIsioSj+Gd35b9bNHxGEAWlXI/BUYkYAAZUTM9FQUUGL2AAaS+xS0FkHjBYLyUMH5ttsVlVbFy8Yl6lt4qGEB6o/fACvQnYADpz94jK6CAAl0FDCBdxbbfv6UAUn+2w69gABl+jxyhAukCBpB0UgsqoIACKxMwgNTTthhAdgXeDDy+/vQHV8EAMriWOCAFVi9gAFm9sUdQQAEFsgQMIPWSrQSQuwIvAV5WTjkus3pgeS/Il8rvRTCJX7fUs/RWwQDSG70HVqA/AQNIf/YeWQEFFOgqYADpKrb9/q0EkBh5vL8j3s0R2yyAxP0gd244rXg07/zN6vVK66tgAFmftUdSYDACBpDBtMKBKKCAAjsVMIDslGinO7QeQOIpVQ8H4jy2lcfYRjjZGEp2CjGQHQwgA2mEw1BgnQIGkHVqeywFFFCgTsAAUucXP91SAJmtesyvgMTb0eOt5rFtfC9Ivc76KxhA1m/uERXoXcAA0nsLHIACCiiwsIABZGGqHe7YQgD5EeArwCvLm8hjzGeUe0AMIPVzwAoKKNCzgAGk5wZ4eAUUUKCDgAGkA9YOdm0hgLwPuA04qrwcME4lXkIYN6EbQOrngBUUUKBnAQNIzw3w8AoooEAHAQNIB6yGA8hs6JvdhG4AqZ8DVlBAgZ4FDCA9N8DDK6CAAh0EDCAdsBoOIPEI3tvnnnwVpzK7H8QAUj8HrKCAAj0LGEB6boCHV0ABBToIGEA6YDUcQF4APAr4tnLZ1XwAicftnlJupv8V4DnA6+tZeqvgTei90XtgBfoTMID0Z++RFVBAga4CBpCuYtvv38I9IHsDzy/h4o3lRvQPlTAS/4wXEcb7QO4C7AIcWc/SWwUDSG/0HliB/gQMIP3Ze2QFFFCgq4ABpKtYmwFkNup9gV8Angf8VbkRvV5gWBUMIMPqh6NRYC0CBpC1MHsQBRRQIEXAAFLP2MIKyMaz3A14BfDj9ac/uAoGkMG1xAEpsHoBA8jqjT2CAgookCVgAKmXbDGA1J/1cCsYQIbbG0emwMoEDCAro7WwAgookC5gAKknbSWA7Am8DTim/pQHXcEAMuj2ODgFViNgAFmNq1UVUECBVQgYQOpVWwkg9wBuBPaoP+VBVzCADLo9Dk6B1QgYQFbjalUFFFBgFQIGkHrVVgLI7uWt5/esP+VBVzCADLo9Dk6B1QgYQFbjalUFFFBgFQIGkHpVA0i9YWYFA0imprUUaETAANJIoxymAgooABhA6qdBKwEkLr36HOAKSH3PraCAAgMTMIAMrCEORwEFFNhCwABSPz2GHkDi3R8/U07zyQaQ+oZbQQEFhidgABleTxyRAgoosCMBA0j93Bh6AHkK8APAhcAbDCD1DbeCAgoMT8AAMryeOCIFFFDAALK6OTD0ADI7c29CX90csLICCvQsYADpuQEeXgEFFOgg4ApIB6wd7GoAqTfMrOBN6Jma1lKgEQEDSCONcpgKKKCAN6GnzIHWAshewG7AXYA7gVuBO1IkhlHEADKMPjgKBdYqYABZK7cHU0ABBaoEXAGp4vvXH24pgETYiNCxcfss8BHgjcDrSyipl+mnggGkH3ePqkCvAgaQXvk9uAIKKNBJwADSiWvTnVsJIDH4A4Fb5gJGrILsCewHPBp4HnAb8CjgE/U0vVQwgPTC7kEV6FfAANKvv0dXQAEFuggYQLpobb5vSwFkZ2f79cCpwMt2tuOAPzeADLg5Dk2BVQkYQFYla10FFFAgX8AAUm/aYgDZBfhK/akPsoIBZJBtcVAKrFbAALJaX6sroIACmQIGkHrNFgPIm4F3AefUn/7gKhhABtcSB6TA6gUMIKs39ggKKKBAloABpF6ytQByAHA18AzgkvrTH1wFA8jgWuKAFFi9gAFk9cYeQQEFFMgSMIDUS7YWQP4AuAfwyPpTH2QFA8gg2+KgFFitgAFktb5WV0ABBTIFDCD1mi0FkNOA04FvBz5df+qDrGAAGWRbHJQCqxUwgKzW1+oKKKBApoABpF6zlQASj9h9OfDdwIfqT3uwFQwgg22NA1NgdQIGkNXZWlkBBRTIFjCA1IsOPYA8FngycEz5593Luz82e/t5PB1rV+Bt9Sy9VTCA9EbvgRXoT8AA0p+9R1ZAAQW6ChhAuoptv//QA8jby6rH9wFvKr8OBr644VTiPHYrASReWNjqZgBptXOOW4EKAQNIBZ4/qoACCqxZwABSDz70ABJvO38W8Jry2N2X1p/yoCsYQAbdHgenwGoEDCCrcbWqAgoosAoBA0i96tADyOwMY1UjVkDeCzy3/rQHW8EAMtjWODAFVidgAFmdrZUVUECBbAEDSL1oKwEkzvQ+wAeAXwdeUX/qg6xgABlkWxyUAqsVMICs1tfqCiigQKaAAaRes6UAEmf7MOBPge8A/qz+9AdXwQAyuJY4IAVWL2AAWb2xR1BAAQWyBAwg9ZKtBZA44/OAuCTr0fWnP7gKBpDBtcQBKbB6AQPI6o09ggIKKJAlYACpl2wxgDygPPHqb4E76wkGVcEAMqh2OBgF1iNgAFmPs0dRQAEFMgQMIPWKLQSQeLzui4Gf3XC6NwH3Kr8X7wB5CvCGepJeKxhAeuX34Ar0I2AA6cfdoyqggALLCBhAllH72p9pJYB8AtgXOB74R+Bm4A+B2TtB4sWE15R96lX6q2AA6c/eIyvQm4ABpDd6D6yAAgp0FjCAdCbb7gdaCCAx6AggDwT+H/Bu4J7AUSV0xNvR9wDuDXwUeDbw5/U0vVQwgPTC7kEV6FfAANKvv0dXQAEFuggYQLpobb5vawEkVjn2L6cy/+/xW58E/hfwcCDenN7iZgBpsWuOWYFKAQNIJaA/roACCqxRwABSj91qAPnGssoRqyL7AJ8F/g54crks6+IFaL4B+I/AXwNxP8kQNgPIELrgGBRYs4ABZM3gHk4BBRSoEDCAVOCVH20xgPwEcDYQN54/orwd/QLgv5fLtBZRiaByPnBt+ZlnAr8DHApcBBwARM3T5p60texni4xnto8BpIuW+yowEgEDyEga6WkooMAkBAwg9W1uLYBcV1Yrng68qdyQfgVwSlkBmV2etZXM1wPxCN/vBD4CRPg4q6yGXA1cXgLOOcBlJZDsDizzWdcOGUC6irm/AiMQMICMoImeggIKTEbAAFLf6hYCyF1KYIjLreK+j/hnvP8j/v3byo3pITG7UX1nKt9S3qR+SdnxocD7gJOAC8uTtG4BDgPOBY4GTljys52NZePnBpCuYu6vwAgEDCAjaKKnoIACkxEwgNS3eugB5CHA/wZi1SKCx43Aq8tjeGPV4qnl32eP5Y19umy7ljer37WsoBwJHFcKhE0cby/gTGCZz7YaS9y7Er/mt3is8CXbtm3j8MMji7gpoMAUBAwgU+iy56iAAmMRMIDUd3LoASQesXsq8BLgVWXVI24evw/wQuD3yssIvwm4raxWLKoSKxzvKj93CPDT5XG+cTnXbLsBOAg4Y8nPtrq5PQJUBJvtNgPIoi10PwXGIWAAGUcfPQsFFJiGgAGkvs9DDyCzMzyw3I/xIeCHyiVYnwbuX0EQ5x7LDK8FPldWQGJFJILNbIub1ON9IxGClvks7lnZ0eYKSEXz/FEFxiRgABlTNz0XBRQYu4ABpL7DrQSQONO4DCtuEI+A8H7g7cDj6gn+9b0i8QjfnyxPwYp7QWZbXNoV4efkJT+LFZQum/eAdNFyXwVGImAAGUkjPQ0FFJiEgAGkvs0tBZA423ga1ZcqT/sY4AlAPM43tvsBsZryPcAvlkfwxu9HMLkS2BOInzlvic9u7zhWA0hHMHdXYAwCBpAxdNFzUECBqQgYQOo73VoAqT/jf7vxOx6pGwEkVlFeDsQ9JE8ErgdOL4/ejfeE7F1+P945ssxnXcdrAOkq5v4KjEDAADKCJnoKCigwGQEDSH2rpxhAQu2xwC8B8UjeuKzreUBcLnU8cClwK3AHcGxZBYmfWfazLl0ygHTRcl8FRiJgABlJIz0NBRSYhIABpL7NUw0gW8nFqscRQLzgMB7DO78t+9minTKALCrlfgqMSMAAMqJmeioKKDB6AQNIfYsNIPWGmRUMIJma1lKgEQEDSCONcpgKKKAAYACpnwYGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAAJIyBwwgKYxpRQwgaZQWUqAdAQNIO71ypAoooIArIPVzwABSb5hZwQCSqWktBRoRMIA00iiHqYACCrgCkjIHDCApjGlFDCBplBZSoB0BA0g7vXKkCiiggCsg9XPAAFJvmFnBAJKpaS0FGhEwgDTSKIepgAIKuAKSMgcMICmMaUUMIGmUFlKgHQEDSDu9cqQKKKCAKyD1c8AAUm+YWcEAkqlpLQUaETCANNIoh6mAAgq4ApIyBwwgKYxpRQwgaZQWUqAdAQNIO71ypAoooIArIPVzwABSb5hZwQCSqWktBRoRMIA00iiHqYACCrgCkjIHDCApjGlFDCBplBZSoB0BA0g7vXKkCiiggCsg9XPAAFJvmFnBwNh9uwAAIABJREFUAJKpaS0FGhEwgDTSKIepgAIKuAKSMgcMICmMaUUMIGmUFlKgHQEDSDu9cqQKKKCAKyD1c8AAUm+YWcEAkqlpLQUaETCANNIoh6mAAgq4ApIyBwwgKYxpRQwgaZQWUqAdAQNIO71ypAoooIArIPVzwABSb5hZwQCSqWktBRoRMIA00iiHqYACCrgCkjIHDCApjGlFDCBplBZSoB0BA0g7vXKkCiiggCsg9XPAAFJvmFnBAJKpaS0FGhEwgDTSKIepgAIKuAKSMgcMICmMaUUMIGmUFlKgHQEDSDu9cqQKKKCAKyD1c8AAUm+YWcEAkqlpLQUaETCANNIoh6mAAgq4ApIyBwwgKYxpRQwgaZQWUqAdAQNIO71ypAoooIArIPVzwABSb5hZwQCSqWktBRoRMIA00iiHqYACCrgCkjIHDCApjGlFDCBplBZSoB0BA0g7vXKkCiiggCsg9XPAAFJvmFnBAJKpaS0FGhEwgDTSKIepgAIKuAKSMgcMICmMaUUMIGmUFlKgHQEDSDu9cqQKKKCAKyD1c8AAUm+YWcEAkqlpLQUaETCANNIoh6mAAgq4ApIyBwwgKYxpRQwgaZQWUqAdAQNIO71ypAoooIArIPVzwABSb5hZwQCSqWktBRoRMIA00iiHqYACCrgCkjIHDCApjGlFDCBplBZSoB0BA0g7vXKkCiiggCsg9XPAAFJvmFnBAJKpaS0FGhEwgDTSKIepgAIKuAKSMgcMICmMaUUMIGmUFlKgHQEDSDu9cqQKKKCAKyD1c8AAUm+YWcEAkqlpLQUaETCANNIoh6mAAgq4ApIyBwwgKYxpRQwgaZQWUqAdAQNIO71ypAoooIArIPVzwABSb5hZwQCSqWktBRoRMIA00iiHqYACCrgCkjIHDCApjGlFDCBplBZSoB0BA0g7vXKkCiiggCsg9XPAAFJvmFnBAJKpaS0FGhEwgDTSKIepgAIKuAKSMgcMICmMaUUMIGmUFlKgHQEDSDu9cqQKKKCAKyD1c8AAUm+YWcEAkqlpLQUaETCANNIoh6mAAgq4ApIyBwwgKYxpRQwgaZQWUqAdAQNIO71ypAoooIArIPVzwABSb5hZwQCSqWktBRoRMIA00iiHqYACCrgCkjIHDCApjGlFDCBplBZSoB0BA0g7vXKkCiiggCsg9XPAAFJvmFnBAJKpaS0FGhEwgDTSKIepgAIKuAKSMgcMICmMaUUMIGmUFlKgHQEDSDu9cqQKKKCAKyD1c8AAUm+YWcEAkqlpLQUaETCANNIoh6mAAgq4ApIyBwwgKYxpRQwgaZQWUqAdAQNIO71ypAoooIArIPVzwABSb5hZwQCSqWktBRoRMIA00iiHqYACCrgCkjIHDCApjGlFDCBplBZSoB0BA0g7vXKkCiiggCsg9XPAAFJvmFnBAJKpaS0FGhEwgDTSKIepgAIKuAKSMgcMICmMaUUMIGmUFlKgHQEDSDu9cqQKKKCAKyD1c2CqAeRJwGuB+wMfA34QuAo4FLgIOAC4ADgNuLMwL/tZly4ZQLpoua8CIxEwgIykkZ6GAgpMQsAAUt/mKQaQBwEfBJ4DvBt4HXA/4DHA1cDlwNnAOcBlJZDsvuRnXTtkAOkq5v4KjEDAADKCJnoKCigwGQEDSH2rpxhAngDcFziv8D0aeBvwVOBCYF/gFuAw4FzgaOCEJT/r2iEDSFcx91dgBAIGkBE00VNQQIHJCBhA6ls9xQCyUS1WQp4LvBE4Ejiu7BA2NwJ7AWcu+dlWHdoHiF/z28HAJdu2bePwwyOLuCmgwBQEDCBT6LLnqIACYxEwgNR3cuoBZDfg48Avlvs+9gBOmWO9ATgIOANY5rObtmjRWSXYbLeLAaR+YltBgZYEDCAtdcuxKqDA1AUMIPUzYOoB5BXA44CHAS8HdgVeOMd6LXAUcOqSn123RYtcAamfv1ZQYBQCBpBRtNGTUECBiQgYQOobPeUAEjedv7kEjCuB08tTsE6aY70ZOBA4ecnPYgWly+Y9IF203FeBkQgYQEbSSE9DAQUmIWAAqW/zVAPI/sAVwIuA1xfGCCRxY3o8gje22CeCyZ7AMUt+dnvHFhlAOoK5uwJjEDCAjKGLnoMCCkxFwABS3+kpBpC7AR8C3rfhcqsvAXHJVKyExLtAzgf2Bp4I7AJcv8RnXTtkAOkq5v4KjEDAADKCJnoKCigwGQEDSH2rpxhA4iWEcenVxi1WPB4KXArcCtwBHFtWQWLf45f8rEuXDCBdtNxXgZEIGEBG0khPQwEFJiFgAKlv8xQDyM7UYtXjiHKJVjyGd35b9rOdHXP2uQFkUSn3U2BEAgaQETXTU1FAgdELGEDqW2wAqTfMrGAAydS0lgKNCBhAGmmUw1RAAQUAA0j9NDCA1BtmVjCAZGpaS4FGBAwgjTTKYSqggAIGkJQ5YABJYUwrYgBJo7SQAu0IGEDa6ZUjVUABBVwBqZ8DBpB6w8wKBpBMTWsp0IiAAaSRRjlMBRRQwBWQlDlgAElhTCtiAEmjtJAC7QgYQNrplSNVQAEFXAGpnwMGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAFZCUOWAASWFMK2IASaO0kALtCBhA2umVI1VAAQVcAamfAwaQesPMCgaQTE1rKdCIgAGkkUY5TAUUUMAVkJQ5YABJYUwrYgBJo7SQAu0IGEDa6ZUjVUABBVwBqZ8DBpB6w8wKBpBMTWsp0IiAAaSRRjlMBRRQwBWQlDlgAElhTCtiAEmjtJAC7QgYQNrplSNVQAEFXAGpnwMGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAFZCUOWAASWFMK2IASaO0kALtCBhA2umVI1VAAQVcAamfAwaQesPMCgaQTE1rKdCIgAGkkUY5TAUUUMAVkJQ5YABJYUwrYgBJo7SQAu0IGEDa6ZUjVUABBVwBqZ8DBpB6w8wKBpBMTWsp0IiAAaSRRjlMBRRQwBWQlDlgAElhTCtiAEmjtJAC7QgYQNrplSNVQAEFXAGpnwMGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAFZCUOWAASWFMK2IASaO0kALtCBhA2umVI1VAAQVcAamfAwaQesPMCgaQTE1rKdCIgAGkkUY5TAUUUMAVkJQ5YABJYUwrYgBJo7SQAu0IGEDa6ZUjVUABBVwBqZ8DBpB6w8wKBpBMTWsp0IiAAaSRRjlMBRRQwBWQlDlgAElhTCtiAEmjtJAC7QgYQNrplSNVQAEFXAGpnwMGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAFZCUOWAASWFMK2IASaO0kALtCBhA2umVI1VAAQVcAamfAwaQesPMCgaQTE1rKdCIgAGkkUY5TAUUUMAVkJQ5YABJYUwrYgBJo7SQAu0IGEDa6ZUjVUABBVwBqZ8DBpB6w8wKBpBMTWsp0IiAAaSRRjlMBRRQwBWQlDlgAElhTCtiAEmjtJAC7QgYQNrplSNVQAEFXAGpnwMGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAFZCUOWAASWFMK2IASaO0kALtCBhA2umVI1VAAQVcAamfAwaQesPMCgaQTE1rKdCIgAGkkUY5TAUUUMAVkJQ5YABJYUwrYgBJo7SQAu0IGEDa6ZUjVUABBVwBqZ8DBpB6w8wKBpBMTWsp0IiAAaSRRjlMBRRQwBWQlDlgAElhTCtiAEmjtJAC7QgYQNrplSNVQAEFXAGpnwMGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAFZCUOWAASWFMK2IASaO0kALtCBhA2umVI1VAAQVcAamfAwaQesPMCgaQTE1rKdCIgAGkkUY5TAUUUMAVkJQ5YABJYUwrYgBJo7SQAu0IGEDa6ZUjVUABBVwBqZ8DBpB6w8wKBpBMTWsp0IiAAaSRRjlMBRRQwBWQlDlgAElhTCtiAEmjtJAC7QgYQNrplSNVQAEFXAGpnwMGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAFZCUOWAASWFMK2IASaO0kALtCBhA2umVI1VAAQVcAamfAwaQesPMCgaQTE1rKdCIgAGkkUY5TAUUUMAVkJQ5YABJYUwrYgBJo7SQAu0IGEDa6ZUjVUABBVwBqZ8DBpB6w8wKBpBMTWsp0IiAAaSRRjlMBRRQwBWQlDlgAElhTCtiAEmjtJAC7QgYQNrplSNVQAEFXAGpnwMGkHrDzAoGkExNaynQiIABpJFGOUwFFFDAFZCUOTDlAPKNwAeBRwOfLJqHAhcBBwAXAKcBd1Z+1qVRBpAuWu6rwEgEDCAjaaSnoYACkxBwBaS+zVMNIBE+3gocCexfAsjuwNXA5cDZwDnAZSWQLPtZ1w4ZQLqKub8CIxAwgIygiZ6CAgpMRsAAUt/qqQaQdwK/B/zyXAA5AbgQ2Be4BTgMOBc4Glj2s64dMoB0FXN/BUYgYAAZQRM9BQUUmIyAAaS+1VMNILHqcU25vGq2AnJmWRE5rrCGzY3AXsCyn23VoX2A+DW/HQxcsm3bNg4/PLKImwIKTEHAADKFLnuOCigwFgEDSH0npxpAZnJxf8csgLwG2AM4ZY71BuAg4IwlP7tpixadVYLNdrsYQOonthUUaEnAANJStxyrAgpMXcAAUj8DDCBfDSCvAnYFXjjHei1wFHDqkp9dt0WLXAGpn79WUGAUAgaQUbTRk1BAgYkIGEDqG20A+WoAOR2Ip2CdNMd6M3AgcPKSn8UKSpfNe0C6aLmvAiMRMICMpJGehgIKTELAAFLfZgPIVwPIY4DzyiN4QzYuzboS2BM4ZsnPbu/YIgNIRzB3V2AMAgaQMXTRc1BAgakIGEDqO20A+WoA2QW4HoiVkHgXyPnA3sATgWU/69ohA0hXMfdXYAQCBpARNNFTUECByQgYQOpbbQD5agAJzeOBS4FbgTuAY8sqSM1nXbpkAOmi5b4KjETAADKSRnoaCigwCQEDSH2bpx5ANhOMVY8jgCvKY3jn91n2s0U7ZQBZVMr9FBiRgAFkRM30VBRQYPQCBpD6FhtA6g0zKxhAMjWtpUAjAgaQRhrlMBVQQAHAAFI/DQwg9YaZFQwgmZrWUqARAQNII41ymAoooIABJGUOGEBSGNOKGEDSKC2kQDsCBpB2euVIFVBAAVdA6ueAAaTeMLOCASRT01oKNCJgAGmkUQ5TAQUUcAUkZQ4YQFIY04oYQNIoLaRAOwIGkHZ65UgVUEABV0Dq54ABpN4ws4IBJFPTWgo0ImAAaaRRDlMBBRRwBSRlDhhAUhjTihhA0igtpEA7AgaQdnrlSBVQQAFXQOrngAGk3jCzggEkU9NaCjQiYABppFEOUwEFFHAFJGUOGEBSGNOKGEDSKC2kQDsCBpB2euVIFVBAAVdA6ueAAaTeMLOCASRT01oKNCJgAGmkUQ5TAQUUcAUkZQ4YQFIY04oYQNIoLaRAOwIGkHZ65UgVUEABV0Dq54ABpN4ws4IBJFPTWgo0ImAAaaRRDlMBBRRwBSRlDhhAUhjTihhA0igtpEA7AgaQdno11ZE6R6faec97MwFXQOrnhQGk3jCzggEkU9NaCjQi4Je7Rho14WE6RyfcfE99OwEDSP2kMIDUG2ZWMIBkalpLgUYE/HLXSKMmPEzn6ISb76kbQFYwBwwgK0CtKGkAqcDzRxVoVcAvd612bjrjdo5Op9ee6c4FXAHZudHO9jCA7ExovZ+vPYCM/T8q0b5PvvLx6+2iR1Ogo8DY/xz6Z7DjhBjg7s7RATbFIfUmYACppzeA1BtmVjCAZGqWWn75WQGqJVMF/HKXymmxFQg4R1eAaslmBQwg9a0zgNQbZlYwgGRqGkBWoGnJVQj45W4VqtbMFHCOZmpaq3UBA0h9Bw0g9YaZFQwgmZoGkBVoWnIVAn65W4WqNTMFnKOZmtZqXcAAUt9BA0i9YWYFA0impgFkBZqWXIWAX+5WoWrNTAHnaKamtVoXMIDUd9AAUm+YWcEAkqlpAFmBpiVXIeCXu1WoWjNTwDmaqWmt1gUMIPUdNIDUG2ZWMIBkahpAVqBpyVUI+OVuFarWzBRwjmZqWqt1AQNIfQcNIPWGmRUMIJmaBpAVaFpyFQJ+uVuFqjUzBZyjmZrWal3AAFLfQQNIvWFmBQNIpqYBZAWallyFgF/uVqFqzUwB52imprVaFzCA1HfQAFJvmFnBAJKpaQBZgaYlVyHgl7tVqFozU8A5mqlprdYFDCD1HTSA1BtmVjCAZGoaQFagaclVCPjlbhWq1swUcI5malqrdQEDSH0HDSD1hpkVDCCZmgaQFWhachUCfrlbhao1MwWco5ma1mpdwABS30EDSL1hZgUDSKamAWQFmpZchYBf7lahas1MAedopqa1WhcwgNR30ABSb5hZwQCSqWkAWYGmJVch4Je7VahaM1PAOZqpaa3WBQwg9R00gNQbZlYwgGRqGkBWoGnJVQj45W4VqtbMFHCOZmpaq3UBA0h9Bw0g9YaZFQwgmZoGkBVoWnIVAn65W4WqNTMFnKOZmtZqXcAAUt9BA0i9YWYFA0impgFkBZqWXIWAX+5WoWrNTAHnaKamtVoXMIDUd9AAUm+YWcEAkqlpAFmBpiVXIeCXu1WoWjNTwDmaqWmt1gUMIPUdNIDUG2ZWMIBkahpAVqBpyVUI+OVuFarWzBRwjmZqWqt1AQNIfQcNIPWGmRUMIJmaBpAVaFpyFQJ+uVuFqjUzBZyjmZrWal3AAFLfQQNIvWFmBQNIpqYBZAWallyFgF/uVqFqzUwB52imprVaFzCA1HfQAFJvmFnBAJKpaQBZgaYlVyHgl7tVqFozU8A5mqlprdYFDCD1HTSA1BtmVjCAZGoaQFaguf6SY//is37R9R/xk698/PoP6hFTBcb+59A5mjpdRl/MAFLfYgNIvWFmBQNIpqYBZAWa6y859i8+6xdd/xH9crd+8+wj+ucwW3S99fwzmOttAKn3NIDUG2ZWMIBkahpAVqC5/pJ+8Vm/efYR/fKTLbr+ev45XL955hH9M5ipCQaQek8DSL1hZgUDSKbmRGqN/T8sfvFpfyKPfY6236Gdn4F/DnduNOQ9/DOY2x0DSL2nAaTeMLOCASRT01oKKDAIAb/8DKINVYMwgFTx9f7D/hnMbYEBpN7TAFJvmFnBAJKpaS0FFBiEwNi//PjlfBDTzEFsITD2P4Prbr4BpF7cAFJvmFnBAJKpaS0FFBiEwNi//BhABjHNHIQBZG1zwABST20AqTfMrGAAydS0lgIKDELAADKINjgIBUYtsM7/nTGA1E8lA0i9YWYFA0imprUUUGAQAuv8YtDHCbsC0oe6x1TgawXW+b8zBpD62WcAqTfMrGAAydS0lgIKKKCAAgpMQsAA0labDSDD6pcBZFj9cDQKKKCAAgoo0ICAAaSBJs0N0QCS169DgYuAA4ALgNOAOzuWN4B0BHN3BRRQQAEFFFDAANLWHDCA5PRrd+Bq4HLgbOAc4LISSLocwQDSRct9FVBAAQUUUEABwADS1jQwgOT06wTgQmBf4BbgMOBc4OiO5Q0gHcHcXQEFFFBAAQUUMIC0NQcMIDn9OhM4EjiulAvXG4G9tii/DxC/5reHxqrJxRdfzCGHHJIzsp1Uefw5713LcTyIAgoooIACCiiwKoG3nfqoVZXeru5VV13FiSeeGL//SOD9azvwiA5kAMlp5muAPYBT5srdABwE3LSDQ5wFRHBxU0ABBRRQQAEFFGhP4GnAG9obdv8jNoDk9OBVwK7AC+fKXQscBVy3g0NstgLy9UAsffwl8MWcoW1Z5WDgEiD+AMU9LG7jF7Dn4+/xZmdo36fXd3tuz6cnsL4zjr903q/c+xtXvLh1FDCAdATbwe6nA/EUrJPmPr8ZOBCIlZChbv96zwlwBPDhoQ7ScaUK2PNUzmaK2fdmWpU2UHueRtlMIXveTKscqAEkZw48BjivPII3Ku4PXAnsCdyec4iVVPF/rFbCOuii9nzQ7VnZ4Oz7ymgHW9ieD7Y1KxuYPV8ZrYWzBQwgOaK7ANcDsRIS7wI5H9gbeGJO+ZVV8X+sVkY72ML2fLCtWenA7PtKeQdZ3J4Psi0rHZQ9XymvxTMFDCB5mscDlwK3AncAx5ZVkLwj5Ffyf6zyTYde0Z4PvUOrGZ99X43rkKva8yF3ZzVjs+ercbXqCgQMILmoseoR91NcUR7Dm1s9v1rcCP9s4NeAz+SXt+IABez5AJuyhiHZ9zUgD+wQ9nxgDVnDcOz5GpA9RI6AASTH0SoKKKCAAgoooIACCiiwgIABZAEkd1FAAQUUUEABBRRQQIEcAQNIjqNVFFBAAQUUUEABBRRQYAEBA8gCSO6igAIKKKCAAgoooIACOQIGkBxHqyiggAIKKKCAAgoooMACAgaQBZDcRYGRCDwJeC1wf+BjwA8CV43k3DyNnQu8A/gt4Dd2vqt7jETgVcCDG3gn1Ui4ez2NHwLOBO4N/DlwMvCJXkfkwRXYQsAAMt3pcWh5aeIBwAXAacCd0+UY/Zk/CPgg8Bzg3cDrgPsBjxz9mXuCIfA04GLgWQaQyUyIhwLvAw7zi+joex7/+/7HwAnA50sQORD4jtGfuSfYrIABpNnWVQ18d+Bq4HLgbOAc4LISSKoK+8ODFXgCcF/gvDLCRwNvA+4+2BE7sCyBvcpLUW8GXmkAyWIddJ27AO8H/gB46aBH6uAyBL4PeEr5FfXiL5Z+p/xvfkZ9ayiQLmAASSdtomD8LcmFwL7ALeVvyM4Fjm5i9A4yQyBWQp5bep9RzxrDFbgI+CJwN+BPDCDDbVTiyJ5X/nLp+eVvxOPyu9sS61tqWAJxmd17gO8CrgH+F/AV4BnDGqajUeCrAgaQac6GuE70SOC4cvoxD24E4m9K3cYvsBvwceAXgV8Z/+lO+gxjpes3gW8tl90ZQMY/HfYsX0L/AXhjuQznHsAxwK3jP/3JnuGvAs8uZx8hJP4bf8NkNTzxwQsYQAbfopUM8DXAHsApc9Xjf6gOAm5ayREtOiSBVwCPAx4GfHlIA3MsqQLxZ/wjwAvK5XZx87kBJJV4kMWeDsSX0XjYRNwPsAvw0fIAitklmIMcuINaWuDhwJuA7y2XV8c9nd8NxO97b+fSrP7gKgUMIKvUHW7teDLKrsAL54Z4LXAUcN1wh+3IEgQeA7y59PrKhHqWGK7AzwH7lRvQY5QGkOH2KnNkLymX4sSf9dn228CnysNGMo9lrWEIxNMN7wB+vAxndlVDzIG/HMYQHYUCXytgAJnmjDgdiKdgnTR3+nGDajw1wyXb8c6J/YErgBcBrx/vaXpmRSAuw7hPuRY8fiseOBDXhUcQiXsE3MYpECsgcX/XI+ZOL/7cXwr88jhPefJnFX2NS6hn/02/J/C5cjP6tsnrCDBIAQPIINuy8kHF34rEUnw8gje2+GIafxse1w7fvvKje4A+BOIG5A+Vx3LOr3x9wSX6PtqxlmPGQybi8pvZ9uoSQCOAxKU5buMUiPdAxPsf4i+a3louy4lV77jENla63cYnEE/Binu9zgA+C8Q7QR5Y/lLRy2zH1+9RnJEBZBRt7HwS8aXk+vIfqHhCzvnA3r6sqrNjSz8QLyGMS682bhE+P9nSiTjWpQW8BGtpuuZ+MB7DGoEz3gHyGeDHgLc0dxYOeFGB+C4X4SOCxz7lRbP/HfiLRQu4nwLrFjCArFt8OMc7vizJx1NR4trRY8sqyHBG6EgUUEABBRRQQAEFRidgABldSzudUKx6HFEuy4jH8LopoIACCiiggAIKKLBSAQPISnktroACCiiggAIKKKCAAvMCBhDngwIKKKCAAgoooIACCqxNwACyNmoPpIACCiiggAIKKKCAAgYQ54ACCiiggAIKKKCAAgqsTcAAsjZqD6SAAgoooIACCiiggAIGEOeAAgoooIACCiiggAIKrE3AALI2ag+kgAIKjFrgYcDdgPeM+iw9OQUUUECBagEDSDWhBRRQQAEFgP8B/ChwCPCVLUR+ErgQ+Czw88ADgR9QUAEFFFBgOgIGkOn02jNVQAEFMgV2B+b/G/INwOuBHwE+Uw50V2BX4Oby/z8W+HXgUOCfgLOAA4AT5wYW+385c6DWUkABBRQYloABZFj9cDQKKKBAKwLXA/cCbt9iwBFAIqjsBtwJfAC479wKSYSWXYDPz9W4GvivrSA4TgUUUECB7gIGkO5m/oQCCiigAPwtcApw+YIYzwZ+vFyiNQstm62ALFjO3RRQQAEFWhUwgLTaOcetgAIK9CvwUSDu53jrAsOIVY64tOo+wKfn9t9RAIkVk9sWqOsuCiiggAINChhAGmyaQ1ZAAQUGIBAB5KeA31tgLN8H/NJObk6fLxNh5SDgCwvUdhcFFFBAgcYEDCCNNczhKqCAAgMR+DjwYuAtAxmPw1BAAQUUaETAANJIoxymAgooMDCBTwEnA3/UYVyvLo/q/Zcd/Ezc1P4E4O0darqrAgoooEBjAgaQxhrmcBVQQIEBCMQ9GhEiHgrEU6sW3eK9H/EUrGfu4Af+HjgJ+ONFC7qfAgoooEB7AgaQ9nrmiBVQQIG+BR5T7v2IFYsu7+x4ObCvAaTv9nl8BRRQoF8BA0i//h5dAQUUaFHgncCNwPd3HHysgLwIuGUHP3dP4HEdHu3b8fDuroACCigwBAEDyBC64BgUUECBdgSOAd4FHAl8qOOwzy6P4t3qEqznLPho346HdncFFFBAgaEIGECG0gnHoYACCrQjEI/I/esVDDdWQGJ15CsrqG1JBRRQQIGBCBhABtIIh6GAAgoooIACCiigwBQEDCBT6LLnqIACCiiggAIKKKDAQAQMIANphMNQQAEFFFBAAQUUUGAKAgaQKXTZc1RAAQUUUEABBRRQYCACBpCBNMJhKKCAAgoooIACCigwBQEDyBS67DkqoIACCiiggAIKKDAQAQPIQBrhMBRQQAEFFFBAAQUUmIKAAWQKXfYcFVCr7PFTAAABKUlEQVRAAQUUUEABBRQYiIABZCCNcBgKKKCAAgoooIACCkxBwAAyhS57jgoooIACCiiggAIKDETAADKQRjgMBRRQQAEFFFBAAQWmIGAAmUKXPUcFFFBAAQUUUEABBQYiYAAZSCMchgIKKKCAAgoooIACUxAwgEyhy56jAgoooIACCiiggAIDETCADKQRDkMBBRRQQAEFFFBAgSkIGECm0GXPUQEFFFBAAQUUUECBgQgYQAbSCIehgAIKKKCAAgoooMAUBAwgU+iy56iAAgoooIACCiigwEAEDCADaYTDUEABBRRQQAEFFFBgCgIGkCl02XNUQAEFFFBAAQUUUGAgAgaQgTTCYSiggAIKKKCAAgooMAUBA8gUuuw5KqCAAgoooIACCigwEIH/Hzx37O7C6JJuAAAAAElFTkSuQmCC" width="640">


### 从图中可以看到，广告5被点击数明显比较多。总奖励也提高了一倍，证明我们的算法很NB

## 五、项目地址

### https://coding.net/u/RuoYun/p/Python-of-machine-learning/git/tree/master/00%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/5.%E8%BF%9B%E9%98%B6%E7%AE%97%E6%B3%95/1.%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/1.%E7%BD%AE%E4%BF%A1%E5%8C%BA%E9%97%B4%E4%B8%8A%E7%95%8C?public=true
