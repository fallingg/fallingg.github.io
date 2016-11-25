---
title:  "Python으로 3D Stacked Barplot 그리기"
date:   2016-11-25
category: Under Pressure
tags: [python, sns, barplot, matplotlib]
---

엑셀은 정말 짱이다. 피벗 테이블 하나 잘 만들어 놓으면 마우스 클릭 몇 번으로 그래프가 시시각각 변하게 만들 수 있고, 나름 보기 정갈하게(예쁜 수준까진 아니지만) 만드는 데에도 큰 시간이나 노력이 들지 않는다. 그런데 최근 3차원 누적 막대그래프를 그릴 일이 생겨서 이것저것 변형해보니, 3차원 막대 그래프는 지원하는데 '누적' 막대 그래프는 지원하지 않는 것 같더라. 그래서 예전 기억을 더듬어가며, 구글링해가며 파이썬으로 그래프를 그려봤다. 같은 값이면 다홍치마라고, 하는김에 색깔도 디자인도 좀 예쁘게 만들자 하는 생각으로 seaborn까지 활용하였다.
{: .text-justify}

<br>

## Libraries


```python
import seaborn as sns
# for styling (based on matplotlib)
```

```python
sns.set_style("whitegrid")
```


```python
import pandas as pd
import numpy as np
# for dataframe & array
```


```python
import matplotlib.pyplot as plt
# for visualization
```


```python
from mpl_toolkits.mplot3d import Axes3D
# for 3D barplot
```

<br>

## Boxplot


```python
box = pd.read_csv("boxplot.csv")
```

예제 데이터는 그냥 정말 아무거나 가져왔다. 뭐든 그려지긴 하겠지. 대략적인 생김새는 아래와 같다.
{: .text-justify}


```python
box[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = sns.boxplot(x = "category", y = "value", data = box)
```


```python
plt.show()

#for save :
#plt.savefig("name.png")
```


![png](/images/2016-11-25/output_10_0.png){: .align-center}

그려보니 아웃라이어가 넘나 심한 것. 그래도 이제와 예제를 바꾸긴 귀찮으니까 그냥 이런 식으로 그리면 된다는 걸 알아두자.
{: .text-justify}

<br>

## Regression Plot


```python
reg1 = pd.read_csv("reg1.csv")  # category : 2-4
```


```python
reg2 = pd.read_csv("reg2.csv") # category : 5-9
```

이 역시도 왠지 그려질 것 같은 데이터를 가져왔다.
{: .text-justify}


```python
reg1[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>category</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006</td>
      <td>2</td>
      <td>38.888889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007</td>
      <td>2</td>
      <td>27.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008</td>
      <td>2</td>
      <td>28.571429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009</td>
      <td>2</td>
      <td>48.484848</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010</td>
      <td>2</td>
      <td>47.368421</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011</td>
      <td>2</td>
      <td>45.833333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2012</td>
      <td>2</td>
      <td>36.666667</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2013</td>
      <td>2</td>
      <td>34.615385</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2014</td>
      <td>2</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015</td>
      <td>2</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
g1 = sns.lmplot(x = "time", y = "value", hue = "category", col = "category", col_wrap = 3, data = reg1)
plt.show()
```


![png](/images/2016-11-25/output_15_0.png){: .align-center}



```python
g2 = sns.lmplot(x = "time", y = "value", hue = "category", col = "category", col_wrap = 4, data = reg2)
plt.show()
```


![png](/images/2016-11-25/output_16_0.png){: .align-center}

꽤나 잘 그려진 것 같다~~(만족)~~
{: .text-justify}

<br>

## 3D Stacked Barplot


```python
data_2006 = pd.read_csv("2006.csv")
data_2006.index = data_2006.pop("index")
data_2007 = pd.read_csv("2007.csv")
data_2007.index = data_2007.pop("index")
data_2008 = pd.read_csv("2008.csv")
data_2008.index = data_2008.pop("index")
data_2009 = pd.read_csv("2009.csv")
data_2009.index = data_2009.pop("index")
data_2010 = pd.read_csv("2010.csv")
data_2010.index = data_2010.pop("index")
data_2011 = pd.read_csv("2011.csv")
data_2011.index = data_2011.pop("index")
data_2012 = pd.read_csv("2012.csv")
data_2012.index = data_2012.pop("index")
data_2013 = pd.read_csv("2013.csv")
data_2013.index = data_2013.pop("index")
data_2014 = pd.read_csv("2014.csv")
data_2014.index = data_2014.pop("index")
data_2015 = pd.read_csv("2015.csv")
data_2015.index = data_2015.pop("index")
```

이건 좀 노가다. 한창 머리 안굴러 갈 새벽시간에 잠 꺨려고 직접 타이핑해보았다. `numpy`를 활용해서 랜덤 변수들을 만들 수도 있었지만, 이건 실제 누적치가 어떻게 반영되는가 궁금해서 무려 10년에 걸친 사짜 데이터를 만들어서 넣어봤다.
{: .text-justify}


```python
data_2006
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>4</td>
      <td>4</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>14</td>
      <td>10</td>
      <td>21</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>106</td>
      <td>52</td>
      <td>10</td>
      <td>64</td>
      <td>20</td>
    </tr>
    <tr>
      <th>7</th>
      <td>66</td>
      <td>41</td>
      <td>21</td>
      <td>63</td>
      <td>32</td>
    </tr>
    <tr>
      <th>8</th>
      <td>17</td>
      <td>15</td>
      <td>9</td>
      <td>13</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
a = {}
b = {}
c = {}
d = {}
e = {}
for i in range(2, 9):
    a[i] = [data_2006['a'][i], data_2007['a'][i], data_2008['a'][i], data_2009['a'][i], data_2010['a'][i], data_2011['a'][i], data_2012['a'][i], data_2013['a'][i], data_2014['a'][i], data_2015['a'][i]]
    b[i] = [data_2006['b'][i], data_2007['b'][i], data_2008['b'][i], data_2009['b'][i], data_2010['b'][i], data_2011['b'][i], data_2012['b'][i], data_2013['b'][i], data_2014['b'][i], data_2015['b'][i]]
    c[i] = [data_2006['c'][i], data_2007['c'][i], data_2008['c'][i], data_2009['c'][i], data_2010['c'][i], data_2011['c'][i], data_2012['c'][i], data_2013['c'][i], data_2014['c'][i], data_2015['c'][i]]
    d[i] = [data_2006['d'][i], data_2007['d'][i], data_2008['d'][i], data_2009['d'][i], data_2010['d'][i], data_2011['d'][i], data_2012['d'][i], data_2013['d'][i], data_2014['d'][i], data_2015['d'][i]]
    e[i] = [data_2006['e'][i], data_2007['e'][i], data_2008['e'][i], data_2009['e'][i], data_2010['e'][i], data_2011['e'][i], data_2012['e'][i], data_2013['e'][i], data_2014['e'][i], data_2015['e'][i]]
```


```python
co = sns.color_palette("muted", n_colors=5) # color select at seaborn
```


```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Z : category (7)
category = [2, 3, 4, 5, 6, 7, 8]


for i, z in zip(category, category):
    
    # X : year (10)
    xs = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'] 

    # y : 5
    ys = np.array(a[i])
    ys2 = np.array(b[i])
    ys3 = np.array(c[i])
    ys4 = np.array(d[i])
    ys5 = np.array(e[i])

    ax.bar(xs, ys, zs=z, zdir='y', color=co[0], alpha=0.8)
    ax.bar(xs, ys2, bottom=ys, zs=z, zdir='y', color=co[1], alpha=0.8)
    ax.bar(xs, ys3, bottom=ys+ys2, zs=z, zdir='y', color=co[2], alpha=0.8)
    ax.bar(xs, ys4, bottom=ys+ys2+ys3, zs=z, zdir='y', color=co[3], alpha=0.8)
    ax.bar(xs, ys5, bottom=ys+ys2+ys3+ys4, zs=z, zdir='y', color=co[4], alpha=0.8)


ax.set_xlabel('year')
ax.set_ylabel('category')
ax.set_zlabel('occurance')


# Angle
ax.view_init(elev=50., azim=-10)
plt.show()


# plt.savefig("overall.png")
```


![png](/images/2016-11-25/output_22_0.png){: .align-center}

원래 목표였던 3D 누적 막대그래프. 가독성(가시성?)이 그렇게 좋지는 못하다. 그래서 각 계열별로 다시 그려봤다.
{: .text-justify}

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Z : category (7)
category = [2, 3, 4, 5, 6, 7, 8]

for i, z in zip(category, category):
    # X : year (10)
    xs = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'] 

    # y : 1 (a)
    ys = np.array(a[i])

    ax.bar(xs, ys, zs=z, zdir='y', color=co[0], alpha=0.8)


ax.set_xlabel('year')
ax.set_ylabel('category')
ax.set_zlabel('occurance')

# Angle
ax.view_init(elev=50., azim=-10)
plt.show()


# plt.savefig("a.png")
```


![png](/images/2016-11-25/output_23_0.png){: .align-center}



```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Z : category (7)
category = [2, 3, 4, 5, 6, 7, 8]

for i, z in zip(category, category):
    # X : year (10)
    xs = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'] 

    # y : 1 (b)
    ys2 = np.array(b[i])

    ax.bar(xs, ys2, zs=z, zdir='y', color=co[1], alpha=0.8)


ax.set_xlabel('year')
ax.set_ylabel('category')
ax.set_zlabel('occurance')

# Angle
ax.view_init(elev=50., azim=-10)
plt.show()


# plt.savefig("b.png")
```


![png](/images/2016-11-25/output_24_0.png){: .align-center}



```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Z : category (7)
category = [2, 3, 4, 5, 6, 7, 8]

for i, z in zip(category, category):
    # X : year (10)
    xs = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'] 

    # y : 1 (c)
    ys3 = np.array(c[i])

    ax.bar(xs, ys3, zs=z, zdir='y', color=co[2], alpha=0.8)

ax.set_xlabel('year')
ax.set_ylabel('category')
ax.set_zlabel('occurance')


# Angle
ax.view_init(elev=50., azim=-10)
plt.show()


# plt.savefig("c.png")
```


![png](/images/2016-11-25/output_25_0.png){: .align-center}



```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Z : category (7)
category = [2, 3, 4, 5, 6, 7, 8]

for i, z in zip(category, category):
    # X : category (10)
    xs = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'] 

    # y : 1 (d)
    ys4 = np.array(d[i])

    ax.bar(xs, ys4, zs=z, zdir='y', color=co[3], alpha=0.8)

ax.set_xlabel('year')
ax.set_ylabel('category')
ax.set_zlabel('occurance')


# Angle
ax.view_init(elev=50., azim=-10)
plt.show()


# plt.savefig("d.png")
```


![png](/images/2016-11-25/output_26_0.png){: .align-center}



```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Z : category (7)
category = [2, 3, 4, 5, 6, 7, 8]

for i, z in zip(category, category):
    # X : year (10)
    xs = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'] 

    # y : 1 (e)
    ys5 = np.array(e[i])

    ax.bar(xs, ys5, zs=z, zdir='y', color=co[4], alpha=0.8)


ax.set_xlabel('year')
ax.set_ylabel('category')
ax.set_zlabel('occurance')


# Angle
ax.view_init(elev=50., azim=-10)
plt.show()


# plt.savefig("e.png")
```


![png](/images/2016-11-25/output_27_0.png){: .align-center}

코딩을 자주 하는 습관을 들여야겠다. 개발자도 아니고 이걸로 먹고 사는 사람도 아니지만, 이렇게 하나씩 정리해두면 언젠가는 도움이 될 날이 올 것 같다. 
{: .text-justify}

## ㅤㅤ