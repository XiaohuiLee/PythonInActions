Python数据挖掘与实战-Chapter7
<!-- TOC -->

- [选择列](#选择列)
    - [1、直接挑选列](#1直接挑选列)
    - [2、根据列名切片选择列](#2根据列名切片选择列)
- [聚类](#聚类)
- [根据索引排序pd.Series](#根据索引排序pdseries)

<!-- /TOC -->
## 选择列
### 1、直接挑选列
```python
choosed_cols = ["count", "min", "max"]
describe_ = describe_[choosed_cols]
```
### 2、根据列名切片选择列
```python
data = df[df.columns[-5:]]
```
## 聚类
导入库包
```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=k, n_jobs = 3)
```
参数解释：

`n_clusters`: 聚类数目,

`n_jobs`: 并行线程数


拟合以及查看标签和聚类中心
```python
km.fit(data)
km.labels_
km.cluster_centers_
```
## 根据索引排序pd.Series
```python
counts = se.value_counts().sort_index()
```