Python数据挖掘与实战
```python
```

# 第六章
## 拉格朗日插值
```python
from scipy.interpolate import lagrange
lagrange(series.index, list(series))(index)
```
## 保存文件
```python
import os
outputfile = os.getcwd() + "\\Chapter6\\output"
model.save(outputfile)
```
## train, test的split
```python
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data[train_col], data[test_col], test_size = 0.3)
```
## keras建立神经网络
```python
from keras.models import Sequential
from keras.layers import Dense, Activation

net = Sequential()
net.add(Dense(units=10, activation = 'relu', input_dim=3))
net.add(Dense(units=1, activation='sigmoid'))
net.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 拟合
net.fit(xtrain, ytrain,batch_size=10, epochs=1000 )
# 预测
# 1、预测类别
y_predict = net.predict_classes(xtest).reshape(len(xtest))
# 2、预测类别概率
y_predict = net.predict(xtest).reshape(len(xtest))
```
## Confusion matrix
```python
from sklearn.metrics import confusion_matrix
```
## 决策树模型
```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(xtrain, ytrain)
# 预测
# 1、预测类别
ytrain_predict = tree.predict(xtrain)
# 2、预测类别概率
ytest_predict = tree.predict_proba(xtest)[:,1]
# 3、评分
tree.score(xtest, ytest)
```

# 第六章
# 第六章
# 第六章
# 第六章
# 第六章
# 第六章
# 第六章
# 第六章