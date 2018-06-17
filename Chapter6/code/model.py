#%%
import pandas as pd
import numpy as np

import os
outputfile = os.getcwd() + "\\Chapter6\\output"
from scipy.interpolate import lagrange
inputData = "Chapter6/data/model.xls"
data = pd.read_excel(inputData)
print(data.head())


#%%
from sklearn.model_selection import train_test_split
train_col = data.columns[:-1]
test_col = data.columns[-1]

xtrain,xtest,ytrain,ytest = train_test_split(data[train_col], data[test_col], test_size = 0.3)
xtrain.head(),xtest.head(),ytrain.head(),ytest.head()

#%%
from keras.models import Sequential
from keras.layers import Dense, Activation

net = Sequential()
net.add(Dense(units=10, activation = 'relu', input_dim=3))
net.add(Dense(units=20, activation = 'relu', input_dim=3))
net.add(Dense(units=1, activation='sigmoid'))
net.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
net.fit(xtrain, ytrain,batch_size=10, epochs=1000 )
#%%
netfile = outputfile + "\\net.model"
net.save_weights(netfile)
#%%
import matplotlib.pyplot as plt
import numpy as np
import itertools
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#%%
from sklearn.metrics import confusion_matrix
y_predict = net.predict_classes(xtest).reshape(len(xtest))
y_predict
#%%
cm_test = confusion_matrix(ytest, y_predict)
plot_confusion_matrix(cm_test,ytest.unique(), normalize=False)


#%%
y_train_predict = net.predict_classes(xtrain).reshape(len(xtrain))
cm_test = confusion_matrix(ytrain, y_train_predict)
plot_confusion_matrix(cm_test,ytrain.unique(), normalize=False)


#%%
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(xtrain, ytrain)
ytrain_predict = tree.predict(xtrain)
ytest_predict = tree.predict(xtest)

from sklearn.externals import joblib
treefile = outputfile + "\\tree.model"
joblib.dump(tree, treefile)
#%%
cm_train_tree = confusion_matrix(ytrain, ytrain_predict)
cm_test_tree = confusion_matrix(ytest, ytest_predict)
plot_confusion_matrix(cm_train_tree, classes=ytrain.unique())
#%%
plot_confusion_matrix(cm_test_tree, classes=ytest.unique())

#%%
from sklearn.metrics import roc_curve

def plot_roc(ytest, y_predict, label):
    fpr, tpr, threshlod = roc_curve(ytest, y_predict, pos_label = 1 )
    # plt.figure(figsize = (9,6))
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.xlim([0,1.05])
    plt.ylim(0, 1.05)
    plt.legend(loc = 4)
    plt.show()
#%%
ytest_predict = tree.predict_proba(xtest)[:,1]

plot_roc(ytest, ytest_predict, "ROC curve of Decision Tree")

tree.score(xtest, ytest)

#%%

y_predict = net.predict(xtest).reshape(len(xtest))
plot_roc(ytest, y_predict, "ROC curve of Net")


#%%
tree.predict_proba(xtest)
ytest_predict