# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 00:05:13 2019

@author: Jose
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from prettytable import PrettyTable
import seaborn as sns


data = pd.read_csv('C:/Users/Jose/OneDrive/Documents/Fall 2019 Courses/CSC450 Senior Research/SpotifyFeat.csv')
data.head()
data.info()

le = preprocessing.LabelEncoder()
data['track_name'] = le.fit_transform(data['track_name'].astype('str'))
data['artist_name'] = le.fit_transform(data['artist_name'].astype('str'))


data.info()


drop_list = ['artist_name']
train = data.drop(drop_list, axis=1)

train.info()

Y = copy.deepcopy(train.Hit)
Y.shape

train1 = train.drop('Hit', axis = 1)


x_train, x_test, y_train, y_test = train_test_split(train1, Y, test_size=0.33, random_state=7)
model = XGBClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = model.score(x_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




from sklearn.model_selection import cross_val_score

def testingModel(model, x_train, y_train):
    scores = cross_val_score(model, x_train,y_train, cv=10)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Std. Dev.", scores.std())
    return scores.mean()


#decision tree
decision_tree = DecisionTreeClassifier(criterion='gini')
decision_tree.fit(train1, Y)
acc_decision_tree = testingModel(decision_tree, train1, Y)


decision_tree1 = DecisionTreeClassifier(criterion='entropy')
decision_tree1.fit(train1,Y)
acc_decision_tree1 = testingModel(decision_tree1, train1,Y)


resultsDT = pd.DataFrame({'Criteria': ['Gini', 'Entropy'], 'Accuracy': [acc_decision_tree, acc_decision_tree1]})
resultsDT.head()
#DTplot = resultsDT.plot(x='Criteria', y='Accuracy', kind = 'bar', title='Accuracy with different criterion', ylim=(.75,.85))

sns.set()
sns.barplot(x='Criteria', y='Accuracy', data=resultsDT).set_title('Accuracy of DT')



tableDT = PrettyTable(['Criteria', 'Accuracy'])
tableDT.add_row(['Gini', round(acc_decision_tree,3)])
tableDT.add_row(['Entropy', round(acc_decision_tree1,3)])
print(tableDT)


#----------------------------DONE with LR-----------------------------
#Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(train1, Y)
acc_log = testingModel(logreg,train1,Y)
#------------------------------^DONE^---------------------------
#-------------------------------DONE with KNN-------------------------
#k-nearest neighbors
#K=3
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train1, Y)
acc_knn = testingModel(knn, train1, Y)

y_predictKNN = knn.predict(x_test)
accuracy_score(y_test,y_predictKNN)

# k = 5
knn5 = KNeighborsClassifier(n_neighbors = 5)
knn5.fit(train1,Y)
acc_knn5 = testingModel(knn5, train1, Y)

#k = 7
knn7 = KNeighborsClassifier(n_neighbors = 7)
knn7.fit(train1,Y)
acc_knn7 = testingModel(knn7, train1, Y)

#k = 9
knn9 = KNeighborsClassifier(n_neighbors = 9)
knn9.fit(train1,Y)
acc_knn9 = testingModel(knn9, train1, Y)

knnResults = pd.DataFrame({'K': [3,5,7,9], 'Accuracy': [acc_knn, acc_knn5, acc_knn7, acc_knn9]})
knnResults.head()

#knn results comparison
#knnResults.plot(x='K', y='Accuracy', kind = 'line', title='Accuracy of different K values in KNN')
sns.set()
sns.lineplot(x='K', y='Accuracy', data=knnResults).set_title('Accuracy of KNN')

from prettytable import PrettyTable
tableKNN = PrettyTable(['K', 'Accuracy'])
tableKNN.add_row(['3', round(acc_knn,3)])
tableKNN.add_row(['5', round(acc_knn5,3)])
tableKNN.add_row(['7', round(acc_knn7,3)])
tableKNN.add_row(['9', round(acc_knn9,3)])
print(tableKNN)

#------------------------^DONE^---------------------------------------

results = pd.DataFrame({'Model': ['Decision Tree Classifier', 'Logistic Regression', 'K-Nearest Neighbors'], 'Accuracy': [acc_decision_tree1, acc_log, acc_knn5]})
results.head()
sns.barplot(x='Model', y='Accuracy', data=results).set_title('Accuracy of Algorithms')


resultSum = PrettyTable(['Model', 'Accuracy'])
resultSum.add_row(['Decision Tree Classifier', round(acc_decision_tree1,3)])
resultSum.add_row(['Logistic Regression', round(acc_log,3)])
resultSum.add_row(['K-Nearest Neighbors', round(acc_knn5,3)])
print(resultSum)



