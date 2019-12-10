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
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image



#dfMain = pd.read_csv('C:/Users/Jose/OneDrive/Documents/Fall 2019 Courses/CSC450 Senior Research/SpotifyFeat.csv') #Small dataset
dfMain = pd.read_csv('C:/Users/Jose/OneDrive/Documents/Fall 2019 Courses/CSC450 Senior Research/SpotifyFeaturesFinal.csv')


dfMain.head()
dfMain.info()

dfOnlyFeatures = dfMain[['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'Hit']] #keep only attributes (floats and ints)
dfOnlyFeatures = dfOnlyFeatures.dropna() #drop any null values
X = dfOnlyFeatures.drop('Hit', axis=1) #drop the Hit column
y = dfOnlyFeatures['Hit'] #only the Hit Column

X.head() #show piece of X without hit column


#X_train will be 75% of data, X_test will be 25%. Same with y train and y test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size=.25)


from sklearn import tree

#---------------------------DT WITH GINI INDEX AS CRITERIA---------------------------------

#Gini index (for impurity) criteria, chooses best split at each node
modelDT = tree.DecisionTreeClassifier(criterion = "gini", splitter = "best")
#fit model with training data
modelDT.fit(X_train, y_train)
#feature importance method
importanceDT = modelDT.feature_importances_
#Feature importance dataFrame for graph for DT
import seaborn as sns
featImportanceDT = pd.DataFrame({'Audio Feature': ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'], 'Importance': [importanceDT[0], importanceDT[1], importanceDT[2], importanceDT[3],
                                                   importanceDT[4],importanceDT[5],importanceDT[6],importanceDT[7],importanceDT[8],importanceDT[9]]})
#featImportanceDT.head()
#sns.set()
from prettytable import PrettyTable
featImpDTTable = PrettyTable(['Audio Feature', 'Importance'])
featImpDTTable.add_row(['acousticness', round(importanceDT[0],3)])
featImpDTTable.add_row(['danceability', round(importanceDT[1],3)])
featImpDTTable.add_row(['duration_ms', round(importanceDT[2],3)])
featImpDTTable.add_row(['energy', round(importanceDT[3],3)])
featImpDTTable.add_row(['instrumentalness', round(importanceDT[4],3)])
featImpDTTable.add_row(['liveness', round(importanceDT[5],3)])
featImpDTTable.add_row(['loudness', round(importanceDT[6],3)])
featImpDTTable.add_row(['speechiness', round(importanceDT[7],3)])
featImpDTTable.add_row(['tempo', round(importanceDT[8],3)])
featImpDTTable.add_row(['valence', round(importanceDT[9],3)])
print(featImpDTTable)

#graphing the feature importance for DT with gini
sns.set(rc={'figure.figsize':(13.0, 10.0)})
sns.barplot(x='Audio Feature', y='Importance', data=featImportanceDT).set_title('Feature Importance in DT with Gini Split')

#print(dict(zip(X.columns, importance)))
#prediction on test data
y_predict = modelDT.predict(X_test)


from sklearn.metrics import accuracy_score
#DT1
accuracyDT = accuracy_score(y_test, y_predict)

#balanced_accuracy_score - DANCIK SUGGESTED
from sklearn.metrics import balanced_accuracy_score
#DT1
balAccuracyDT = balanced_accuracy_score(y_test,y_predict)

from sklearn.model_selection import cross_val_score
cvScores = cross_val_score(modelDT, X_train, y_train, cv = 5)
cvMeanScore = cvScores.mean()

#maybe include
#from sklearn.metrics import precision_score
#presScore = precision_score(y_test, y_predict)
#------------------------------------------------------------------------------------------------
#--------------------------------------DT WITH ENTROPY AS CRITERION-------------------------------------

#Entropy (information gain) criteria, best split at each node
modelDT2 = tree.DecisionTreeClassifier(criterion = "entropy", splitter = "best")
#fit model
modelDT2.fit(X_train, y_train)
#predict for DT2
y_predict2 = modelDT2.predict(X_test)
#accuracy
accuracyDT2 = accuracy_score(y_test, y_predict2)
#bal acc
balAccuracyDT2 = balanced_accuracy_score(y_test,y_predict2)

cvScores2 = cross_val_score(modelDT2, X_train, y_train, cv = 5)
cvMeanScore2 = cvScores2.mean()

importanceDT2 = modelDT2.feature_importances_
#Feature importance dataFrame for graph

featImportanceDT2 = pd.DataFrame({'Audio Feature': ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'], 'Importance': [importanceDT2[0], importanceDT2[1], importanceDT2[2], importanceDT2[3],
                                                   importanceDT2[4],importanceDT2[5],importanceDT2[6],importanceDT2[7],importanceDT2[8],importanceDT2[9]]})
#featImportanceDT.head()
#sns.set()

from prettytable import PrettyTable
featImpDT2Table = PrettyTable(['Audio Feature', 'Importance'])
featImpDT2Table.add_row(['acousticness', round(importanceDT2[0],3)])
featImpDT2Table.add_row(['danceability', round(importanceDT2[1],3)])
featImpDT2Table.add_row(['duration_ms', round(importanceDT2[2],3)])
featImpDT2Table.add_row(['energy', round(importanceDT2[3],3)])
featImpDT2Table.add_row(['instrumentalness', round(importanceDT2[4],3)])
featImpDT2Table.add_row(['liveness', round(importanceDT2[5],3)])
featImpDT2Table.add_row(['loudness', round(importanceDT2[6],3)])
featImpDT2Table.add_row(['speechiness', round(importanceDT2[7],3)])
featImpDT2Table.add_row(['tempo', round(importanceDT2[8],3)])
featImpDT2Table.add_row(['valence', round(importanceDT2[9],3)])
print(featImpDT2Table)


#graphing the feature importance for DT with gini
sns.set(rc={'figure.figsize':(13.0, 10.0)})
sns.barplot(x='Audio Feature', y='Importance', data=featImportanceDT2).set_title('Feature Importance in DT with entropy split')

#--------------------------------------------------------------------------------------------

#-----------------DT summaries-------------------------------

tableDT = PrettyTable(['Criteria', 'Accuracy', 'Balanced Accuracy', 'Cross Validation Score'])
tableDT.add_row(['Gini Index', round(accuracyDT,3), round(balAccuracyDT,3), round(cvMeanScore,3)])
tableDT.add_row(['Entropy', round(accuracyDT2,3), round(balAccuracyDT2,3), round(cvMeanScore2,3)])
print(tableDT)

resultDT = pd.DataFrame({'Criteria': ['Gini', 'Entropy'], 'Accuracy': [balAccuracyDT, balAccuracyDT2]})
resultDT.head()
#DTplot = resultsDT.plot(x='Criteria', y='Accuracy', kind = 'bar', title='Accuracy with different criterion', ylim=(.75,.85))

sns.set()
sns.barplot(x='Criteria', y='Accuracy', data=resultDT).set_title('Balanced Accuracy of Decision Tree')



from sklearn.metrics import confusion_matrix

#confusion matrix for DT with GINI
cfDT1 = pd.DataFrame(confusion_matrix(y_test, y_predict), columns=['Pred. Non Hit', 'Pred. Hit'], index=['True Non Hit', 'True Hit'])
cfDT1
#confusion matrix for DT with Entropys
cfDT2 = pd.DataFrame(confusion_matrix(y_test, y_predict2), columns=['Pred. Non Hit', 'Pred. Hit'], index=['True Non Hit', 'True Hit'])
cfDT2
#BUILDING THE TREE - DOES NOT WORK


#dot_data = StringIO()
#export_graphviz(model, out_file='tree1.dot', filled = True, rounded = True, special_characters = True, feature_names = ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'], class_names = ['No Hit', 'Hit'], label='all')
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)
#-------------------------------------------------------------------
#-----------------------------------------------LOGISTIC REGRESSION - DEFAULT -----------------------------
from sklearn.linear_model import LogisticRegression


from sklearn import preprocessing

toNorm = X_train
cols =toNorm.columns

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(toNorm)
df_norm = pd.DataFrame(np_scaled, columns = cols)
df_norm

lRegT = LogisticRegression()
lRegT.fit(df_norm, y_train)
y_predictLRT = lRegT.predict(X_test)
importanceLRT = lRegT.coef_
#lReg = LogisticRegression()
#lReg.fit(X_train, y_train)
#y_predictLR = lReg.predict(X_test)
#importanceLR = lReg.coef_

importanceLRdf = pd.DataFrame({"Feature": ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence'], 
                               "Importance": [importanceLRT[0][0], importanceLRT[0][1], importanceLRT[0][2], importanceLRT[0][3], importanceLRT[0][4],importanceLRT[0][5],importanceLRT[0][6],importanceLRT[0][7],importanceLRT[0][8],importanceLRT[0][9]]})
sns.set(rc={'figure.figsize':(13.0, 10.0)})
sns.barplot(x='Feature', y='Importance', data=importanceLRdf).set_title('Feature Importance in LR')


accuracyLR = accuracy_score(y_test, y_predictLRT)
balAccuracyLR = balanced_accuracy_score(y_test, y_predictLRT)
cvScoresLR = cross_val_score(lRegT, X_train, y_train, cv=5)
cvMeanScoreLR = cvScores.mean()

tableLR = PrettyTable(['Algorithm', 'Accuracy', 'Bal. Accuracy Score', 'CV Score'])
tableLR.add_row(['Logistic Regression', round(accuracyLR,3), round(balAccuracyLR,3), round(cvMeanScoreLR,3)])
print(tableLR)

#----------------------------------------------------------------------------------------------------------

#------------------------KNN-----------------------

#k=3
from sklearn.neighbors import KNeighborsClassifier
knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(X_train, y_train)

y_predKnn3 = knn3.predict(X_test)

accuracyKnn3 = accuracy_score(y_test, y_predKnn3)
#bal acc
balAccKnn3 = balanced_accuracy_score(y_test,y_predKnn3)

cvScoresKnn3 = cross_val_score(knn3, X_train, y_train, cv = 5)
cvMeanScoreKnn3 = cvScoresKnn3.mean()

#k = 5
knn5 = KNeighborsClassifier(n_neighbors = 5)
knn5.fit(X_train, y_train)

y_predKnn5 = knn5.predict(X_test)

accuracyKnn5 = accuracy_score(y_test, y_predKnn5)
#bal acc
balAccKnn5 = balanced_accuracy_score(y_test,y_predKnn5)

cvScoresKnn5 = cross_val_score(knn5, X_train, y_train, cv = 5)
cvMeanScoreKnn5 = cvScoresKnn5.mean()

#k = 7
knn7 = KNeighborsClassifier(n_neighbors = 7)
knn7.fit(X_train, y_train)

y_predKnn7 = knn7.predict(X_test)

accuracyKnn7 = accuracy_score(y_test, y_predKnn7)
#bal acc
balAccKnn7 = balanced_accuracy_score(y_test,y_predKnn7)

cvScoresKnn7 = cross_val_score(knn7, X_train, y_train, cv = 5)
cvMeanScoreKnn7 = cvScoresKnn7.mean()

#k = 9
knn9 = KNeighborsClassifier(n_neighbors = 9)
knn9.fit(X_train, y_train)

y_predKnn9 = knn9.predict(X_test)

accuracyKnn9 = accuracy_score(y_test, y_predKnn9)
#bal acc
balAccKnn9 = balanced_accuracy_score(y_test,y_predKnn9)

cvScoresKnn9 = cross_val_score(knn9, X_train, y_train, cv = 5)
cvMeanScoreKnn9 = cvScoresKnn9.mean()


tableKNN = PrettyTable(['K', 'Accuracy', 'Bal. Acc Score', 'CV Score'])
tableKNN.add_row(['3', round(accuracyKnn3,3),round(balAccKnn3,3),round(cvMeanScoreKnn3,3)])
tableKNN.add_row(['5', round(accuracyKnn5,3),round(balAccKnn5,3),round(cvMeanScoreKnn5,3)])
tableKNN.add_row(['7', round(accuracyKnn7,3),round(balAccKnn7,3),round(cvMeanScoreKnn7,3)])
tableKNN.add_row(['9', round(accuracyKnn9,3),round(balAccKnn9,3),round(cvMeanScoreKnn9,3)])
print(tableKNN)
 
knnResult = pd.DataFrame({'K': [3,5,7,9], 'Accuracy': [balAccKnn3, balAccKnn5, balAccKnn7, balAccKnn9]})
knnResult.head()

#knn results comparison
#knnResults.plot(x='K', y='Accuracy', kind = 'line', title='Accuracy of different K values in KNN')
sns.set()
sns.lineplot(x='K', y='Accuracy', data=knnResult).set_title('Accuracy of KNN with different values of K')

summaryTable = PrettyTable(['Algorithm', 'Accuracy', 'Bal. Accuracy Score', 'CV Score'])
summaryTable.add_row(['Decision Tree (gini)', round(accuracyDT,3), round(balAccuracyDT,3), round(cvMeanScore,3)])
summaryTable.add_row(['Decision Tree (entropy)', round(accuracyDT2,3), round(balAccuracyDT2,3), round(cvMeanScore2,3)])
summaryTable.add_row(['Logistic Regression', round(accuracyLR,3), round(balAccuracyLR,3), round(cvMeanScoreLR,3)])
summaryTable.add_row(['KNN (k=3)', round(accuracyKnn3,3),round(balAccKnn3,3),round(cvMeanScoreKnn3,3)])
summaryTable.add_row(['KNN (k=5)', round(accuracyKnn5,3),round(balAccKnn5,3),round(cvMeanScoreKnn5,3)])
print(summaryTable)

results = pd.DataFrame({'Model': ['Decision Tree (gini)', 'Decision Tree (entropy)', 'Logistic Regression', 'KNN (k=3)', 'KNN (k=5)'], 'Balanced Accuracy': [balAccuracyDT, balAccuracyDT2, balAccuracyLR, balAccKnn3, balAccKnn5]})
results.head()

sns.set(rc={'figure.figsize':(11.0, 8.0)})
sns.barplot(x='Model', y='Balanced Accuracy', data=results).set_title('Balanced Accuracy of Algorithms')






