#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns

df_rf = sns.load_dataset("titanic")

print(df_rf.head())

df_rf.isnull().sum()

X = df_rf.drop(columns=['survived'])
y = df_rf['survived']
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=98)
   
rf_clf = RandomForestClassifier(
    criterion="gini",

    min_samples_leaf=1,
    min_samples_split=10,
    max_features='sqrt',
    random_state=1
    )

rf_clf.fit(X_train, y_train)

rf_clf.predict(X_test)

rf_y_true = y_test

y_pred = rf_clf.predict(X_test)
    
accuracy = rf_clf.score(X_test, y_test)
precision = precision_score(y_test, y_pred, average='weighted') 
recall = recall_score(y_test, y_pred, average='weighted') 
f1_score = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy) 
print("Precision Score:", precision)
print("Recall Score: ", recall) 
print("F1 Score: ", f1_score) 

from sklearn.metrics import roc_curve, auc

rf_auc = 0

def plot_roc(rf_y_true, rf_probs):
    
    
    rf_fpr, rf_tpr, rf_threshold = roc_curve(rf_y_true, rf_probs)

    rf_auc_val = metrics.auc(rf_fpr, rf_tpr) 
    
    plt.plot(rf_fpr, rf_tpr, label = 'AUC=%0.2f'%rf_auc_val, color = 'darkorange')
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'b--')
    plt.xlim([0,1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return rf_auc_val

rf_probs = rf_clf.predict_proba(X_test) [:,1] 
rf_auc = plot_roc(rf_y_true, rf_probs) 

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

max_acc, max_k = 0, 0

for k in range(2, 11):

    skfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = 100) 
    
    results_skfold_acc = (cross_val_score(rf_clf, X, y, cv = skfold)).mean() * 100.0
    
    if results_skfold_acc > max_acc: 
        max_acc = results_skfold_acc
        max_k = k
        
    print("Accuracy: %.2f%%" % (results_skfold_acc)) 

print("I love pavan")
