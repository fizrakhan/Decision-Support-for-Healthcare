# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:32:39 2023

@author: HP
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

##Exploratory Data Analysis

#Q1.1
file = pd.read_csv("C:/Users/HP/Documents/Master's/Decision Support/Exercise4/diabetes.csv")
file.describe()

#Q1.2
features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
'DiabetesPedigreeFunction','Age']
target_feature=['Outcome']
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
figure=sns.boxplot(data=file[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']])

#Q1.3
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
file_features_normalized=pd.DataFrame(scaler.fit_transform(file[features]),columns=features)

normalized_features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
'DiabetesPedigreeFunction','Age']
target_feature=['Outcome']
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
figure=sns.boxplot(data=file_features_normalized[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']])

#Q1.4
file['Outcome'].value_counts()

##Building the classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

##employing Logistic Regression model
logreg_clf=LogisticRegression(random_state=42, class_weight='balanced')

#Q2.1: Stratification
from sklearn.model_selection import train_test_split

y=file['Outcome'].copy()
X=file.drop('Outcome',axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42, 
stratify=y)

y_train.value_counts()
y_test.value_counts()

#Q2.2, 1: results of logistic regression classifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

std_scaler=StandardScaler()
logreg_clf_pipe=Pipeline([('scaler', std_scaler), ('clf', logreg_clf)])
logreg_clf_pipe.fit(X_train,np.ravel(y_train))

from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score
y_pred_test=logreg_clf_pipe.predict(X_test)
print('Test accuracy', accuracy_score(y_test, y_pred_test))
print('Test sensitivity/recall', recall_score(y_test, y_pred_test))
print('Test precision', precision_score(y_test, y_pred_test))
print('Balanced accuracy score', balanced_accuracy_score(y_test, y_pred_test))

#q2.2, 2: confusion matrix of logistic regression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred_test)
df_cm = pd.DataFrame(cm, index= logreg_clf_pipe.classes_, columns= logreg_clf_pipe.classes_)
sns.heatmap(df_cm, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

#employing Random Forest model
rf_clf=RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_split=8, 
min_samples_leaf=3, bootstrap= True, random_state=42)

#Stratification
from sklearn.model_selection import train_test_split

y=file['Outcome'].copy()
X=file.drop('Outcome',axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42, 
stratify=y)

#Q3.1: results of random forest classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

std_scaler=StandardScaler()
ranfor_clf_pipe=Pipeline([('scaler', std_scaler), ('clf', rf_clf)])
ranfor_clf_pipe.fit(X_train,np.ravel(y_train))

from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score
y_pred_test=ranfor_clf_pipe.predict(X_test)
print('Test accuracy', accuracy_score(y_test, y_pred_test))
print('Test sensitivity/recall', recall_score(y_test, y_pred_test))
print('Test precision', precision_score(y_test, y_pred_test))
print('Balanced accuracy', balanced_accuracy_score(y_test, y_pred_test))

#confusion matrix of random forest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred_test)
df_cm = pd.DataFrame(cm, index= ranfor_clf_pipe.classes_, columns= ranfor_clf_pipe.classes_)
sns.heatmap(df_cm, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

#Q3.2:adding class imbalanced parameter to random forest
rf_clf_balanced1=RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_split=8, 
min_samples_leaf=3, bootstrap= True, class_weight='balanced', 
random_state=42)

from sklearn.model_selection import train_test_split

y1=file['Outcome'].copy()
X1=file.drop('Outcome',axis=1)
X_train1, X_test1, y_train1, y_test1=train_test_split(X1,y1, test_size=0.3, random_state=42, 
stratify=y1)

#Q3.1: results of 2nd random forest classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

std_scaler1=StandardScaler()
ranfor_clf_pipe1=Pipeline([('scaler', std_scaler1), ('clf', rf_clf_balanced1)])
ranfor_clf_pipe1.fit(X_train1,np.ravel(y_train1))

from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score
y_pred_test1=ranfor_clf_pipe1.predict(X_test1)
print('Test accuracy', accuracy_score(y_test1, y_pred_test1))
print('Test sensitivity/recall', recall_score(y_test1, y_pred_test1))
print('Test precision', precision_score(y_test1, y_pred_test1))
print('Balanced accuracy', balanced_accuracy_score(y_test1, y_pred_test1))

#confusion matrix of 2nd random forest
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
cm1 = confusion_matrix(y_test1, y_pred_test1)
df_cm1 = pd.DataFrame(cm1, index= ranfor_clf_pipe1.classes_, columns= ranfor_clf_pipe1.classes_)
sns.heatmap(df_cm1, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

#Q4: Cross validation approach for logistic regression classifier
from sklearn.model_selection import cross_validate, StratifiedKFold
# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Define scoring metrics
scoring_metrics = ['accuracy', 'precision', 'recall']
# Perform 5-fold stratified cross-validation with multiple metrics
cv_results = cross_validate(logreg_clf_pipe, X_train, y_train, cv=stratified_kfold, 
scoring=scoring_metrics)
# Output the cross-validation scores for each metric
for metric, scores in cv_results.items():
    if 'test_' in metric: # We're interested in test scores
        print(f"{metric}: {scores}")
        print(f"Mean {metric}: {np.mean(scores)}")
        print(f"Standard Deviation of {metric}: {np.std(scores)}")

