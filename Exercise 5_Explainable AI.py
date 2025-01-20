# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:38:28 2023

@author: HP
"""

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file = pd.read_csv("C:/Users/HP/Documents/Master's/Decision Support/Exercise4/diabetes.csv")
file.describe()

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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

std_scaler=StandardScaler()
logreg_clf_pipe=Pipeline([('scaler', std_scaler), ('clf', logreg_clf)])
logreg_clf_pipe.fit(X_train,np.ravel(y_train))

importance = logreg_clf_pipe['clf'].coef_[0]

# plot feature importance
plt.bar([x for x in range(len(importance))], importance, tick_label=features)
plt.xticks(rotation=70)
plt.grid(axis='y', linestyle='--', )
plt.ylabel('LogReg coefficients value')
plt.xlabel('Features')
plt.show()

#building decision tree classifier
from sklearn import tree

dt_clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=5, min_samples_split=8, min_samples_leaf=3, min_weight_fraction_leaf=0.0, random_state=42, min_impurity_decrease=0.0, class_weight= 'balanced')

from sklearn.model_selection import train_test_split

y=file['Outcome'].copy()
X=file.drop('Outcome',axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42, 
stratify=y)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

std_scaler=StandardScaler()
dt_clf_pipe=Pipeline([('scaler', std_scaler), ('clf', dt_clf)])
dt_clf_pipe.fit(X_train,np.ravel(y_train))

from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score
y_pred_test=dt_clf_pipe.predict(X_test)
print('Test accuracy', accuracy_score(y_test, y_pred_test))
print('Test sensitivity/recall', recall_score(y_test, y_pred_test))
print('Test precision', precision_score(y_test, y_pred_test))
print('Balanced accuracy score', balanced_accuracy_score(y_test, y_pred_test))

rf_clf=RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_split=8, 
min_samples_leaf=3, bootstrap= True, random_state=42)

from sklearn.model_selection import train_test_split

y=file['Outcome'].copy()
X=file.drop('Outcome',axis=1)
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=42, 
stratify=y)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

std_scaler=StandardScaler()
rf_clf_pipe=Pipeline([('scaler', std_scaler), ('clf', rf_clf)])
rf_clf_pipe.fit(X_train,np.ravel(y_train))


importance1 = rf_clf_pipe['clf'].feature_importances_

# plot feature importance1
plt.bar([x for x in range(len(importance1))], importance1, tick_label=features)
plt.xticks(rotation=70)
plt.grid(axis='y', linestyle='--', )
plt.ylabel('Random Forest coefficients value')
plt.xlabel('Features')
plt.show()

#Feature permutation importance
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
# Compute permutation importance
result = permutation_importance(rf_clf_pipe, X_train, y_train, n_repeats=30, random_state=42)
# Sort features by importance
sorted_idx = result.importances_mean.argsort()
# Create a DataFrame to hold the results
feature_importance_df = pd.DataFrame({
 'Feature': np.array(features)[sorted_idx], # Assuming 'features' contains the feature names
 'Importance_Mean': result.importances_mean[sorted_idx],
 'Importance_Std': result.importances_std[sorted_idx]
})
# Sort the DataFrame by Importance_Mean in descending order
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance_Mean', 
ascending=False)
print(sorted_feature_importance_df)
# Plotting
plt.figure(figsize=(12, 8))
plt.barh(sorted_feature_importance_df['Feature'], 
sorted_feature_importance_df['Importance_Mean'], 
xerr=sorted_feature_importance_df['Importance_Std'], align='center', alpha=0.5, 
ecolor='black', capsize=10)
plt.xlabel('Importance', fontsize=20)
plt.ylabel('Feature', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Sorted Feature Permutation Importance', fontsize=20)
plt.gca().invert_yaxis() # Reverse the order to have the most important feature at the top
plt.show()

#Q2.3

from sklearn.inspection import PartialDependenceDisplay

fig_train, (ax1,ax2,ax3,ax4) = plt.subplots(4, 2, figsize=(20, 12))
PartialDependenceDisplay.from_estimator(rf_clf_pipe, X_train,X_train.columns, ax=[ax1,ax2,ax3,ax4])




