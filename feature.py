#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.naive_bayes import *
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import loguniform
from skopt import BayesSearchCV
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier 



# In[ ]:


def feature_selection_with_RandomForest(X, y, test_size=0.3, random_state=42, threshold=0.05, n_estimators=100):

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # For training
    # Oversampling the minority class
    oversampler = RandomOverSampler(sampling_strategy='minority')
    X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)

    # Undersampling the majority class
    undersampler = RandomUnderSampler(sampling_strategy='majority')
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train_over, y_train_over)

    # Display feature importance
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    print("Feature Importance:")
    print(feature_importances)

    # Select features with importance above a certain threshold
    sfm = SelectFromModel(rf, threshold=threshold)
    sfm.fit(X_train, y_train)

    # Transform the training set and test set
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)

    # Train a classifier on the selected features
    selected_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    selected_rf.fit(X_train_selected, y_train)

    # Make predictions on the test set
    y_pred = selected_rf.predict(X_test_selected)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set with selected features using Random Forest: {accuracy:.2f}")

    return accuracy






# In[ ]:


def feature_selection_with_XGBoost(X, y, test_size=0.3, random_state=42, threshold=0.05, n_estimators=100):

    # Assuming you have your features X and target variable y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # XGBoost for feature importance
    xgb = XGBClassifier(n_estimators=100, random_state=random_state)
    xgb.fit(X_train, y_train)

    # Display feature importance
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': xgb.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    print("Feature Importance:")
    print(feature_importances)

    # Select features with importance above a certain threshold
    sfm = SelectFromModel(xgb, threshold=0.05)
    sfm.fit(X_train, y_train)

    # Transform the training set and test set
    X_train_selected = sfm.transform(X_train)
    X_test_selected = sfm.transform(X_test)

    # Train a classifier on the selected features with XGBoost
    selected_xgb = XGBClassifier(n_estimators=100, random_state=42)
    selected_xgb.fit(X_train_selected, y_train)

    # Make predictions on the test set
    y_pred_xgb = selected_xgb.predict(X_test_selected)

    # Evaluate the accuracy with XGBoost
    accuracy = accuracy_score(y_test, y_pred_xgb)
    print(f"Accuracy on the test set with selected features using XGBoost: {accuracy:.2f}")

    return accuracy



# In[ ]:


def feature_selection_with_AdaBoost(X, y, test_size=0.3, random_state=42, threshold=0.05, n_estimators=100):

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # AdaBoost for feature importance
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=random_state)
    adaboost.fit(X_train, y_train)

    # Display feature importance
    feature_importances_ada = pd.DataFrame({'Feature': X.columns, 'Importance': adaboost.feature_importances_})
    feature_importances_ada = feature_importances_ada.sort_values(by='Importance', ascending=False)
    print("Feature Importance with AdaBoost:")
    print(feature_importances_ada)

    # Select features with importance above a certain threshold
    sfm_ada = SelectFromModel(adaboost, threshold=0.05)
    sfm_ada.fit(X_train, y_train)

    # Transform the training set and test set
    X_train_selected_ada = sfm_ada.transform(X_train)
    X_test_selected_ada = sfm_ada.transform(X_test)

    # Train a classifier on the selected features with AdaBoost
    selected_ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    selected_ada.fit(X_train_selected_ada, y_train)

    # Make predictions on the test set
    y_pred_ada = selected_ada.predict(X_test_selected_ada)

    # Evaluate the accuracy with AdaBoost
    accuracy = accuracy_score(y_test, y_pred_ada)
    print(f"Accuracy on the test set with selected features using AdaBoost: {accuracy:.2f}")


    return accuracy



