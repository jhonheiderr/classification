# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.dummy import DummyClassifier
import seaborn as sns

# Loading the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the classifiers
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
lr_classifier = LogisticRegression(max_iter=10000, random_state=42)  # Increase max_iter
dt_classifier = DecisionTreeClassifier(random_state=42)

# Random Classifier Model
random_classifier = DummyClassifier(strategy="uniform", random_state=42)
random_classifier.fit(X_train, y_train)

# Training the models
rf_classifier.fit(X_train, y_train)
lr_classifier.fit(X_train, y_train)
dt_classifier.fit(X_train, y_train)

random_classifier = DummyClassifier(strategy="uniform", random_state=42)
random_classifier.fit(X_train, y_train)

# Model estimations on the test data
rf_y_pred = rf_classifier.predict_proba(X_test)[:, 1]
lr_y_pred = lr_classifier.predict_proba(X_test)[:, 1]
dt_y_pred = dt_classifier.predict_proba(X_test)[:, 1]
random_y_pred = random_classifier.predict_proba(X_test)[:, 1]

# Calculating ROC curve and AUC for each model
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred)
rf_auc = auc(rf_fpr, rf_tpr)

lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_pred)
lr_auc = auc(lr_fpr, lr_tpr)

dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_y_pred)
dt_auc = auc(dt_fpr, dt_tpr)

random_fpr, random_tpr, _ = roc_curve(y_test, random_y_pred)
random_auc = auc(random_fpr, random_tpr)

# Plotting ROC curves
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.2f})')
plt.plot(random_fpr, random_tpr, 'k--', label=f'Random Classifier (AUC = {random_auc:.2f})')

# Configurando o gráfico
plt.title('ROC Curve - Model Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# #### Calculating and displaying the confusion matrix and metrics for each model

rf_conf_matrix = confusion_matrix(y_test, rf_classifier.predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Value')
plt.ylabel('Real Value')
plt.show()

lr_conf_matrix = confusion_matrix(y_test, lr_classifier.predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(lr_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Value')
plt.ylabel('Real Value')
plt.show()

dt_conf_matrix = confusion_matrix(y_test, dt_classifier.predict(X_test))
plt.figure(figsize=(8, 6))
sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted Value')
plt.ylabel('Real Value')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(dt_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=data.target_names, yticklabels=data.target_names)

# Adicionando rótulos aos elementos da matriz
for i in range(dt_conf_matrix.shape[0]):
    for j in range(dt_conf_matrix.shape[1]):
        if i == 0 and j == 0:
            plt.text(j + 0.5, i + 0.6, f"True Negatives", horizontalalignment='center', verticalalignment='center')
        elif i == 1 and j == 0:
            plt.text(j + 0.5, i + 0.6, f"False Negatives", horizontalalignment='center', verticalalignment='center')
        elif i == 0 and j == 1:
            plt.text(j + 0.5, i + 0.6, f"False Positives", horizontalalignment='center', verticalalignment='center')
        elif i == 1 and j == 1:
            plt.text(j + 0.5, i + 0.6, f"True Positives", horizontalalignment='center', verticalalignment='center')

plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.show()

