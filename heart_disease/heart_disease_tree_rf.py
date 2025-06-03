import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Load dataset
data = pd.read_csv("heart.csv")

# Show basic info
print("Dataset Shape:", data.shape)
print(data.head())

# Split data
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------- 1. Decision Tree -----------------
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dtree, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'])
plt.title("Decision Tree (max_depth=3)")
plt.show()

# Accuracy
y_pred_dt = dtree.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# ----------------- 2. Random Forest -----------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Accuracy
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# ----------------- 3. Feature Importances -----------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(importances)

# Plot feature importances
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importances (Random Forest)")
plt.show()

# ----------------- 4. Cross-Validation -----------------
dtree_cv_scores = cross_val_score(dtree, X, y, cv=5)
rf_cv_scores = cross_val_score(rf, X, y, cv=5)

print("\nDecision Tree Cross-Validation Accuracy: ", np.mean(dtree_cv_scores))
print("Random Forest Cross-Validation Accuracy: ", np.mean(rf_cv_scores))
