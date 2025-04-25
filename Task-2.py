# ml_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv("student_scores(task-2).csv")
print("\nFirst 5 Rows:")
print(df.head())

# Step 2: Explore the data
print("\nData Info:")
print(df.info())
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation")
plt.show()

# Step 3: Split into features (X) and target (y)
X = df.drop("WillPass", axis=1)
y = df["WillPass"]

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a classification model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Feature Importance
importance = model.feature_importances_
features = X.columns
plt.barh(features, importance)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()
