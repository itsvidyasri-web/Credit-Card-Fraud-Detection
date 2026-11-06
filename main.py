
import pandas as pd

# Load the credit card dataset
data = pd.read_csv("creditcard.csv")

# Display first 5 rows
print(data.head())

# Display dataset size
print("Dataset Shape:", data.shape)


# Display column names
print("\nColumn Names:")
print(data.columns)

# Check dataset information
print("\nDataset Information:")
data.info()

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())


# Count genuine and fraudulent transactions
print("\nTransaction Class Count:")
print(data["Class"].value_counts())

# Calculate fraud percentage
fraud_percentage = (data["Class"].sum() / len(data)) * 100

print("\nFraud Transaction Percentage:")
print(f"{fraud_percentage:.4f}%")


from sklearn.model_selection import train_test_split

# Separate input features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=5000)

# Train the model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Predict test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
print("\nModel Evaluation Results")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred, zero_division=0))

print("Recall:", recall_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

model.fit(X_train, y_train)

print("Improved Model Training Completed!")


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Predict test transactions
y_pred = model.predict(X_test)

# Display evaluation metrics
print("\nImproved Model Evaluation")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


import pickle

# Save the trained model
with open("fraud_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nFraud Detection Model Saved Successfully!")
