adhd_detection_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create Synthetic Dataset (Replace with real data)
def create_synthetic_data(n_samples=1000):
    # Randomly generate features (replace with actual features from your dataset)
    np.random.seed(42)
    data = {
        'age': np.random.randint(6, 18, size=n_samples),  # Age of the subject
        'hyperactivity_score': np.random.rand(n_samples) * 10,  # Hyperactivity behavior score
        'impulsivity_score': np.random.rand(n_samples) * 10,  # Impulsivity behavior score
        'attention_score': np.random.rand(n_samples) * 10,  # Attention score
        'parental_concern': np.random.rand(n_samples) * 10,  # Parental concern score
    }
    df = pd.DataFrame(data)
    
    # Simulating labels: 1 for ADHD, 0 for non-ADHD
    df['label'] = np.where(
        (df['hyperactivity_score'] > 7) & (df['attention_score'] < 3), 1, 0)
    
    return df

# Step 2: Load and Prepare Data
df = create_synthetic_data()

# Features and target variable
X = df.drop(columns=['label'])
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Standardize the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train the Model - Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test_scaled)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No ADHD", "ADHD"], yticklabels=["No ADHD", "ADHD"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature Importance Visualization
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Step 6: Save the Model (optional)
import joblib
joblib.dump(model, 'adhd_detection_model.pkl')
print("Model saved as 'adhd_detection_model.pkl'")
