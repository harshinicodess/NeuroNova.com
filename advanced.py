# adhd_detection_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Step 1: Generate Synthetic Data (or replace this with your real ADHD dataset)
def create_synthetic_data(n_samples=1000):
    """
    Creates a synthetic dataset for ADHD detection.
    Replace this with actual data loading and preprocessing when available.
    """
    np.random.seed(42)
    
    # Generate synthetic features (replace with your real features)
    data = {
        'age': np.random.randint(6, 18, size=n_samples),  # Age of the subject
        'hyperactivity_score': np.random.rand(n_samples) * 10,  # Hyperactivity score
        'impulsivity_score': np.random.rand(n_samples) * 10,  # Impulsivity score
        'attention_score': np.random.rand(n_samples) * 10,  # Attention score
        'parental_concern': np.random.rand(n_samples) * 10,  # Parental concern score
    }
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # Generate labels based on some criteria (you would replace this with actual labels in real-world data)
    df['label'] = np.where(
        (df['hyperactivity_score'] > 7) & (df['attention_score'] < 3), 1, 0)  # ADHD or Non-ADHD
    
    return df

# Step 2: Load and Prepare Data
df = create_synthetic_data()

# Display a sample of the data (helps in understanding the features)
print("Sample Data:")
print(df.head())

# Features and target variable
X = df.drop(columns=['label'])
y = df['label']

# Step 3: Split the Data into Training and Testing Sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Standardize the Features (important for models sensitive to feature scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Model - Random Forest Classifier with Hyperparameter Tuning
# Here, we'll use GridSearchCV for hyperparameter tuning to optimize the model.
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required at a leaf node
}

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Step 6: Evaluate the Model
y_pred = best_model.predict(X_test_scaled)

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No ADHD", "ADHD"], yticklabels=["No ADHD", "ADHD"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 7: Visualize Feature Importance
feature_importance = best_model.feature_importances_
features = X.columns

# Plot the feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance in ADHD Detection")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Step 8: Save the Model (optional)
joblib.dump(best_model, 'adhd_detection_model.pkl')
print("Model saved as 'adhd_detection_model.pkl'")

# Step 9: Predict for New Samples (Example Usage)
new_data = np.array([[12, 8.5, 6.2, 3.1, 7.8]])  # Example new sample data
new_data_scaled = scaler.transform(new_data)  # Don't forget to scale the new data
prediction = best_model.predict(new_data_scaled)
print("\nPredicted label for new data:", "ADHD" if prediction[0] == 1 else "No ADHD")
