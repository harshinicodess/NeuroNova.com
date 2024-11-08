# adhd_detection_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import logging
from shap import TreeExplainer

# Set up logging
logging.basicConfig(level=logging.INFO)

# 1. Function to generate synthetic data (Replace with real dataset)
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(6, 18, size=n_samples),  # Age of the subject
        'hyperactivity_score': np.random.rand(n_samples) * 10,  # Hyperactivity score
        'impulsivity_score': np.random.rand(n_samples) * 10,  # Impulsivity score
        'attention_score': np.random.rand(n_samples) * 10,  # Attention score
        'parental_concern': np.random.rand(n_samples) * 10,  # Parental concern score
    }
    
    df = pd.DataFrame(data)
    
    # Generate synthetic labels
    df['label'] = np.where(
        (df['hyperactivity_score'] > 7) & (df['attention_score'] < 3), 1, 0)  # ADHD or Non-ADHD
    
    return df

# 2. Preprocess Data (train/test split + scaling)
def preprocess_data(df):
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Train-test split (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 3. Train and Tune Model
def train_model(X_train, y_train):
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=42)
    
    # Using GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

# 4. Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
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

# 5. Visualize Feature Importance
def plot_feature_importance(model, X_train):
    feature_importance = model.feature_importances_
    features = X_train.columns

    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importance, y=features)
    plt.title("Feature Importance in ADHD Detection")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()

# 6. Model Interpretability using SHAP
def interpret_model(model, X_train):
    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Summary plot
    shap.summary_plot(shap_values[1], X_train)  # SHAP values for ADHD class (1)

# 7. Save Model & Scaler
def save_model_and_scaler(model, scaler, filename="adhd_detection_model.pkl"):
    joblib.dump(model, filename)
    joblib.dump(scaler, "scaler.pkl")
    logging.info(f"Model and Scaler saved as '{filename}' and 'scaler.pkl'.")

# 8. Predict New Data
def predict_new_data(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)  # Scale the data
    prediction = model.predict(new_data_scaled)
    return "ADHD" if prediction[0] == 1 else "No ADHD"

# Main Execution
if __name__ == "__main__":
    # Step 1: Load and prepare data
    df = create_synthetic_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
    
    # Step 2: Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Step 3: Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)
    
    # Step 4: Visualize feature importance
    plot_feature_importance(model, df.drop(columns=["label"]))
    
    # Step 5: Interpret the model
    interpret_model(model, df.drop(columns=["label"]))
    
    # Step 6: Save the model and scaler
    save_model_and_scaler(model, scaler)

    # Step 7: Predict for new samples
    new_data = np.array([[12, 8.5, 6.2, 3.1, 7.8]])  # Example new sample data
    prediction = predict_new_data(model, scaler, new_data)
    print(f"\nPredicted label for new data: {prediction}")
