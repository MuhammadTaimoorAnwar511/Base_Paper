import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib # Save the scaler objects for later use
# Path to the dataset
dataset_path = r"Data\bitcoin_data.csv"

# Load the CSV file
data = pd.read_csv(dataset_path)

# Strip column names
data.columns = data.columns.str.strip()

# Convert 'Open Time' to datetime
if 'Open Time' in data.columns:
    data['Open Time'] = pd.to_datetime(data['Open Time'])

# Define feature columns (X) and target column (Y)
feature_columns = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
target_column = 'Close'

X_data = data[feature_columns]  # Features
y_data = data[target_column]    # Target

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Initialize MinMaxScaler for features and target
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Scale the training features and target
X_train_scaled = feature_scaler.fit_transform(X_train)
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))  # Reshape y_train to 2D

# Scale the testing features and target
X_test_scaled = feature_scaler.transform(X_test)
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))  # Reshape y_test to 2D

# Display the split sizes
print("\n======================================\n")
print("Training set size (features):", X_train_scaled.shape)
print("Training set size (target):", y_train_scaled.shape)
print("Testing set size (features):", X_test_scaled.shape)
print("Testing set size (target):", y_test_scaled.shape)

# Print the scaled data for verification
print("\n======================================\n")
print("Scaled training features (X_train_scaled):")
print(X_train_scaled[:5])  # Display first 5 rows of scaled training features

print("\nScaled training target (y_train_scaled):")
print(y_train_scaled[:5])  # Display first 5 rows of scaled training target

print("\nScaled testing features (X_test_scaled):")
print(X_test_scaled[:5])  # Display first 5 rows of scaled testing features

print("\nScaled testing target (y_test_scaled):")
print(y_test_scaled[:5])  # Display first 5 rows of scaled testing target

# Ensure the output directory exists
output_dir = 'Model/data'
os.makedirs(output_dir, exist_ok=True)

# Save the scaled data and scaler objects as .npy and .pkl files
np.save(os.path.join(output_dir, 'X_train_scaled.npy'), X_train_scaled)
np.save(os.path.join(output_dir, 'X_test_scaled.npy'), X_test_scaled)
np.save(os.path.join(output_dir, 'y_train_scaled.npy'), y_train_scaled)
np.save(os.path.join(output_dir, 'y_test_scaled.npy'), y_test_scaled)

joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))

# Export variables to be imported in other scripts
__all__ = ['X_train_scaled', 'X_test_scaled', 'y_train_scaled', 'y_test_scaled', 'feature_scaler', 'target_scaler']
