import numpy as np
import json
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from data_processing import X_test_scaled, y_test_scaled, target_scaler

# Load the pre-trained model
model = load_model('Model/model/lstm_model.h5')

# Make predictions on the test set
y_pred_scaled = model.predict(X_test_scaled)

# Inverse transform the predicted values and the actual values
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_actual = target_scaler.inverse_transform(y_test_scaled)

# Compute evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)

# Prepare evaluation results as a dictionary
evaluation_results = {
    "Mean Squared Error": mse,
    "Mean Absolute Error": mae,
    "First 5 Predictions": y_pred[:5].tolist(),
    "First 5 Actual Values": y_test_actual[:5].tolist()
}

# Save the evaluation results to a JSON file
with open('Model/model/evaluation_results.json', 'w') as json_file:
    json.dump(evaluation_results, json_file, indent=4)

print("Model evaluation complete and results saved to evaluation_results.json!")
