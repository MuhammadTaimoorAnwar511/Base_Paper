import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from data_processing import X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler

# Function to prepare the data for LSTM input
def create_dataset(features, target, time_steps=60):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:(i + time_steps)])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

# Reshape data for LSTM input: [samples, time steps, features]
time_steps = 60  # Use 60 previous time steps to predict the next one
X_train_lstm, y_train_lstm = create_dataset(X_train_scaled, y_train_scaled, time_steps)
X_test_lstm, y_test_lstm = create_dataset(X_test_scaled, y_test_scaled, time_steps)

# Reshape input to be [samples, time steps, features]
X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], X_train_lstm.shape[2])
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], X_test_lstm.shape[2])

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(units=1))  # Predict a single value (Close price)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
print("Model Summary:")
print(model.summary())

# Train the model
history = model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test_lstm), verbose=1)

# Save the trained model
model.save('Model/model/lstm_model.h5')

# Plot Training and Validation Loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("Model training complete and model saved!")
