import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# Config
TICKER = 'GOOGL'
START_DATE = '2009-01-01'
END_DATE = '2023-01-01'
SEQUENCE_LENGTH = 150
EPOCHS = 50
BATCH_SIZE = 32

# Load data
df = yf.download(TICKER, start=START_DATE, end=END_DATE)
data = df[['Close']]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Save scaler
joblib.dump(scaler, 'scaler.save')

# Create training sequences
x_train, y_train = [], []
for i in range(SEQUENCE_LENGTH, len(data_scaled)):
    x_train.append(data_scaled[i - SEQUENCE_LENGTH:i])
    y_train.append(data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Save model
model.save('keras_model_150.h5')
print("✅ Saved keras_model_150.h5 and scaler.save")
