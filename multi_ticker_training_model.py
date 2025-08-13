import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ===== Cấu hình =====
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'NFLX', 'BRK-B', 'JPM',
    'V', 'MA', 'PG', 'KO', 'PEP',
    'VCB.VN', 'CTG.VN', 'BID.VN', 'TCB.VN', 'VPB.VN',
]

START_DATE = '2009-01-01'
END_DATE = '2023-01-01'
SEQUENCE_LENGTH = 150
EPOCHS = 50
BATCH_SIZE = 32

# ===== Tải và gộp dữ liệu =====
all_close = []
for ticker in TICKERS:
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if df.empty:
        print(f"⚠ Không có dữ liệu: {ticker}")
        continue
    all_close.append(df[['Close']].rename(columns={'Close': f'Close_{ticker}'}))

if not all_close:
    raise RuntimeError("❌ Không có dữ liệu nào được tải")

# Gộp theo cột, chỉ lấy phần giao nhau thời gian
data = all_close[0]
for df in all_close[1:]:
    data = data.join(df, how='inner')

# ===== Chuẩn hóa =====
stacked = pd.DataFrame({'Close': pd.concat([data[c] for c in data.columns])})
scaler = MinMaxScaler()
scaler.fit(stacked[['Close']])
joblib.dump(scaler, 'scaler.save')

# ===== Tạo chuỗi thời gian huấn luyện =====
def create_sequences(series, seq_len):
    scaled = scaler.transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y)

X_list, y_list = [], []
for col in data.columns:
    s = data[col].dropna()
    Xi, yi = create_sequences(s.to_frame(), SEQUENCE_LENGTH)
    if len(Xi) > 0:
        X_list.append(Xi)
        y_list.append(yi)

X_train = np.vstack(X_list)
y_train = np.hstack(y_list)

print("✅ Training shape:", X_train.shape, y_train.shape)

# ===== Xây dựng mô hình LSTM =====
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# ===== Lưu mô hình =====
model.save('multi_ticket_train.h5')
print("✅ Saved keras_model_150.h5 and scaler.save")
