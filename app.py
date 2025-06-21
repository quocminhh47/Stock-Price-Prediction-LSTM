# app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta, date

# ====== Cấu hình ======
SEQUENCE_LENGTH = 150
START_DATE = "2009-01-01"
END_DATE = date.today()

# ====== Giao diện ======
st.title('Stock Closing Price Prediction')
ticker = st.text_input("Nhập mã cổ phiếu", "GOOGL")

# ====== Tải dữ liệu ======
df = yf.download(ticker, start=START_DATE, end=END_DATE)
if df.empty:
    st.error("Không thể tải dữ liệu. Kiểm tra mã cổ phiếu hoặc kết nối mạng.")
    st.stop()

st.subheader(f'Dữ liệu từ {START_DATE} đến {END_DATE}')
st.write(df.describe())

# ====== Hiển thị biểu đồ cơ bản ======
st.subheader('Biểu đồ giá đóng cửa')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig1)

# ====== Trung bình động ======
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

st.subheader('Giá đóng cửa + MA100')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df.Close, 'r', label='Giá đóng cửa')
plt.plot(ma100, 'g', label='MA100')
plt.legend()
st.pyplot(fig2)

st.subheader('Giá đóng cửa + MA100 + MA200')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Giá đóng cửa')
plt.plot(ma100, 'g', label='MA100')
plt.plot(ma200, 'b', label='MA200')
plt.legend()
st.pyplot(fig3)

# ====== Tách train/test ======
train_df = pd.DataFrame(df['Close'][0:int(len(df) * 0.85)])
test_df = pd.DataFrame(df['Close'][int(len(df) * 0.85):])

# ====== Tải lại scaler và model đã train ======
scaler = joblib.load("scaler.save")
model = load_model("keras_model_150.h5")

# ====== Tiền xử lý dữ liệu ======
past_days = train_df.tail(SEQUENCE_LENGTH)
final_df = past_days._append(test_df, ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(SEQUENCE_LENGTH, input_data.shape[0]):
    x_test.append(input_data[i - SEQUENCE_LENGTH:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# ====== Dự đoán test set ======
y_pred = model.predict(x_test)
scale = scaler.scale_
scale_factor = 1 / scale[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# ====== Biểu đồ test set (dự đoán vs thực tế) ======
st.subheader('Giá dự đoán vs Thực tế (Test Set)')
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=test_df.index[-len(y_test):],
    y=y_test,
    name='Giá thực tế',
    line=dict(color='green')
))
fig4.add_trace(go.Scatter(
    x=test_df.index[-len(y_test):],
    y=y_pred.flatten(),
    name='Giá dự đoán',
    line=dict(color='red')
))
fig4.update_layout(
    title='Dự đoán vs Thực tế',
    xaxis_title='Ngày',
    yaxis_title='Giá cổ phiếu',
    hovermode='x unified'
)
st.plotly_chart(fig4, use_container_width=True)

# ====== Dự đoán 30 ngày tương lai ======
last_sequence = scaler.transform(df['Close'].tail(SEQUENCE_LENGTH).values.reshape(-1, 1))
future_input = list(last_sequence)
future_output = []

for _ in range(30):
    input_array = np.array(future_input[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
    pred = model.predict(input_array)[0][0]
    future_output.append(pred)
    future_input.append([pred])

# Scale lại về giá thật
future_output = np.array(future_output).reshape(-1, 1) * scale_factor
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

# ====== Biểu đồ dự đoán tương lai ======
st.subheader('Dự đoán giá 30 ngày tương lai')
fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=future_dates,
    y=future_output.flatten(),
    name='Dự đoán tương lai',
    line=dict(color='blue', dash='dash')
))
fig5.update_layout(
    title='Dự đoán 30 ngày tới',
    xaxis_title='Ngày',
    yaxis_title='Giá cổ phiếu',
    hovermode='x unified'
)
st.plotly_chart(fig5, use_container_width=True)
