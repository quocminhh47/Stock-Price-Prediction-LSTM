# ===== File: app.py (Updated with BiLSTM + ARIMA) =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import yfinance as yf
import plotly.graph_objects as go
from datetime import timedelta, date
from statsmodels.tsa.arima.model import ARIMA

# ====== Cấu hình ======
SEQUENCE_LENGTH = 150
START_DATE = "2009-01-01"
END_DATE = date.today()

st.set_page_config(page_title="Stock Closing Price Prediction", layout="wide")
st.title('Stock Closing Price Prediction')

with st.sidebar:
    st.header("Cấu hình")
    ticker = st.text_input("Nhập mã cổ phiếu", "GOOGL")

# ====== Tải dữ liệu ======
df = yf.download(ticker, start=START_DATE, end=END_DATE)
if df.empty:
    st.error("Không thể tải dữ liệu. Kiểm tra mã cổ phiếu hoặc kết nối mạng.")
    st.stop()

st.subheader(f'Dữ liệu từ {START_DATE} đến {END_DATE}')
st.write(df.describe())

st.subheader('Biểu đồ giá đóng cửa')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig1)

# ===== Trung bình động =====
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

# ===== Train/Test =====
train_df = pd.DataFrame(df['Close'][0:int(len(df) * 0.85)])
test_df = pd.DataFrame(df['Close'][int(len(df) * 0.85):])

scaler = joblib.load("scaler.save")
model = load_model("bilstm_model_150.h5")

# ===== Tiền xử lý dữ liệu =====
past_days = train_df.tail(SEQUENCE_LENGTH)
final_df = past_days._append(test_df, ignore_index=True)
input_data = scaler.transform(final_df)

x_test, y_test = [], []
for i in range(SEQUENCE_LENGTH, input_data.shape[0]):
    x_test.append(input_data[i - SEQUENCE_LENGTH:i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# ===== Dự đoán BiLSTM =====
y_pred = model.predict(x_test)
scale = scaler.scale_
scale_factor = 1 / scale[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

mae = mean_absolute_error(y_test, y_pred.flatten())
daily_abs_err = np.abs(y_pred.flatten() - y_test)
daily_pct_err = np.where(y_test != 0, daily_abs_err / np.abs(y_test) * 100, np.nan)
customdata_pred = np.column_stack([y_test, daily_abs_err, daily_pct_err])

st.subheader('Giá dự đoán vs Thực tế (Test Set)')
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=test_df.index[-len(y_test):], y=y_test, name='Giá thực tế',
    line=dict(color='green'),
    hovertemplate='Ngày: %{x}<br>Giá thực tế: %{y:.2f}<extra></extra>'
))
fig4.add_trace(go.Scatter(
    x=test_df.index[-len(y_test):], y=y_pred.flatten(), name='BiLSTM Dự đoán',
    line=dict(color='red'), customdata=customdata_pred,
    hovertemplate='Ngày: %{x}<br>Giá dự đoán: %{y:.2f}<br>Thực tế: %{customdata[0]:.2f}<br>Lệch: %{customdata[1]:.2f}<br>Lệch %%: %{customdata[2]:.2f}%%<extra></extra>'
))
fig4.update_layout(title='Dự đoán vs Thực tế', xaxis_title='Ngày', yaxis_title='Giá cổ phiếu', hovermode='x unified')
st.plotly_chart(fig4, use_container_width=True)

# ===== Dự đoán 30 ngày với ARIMA =====
st.subheader('Dự đoán giá 30 ngày tương lai (ARIMA)')

model_arima = ARIMA(df['Close'], order=(5, 1, 0))
model_arima_fit = model_arima.fit()
forecast = model_arima_fit.forecast(steps=30)
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=future_dates, y=forecast, name='ARIMA Dự đoán',
    line=dict(color='blue', dash='dash')
))
fig5.update_layout(title='Dự đoán 30 ngày tới (ARIMA)', xaxis_title='Ngày', yaxis_title='Giá cổ phiếu', hovermode='x unified')
st.plotly_chart(fig5, use_container_width=True)

st.caption(f"MAE (BiLSTM): {mae:.2f}")
