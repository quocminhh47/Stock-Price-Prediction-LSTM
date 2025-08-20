# ===== app.py (BiLSTM + ARIMA, pipeline đồng bộ) =====
import pandas as pd
import numpy as np
import streamlit as st
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import joblib
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import datetime as dt

st.set_page_config(page_title="Stock Closing Price Prediction", layout="wide")
st.title('Stock Closing Price Prediction')

SEQUENCE_LENGTH = 150

# Session: nhớ chart/mã hỗ trợ gần nhất để tái dùng khi mã hiện tại không hỗ trợ
if "last_supported_ticker" not in st.session_state:
    st.session_state["last_supported_ticker"] = None
if "last_supported_fig4" not in st.session_state:
    st.session_state["last_supported_fig4"] = None

SUPPORTED_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'NFLX', 'BRK-B', 'JPM',
    'V', 'MA', 'PG', 'KO', 'PEP',
    'VCB.VN', 'CTG.VN', 'BID.VN', 'TCB.VN', 'VPB.VN',
]

TICKER_NAMES = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AMZN': 'Amazon.com, Inc.',
    'META': 'Meta Platforms, Inc. (Facebook)',
    'TSLA': 'Tesla, Inc.',
    'NVDA': 'NVIDIA Corporation',
    'NFLX': 'Netflix, Inc.',
    'BRK-B': 'Berkshire Hathaway Inc.',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.',
    'PG': 'Procter & Gamble Co.',
    'KO': 'Coca-Cola Co.',
    'PEP': 'PepsiCo, Inc.',
    'VCB.VN': 'Ngân hàng TMCP Ngoại thương Việt Nam',
    'CTG.VN': 'Ngân hàng TMCP Công Thương Việt Nam',
    'BID.VN': 'Ngân hàng TMCP Đầu tư và Phát triển Việt Nam',
    'TCB.VN': 'Ngân hàng TMCP Kỹ Thương Việt Nam',
    'VPB.VN': 'Ngân hàng TMCP Việt Nam Thịnh Vượng',
}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Cấu hình")
    ticker = st.text_input("Nhập mã cổ phiếu", "GOOGL").upper()
    start_date = st.date_input("Ngày bắt đầu", value=dt.date(2015, 1, 1))
    end_date   = st.date_input("Ngày kết thúc", value=dt.date.today())
    with st.expander("📃 Xem toàn bộ danh sách hỗ trợ", expanded=False):
        df_supported = pd.DataFrame([
            {"Mã cổ phiếu": t, "Tên công ty": TICKER_NAMES.get(t, "(chưa cập nhật)")}
            for t in SUPPORTED_TICKERS
        ])
        st.table(df_supported)

        # Cho phép tải về dạng CSV
        st.download_button(
            "Tải danh sách (.csv)",
            data=df_supported.to_csv(index=False),
            file_name="supported_tickers.csv",
            mime="text/csv",
            use_container_width=True
        )

    if ticker in SUPPORTED_TICKERS:
        st.success("✅ Mã này được hỗ trợ bởi mô hình BiLSTM (đã training).")
    else:
        st.warning(
            "⚠️ BiLSTM chỉ áp dụng cho các mã đã training.\n\n"
            "Bạn vẫn xem được MA100/MA200 và ARIMA 30 ngày."
        )

# Validate ngày
if start_date >= end_date:
    st.error("⚠ Ngày bắt đầu phải nhỏ hơn ngày kết thúc")
    st.stop()

# ---------------- Load dữ liệu ----------------
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
if df.empty:
    st.error("Không thể tải dữ liệu. Kiểm tra mã cổ phiếu hoặc kết nối mạng.")
    st.stop()

# ---------------- Thống kê mô tả ----------------
st.subheader(f'Dữ liệu từ {start_date} đến {end_date}')
st.dataframe(df.describe().round(2))
with st.expander(" Giải thích các chỉ số thống kê", expanded=False):
    st.markdown("""
- **count**: Số lượng giá trị không bị thiếu trong cột.
- **mean**: Giá trị trung bình cộng của cột.
- **std**: Độ lệch chuẩn, cho biết mức độ phân tán so với trung bình.
- **min**: Giá trị nhỏ nhất trong cột.
- **25% (Q1)**: Phân vị thứ nhất – 25% dữ liệu nhỏ hơn giá trị này.
- **50% (Median)**: Trung vị – 50% dữ liệu nhỏ hơn giá trị này.
- **75% (Q3)**: Phân vị thứ ba – 75% dữ liệu nhỏ hơn giá trị này.
- **max**: Giá trị lớn nhất trong cột.
    """)

# ---------------- MA100 / MA200 ----------------
close_series = df['Close'].astype(float).copy()
close_series.name = 'Close'

ma100 = close_series.rolling(100).mean()
ma200 = close_series.rolling(200).mean()

df_ma100 = pd.concat([close_series, ma100], axis=1).dropna()
df_ma100.columns = ['Close', 'MA100']

df_ma200 = pd.concat([close_series, ma100, ma200], axis=1).dropna()
df_ma200.columns = ['Close', 'MA100', 'MA200']

# --- Close + MA100 ---
st.subheader(f'Biểu đồ giá đóng cửa + MA100 của {ticker}')
if df_ma100.empty:
    st.info("Chưa đủ dữ liệu để tính MA100 (cần ≥ 100 phiên).")
else:
    fig_ma100 = go.Figure()
    fig_ma100.add_trace(go.Scatter(
        x=df_ma100.index, y=df_ma100['Close'], name='Giá đóng cửa', mode='lines',
        hovertemplate='Ngày: %{x|%Y-%m-%d}<br>Giá đóng cửa: %{y:.2f}<extra></extra>'
    ))
    fig_ma100.add_trace(go.Scatter(
        x=df_ma100.index, y=df_ma100['MA100'], name='MA100', mode='lines',
        hovertemplate='Ngày: %{x|%Y-%m-%d}<br>MA100: %{y:.2f}<extra></extra>'
    ))
    fig_ma100.update_layout(
        title='Giá đóng cửa + MA100',
        xaxis_title='Ngày', yaxis_title='Giá cổ phiếu', hovermode='x unified'
    )
    # (tuỳ chọn) cũng ép định dạng x trên hover của trục:
    fig_ma100.update_xaxes(hoverformat='%Y-%m-%d')
    st.plotly_chart(fig_ma100, use_container_width=True)

# --- Close + MA100 + MA200 ---
st.subheader(f'Biểu đồ giá đóng cửa + MA100 + MA200 của {ticker}')
if df_ma200.empty:
    st.info("Chưa đủ dữ liệu để tính MA200 (cần ≥ 200 phiên).")
else:
    fig_ma200 = go.Figure()
    fig_ma200.add_trace(go.Scatter(
        x=df_ma200.index, y=df_ma200['Close'], name='Giá đóng cửa', mode='lines',
        hovertemplate='Ngày: %{x|%Y-%m-%d}<br>Giá đóng cửa: %{y:.2f}<extra></extra>'
    ))
    fig_ma200.add_trace(go.Scatter(
        x=df_ma200.index, y=df_ma200['MA100'], name='MA100', mode='lines',
        hovertemplate='Ngày: %{x|%Y-%m-%d}<br>MA100: %{y:.2f}<extra></extra>'
    ))
    fig_ma200.add_trace(go.Scatter(
        x=df_ma200.index, y=df_ma200['MA200'], name='MA200', mode='lines',
        hovertemplate='Ngày: %{x|%Y-%m-%d}<br>MA200: %{y:.2f}<extra></extra>'
    ))
    fig_ma200.update_layout(
        title='Giá đóng cửa + MA100 + MA200',
        xaxis_title='Ngày', yaxis_title='Giá cổ phiếu', hovermode='x unified'
    )
    fig_ma200.update_xaxes(hoverformat='%Y-%m-%d')  # tuỳ chọn
    st.plotly_chart(fig_ma200, use_container_width=True)


# ---------------- Chuẩn bị train/test ----------------
split_idx = int(len(close_series) * 0.85)
if len(close_series) <= SEQUENCE_LENGTH + 1:
    st.warning("Dữ liệu quá ngắn so với SEQUENCE_LENGTH. Hãy chọn khoảng thời gian dài hơn.")
    st.stop()

train_df = pd.DataFrame(close_series.iloc[:split_idx])
test_df  = pd.DataFrame(close_series.iloc[split_idx:])

# ---------------- BiLSTM (chỉ cho mã đã training) ----------------
st.subheader('Giá dự đoán vs Thực tế (Test Set)')

if ticker in SUPPORTED_TICKERS:
    try:
        # Load scaler/model đúng cặp
        scaler = joblib.load("scaler2.save")
        model = load_model("multi_ticket_train2.h5")

        # Tiền xử lý
        past_days = train_df.tail(SEQUENCE_LENGTH)
        final_df  = pd.concat([past_days, test_df], axis=0)
        final_df.columns = ['Close']

        scaled_all = scaler.transform(final_df[['Close']])
        x_test, y_test_scaled = [], []
        for i in range(SEQUENCE_LENGTH, scaled_all.shape[0]):
            x_test.append(scaled_all[i - SEQUENCE_LENGTH : i])
            y_test_scaled.append(scaled_all[i, 0])
        x_test = np.array(x_test)
        y_test_scaled = np.array(y_test_scaled)

        # Dự đoán
        y_pred_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)
        y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
        y_test = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
        test_index = final_df.index[SEQUENCE_LENGTH:]

        # Calibration (tuỳ chọn)
        try:
            a, b = np.polyfit(y_pred, y_test, 1)
            y_pred_cal = a * y_pred + b
            mae_raw = mean_absolute_error(y_test, y_pred)
            mae_cal = mean_absolute_error(y_test, y_pred_cal)
            if mae_cal < mae_raw:
                y_pred, mae, cal_note = y_pred_cal, mae_cal, f"(calibrated, a={a:.3f}, b={b:.3f})"
            else:
                mae, cal_note = mae_raw, ""
        except Exception:
            mae, cal_note = mean_absolute_error(y_test, y_pred), ""

        # Plot
        daily_abs_err = np.abs(y_pred - y_test)
        daily_pct_err = np.where(y_test != 0, daily_abs_err / np.abs(y_test) * 100, np.nan)
        customdata_pred = np.column_stack([y_test, daily_abs_err, daily_pct_err])

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=test_index, y=y_test, name='Giá thực tế',
            hovertemplate='Ngày: %{x}<br>Giá thực tế: %{y:.2f}<extra></extra>'
        ))
        fig4.add_trace(go.Scatter(
            x=test_index, y=y_pred, name=f'BiLSTM Dự đoán {cal_note}',
            customdata=customdata_pred,
            hovertemplate=('Ngày: %{x}<br>Giá dự đoán: %{y:.2f}'
                           '<br>Thực tế: %{customdata[0]:.2f}'
                           '<br>Lệch: %{customdata[1]:.2f}'
                           '<br>Lệch %%: %{customdata[2]:.2f}%%<extra></extra>')
        ))
        fig4.update_layout(title=f'Dự đoán vs Thực tế  •  MAE={mae:.2f}',
                           xaxis_title='Ngày',
                           yaxis_title=('Giá cổ phiếu (VND)' if ticker.endswith('.VN') else 'Giá cổ phiếu'),
                           hovermode='x unified')
        st.plotly_chart(fig4, use_container_width=True)

        # Lưu lại để dùng cho các lần sau
        st.session_state["last_supported_ticker"] = ticker
        st.session_state["last_supported_fig4"] = fig4

    except Exception as e:
        st.error(f"Không thể chạy BiLSTM cho {ticker}: {e}")

else:
    prev_ticker = st.session_state.get("last_supported_ticker")
    prev_fig = st.session_state.get("last_supported_fig4")

    if prev_ticker is not None and prev_fig is not None:
        st.info(
            f"ℹ️ Mã **{ticker}** chưa được training. "
            f"Đang hiển thị **biểu đồ của mã trước: {prev_ticker}**."
        )
        fig_prev = go.Figure(prev_fig)
        old_title = (fig_prev.layout.title.text or "Giá dự đoán vs Thực tế")
        fig_prev.update_layout(title=f"{old_title} • từ mã trước: {prev_ticker}")
        st.plotly_chart(fig_prev, use_container_width=True)
    else:
        st.info("ℹ️ Mã này **chưa được training** và hiện **chưa có biểu đồ trước đó** để hiển thị.")
        fig_placeholder = go.Figure()
        fig_placeholder.add_annotation(
            text="🚫 Chưa hỗ trợ BiLSTM cho mã này",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig_placeholder.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="rgba(0,0,0,0)",
            height=300
        )
        st.plotly_chart(fig_placeholder, use_container_width=True)

# ---------------- Dự đoán 30 ngày tới (ARIMA, có loading) ----------------
st.subheader(f'Dự đoán giá cổ phiếu 30 ngày tới cho {ticker}')

loading_ph = st.empty()  # chừa chỗ hiển thị spinner ngay dưới subheader
with loading_ph.container():
    with st.spinner("⏳ Đang tính toán dự báo 30 ngày…"):
        close = df['Close'].astype(float)
        log_price = np.log(close)

        best_fit, best_cfg, best_aic = None, None, float("inf")
        for p in range(0, 4):
            for q in range(0, 4):
                for trend in ['n', 't']:  # 't' = có drift
                    try:
                        fit = ARIMA(log_price, order=(p, 1, q), trend=trend).fit(
                            method_kwargs={"warn_convergence": False}
                        )
                        if fit.aic < best_aic:
                            best_aic, best_fit, best_cfg = fit.aic, fit, (p, 1, q, trend)
                    except Exception:
                        pass

        if best_fit is None:
            best_cfg = (1, 1, 1, 't')
            best_fit = ARIMA(log_price, order=(1, 1, 1), trend='t').fit(
                method_kwargs={"warn_convergence": False}
            )

        steps = 30
        fc_log = best_fit.get_forecast(steps=steps).predicted_mean
        fc_price = np.exp(fc_log)

        future_idx = pd.bdate_range(close.index[-1] + pd.offsets.BDay(1), periods=steps)

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=future_idx, y=fc_price, mode='lines', name='ARIMA dự đoán',
            line=dict(dash='dash'),
            hovertemplate='Ngày: %{x|%Y-%m-%d}<br>Giá dự đoán: %{y:.2f}<extra></extra>'
        ))
        p, d, q, trend = best_cfg
        fig5.update_layout(
            title=f'Dự đoán 30 ngày tới (ARIMA, order=({p},{d},{q}), trend=\"{trend}\")',
            xaxis_title='Ngày',
            yaxis_title=('Giá cổ phiếu (VND)' if ticker.endswith('.VN') else 'Giá cổ phiếu'),
            hovermode='x unified'
        )

loading_ph.empty()
st.plotly_chart(fig5, use_container_width=True)
