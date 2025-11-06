# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import base64
from io import BytesIO

st.set_page_config(page_title="48h Temperature Forecast", layout="wide")
st.title("48-Hour Temperature Forecast")

# ---------------- Helpers ----------------
@st.cache_resource
def load_model(path="xgb_model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Could not load model '{path}': {e}")
        return None

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")

def download_link(df: pd.DataFrame, filename="forecast_48h.csv"):
    b64 = base64.b64encode(to_csv_bytes(df)).decode()
    return f"data:file/csv;base64,{b64}"

def simulate_ssr(hour, amplitude=800):
    # Simple diurnal proxy: peak midday, zero at night
    return max(0.0, np.sin((hour - 6) / 24.0 * 2 * np.pi)) * amplitude

# ---------------- Sidebar ----------------
st.sidebar.header("Input & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV (feature rows, last row used)", type=["csv"])
horizon = st.sidebar.number_input("Forecast horizon (hours)", min_value=1, max_value=168, value=48, step=1)
confidence = st.sidebar.number_input("Confidence band (± °C)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
show_table = st.sidebar.checkbox("Show forecast table", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("CSV should contain the same feature columns used for training (lag, rolling, and time features).")

# ---------------- Load model ----------------
model = load_model("xgb_model.pkl")
if model is None:
    st.stop()

# Try to get feature names from booster
try:
    model_feature_names = model.get_booster().feature_names
except Exception:
    model_feature_names = None

# ---------------- Upload / prepare input ----------------
if uploaded is None:
    st.info("Please upload a CSV containing recent feature rows (the last row will be used as the starting point).")
    st.stop()

try:
    df_input = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Error reading uploaded CSV: {e}")
    st.stop()

st.subheader("Uploaded data (last row will be used)")
st.dataframe(df_input.tail(5))

start_row = df_input.iloc[[-1]].copy()  # last row as dataframe

# validate model features (best-effort)
if model_feature_names is not None:
    missing = [c for c in model_feature_names if c not in start_row.columns]
    extra = [c for c in start_row.columns if c not in model_feature_names]
    st.write(f"Model expects {len(model_feature_names)} features.")
    if missing:
        st.warning(f"{len(missing)} required columns are missing (example: {missing[:6]})")
    else:
        st.success("All required model features are present.")
    if extra:
        st.info(f"Uploaded CSV has {len(extra)} extra columns (they will be ignored).")

# prepare exogenous history (last 24 rows if present)
exog_candidates = ['ssr','hcc','lcc','mcc','u10','v10','sp']
exog_cols = [c for c in exog_candidates if c in df_input.columns]
exog_hist = None
if len(exog_cols) > 0 and len(df_input) >= 24:
    exog_hist = df_input[exog_cols].iloc[-24:].reset_index(drop=True)

# ---------------- Forecast button ----------------
if st.button("Run Forecast"):
    # Prepare current_row matching model feature order if possible
    if model_feature_names is not None:
        current_row = pd.DataFrame([{c: start_row[c].values[0] if c in start_row.columns else np.nan for c in model_feature_names}])
    else:
        current_row = start_row.copy()

    # ensure numeric where applicable
    for col in current_row.columns:
        try:
            current_row[col] = pd.to_numeric(current_row[col])
        except Exception:
            pass

    # find last timestamp if available
    last_time = None
    if 'valid_time' in df_input.columns:
        try:
            last_time = pd.to_datetime(df_input['valid_time'].iloc[-1])
        except Exception:
            last_time = pd.Timestamp.now()
    elif 'datetime' in df_input.columns:
        try:
            last_time = pd.to_datetime(df_input['datetime'].iloc[-1])
        except Exception:
            last_time = pd.Timestamp.now()
    else:
        last_time = pd.Timestamp.now()

    future_times = []
    future_preds = []
    exog_idx = 0

    # recursive forecast
    for i in range(1, int(horizon)+1):
        try:
            pred = model.predict(current_row.values)[0]
        except Exception as e:
            st.error(f"Prediction error at step {i}: {e}")
            pred = np.nan

        next_time = last_time + pd.Timedelta(hours=i)
        future_times.append(next_time)
        future_preds.append(pred)

        # update t2m lags
        for k in range(48, 1, -1):
            to_col, from_col = f"t2m_lag_{k}", f"t2m_lag_{k-1}"
            if from_col in current_row.columns and to_col in current_row.columns:
                current_row.iloc[0, current_row.columns.get_loc(to_col)] = current_row.iloc[0, current_row.columns.get_loc(from_col)]
        if 't2m_lag_1' in current_row.columns:
            current_row.iloc[0, current_row.columns.get_loc('t2m_lag_1')] = pred

        # update rolling stats
        lag_24 = [f't2m_lag_{j}' for j in range(1,25) if f't2m_lag_{j}' in current_row.columns]
        if lag_24:
            vals = current_row[lag_24].values.flatten().astype(float)
            if 't2m_roll_mean_24' in current_row.columns:
                current_row.iloc[0, current_row.columns.get_loc('t2m_roll_mean_24')] = np.nanmean(vals)
            if 't2m_roll_std_24' in current_row.columns:
                current_row.iloc[0, current_row.columns.get_loc('t2m_roll_std_24')] = np.nanstd(vals)

        # update exogenous values: rotate history if available, otherwise simulate ssr + small jitter
        if exog_hist is not None:
            hist_row = exog_hist.iloc[exog_idx % len(exog_hist)]
            for c in exog_cols:
                if c in current_row.columns:
                    current_row.iloc[0, current_row.columns.get_loc(c)] = hist_row[c]
            exog_idx += 1
        else:
            if 'ssr' in current_row.columns:
                current_row.iloc[0, current_row.columns.get_loc('ssr')] = simulate_ssr(next_time.hour)
            # small noise for clouds/pressure/wind
            for c in ['hcc','lcc','mcc','sp','u10','v10']:
                if c in current_row.columns:
                    base = float(current_row.iloc[0, current_row.columns.get_loc(c)])
                    if 'cc' in c or 'mcc' in c:
                        jitter = np.random.uniform(-0.03, 0.03)
                        current_row.iloc[0, current_row.columns.get_loc(c)] = np.clip(base + jitter, 0, 1)
                    else:
                        jitter = np.random.uniform(-0.5, 0.5)
                        current_row.iloc[0, current_row.columns.get_loc(c)] = base + jitter

        # update time features
        if 'hour' in current_row.columns:
            current_row.iloc[0, current_row.columns.get_loc('hour')] = next_time.hour
        if 'day' in current_row.columns:
            current_row.iloc[0, current_row.columns.get_loc('day')] = next_time.day
        if 'month' in current_row.columns:
            current_row.iloc[0, current_row.columns.get_loc('month')] = next_time.month
        if 'day_of_week' in current_row.columns:
            current_row.iloc[0, current_row.columns.get_loc('day_of_week')] = next_time.dayofweek
        if 'year' in current_row.columns:
            current_row.iloc[0, current_row.columns.get_loc('year')] = next_time.year

    # build forecast dataframe
    forecast_df = pd.DataFrame({'valid_time': future_times, 'predicted_temperature(°C)': future_preds})
    forecast_df = forecast_df.set_index('valid_time')

    st.subheader("Forecast Summary")
    st.write(f"Forecast horizon: {horizon} hours (starting from {last_time})")
    if show_table:
        st.dataframe(forecast_df)

    # Plot observed last 24h if available + forecast
    plt.figure(figsize=(12,5))
    if 'valid_time' in df_input.columns and 't2m' in df_input.columns:
        try:
            obs = df_input.set_index(pd.to_datetime(df_input['valid_time']))['t2m']
            obs_last = obs.iloc[-24:]
            plt.plot(obs_last.index, obs_last.values, label='Observed (last 24h)', color='blue', linewidth=2)
        except Exception:
            pass

    plt.plot(forecast_df.index, forecast_df['predicted_temperature(°C)'], label='Forecast', color='orange', linestyle='--', linewidth=2)

    upper = forecast_df['predicted_temperature(°C)'] + confidence
    lower = forecast_df['predicted_temperature(°C)'] - confidence
    plt.fill_between(forecast_df.index, lower, upper, color='orange', alpha=0.15, label=f'±{confidence}°C')

    plt.title('48-Hour Temperature Forecast')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(plt)

    # Download link
    dl = download_link(forecast_df, filename="forecast_48h.csv")
    st.markdown(f"[⬇️ Download forecast CSV]({dl})", unsafe_allow_html=True)

    st.success("Forecast generated successfully.")
