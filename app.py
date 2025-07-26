import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction (Next 7 Days)")

# File uploader
uploaded_file = st.file_uploader("C:\\Users\\yashi\\OneDrive\\Desktop\\stock-price-predictor\\NSE-Tata-Global-Beverages-Limited.csv", type=['csv'])

# Load default if no file
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    df = pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
    st.info("Using default dataset: NSE-Tata-Global-Beverages-Limited.csv")

# Check necessary columns
if 'Date' not in df.columns or 'Close' not in df.columns:
    st.error("CSV must contain 'Date' and 'Close' columns.")
    st.stop()

# Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']].dropna()
df.sort_values('Date', inplace=True)
df['Date_Ordinal'] = df['Date'].map(lambda date: date.toordinal())

# Train model
X = df[['Date_Ordinal']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Predict next 7 days
last_date = df['Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
future_ordinals = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
future_predictions = model.predict(future_ordinals)

# Show future predictions
st.subheader("📅 Predicted Prices for Next 7 Days:")
future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close Price (₹)": np.round(future_predictions, 2)
})
st.dataframe(future_df)

# Plot
fig, ax = plt.subplots(figsize=(14,6))
ax.plot(df['Date'], y, label="Actual Price")
ax.plot(df['Date'].iloc[-len(y_test):], y_pred, label="Predicted Price", linestyle='--')
ax.plot(future_dates, future_predictions, marker='o', label="Next 7 Days", linestyle='dashed', color='green')
ax.set_title("Stock Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.caption(f"Model used: Linear Regression | MSE: {mse:.2f}")
