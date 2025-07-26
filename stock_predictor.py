import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Load dataset
df = pd.read_csv('C:\\Users\\yashi\\OneDrive\\Desktop\\stock-price-predictor\\NSE-Tata-Global-Beverages-Limited.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Close']].dropna()
df.sort_values('Date', inplace=True)

# Convert Date to numeric ordinal
df['Date_Ordinal'] = df['Date'].map(lambda date: date.toordinal())

# Features and Target
X = df[['Date_Ordinal']]
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions for test
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# ✅ Predict Next 7 Days
last_date = df['Date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
future_ordinals = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
future_predictions = model.predict(future_ordinals)

# Show predicted future values
for i in range(7):
    print(f"{future_dates[i].date()} ➜ Predicted Close: ₹{future_predictions[i]:.2f}")

# ✅ Plotting
plt.figure(figsize=(14,6))

# Plot actual data
plt.plot(df['Date'], y, label='Actual Price')

# Plot model predictions on test data
plt.plot(df['Date'].iloc[-len(y_test):], y_pred, label='Predicted Price (Test)', linestyle='--')

# Plot future predictions
plt.plot(future_dates, future_predictions, label='Next 7 Days Forecast', marker='o', linestyle='dashed', color='green')

plt.title('Stock Price Prediction (Next 7 Days Forecast Included)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
