import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Constants
STARTING_BALANCE = 10000  # Initial trading capital
TRANSACTION_FEE = 0.01  # 1% fee per transaction
MAX_EXPECTED_CHANGE = 0.05  # Assume max expected daily pct change is 5%

# Function to compute the Relative Strength Index (RSI) without rolling
def compute_rsi(series, period=7):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def generate_signals(predicted_pct_change, buy_threshold=0.01, sell_threshold=-0.01):
    if predicted_pct_change > buy_threshold:
        return "Buy"
    elif predicted_pct_change < sell_threshold:
        return "Sell"
    else:
        return "Hold"

# Function to execute trades based on generated signals and confidence level
def execute_trade(signal, execution_price, balance, shares_owned, confidence=0.0):
    # Confidence-based invest ratio between 10% and 80%
    min_ratio = 0.1
    max_ratio = 0.8
    invest_ratio = min_ratio + (max_ratio - min_ratio) * confidence

    amt = (balance * invest_ratio) // execution_price  # Shares to buy based on confidence

    if signal == "Buy" and amt > 0:
        if balance - ((amt * execution_price) * (1 + TRANSACTION_FEE)) < 0:
            amt = max(0, amt - 1)

        buy_amount = amt * execution_price
        transaction_cost = buy_amount * TRANSACTION_FEE
        shares_bought = amt
        shares_owned += shares_bought
        order = f"Buy: {shares_bought} shares at ${execution_price:.2f} each."
        balance -= (buy_amount + transaction_cost)

    elif signal == "Sell" and shares_owned >= 1:
        shares_to_sell = max(1, shares_owned // 2)
        sell_value = shares_to_sell * execution_price
        transaction_cost = sell_value * TRANSACTION_FEE
        balance += (sell_value - transaction_cost)
        shares_owned -= shares_to_sell
        order = f"Sell: {shares_to_sell} shares at ${execution_price:.2f} each"

    else:
        order = "Hold"

    return balance, shares_owned, order

# Load historical Tesla stock data (2010 - 2022)
train_df = pd.read_csv("TSLA.csv")

# Fetch recent Tesla stock data (2024 - 2025) from Yahoo Finance
recent_data = yf.download("TSLA", start="2024-03-25", end="2025-03-22")
recent_data.reset_index(inplace=True)

# Merge historical and recent data
train_df = pd.concat([train_df, recent_data], ignore_index=True)
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df.set_index('Date', inplace=True)
train_df = train_df.sort_index()

# Adjust for Tesla's 3-for-1 stock split in August 2022
split_date = pd.to_datetime("2022-08-25")
train_df.loc[train_df.index < split_date, ['Open', 'High', 'Low', 'Close']] /= 3

# Feature Engineering (Momentum & Volatility)
train_df['Price_Change_1D'] = train_df['Close'].diff(1)
train_df['Price_Change_2D'] = train_df['Close'].diff(2)
train_df['Price_Change_3D'] = train_df['Close'].diff(3)
train_df['High_Low_Range'] = train_df['High'] - train_df['Low']
train_df['Momentum_3'] = train_df['Close'] - train_df['Close'].shift(3)
train_df['RSI_7'] = compute_rsi(train_df['Close'], 7)

# Target Variable: Predict next day's Open price movement
train_df['Target'] = (train_df['Open'].shift(-1) - train_df['Open']) / train_df['Open']

# Handle missing values
train_df.ffill(inplace=True)
train_df.bfill(inplace=True)

# Prepare training data
train_data = train_df.dropna(subset=['Target'])
features = ['Price_Change_1D', 'Price_Change_2D', 'Price_Change_3D', 'High_Low_Range', 'Momentum_3', 'RSI_7']
X_train = train_data[features]
y_train = train_data['Target']

# Train XGBoost Regression Model
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=6, objective='reg:squarederror')
model.fit(X_train, y_train)
joblib.dump(model, "trained_model.pkl")

# Simulate trading from March 24 to March 28, 2025
prediction_dates = pd.date_range("2025-03-24", "2025-03-28", freq="B")
model = joblib.load("trained_model.pkl")

# Fetch Tesla's last known stock price from March 21, 2025
latest_price_data = yf.download("TSLA", start="2025-03-21", end="2025-03-22")
last_known_open_price = float(latest_price_data["Open"].values[0])

# === START Threshold Optimization === #
best_balance = 0
best_thresholds = (0, 0)
best_results = []

buy_thresholds = np.arange(0.005, 0.03, 0.005)  # From 0.5% to 3%
sell_thresholds = np.arange(-0.02, -0.005, 0.005)  # From -2% to -0.5%

for buy in buy_thresholds:
    for sell in sell_thresholds:
        balance = STARTING_BALANCE
        shares_owned = 0
        last_known_open = last_known_open_price
        temp_results = []

        for trade_date in prediction_dates:
            latest_features = train_df[features].iloc[-1].values.reshape(1, -1)
            predicted_pct_change = float(model.predict(latest_features)[0])
            predicted_open_price = last_known_open * (1 + predicted_pct_change)

            # Calculate confidence based on magnitude of prediction
            confidence = min(abs(predicted_pct_change) / MAX_EXPECTED_CHANGE, 1.0)

            signal = generate_signals(predicted_pct_change, buy_threshold=buy, sell_threshold=sell)

            balance, shares_owned, order = execute_trade(
                signal,
                predicted_open_price,
                balance,
                shares_owned,
                confidence=confidence
            )

            temp_results.append([
                trade_date.date(),
                f"${predicted_open_price:.2f}",
                f"{predicted_pct_change:.3f}",
                f"{confidence:.3f}",
                signal,
                order,
                f"${balance:.2f}",
                shares_owned,
                f"${shares_owned * predicted_open_price:.2f}",
                f"${balance + (shares_owned * predicted_open_price):.2f}"
            ])

            last_known_open = predicted_open_price  # Update for next prediction

        total_balance = balance + (shares_owned * last_known_open)

        if total_balance > best_balance:
            best_balance = total_balance
            best_thresholds = (buy, sell)
            best_results = temp_results

print(f"Best thresholds found: Buy > {best_thresholds[0]:.3f}, Sell < {best_thresholds[1]:.3f} | Total Balance: ${best_balance:.2f}")

# === END Threshold Optimization === #

# Convert best results to DataFrame and display
results_df = pd.DataFrame(best_results, columns=[
    "Date",
    "Predicted_Open_Price",
    "Predicted_Change",
    "Confidence",
    "Signal",
    "Order",
    "Balance",
    "Shares_Owned",
    "Tesla_Capital",
    "Total_Balance"
])
results_df.set_index("Date", inplace=True)

from IPython.display import display
display(results_df)
