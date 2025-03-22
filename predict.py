import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load historical stock data (2010 - 2022)
train_df = pd.read_csv("TSLA.csv")

# Fetch recent stock prices (2022 - 2025) from Yahoo Finance
recent_data = yf.download("TSLA", start="2022-03-25", end="2025-03-22")
recent_data.reset_index(inplace=True)

# Merge with old dataset
recent_data.rename(columns={"Date": "Date", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"}, inplace=True)
train_df = pd.concat([train_df, recent_data], ignore_index=True)

# Convert Date column to datetime and sort
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df.set_index('Date', inplace=True)
train_df = train_df.sort_index()

# Adjust prices for Tesla's 3-for-1 stock split in August 2022
split_date = pd.to_datetime("2022-08-25")
train_df.loc[train_df.index < split_date, ['Open', 'High', 'Low', 'Close']] /= 3  # Adjust pre-split prices

# Feature Engineering (No Rolling Averages)
train_df['Price_Change_1D'] = train_df['Close'] - train_df['Close'].shift(1)  
train_df['Price_Change_2D'] = train_df['Close'] - train_df['Close'].shift(2)  
train_df['Price_Change_3D'] = train_df['Close'] - train_df['Close'].shift(3)  
train_df['High_Low_Range'] = train_df['High'] - train_df['Low']  
train_df['Momentum_3'] = train_df['Close'] - train_df['Close'].shift(3)  

# Relative Strength Index (RSI) without Rolling
def compute_rsi(series, period=7):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

train_df['RSI_7'] = compute_rsi(train_df['Close'], 7)

# Target Variable (Predict next day's Open price movement)
train_df['Target'] = (train_df['Open'].shift(-1) - train_df['Open']) / train_df['Open']

# Fill missing values
train_df.ffill(inplace=True)
train_df.bfill(inplace=True)

# Drop NaN targets (only for training)
train_data = train_df.dropna(subset=['Target'])

# Select Features
features = ['Price_Change_1D', 'Price_Change_2D', 'Price_Change_3D', 'High_Low_Range', 'Momentum_3', 'RSI_7']
X_train = train_data[features]
y_train = train_data['Target']

# Train XGBoost Model
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=6, objective='reg:squarederror')
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "trained_model.pkl")

# Simulate March 24 to March 28, 2025
prediction_dates = pd.date_range("2025-03-24", "2025-03-28", freq="B")

# Load the trained model
model = joblib.load("trained_model.pkl")

# Fetch Tesla's actual March 21, 2025, stock price from Yahoo Finance
latest_price_data = yf.download("TSLA", start="2025-03-21", end="2025-03-22")
last_known_open = float(latest_price_data["Open"].values[0])  

# Set up starting conditions for the trading simulation
starting_balance = 10000
balance = starting_balance
shares_owned = 0
transaction_fee = 0.01  

# Function to generate Buy, Sell, or Hold signal
def generate_signals(predicted_pct_change, buy_threshold=0.005, sell_threshold=-0.010):
    if predicted_pct_change > buy_threshold:
        return "Buy"
    elif predicted_pct_change < sell_threshold:
        return "Sell"
    else:
        return "Hold"

# Function to execute the trade
def execute_trade(signal, execution_price, balance, shares_owned):
    if signal == "Buy":
        buy_amount = balance * 0.1  
        transaction_cost = buy_amount * transaction_fee
        shares_bought = buy_amount / execution_price
        balance -= (buy_amount + transaction_cost)  
        shares_owned += shares_bought
        order = f"Buy: ${buy_amount:.2f}"

    elif signal == "Sell" and shares_owned > 0:
        shares_to_sell = shares_owned * 0.2  
        sell_value = shares_to_sell * execution_price
        transaction_cost = sell_value * transaction_fee
        balance += (sell_value - transaction_cost)  
        shares_owned -= shares_to_sell
        order = f"Sell: {shares_to_sell:.2f} shares"
    else:
        order = "Hold"

    return balance, shares_owned, order

# Predict and Execute for Each Day
results = []

for trade_date in prediction_dates:
    latest_features = train_df[features].iloc[-1].values.reshape(1, -1)  

    # Predict % change for the target day (March 24 - March 28)
    predicted_pct_change = float(model.predict(latest_features)[0])  
    predicted_open_price = last_known_open * (1 + predicted_pct_change)

    # Generate trading signal (Buy, Sell, Hold)
    signal = generate_signals(predicted_pct_change)

    # Execute the trade at 10 AM
    balance, shares_owned, order = execute_trade(signal, predicted_open_price, balance, shares_owned)

    # Store results in a list
    results.append([trade_date.date(), predicted_open_price, predicted_pct_change, signal, order, balance])

    # Update last known Open price for the next day's prediction
    last_known_open = predicted_open_price  

# Convert results to a DataFrame and display as a table
results_df = pd.DataFrame(results, columns=["Date", "Predicted Open Price", "Predicted Change", "Signal", "Order", "Balance"])
results_df.set_index("Date", inplace=True)

from IPython.display import display
display(results_df)
