import requests
import pandas as pd

def get_candlestick_data(symbol="BTCUSDT", interval="1h", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()

    # Transformar em DataFrame
    columns = ["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"]
    df = pd.DataFrame(data, columns=columns)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

df = get_candlestick_data()
print(df.head())
