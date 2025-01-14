import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- 1. Coleta de Dados ---
def get_candlestick_data(symbol="BTCUSDT", interval="1h", limit=500):
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

# --- 2. Cálculo de Indicadores Técnicos ---
def calculate_indicators(df):
    df["SMA_50"] = df["close"].rolling(window=50).mean()
    df["SMA_200"] = df["close"].rolling(window=200).mean()
    df["RSI"] = calculate_rsi(df["close"], window=14)
    df.dropna(inplace=True)  # Remove linhas com valores NaN
    return df

def calculate_rsi(series, window=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 3. Definição do Alvo ---
def define_target(df):
    # Prever o preço futuro (próximo fechamento)
    df["future_price"] = df["close"].shift(-1)
    df.dropna(inplace=True)
    return df

# --- 4. Treinamento do Modelo ---
def train_model(df, features):
    X = df[features]
    y = df["future_price"]  # Preço futuro como alvo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Avaliação do modelo
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Erro médio absoluto: {mae:.2f}")
    return model

# --- 5. Tomada de Decisão ---
def make_decision(latest_price, predicted_price):
    if predicted_price > latest_price:
        return f"COMPRAR agora por {latest_price:.2f} e vender por {predicted_price:.2f}"
    else:
        return f"ESPERAR, pois o preço futuro esperado é {predicted_price:.2f}"

# --- 6. Análise em Tempo Real ---
def real_time_analysis(model, features, symbol="BTCUSDT"):
    df = get_candlestick_data(symbol)
    df = calculate_indicators(df)

    latest_data = df[features].iloc[-1:].dropna()

    if latest_data.empty:
        return "Dados insuficientes para análise."

    latest_price = df["close"].iloc[-1]
    predicted_price = model.predict(latest_data)[0]
    decision = make_decision(latest_price, predicted_price)
    return decision

# --- Execução Principal ---
if __name__ == "__main__":
    # Coleta de dados
    df = get_candlestick_data()

    # Cálculo de indicadores e definição do alvo
    df = calculate_indicators(df)
    df = define_target(df)

    # Seleção de recursos (features)
    features = ["close", "SMA_50", "SMA_200", "RSI"]

    # Treinamento do modelo
    model = train_model(df, features)

    # Análise em tempo real
    decision = real_time_analysis(model, features, symbol="BTCUSDT")
    print(decision)
