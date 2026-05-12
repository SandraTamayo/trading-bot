import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def descargar_datos(ticker, period="2y"):
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    df["Ticker"] = ticker
    return df

def indicadores(df):
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["RET_5"] = df["Close"].pct_change(5)
    return df

def crear_target(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df

def entrenar_modelo(df_all):
    df_all = df_all.dropna()
    features = ["SMA_20", "SMA_50", "RSI", "RET_5"]

    X = df_all[features]
    y = df_all["Target"]

    split = int(len(df_all) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    print("Accuracy:", model.score(X_test, y_test))
    return model

def predecir_acciones(model, dfs):
    resultados = []

    for ticker, df in dfs.items():
        df = df.dropna()
        if len(df) == 0:
            continue

        last = df.iloc[-1]
        X = last[["SMA_20", "SMA_50", "RSI", "RET_5"]].values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]

        if prob > 0.6:
            señal = "COMPRAR"
        elif prob < 0.4:
            señal = "VENDER"
        else:
            señal = "MANTENER"

        resultados.append({
            "Ticker": ticker,
            "Prob_Subida": prob,
            "Señal": señal
        })

    return pd.DataFrame(resultados).sort_values(by="Prob_Subida", ascending=False)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "NVDA"]

    dfs = {}
    df_all = pd.DataFrame()

    for t in tickers:
        df = descargar_datos(t)
        df = indicadores(df)
        df = crear_target(df)

        dfs[t] = df
        df_all = pd.concat([df_all, df])

    model = entrenar_modelo(df_all)
    ranking = predecir_acciones(model, dfs)

    print("\nRanking:")
    print(ranking)

    ranking.to_csv("ranking.csv", index=False)
