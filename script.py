import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# 1. Descargar datos
# -------------------------
def descargar_datos(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)

        if df is None or df.empty:
            print(f"⚠️ No data for {ticker}")
            return None

        df["Ticker"] = ticker
        df = df.reset_index()
        return df

    except Exception as e:
        print(f"❌ Error descargando {ticker}: {e}")
        return None


# -------------------------
# 2. Indicadores
# -------------------------
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


# -------------------------
# 3. Target
# -------------------------
def crear_target(df):
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


# -------------------------
# 4. Entrenar modelo
# -------------------------
def entrenar_modelo(df_all):
    df_all = df_all.dropna()

    if df_all.empty:
        raise ValueError("No hay datos para entrenar el modelo")

    features = ["SMA_20", "SMA_50", "RSI", "RET_5"]

    X = df_all[features]
    y = df_all["Target"]

    split = int(len(df_all) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"📊 Accuracy: {acc:.2f}")

    return model


# -------------------------
# 5. Predicciones
# -------------------------
def predecir_acciones(model, dfs):
    resultados = []

    for ticker, df in dfs.items():
        try:
            df = df.dropna()

            if df.empty:
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
                "Prob_Subida": round(prob, 3),
                "Señal": señal
            })

        except Exception as e:
            print(f"❌ Error prediciendo {ticker}: {e}")

    if len(resultados) == 0:
        raise ValueError("No se generaron resultados")

    return pd.DataFrame(resultados).sort_values(by="Prob_Subida", ascending=False)


# -------------------------
# 6. MAIN
# -------------------------
if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "TSLA", "SPY", "NVDA"]

    dfs = {}
    df_all = pd.DataFrame()

    print("📥 Descargando datos...")

    for t in tickers:
        df = descargar_datos(t)

        if df is None:
            continue

        try:
            df = indicadores(df)
            df = crear_target(df)

            dfs[t] = df
            df_all = pd.concat([df_all, df])

            print(f"✅ {t} listo")

        except Exception as e:
            print(f"❌ Error procesando {t}: {e}")

    print("\n🤖 Entrenando modelo...")
    model = entrenar_modelo(df_all)

    print("\n📊 Generando ranking...")
    ranking = predecir_acciones(model, dfs)

    print("\n📈 RESULTADO FINAL:")
    print(ranking)

    # Guardar resultado
    ranking.to_csv("ranking.csv", index=False)
    print("\n💾 Archivo guardado: ranking.csv")
