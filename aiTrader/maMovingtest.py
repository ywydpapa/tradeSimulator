import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_upbit_ohlcv(ticker='KRW-BTC', count=200, candle_unit='15m'):
    candle_map = {
        '1d': ('days', ''),
        '4h': ('minutes', 240),
        '1h': ('minutes', 60),
        '30m': ('minutes', 30),
        '15m': ('minutes', 15),
        '10m': ('minutes', 10),
        '5m': ('minutes', 5),
        '3m': ('minutes', 3),
        '1m': ('minutes', 1),
    }
    if candle_unit not in candle_map:
        raise ValueError(f"지원하지 않는 단위입니다: {candle_unit}")
    api_type, minute = candle_map[candle_unit]
    if api_type == 'days':
        url = f'https://api.upbit.com/v1/candles/days?market={ticker}&count={count}'
    else:
        url = f'https://api.upbit.com/v1/candles/minutes/{minute}?market={ticker}&count={count}'
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"API 요청 실패: {response.status_code}, {response.text}")
    data = response.json()
    df = pd.DataFrame(data)
    df = df.rename(columns={
        'candle_date_time_kst': 'date',
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    })
    needed_cols = [col for col in ['date', 'open', 'high', 'low', 'close', 'volume'] if col in df.columns]
    df = df[needed_cols]
    df = df[::-1].reset_index(drop=True)
    return df

def predict_next_n_var(train_df, n=2, columns=['open','high','low','close','volume'], maxlags=5):
    subdf = train_df[columns].dropna()
    model = VAR(subdf)
    fit = model.fit(maxlags=maxlags)
    forecast_input = subdf.values[-fit.k_ar:]
    forecast = fit.forecast(y=forecast_input, steps=n)
    return forecast  # shape: (n, len(columns))

def eval_var_predict(df, columns=['open','high','low','close','volume'], train_size=10, pred_steps=2):
    # 최근 train_size+pred_steps 만큼 사용
    subdf = df[columns].dropna().iloc[-(train_size+pred_steps):]
    train = subdf.iloc[:train_size]
    test = subdf.iloc[train_size:train_size+pred_steps]
    preds = predict_next_n_var(train, n=pred_steps, columns=columns, maxlags=3)
    preds_df = pd.DataFrame(preds, columns=columns, index=test.index)

    results = {}
    for col in columns:
        mae = mean_absolute_error(test[col], preds_df[col])
        mse = mean_squared_error(test[col], preds_df[col])
        results[col] = {'MAE': mae, 'MSE': mse}
    return results, preds_df, test

if __name__ == "__main__":
    df = get_upbit_ohlcv("KRW-XRP", 200, "10m")
    columns = ['open','high','low','close','volume']
    results, preds_df, test_df = eval_var_predict(df, columns=columns, train_size=15, pred_steps=2)

    print("실제값:")
    print(test_df)
    print("\n예측값:")
    print(preds_df)
    print("\n평가 결과 (MAE, MSE):")
    for col, metrics in results.items():
        print(f"{col}: MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}")
