import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

def get_upbit_ohlcv(ticker='KRW-BTC', count=180, candle_unit='15m'):
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

def detect_pullback_pattern(df, ma_period=15, up_count=3, pullback_count=2):
    if len(df) < (ma_period + up_count + pullback_count + 1):
        return []
    df['ma'] = df['close'].rolling(ma_period).mean()
    df['above_ma'] = df['close'] > df['ma']
    df['is_bull'] = df['close'] > df['open']
    df['is_bear'] = df['close'] < df['open']
    signals = []
    for i in range(ma_period + up_count + pullback_count, len(df)):
        if not df['above_ma'].iloc[i - up_count - pullback_count:i].all():
            continue
        if not df['is_bull'].iloc[i - pullback_count - up_count:i - pullback_count].all():
            continue
        if not (df['is_bear'].iloc[i - pullback_count:i].sum() >= 1):
            continue
        if not df['is_bull'].iloc[i]:
            continue
        signals.append(i)
    return signals

def detect_sell_pattern(df, ma_period=15, down_count=3, rebound_count=2):
    if len(df) < (ma_period + down_count + rebound_count + 1):
        return []
    df['ma'] = df['close'].rolling(ma_period).mean()
    df['below_ma'] = df['close'] < df['ma']
    df['is_bull'] = df['close'] > df['open']
    df['is_bear'] = df['close'] < df['open']
    signals = []
    for i in range(ma_period + down_count + rebound_count, len(df)):
        if not df['below_ma'].iloc[i - down_count - rebound_count:i].all():
            continue
        if not df['is_bear'].iloc[i - rebound_count - down_count:i - rebound_count].all():
            continue
        if not (df['is_bull'].iloc[i - rebound_count:i].sum() >= 1):
            continue
        if not df['is_bull'].iloc[i]:
            continue
        signals.append(i)
    return signals

def get_position_states(df, buy_signals, sell_signals):
    position_states = []
    state = '관망'
    for i in range(len(df)):
        if i in buy_signals:
            state = '매수'
        elif i in sell_signals:
            state = '매도'
        position_states.append(state)
    return position_states

def predict_next_n_var(df, n=2, columns=['open','high','low','close','volume'], maxlags=5):
    # 결측치 제거
    subdf = df[columns].dropna()
    model = VAR(subdf)
    fit = model.fit(maxlags=maxlags)
    # 마지막 부분의 데이터로 예측
    forecast_input = subdf.values[-fit.k_ar:]
    forecast = fit.forecast(y=forecast_input, steps=n)
    return forecast  # shape: (n, len(columns))

if __name__ == "__main__":
    df = get_upbit_ohlcv("KRW-XRP", 200, "15m")
    buy_signals = detect_pullback_pattern(df)
    sell_signals = detect_sell_pattern(df)
    states = get_position_states(df, buy_signals, sell_signals)
    df['state'] = states

    # VAR로 미래 2개 캔들 예측 (open, high, low, close, volume 모두)
    columns = ['open','high','low','close','volume']
    preds = predict_next_n_var(df, n=2, columns=columns, maxlags=5)
    last_date = pd.to_datetime(df['date'].iloc[-1])
    last_state = df['state'].iloc[-1]
    pred_rows = []
    for i, pred in enumerate(preds):
        next_date = last_date + pd.Timedelta(minutes=15*(i+1))
        pred_row = {
            'date': next_date.strftime('%Y-%m-%d %H:%M:%S'),
            'open': pred[0],
            'high': pred[1],
            'low': pred[2],
            'close': pred[3],
            'volume': pred[4],
            'state': last_state  # 예측구간은 직전 상태 유지
        }
        pred_rows.append(pred_row)
    df_pred = pd.DataFrame(pred_rows)
    df_final = pd.concat([df, df_pred], ignore_index=True)

    print(df_final.tail(10))
