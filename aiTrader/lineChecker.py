import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


def peak_trade(
        ticker='KRW-BTC',
        short_window=3,
        mid_window=3,
        long_window=20,
        count=180,
        candle_unit='1h'
):
    # 0. 캔들 단위 변환
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

    # 1. 데이터 가져오기
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df['candle_date_time_kst'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
    df.set_index('candle_date_time_kst', inplace=True)
    df = df.sort_index(ascending=True)
    df = df[['trade_price', 'candle_acc_trade_volume']]

    # 2. VWMA 및 MA 계산
    df[f'VWMA_{short_window}'] = (
            (df['trade_price'] * df['candle_acc_trade_volume'])
            .rolling(window=short_window).sum() /
            df['candle_acc_trade_volume'].rolling(window=short_window).sum()
    )
    df[f'VWMA_{mid_window}'] = (
            (df['trade_price'] * df['candle_acc_trade_volume'])
            .rolling(window=mid_window).sum() /
            df['candle_acc_trade_volume'].rolling(window=mid_window).sum()
    )
    df[f'VWMA_{long_window}'] = (
            (df['trade_price'] * df['candle_acc_trade_volume'])
            .rolling(window=long_window).sum() /
            df['candle_acc_trade_volume'].rolling(window=long_window).sum()
    )
    df[f'MA_{short_window}'] = df['trade_price'].rolling(window=short_window).mean()
    df[f'MA_{mid_window}'] = df['trade_price'].rolling(window=mid_window).mean()
    df[f'MA_{long_window}'] = df['trade_price'].rolling(window=long_window).mean()

    # === 상승/하강봉 평균 변화율 계산 추가 ===
    df['prev_price'] = df['trade_price'].shift(1)
    df['change'] = df['trade_price'] - df['prev_price']
    df['rate'] = (df['trade_price'] - df['prev_price']) / df['prev_price']

    up_candles = df[df['change'] > 0]
    down_candles = df[df['change'] < 0]

    avg_up_rate = up_candles['rate'].mean() * 100  # %
    avg_down_rate = down_candles['rate'].mean() * 100  # %
    spike_rate = (abs(avg_down_rate/100)*2.5) #평균하강율의 3.5배를 급변으로 인식

    print(f"급변 인식 : {spike_rate:.3f}%")
    print(f"상승봉 평균 상승률: {avg_up_rate:.3f}%")
    print(f"하강봉 평균 하강률: {avg_down_rate:.3f}%")
    # === 급상승/급하강 구간 표시 ===
    spike_up = df[df['rate'] > spike_rate]
    spike_down = df[df['rate'] < -spike_rate]
    ma2 = df['MA_2']
    vwma9 = df['VWMA_45']
    signal = ma2 > vwma9
    df['golden_cross'] = (signal & (~signal.shift(1, fill_value=False)))
    df['dead_cross'] = ((~signal) & signal.shift(1, fill_value=False))

    # === 골든/데드크로스 쌍 중 의미 없는 것 제거 ===
    golden_list = df.index[df['golden_cross']].tolist()
    dead_list = df.index[df['dead_cross']].tolist()
    all_cross = sorted([(idx, 'golden') for idx in golden_list] + [(idx, 'dead') for idx in dead_list],
                       key=lambda x: x[0])

    remove_idx = set()
    mean_rate = abs(df['rate'].mean())

    i = 0
    while i < len(all_cross) - 1:
        idx1, type1 = all_cross[i]
        idx2, type2 = all_cross[i + 1]
        # 정확히 "서로 다른" 신호만 쌍으로 묶음
        if type1 != type2:
            price1 = df.loc[idx1, 'trade_price']
            price2 = df.loc[idx2, 'trade_price']
            rate = abs((price2 - price1) / price1)
            if rate <= mean_rate:
                remove_idx.add(idx1)
                remove_idx.add(idx2)
            i += 2  # 쌍으로 처리했으니 두 칸 이동
        else:
            i += 1  # 같은 종류면 한 칸만 이동

    df.loc[df.index.isin(remove_idx), 'golden_cross'] = False
    df.loc[df.index.isin(remove_idx), 'dead_cross'] = False

    now_price = df['trade_price'].iloc[-1]
    future_price_xgb = predict_future_price_xgb(df, periods=3, n_lags=5, method='xgb')
    pred_rate_xgb = (future_price_xgb - now_price) / now_price * 100 if future_price_xgb is not None else None
    print(f"현재가: {now_price:.2f}")
    print(f"[XGBoost] 3캔들 뒤 예측가: {future_price_xgb:.2f} / 예측 변화율: {pred_rate_xgb:.3f}%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    plt.figure(figsize=(16, 7))
    plt.plot(df.index, df['trade_price'], label='Price', color='black', linewidth=1)
    #plt.plot(df.index, df[f'VWMA_{short_window}'], label=f'VWMA {short_window}', linestyle='--')
    plt.plot(df.index, df[f'VWMA_{mid_window}'], label=f'VWMA {mid_window}', linestyle='--')
    plt.plot(df.index, df[f'VWMA_{long_window}'], label=f'VWMA {long_window}', linestyle='--')
    plt.plot(df.index, df[f'MA_{short_window}'], label=f'MA {short_window}', linestyle=':')
    #plt.plot(df.index, df[f'MA_{mid_window}'], label=f'MA {mid_window}', linestyle=':')
    #plt.plot(df.index, df[f'MA_{long_window}'], label=f'MA {long_window}', linestyle=':')
    plt.scatter(df[df['golden_cross']].index, df[df['golden_cross']]['trade_price'],
                color='gold', label='Golden Cross', marker='*', s=200, edgecolor='black')
    plt.scatter(df[df['dead_cross']].index, df[df['dead_cross']]['trade_price'],
                color='gray', label='Dead Cross', marker='X', s=120, edgecolor='black')
    plt.scatter(spike_up.index, spike_up['trade_price'], color='red', label='Spike Up', marker='^', s=100)
    plt.scatter(spike_down.index, spike_down['trade_price'], color='blue', label='Spike Down', marker='v', s=100)
    plt.title(f'{ticker} Price & Moving Averages\n(Spike: {spike_rate * 100:.1f}%)')
    plt.xlabel('Datetime (KST)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_features(df, n_lags=5):
    df_feat = df.copy()
    for i in range(1, n_lags+1):
        df_feat[f'lag_{i}'] = df_feat['trade_price'].shift(i)
    df_feat['volume'] = df_feat['candle_acc_trade_volume']
    df_feat['ma_3'] = df_feat['trade_price'].rolling(window=3).mean()
    df_feat['ma_7'] = df_feat['trade_price'].rolling(window=7).mean()
    df_feat['std_3'] = df_feat['trade_price'].rolling(window=3).std()
    df_feat['std_7'] = df_feat['trade_price'].rolling(window=7).std()
    df_feat = df_feat.dropna()
    return df_feat

def predict_future_price_xgb(df, periods=3, n_lags=5, method='xgb'):
    df_feat = create_features(df, n_lags)
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['volume', 'ma_3', 'ma_7', 'std_3', 'std_7']
    X = df_feat[feature_cols]
    y = df_feat['trade_price']
    last_row = X.iloc[[-1]].copy()  # DataFrame 형태 유지
    if method == 'xgb':
        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    else:
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, random_state=42, verbosity=0)
    model.fit(X, y)
    preds = []
    recent_prices = list(df_feat['trade_price'].iloc[-n_lags:].values)
    current_feats = last_row.copy()
    for _ in range(periods):
        pred = model.predict(current_feats)[0]
        preds.append(pred)
        new_lags = np.roll(current_feats.values[0][:n_lags], 1)
        new_lags[0] = pred
        for i in range(n_lags):
            current_feats.iloc[0, i] = new_lags[i]
        recent_prices = list(new_lags[:n_lags])
        current_feats.iloc[0, n_lags + 1] = np.mean(recent_prices[-3:])  # ma_3
        current_feats.iloc[0, n_lags + 2] = np.mean(recent_prices[-7:])  # ma_7
        current_feats.iloc[0, n_lags + 3] = np.std(recent_prices[-3:])  # std_3
        current_feats.iloc[0, n_lags + 4] = np.std(recent_prices[-7:])  # std_7
    return np.mean(preds)

peak_trade("KRW-ETH", 2,10, 45,200, '3m')
peak_trade("KRW-ETH", 2,10, 45,200, '5m')
peak_trade("KRW-ETH", 2,10, 45,200, '15m')
peak_trade("KRW-ETH", 2,10, 45,200, '30m')
peak_trade("KRW-ETH", 2,10, 45,200, '1h')
