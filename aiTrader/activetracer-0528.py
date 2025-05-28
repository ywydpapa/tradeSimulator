import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import time


def compute_stoch_rsi(series, window=14, smooth_k=3, smooth_d=3):
    # 1. RSI 먼저 계산
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # 2. StochRSI 계산
    min_rsi = rsi.rolling(window).min()
    max_rsi = rsi.rolling(window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)

    # 3. %K, %D smoothing
    stoch_k = stoch_rsi.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(smooth_d).mean()

    return stoch_rsi, stoch_k, stoch_d


def peak_trade(
        ticker='KRW-BTC',
        short_window=3,
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
    df[f'VWMA_{long_window}'] = (
            (df['trade_price'] * df['candle_acc_trade_volume'])
            .rolling(window=long_window).sum() /
            df['candle_acc_trade_volume'].rolling(window=long_window).sum()
    )
    df[f'MA_{short_window}'] = df['trade_price'].rolling(window=short_window).mean()
    df[f'MA_{long_window}'] = df['trade_price'].rolling(window=long_window).mean()

    # === 상승/하강봉 평균 변화율 계산 추가 ===
    df['prev_price'] = df['trade_price'].shift(1)
    df['change'] = df['trade_price'] - df['prev_price']
    df['rate'] = (df['trade_price'] - df['prev_price']) / df['prev_price']

    up_candles = df[df['change'] > 0]
    down_candles = df[df['change'] < 0]

    avg_up_rate = up_candles['rate'].mean() * 100  # %
    avg_down_rate = down_candles['rate'].mean() * 100  # %

    print(f"상승봉 평균 상승률: {avg_up_rate:.3f}%")
    print(f"하강봉 평균 하강률: {avg_down_rate:.3f}%")
    # =====================================

    # 3. 크로스 포인트 계산
    golden_cross = df[
        (df[f'VWMA_{short_window}'] > df[f'VWMA_{long_window}']) &
        (df[f'VWMA_{short_window}'].shift(1) <= df[f'VWMA_{long_window}'].shift(1))
        ]
    dead_cross = df[
        (df[f'VWMA_{short_window}'] < df[f'VWMA_{long_window}']) &
        (df[f'VWMA_{short_window}'].shift(1) >= df[f'VWMA_{long_window}'].shift(1))
        ]
    # RSI 추가
    stoch_rsi, stoch_k, stoch_d = compute_stoch_rsi(df['trade_price'], window=14, smooth_k=3, smooth_d=3)
    df['stoch_rsi'] = stoch_rsi
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    valid_idx = df.index[df[f'VWMA_{long_window}'].notna()]
    ax1.plot(df['trade_price'], label='Price')
    ax1.plot(valid_idx, df.loc[valid_idx, f'VWMA_{short_window}'], label=f'VWMA {short_window}', color='blue')
    ax1.plot(valid_idx, df.loc[valid_idx, f'VWMA_{long_window}'], label=f'VWMA {long_window}', color='orange')
    ax1.scatter(golden_cross.index, golden_cross[f'VWMA_{short_window}'], color='green', label='Golden Cross',
                marker='o', s=100)
    ax1.scatter(dead_cross.index, dead_cross[f'VWMA_{short_window}'], color='red', label='Dead Cross', marker='x',
                s=100)
    ax1.set_title(f'{ticker} VWMA Cross Points ({candle_unit})')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend()

    # === 마지막 10개 값의 방향 막대 추가 ===
    last10_idx = df.index[-10:]
    last10_prices = df['trade_price'].iloc[-10:]
    price_diff = last10_prices.diff().fillna(0)
    circle_colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in price_diff]

    circle_y = df['trade_price'].min() - (df['trade_price'].max() - df['trade_price'].min()) * 0.05

    ax1.scatter(
        last10_idx,
        [circle_y] * 10,
        s=400,
        color=circle_colors,
        marker='o',
        edgecolors='black',
        linewidths=1.5,
        label='Last 10 Change Signal'
    )
    # =======================

    # STOCH RSI
    ax2.plot(df.index, df['stoch_k'], label='StochRSI %K', color='purple')
    ax2.plot(df.index, df['stoch_d'], label='StochRSI %D', color='magenta', linestyle='--')
    ax2.axhline(0.8, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(0.2, color='blue', linestyle=':', alpha=0.5)
    ax2.set_ylabel('StochRSI')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    plt.close()

    # 최근 크로스 판단
    recent = df.tail(5)
    # print("최근 5개 캔들")
    # print(recent[['trade_price', f'VWMA_{short_window}', f'VWMA_{long_window}']])

    # 최근 크로스 판단
    now = df.index[-1]
    now_price = df['trade_price'].iloc[-1]
    golden_times = golden_cross.index[golden_cross.index <= now]
    dead_times = dead_cross.index[dead_cross.index <= now]

    last_golden = golden_times[-1] if len(golden_times) > 0 else None
    last_dead = dead_times[-1] if len(dead_times) > 0 else None

    if last_golden and last_dead:
        if last_golden > last_dead:
            last_cross_type = 'golden'
            last_cross_time = last_golden
        else:
            last_cross_type = 'dead'
            last_cross_time = last_dead
    elif last_golden:
        last_cross_type = 'golden'
        last_cross_time = last_golden
    elif last_dead:
        last_cross_type = 'dead'
        last_cross_time = last_dead
    else:
        last_cross_type = None
        last_cross_time = None

    recent5_idx = df.index[-5:]
    recent3_idx = df.index[-3:]

    recent_golden = [idx for idx in golden_cross.index if idx in recent3_idx]
    recent_dead = [idx for idx in dead_cross.index if idx in recent3_idx]
    recent_golden_5 = [idx for idx in golden_cross.index if idx in recent5_idx]
    recent_dead_5 = [idx for idx in dead_cross.index if idx in recent5_idx]

    if recent_golden_5 and recent_dead_5:
        print("최근 5개 캔들에 골든/데드가 모두 있습니다. 매매 대기!")
        return

    if recent_golden:
        print("최근 3개 캔들에 골든크로스 발생! 매수 신호! 보유하고 있지 않다면 매수")
        now_price = df['trade_price'].iloc[-1]
        volum = 500_000 / now_price
        print(f"매수 실행: {now_price}에 {volum:.6f}코인")
        # buy_crypto(...) 호출 위치!
        return

    if recent_dead:
        print("최근 3개 캔들에 데드크로스 발생! 매도 신호! 보유중인 코인 판매")
        now_price = df['trade_price'].iloc[-1]
        print(f"매도 실행: {now_price}에 보유코인 전량")
        # sell_crypto(...) 호출 위치!
        return

    if last_cross_type is not None:
        analyze_cross_with_peak_and_vwma(
            df, last_cross_type, last_cross_time,
            short_window=short_window,
            long_window=long_window,
            threshold=0.03,  # 하락/상승 임계값 3%
            close_threshold=0.001  # VWMA 근접 임계값 0.1%
        )
    else:
        print("아직 골든/데드 크로스가 없습니다.")


def analyze_cross_with_peak_and_vwma(
    df,
    last_cross_type,
    last_cross_time,
    short_window,
    long_window,
    threshold=0.015,
    close_threshold=0.001
):
    # 1. 구간 결정
    if last_cross_type is not None and last_cross_time is not None:
        # 크로스가 있으면, 크로스 이후 데이터만 사용
        prices = df.loc[last_cross_time:]['trade_price']
        vwmashort = df.loc[last_cross_time:][f'VWMA_{short_window}']
        vwmalong = df.loc[last_cross_time:][f'VWMA_{long_window}']
        print(f"최근 크로스({last_cross_type})가 {last_cross_time}에 발생, 이후 데이터 기준으로 판단합니다.")
    else:
        # 크로스가 없으면 전체 데이터 사용
        prices = df['trade_price']
        vwmashort = df[f'VWMA_{short_window}']
        vwmalong = df[f'VWMA_{long_window}']
        print("최근 크로스가 없습니다. 전체 데이터에서 최고점/최저점 기준으로 판단합니다.")

    now_price = prices.iloc[-1]
    now_vwmalong = vwmalong.iloc[-1]
    vwma_gap = abs(now_price - now_vwmalong) / now_vwmalong

    # 꼭지점 찾기
    peak_indices, _ = find_peaks(prices)
    valley_indices, _ = find_peaks(-prices)

    if len(peak_indices) > 0:
        last_peak_time = prices.index[peak_indices[-1]]
        last_peak_value = prices.iloc[peak_indices[-1]]
        print(f"마지막 최고점: {last_peak_time} / {last_peak_value}")

    if len(valley_indices) > 0:
        last_valley_time = prices.index[valley_indices[-1]]
        last_valley_value = prices.iloc[valley_indices[-1]]
        print(f"마지막 최저점: {last_valley_time} / {last_valley_value}")

    # 최고점, 최저점 및 신호 판단
    max_price = prices.max()
    max_time = prices.idxmax()
    min_price = prices.min()
    min_time = prices.idxmin()

    fall_rate = (max_price - now_price) / max_price
    rise_rate = (now_price - min_price) / min_price

    print(f"최고가: {max_price:.2f} ({max_time}), 최저가: {min_price:.2f} ({min_time}), 현재가: {now_price:.2f}")
    print(f"최고가 대비 하락률: {fall_rate * 100:.2f}%")
    print(f"최저가 대비 상승률: {rise_rate * 100:.2f}%")

    # 신호 판단 (최고점/최저점 모두 체크)
    if fall_rate >= threshold:
        print(f"→ {threshold*100:.1f}% 이상 하락! 매도 신호!")
        print(f"→최고가 대비  {threshold * 100:.1f}% 이상 하락으로 보유 코인 전액 현재가 {now_price} 매도 실행!")
    elif vwma_gap <= close_threshold:
        print(f"→ 가격이 long VWMA({long_window})와 0.1% 이내로 접근! 추가 매도 신호!")
        print(f"→ 가격이 long VWMA({long_window})와 0.1% 이내로 접근 보유코인이 있을 경우 전액 현재가 {now_price} 매도 실행!")
    else:
        print("→ 아직 매도 신호 아님(최고가 하락 미달, long VWMA 접근 미달)")

    if rise_rate >= threshold:
        print(f"→최저가 대비  {threshold*100:.1f}% 이상 상승! 매수 신호!")
        print(f"현재가 {now_price}로 500,000원 매수")
    elif vwma_gap <= close_threshold:
        print(f"→ 가격이 long VWMA({long_window})와 0.1% 이내로 접근! 추가 매수 신호!")
        print(f"보유 코인 없을 경우 현재가 {now_price}로 500,000원 매수")
    else:
        print("→ 아직 매수 신호 아님(최저가 상승 미달, long VWMA 접근 미달)")

def get_wallet(uno):
    url = f'http://ywydpapa.iptime.org:8000/restwallet/{uno}'
    response = requests.get(url)
    data = response.json()
    return data


def buy_crypto(ticker,uno):
    url = f'http://ywydpapa.iptime.org:8000/restbuymarket{uno}/{ticker}'
    response = requests.get(url)


# 사용 예시
while True:

    ticker = 'KRW-SUI'
    peak_trade(ticker, 1, 20, 200, '30m')
    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    peak_trade(ticker, 1, 20, 200, '3m')
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    peak_trade(ticker, 1, 20, 200, '1m')
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    wallets = get_wallet(1)
    avgprices = wallets['data']['wallet_dict']
    for aitem in avgprices.items():
        print(aitem)
    wallteitems = wallets['data']['wallet_list']
    for witem in wallteitems:
        print(witem[9])
    time.sleep(60)
