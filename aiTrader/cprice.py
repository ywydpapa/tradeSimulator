import numpy as np
import requests
from datetime import datetime, timezone

def all_cprice():
    server_url = "https://api.upbit.com"
    params = {"quote_currencies": "KRW"}
    res = requests.get(server_url + "/v1/ticker/all", params=params)
    data = res.json()
    result = []
    for item in data:
        market = item.get("market")
        trade_price = item.get("trade_price")
        timestamp = item.get("timestamp")
        if market and trade_price and timestamp:
            time_str = datetime.fromtimestamp(timestamp / 1000, timezone.utc ).strftime('%Y-%m-%d %H:%M:%S')
            result.append({"market": market,"trade_price": trade_price,"time_str": time_str })
    return result


def get_upbit_orderbooks(market="KRW-BTC"):
    url = "https://api.upbit.com/v1/orderbook"
    params = {"markets": market}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def get_krw_tickers():
    url = "https://api.upbit.com/v1/market/all"
    data = requests.get(url).json()
    krw_tickers = [item['market'] for item in data if item['market'].startswith('KRW-')]
    return krw_tickers


def find_walls(sizes, threshold=2):
    sizes = np.array(sizes)
    mean = np.mean(sizes)
    std = np.std(sizes)
    wall_level = mean + threshold * std
    walls = [(i, size) for i, size in enumerate(sizes) if size >= wall_level]
    return walls, wall_level


def check_orderbook(ticker, wall_threshold=2):
    lines = get_upbit_orderbooks(ticker)
    if lines:
        print("MARKET:", lines[0]["market"])
        total_ask = lines[0]["total_ask_size"]
        total_bid = lines[0]["total_bid_size"]
        print("ASK:", total_ask)
        print("BID:", total_bid)
        # 가격 내림차순 정렬
        units = sorted(lines[0]["orderbook_units"], key=lambda x: x["ask_price"], reverse=True)
        print(f"{'BID(%)':>8} {'BID_SIZE':>10} {'BID_PRICE':>12} | {'ASK_PRICE':<12} {'ASK_SIZE':<10} {'ASK(%)':<7}")
        print("-" * 70)

        # 벽 감지를 위한 사이즈 리스트 생성
        bid_sizes = [unit['bid_size'] for unit in units]
        ask_sizes = [unit['ask_size'] for unit in units]
        bid_prices = [unit['bid_price'] for unit in units]
        ask_prices = [unit['ask_price'] for unit in units]

        # 벽 감지
        bid_walls, bid_wall_level = find_walls(bid_sizes, threshold=wall_threshold)
        ask_walls, ask_wall_level = find_walls(ask_sizes, threshold=wall_threshold)

        for idx, unit in enumerate(units):
            bid_size = unit['bid_size']
            ask_size = unit['ask_size']
            bid_share = (bid_size / total_bid * 100) if total_bid > 0 else 0
            ask_share = (ask_size / total_ask * 100) if total_ask > 0 else 0

            bid_wall_mark = "⬅️벽" if any(idx == w[0] for w in bid_walls) else ""
            ask_wall_mark = "벽➡️" if any(idx == w[0] for w in ask_walls) else ""

            print(
                f"{bid_share:>7.2f}% {bid_size:>10.4f} {unit['bid_price']:>12,.0f} | {unit['ask_price']:<12,.0f} {ask_size:<10.4f} {ask_share:<6.2f}% {bid_wall_mark:3} {ask_wall_mark:3}")

        # 벽 요약 출력
        print("\n[매수벽 감지 기준치: {:.4f}]".format(bid_wall_level))
        if bid_walls:
            for idx, size in bid_walls:
                print(f"  ⬅️ 매수벽: 가격 {bid_prices[idx]:,.0f}, 수량 {size:.4f}")
        else:
            print("  매수벽 없음")
        print("[매도벽 감지 기준치: {:.4f}]".format(ask_wall_level))
        if ask_walls:
            for idx, size in ask_walls:
                print(f"  매도벽➡️: 가격 {ask_prices[idx]:,.0f}, 수량 {size:.4f}")
        else:
            print("  매도벽 없음")

check_orderbook("KRW-SUI")