from datetime import datetime
import requests
import time


def add_amt(bidamt, askamt):
    url = f'http://ywydpapa.iptime.org:8000/restaddtradeamt/{bidamt}/{askamt}'
    response = requests.get(url)
    return response

def add_orderbook(datetag, idxrow, coinn, bidamt, askamt, totalamt, amtdiff):
    url = f'http://ywydpapa.iptime.org:8000/restaddorderbookamt/{datetag}/{idxrow}/{coinn}/{bidamt}/{askamt}/{totalamt}/{amtdiff}'
    response = requests.get(url)
    return response

def format_korean_currency(amount):
    if amount >= 100_000_000:
        return f"{amount // 100_000_000}억 {((amount % 100_000_000) // 10_000)}만"
    elif amount >= 10_000:
        return f"{amount // 10_000}만"
    else:
        return f"{amount}원"

def hotmain():
    markets_url = "https://api.upbit.com/v1/market/all"
    markets = requests.get(markets_url).json()
    krw_markets = [m['market'] for m in markets if m['market'].startswith('KRW-')]

    results = []

    for market in krw_markets:
        try:
            orderbook_url = f"https://api.upbit.com/v1/orderbook?markets={market}"
            res = requests.get(orderbook_url).json()
            units = res[0]['orderbook_units']
            total_bid_amount = sum(u['bid_price'] * u['bid_size'] for u in units)
            total_ask_amount = sum(u['ask_price'] * u['ask_size'] for u in units)
            total_amount = total_bid_amount + total_ask_amount
            total_diff = total_bid_amount / total_ask_amount * 100
            results.append({'market': market,'total_bid_amount': total_bid_amount,'total_ask_amount': total_ask_amount,'total_amount': total_amount,'total_diff': total_diff})
            time.sleep(0.1)  # API rate limit 대응
        except Exception as e:
            print(f"Error for {market}: {e}")
    sorted_results = sorted(results, key=lambda x: x['total_amount'], reverse=True)
    # === 전체 오더북 매수·매도금액 합계 구하는 부분 추가 ===
    total_bid_all = sum(r['total_bid_amount'] for r in results)
    total_ask_all = sum(r['total_ask_amount'] for r in results)
    print(f"\n전체 KRW마켓 오더북 매수금액 합계: {total_bid_all:,.0f}원")
    print(f"전체 KRW마켓 오더북 매도금액 합계: {total_ask_all:,.0f}원\n")
    add_amt(int(total_bid_all), int(total_ask_all))
    now = datetime.now()
    datetag = now.strftime("%Y%m%d%H%M%S")
    idxr = 0
    for r in sorted_results[:30]:
        idxr += 1
        print(f"{r['market']}: 매수금액 {r['total_bid_amount']:,.0f}원, 매도금액 {r['total_ask_amount']:,.0f}원, 비율 {r['total_diff']:,.0f}%")
        add_orderbook( datetag, idxr, r['market'], int(r['total_bid_amount']), int(r['total_ask_amount']), int(r['total_amount']), r['total_diff'])


while True:
    hotmain()
    time.sleep(3600)
