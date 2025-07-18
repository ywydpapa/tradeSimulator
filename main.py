import asyncio
import time
from typing import Dict
import math
import aiohttp
from fastapi import FastAPI, Depends, Request, Form, Response, HTTPException, status, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import text
import dotenv
import os
import requests
import jinja2
from datetime import datetime
from aiTrader.vwmatrend import vwma_ma_cross_and_diff_noimage
from fastapi import WebSocket, WebSocketDisconnect
import httpx
import websockets
import json

dotenv.load_dotenv()
DATABASE_URL = os.getenv("dburl")
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="supersecretkey")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

tradetrend: Dict = {}


def format_currency(value):
    if isinstance(value, (int, float)):
        return "{:,.0f}".format(value)
    return value


templates.env.filters['currency'] = format_currency


# 데이터베이스 세션 생성
async def get_db():
    async with async_session() as session:
        yield session


async def get_current_prices():
    server_url = "https://api.upbit.com"
    params = {"quote_currencies": "KRW"}
    res = requests.get(server_url + "/v1/ticker/all", params=params)
    data = res.json()
    result = []
    for item in data:
        market = item.get("market")
        trade_price = item.get("trade_price")
        if market and trade_price:
            result.append({"market": market, "trade_price": trade_price})
    return result


async def get_current_price(coinn):
    server_url = "https://api.upbit.com"
    params = {"quote_currencies": "KRW"}
    res = requests.get(server_url + "/v1/ticker/all", params=params)
    data = res.json()
    result = []
    for item in data:
        market = item.get("market")
        trade_price = item.get("trade_price")
        if market and trade_price and market == coinn:
            result.append({"market": market, "trade_price": trade_price})
    return result


async def get_krw_tickers():
    url = "https://api.upbit.com/v1/market/all"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            krw_tickers = [item for item in data if item['market'].startswith('KRW-')]
            return krw_tickers


async def update_tradetrend():
    global tradetrend
    while True:
        async for db in get_db():
            try:
                query = text("SELECT distinct (coinName) FROM polarisSets WHERE attrib not like :attxxx")
                coinlist = await db.execute(query, {"attxxx": "%XXX%"})
                coinlist = coinlist.fetchall()
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                timeframes = ['1d', '4h', '1h', '30m', '3m', '1m']
                result: Dict[str, Dict[str, dict]] = {}

                for coin in coinlist:
                    try:
                        result[coin[0]] = {}
                        for tf in timeframes:
                            try:
                                df, reversal_points, reversal_distances, slope, angle_deg, delta_x = vwma_ma_cross_and_diff_noimage(
                                    coin[0], 3, 35, 150, tf)
                                if math.isinf(slope):
                                    if slope > 0:
                                        slope = 1e10
                                    else:
                                        slope = -1e10
                                result[coin[0]][tf] = {
                                    "slope": slope,
                                    "angle_deg": angle_deg,
                                    "reversal_count": len(reversal_points),
                                    "reversal_distances": reversal_distances,
                                    "deltax": delta_x
                                }
                                time.sleep(0.15)
                            except Exception as e:
                                print(f"코인트렌드 타임프레임 처리 중 오류 발생 {tf} for {coin[0]}: {str(e)}")
                                continue
                    except Exception as e:
                        print(f"코인트렌드 처리 중 오류 발생 {coin[0]}: {str(e)}")
                        continue
                tradetrend = result
                print(f"[{now}] tradetrend updated")
            except Exception as e:
                print(f"update_tradetrend 오류 발생: {str(e)}")
            finally:
                await db.close()
        await asyncio.sleep(90)


def get_signal_class(slope: float) -> dict:
    if slope < -44.9:
        return {'cls': 'black', 'label': '⚫'}
    elif slope > 45:
        return {'cls': 'black', 'label': '⚫'}
    elif slope < 0:
        return {'cls': 'red', 'label': '🔴'}
    elif slope < 0.2:
        return {'cls': 'orange', 'label': '🟠'}
    else:
        return {'cls': 'green', 'label': '🟢'}


def make_signal_bulbs(tfs: dict) -> str:
    bulbs = ""
    tf_order = ['1d', '4h', '1h', '30m', '3m', '1m']
    tf_label = {'1d': '1D', '4h': '4H', '1h': '1H', '30m': '30', '3m': '3', '1m': '1'}
    for tf in tf_order:
        if tf in tfs:
            slope = tfs[tf]['slope']
            sig = get_signal_class(slope)
            bulbs += f'<span class="signal-bulb {sig["cls"]}" title="{tf}"></span>'
    return bulbs


async def buy_crypto(request, uno, coinn, price, volum, db: AsyncSession = Depends(get_db)):
    global walletkrw, walletvolum
    try:
        walletvolum = 0.0
        costkrw = volum * price
        costfee = costkrw * 0.0005
        totalcost = costkrw + costfee
        seckey = request.session.get("setupKey")
        if seckey is None:
            seckey = await get_seckey(uno, db)
        wallets = await get_current_balance(uno, db)
        for wallet in wallets[0]:
            if wallet[5] == "KRW":
                walletkrw = wallet[9]
            elif wallet[5] == coinn:
                walletvolum = wallet[9]
        if walletkrw < totalcost:
            return False
        else:
            walletkrw = walletkrw - totalcost
            sumvolum = walletvolum + volum
            ctype = "BUY-" + coinn
            query = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": "KRW"})
            await db.commit()
            query2 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,outAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,'KRW', 1 , :costkrw, :remkrw, :seckey)")
            await db.execute(query2,
                             {"uno": uno, "ctype": ctype, "costkrw": totalcost, "remkrw": walletkrw, "seckey": seckey})
            await db.commit()
            query4 = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query4, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": coinn})
            await db.commit()
            query3 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,inAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,:coinn, :uprice, :inamt, :remamt, :seckey)")
            await db.execute(query3, {"uno": uno, "ctype": ctype, "coinn": coinn, "uprice": price, "inamt": volum,
                                      "remamt": sumvolum, "seckey": seckey})
            await db.commit()
    except Exception as e:
        print("Error!!", e)
    return True


async def get_seckey(uno: int, db: AsyncSession = Depends(get_db)):
    try:
        query = text("select setupKey from trUser where userNo = :uno")
        result = await db.execute(query, {"uno": uno})
        skey = result.fetchone()
        return skey[0]
    except Exception as e:
        print("Error!!", e)
        return None


async def sell_crypto(request, uno, coinn, price, volum, db: AsyncSession = Depends(get_db)):
    global walletkrw, walletvolum
    try:
        walletvolum = 0.0
        costkrw = volum * price
        costfee = costkrw * 0.0005
        totalcost = costkrw - costfee
        seckey = request.session.get("setupKey")
        if seckey is None:
            seckey = await get_seckey(uno, db)
        wallets = await get_current_balance(uno, db)
        for wallet in wallets[0]:
            if wallet[5] == "KRW":
                walletkrw = wallet[9]
            elif wallet[5] == coinn:
                walletvolum = wallet[9]
        if walletvolum < volum:
            return False
        else:
            walletkrw = walletkrw + totalcost
            sumvolum = walletvolum - volum
            ctype = "SELL-" + coinn
            query = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": "KRW"})
            await db.commit()
            query2 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,inAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,'KRW', 1 , :costkrw, :remkrw, :seckey)")
            await db.execute(query2,
                             {"uno": uno, "ctype": ctype, "costkrw": totalcost, "remkrw": walletkrw, "seckey": seckey})
            await db.commit()
            query4 = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query4, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": coinn})
            await db.commit()
            query3 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,outAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,:coinn, :uprice, :inamt, :remamt, :seckey)")
            await db.execute(query3, {"uno": uno, "ctype": ctype, "coinn": coinn, "uprice": price, "inamt": volum,
                                      "remamt": sumvolum, "seckey": seckey})
            await db.commit()
    except Exception as e:
        print("Error!!", e)
    return True


async def rest_sell_crypto(request, uno, coinn, price, volum, db: AsyncSession = Depends(get_db)):
    global walletkrw, walletvolum
    try:
        walletvolum = 0.0
        costkrw = volum * price
        costfee = costkrw * 0.0005
        totalcost = costkrw - costfee
        seckey = await get_seckey(uno, db)
        wallets = await get_current_balance(uno, db)
        for wallet in wallets[0]:
            if wallet[5] == "KRW":
                walletkrw = wallet[9]
            elif wallet[5] == coinn:
                walletvolum = wallet[9]
        if walletvolum < volum:
            return False
        else:
            walletkrw = walletkrw + totalcost
            sumvolum = walletvolum - volum
            ctype = "SELL-" + coinn
            query = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": "KRW"})
            await db.commit()
            query2 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,inAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,'KRW', 1 , :costkrw, :remkrw, :seckey)")
            await db.execute(query2,
                             {"uno": uno, "ctype": ctype, "costkrw": totalcost, "remkrw": walletkrw, "seckey": seckey})
            await db.commit()
            query4 = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query4, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": coinn})
            await db.commit()
            query3 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,outAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,:coinn, :uprice, :inamt, :remamt, :seckey)")
            await db.execute(query3, {"uno": uno, "ctype": ctype, "coinn": coinn, "uprice": price, "inamt": volum,
                                      "remamt": sumvolum, "seckey": seckey})
            await db.commit()
    except Exception as e:
        print("Error!!", e)
    return True


async def rest_cut_crypto(request, uno, coinn, price, volum, db: AsyncSession = Depends(get_db)):
    global walletkrw, walletvolum
    try:
        walletvolum = 0.0
        costkrw = volum * price
        costfee = costkrw * 0.0005
        totalcost = costkrw - costfee
        seckey = await get_seckey(uno, db)
        wallets = await get_current_balance(uno, db)
        for wallet in wallets[0]:
            if wallet[5] == "KRW":
                walletkrw = wallet[9]
            elif wallet[5] == coinn:
                walletvolum = wallet[9]
        if walletvolum < volum:
            return False
        else:
            walletkrw = walletkrw + totalcost
            sumvolum = walletvolum - volum
            ctype = "CUT-" + coinn
            query = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": "KRW"})
            await db.commit()
            query2 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,inAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,'KRW', 1 , :costkrw, :remkrw, :seckey)")
            await db.execute(query2,
                             {"uno": uno, "ctype": ctype, "costkrw": totalcost, "remkrw": walletkrw, "seckey": seckey})
            await db.commit()
            query4 = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query4, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": coinn})
            await db.commit()
            query3 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,outAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,:coinn, :uprice, :inamt, :remamt, :seckey)")
            await db.execute(query3, {"uno": uno, "ctype": ctype, "coinn": coinn, "uprice": price, "inamt": volum,
                                      "remamt": sumvolum, "seckey": seckey})
            await db.commit()
    except Exception as e:
        print("Error!!", e)
    return True


async def rest_buy_crypto(request, uno, coinn, price, volum, db: AsyncSession = Depends(get_db)):
    global walletkrw, walletvolum
    try:
        walletvolum = 0.0
        costkrw = volum * price
        costfee = costkrw * 0.0005
        totalcost = costkrw + costfee
        seckey = await get_seckey(uno, db)
        wallets = await get_current_balance(uno, db)
        for wallet in wallets[0]:
            if wallet[5] == "KRW":
                walletkrw = wallet[9]
            elif wallet[5] == coinn:
                walletvolum = wallet[9]
        if walletkrw < totalcost:
            return False
        else:
            walletkrw = walletkrw - totalcost
            sumvolum = walletvolum + volum
            ctype = "BUY-" + coinn
            query = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": "KRW"})
            await db.commit()
            query2 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,outAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,'KRW', 1 , :costkrw, :remkrw, :seckey)")
            await db.execute(query2,
                             {"uno": uno, "ctype": ctype, "costkrw": totalcost, "remkrw": walletkrw, "seckey": seckey})
            await db.commit()
            query4 = text("UPDATE trWallet set attrib = :xxxup WHERE userNo = :uno and currency = :coin")
            await db.execute(query4, {"xxxup": 'XXXUPXXXUP', "uno": uno, "coin": coinn})
            await db.commit()
            query3 = text("INSERT INTO trWallet (userNo,changeType,currency,unitPrice,inAmt, remainAmt, linkNo) "
                          "values (:uno, :ctype ,:coinn, :uprice, :inamt, :remamt, :seckey)")
            await db.execute(query3, {"uno": uno, "ctype": ctype, "coinn": coinn, "uprice": price, "inamt": volum,
                                      "remamt": sumvolum, "seckey": seckey})
            await db.commit()
    except Exception as e:
        print("Error!!", e)
    return True


async def rest_add_trade_amt(bidamt, askamt, db):
    try:
        query = text("INSERT INTO tradeAmt (bidAmt, askAmt) VALUES (:bidamt, :askamt)")
        await db.execute(query, {"bidamt": bidamt, "askamt": askamt})
        await db.commit()
        return True
    except Exception as e:
        print("Error!!", e)
        return False


async def rest_add_orderbook_amt(datetag, idxrow, coinn, bidamt, askamt, totalamt, amtdiff, db):
    try:
        query = text(
            "INSERT INTO orderbookAmt (dateTag, idxRow, coinName, bidAmt, askAmt, totalAmt, amtDiff) values (:dateTag, :idxRow, :coinName, :bidAmt, :askAmt, :totalAmt, :amtDiff)")
        await db.execute(query,
                         {"dateTag": datetag, "idxRow": idxrow, "coinName": coinn, "bidAmt": bidamt, "askAmt": askamt,
                          "totalAmt": totalamt, "amtDiff": amtdiff})
        await db.commit()
        return True
    except Exception as e:
        print("Error!!", e)
        return False


async def rest_predict(dateTag,coinName,avgUprate,avgDownrate,currentPrice,predictA,predictB,predictC,predictD,rateA,rateB,rateC,rateD,intV,db):
    try:
        query = text("INSERT into predictPrice (dateTag,coinName,avgUprate,avgDownrate,currentPrice,predictA,predictB,predictC,predictD,rateA,rateB,rateC,rateD,intV) values (:dateTag,:coinName,:avgUprate,:avgDownrate,:currentPrice,:predictA,:predictB,:predictC,:predictD,:rateA,:rateB,:rateC,:rateD,:intv)")
        await db.execute(query,{"dateTag":dateTag, "coinName": coinName, "avgUprate": avgUprate, "avgDownrate": avgDownrate, "currentPrice": currentPrice, "predictA": predictA,"predictB": predictB,"predictC": predictC,"predictD": predictD, "rateA":rateA,"rateB":rateB,"rateC":rateC,"rateD":rateD,"intv":intV})
        await db.commit()
        return True
    except Exception as e:
        print("Error!!", e)
        return False


async def get_hotcoins(request, db):
    try:
        query = text("SELECT * FROM orderbookAmt where dateTag = (select max(dateTag) from orderbookAmt)")
        result = await db.execute(query)
        orderbooks = result.fetchall()
        return orderbooks
    except Exception as e:
        print("Error!!", e)
        return False


async def get_predicts(request, db):
    try:
        query = text("SELECT * FROM predictPrice where dateTag = (select max(dateTag) from predictPrice)")
        result = await db.execute(query)
        predicts = result.fetchall()
        return predicts
    except Exception as e:
        print("Error!!", e)
        return False


async def get_hotamt(request, db):
    try:
        query = text("select * from tradeAmt order by regDate desc limit 1")
        result = await db.execute(query)
        hotamt = result.fetchone()
        return hotamt
    except Exception as e:
        print("Error!!", e)
        return False


async def get_current_balance(uno, db: AsyncSession = Depends(get_db)):
    try:
        query = text("SELECT * FROM trWallet where userNo = :uno and attrib not like :attxx order by currency ")
        result = await db.execute(query, {"uno": uno, "attxx": "%XXX%"})
        mycoins = result.fetchall()
        coinprice = {}
        gcprice = await get_current_prices()
        price_dict = {item['market']: item['trade_price'] for item in gcprice}
        for coin in mycoins:
            if coin[5] != "KRW":
                cprice = price_dict.get(coin[5], None)
            else:
                cprice = 1.0
            coinprice[coin[5]] = cprice
    except Exception as e:
        print("Error!!", e)
    finally:
        return mycoins, coinprice


async def get_trsetups(uno, db: AsyncSession = Depends(get_db)):
    try:
        query = text("SELECT * FROM polarisSets where userNo = :uno and attrib not like :attxx")
        result = await db.execute(query, {"uno": uno, "attxx": "%XXX%"})
        mysetups = result.fetchall()
        mysets = []
        for setup in mysetups:
            mysets.append({
                "setupNo": setup[0],
                "coinName": setup[2],
                "stepAmt": setup[3],
                "tradeType": setup[4],
                "maxAmt": setup[5],
                "useYN": setup[6],
            })
    except Exception as e:
        print("Get Setup Error!!", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        print(mysets)
    return mysets


async def get_logbook(request, uno, coinn, db: AsyncSession = Depends(get_db)):
    global mylogs
    try:
        query = text(
            "SELECT changeType, currency,unitPrice,inAmt,outAmt,remainAmt,regDate FROM trWallet where userNo = :uno and currency = :coinn and linkNo = :seckey order by regDate ")
        result = await db.execute(query, {"uno": uno, "coinn": coinn, "seckey": request.session.get("setupKey")})
        rows = result.fetchall()
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in rows]
        return jsonable_encoder(data)
    except Exception as e:
        print("Error!!", e)


async def get_avg_price(uno, setkey, coinn, db: AsyncSession = Depends(get_db)):
    try:
        query = text(
            "SELECT linkNo, regDate, changeType, currency, unitPrice, inAmt, outAmt, remainAmt, session_id, IFNULL(누적매수금액 / NULLIF(누적매수수량,0), 0) AS 매수평균단가 FROM "
            "(SELECT *,SUM(CASE WHEN changeType LIKE 'BUY%' THEN unitPrice * inAmt ELSE 0 END) OVER (PARTITION BY session_id ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS 누적매수금액,SUM(CASE WHEN changeType LIKE 'BUY%' THEN inAmt ELSE 0 END) OVER (PARTITION BY session_id ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS 누적매수수량 "
            "FROM ( SELECT *, SUM(is_zero) OVER (ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_id "
            "FROM ( SELECT *, CASE WHEN remainAmt = 0 THEN 1 ELSE 0 END AS is_zero FROM trWallet WHERE currency = :coinn and userNo = :uno ORDER BY regDate, linkNo ) t1 ) t2 ) t3 WHERE linkNo = :linkno ORDER BY regDate DESC, linkNo DESC LIMIT 1")
        result = await db.execute(query, {"coinn": coinn, "linkno": setkey, "uno": uno})
        mycoin = result.fetchone()
        return mycoin
    except Exception as e:
        print("Error!!", e)
        return None


async def get_avg_by_coin(uno, setkey, db: AsyncSession = Depends(get_db)):
    try:
        query = text(
            "SELECT currency,IFNULL(누적매수금액 / NULLIF(누적매수수량,0), 0) AS avg_price FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY currency ORDER BY regDate DESC, linkNo DESC) AS rn,SUM(CASE WHEN changeType LIKE 'BUY%' THEN unitPrice * inAmt ELSE 0 END) OVER (PARTITION BY currency, session_id ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS 누적매수금액, SUM(CASE WHEN changeType LIKE 'BUY%' THEN inAmt ELSE 0 END)                OVER (PARTITION BY currency, session_id ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS 누적매수수량    FROM (        SELECT *,               SUM(is_zero) OVER (PARTITION BY currency ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_id FROM ( SELECT *, CASE WHEN remainAmt = 0 THEN 1 ELSE 0 END AS is_zero FROM trWallet WHERE userNo = :uno and linkNo = :linkno ORDER BY regDate, linkNo ) t1 ) t2) t3 WHERE rn = 1")
        result = await db.execute(query, {"uno": uno, "linkno": setkey})
        rows = result.fetchall()
        return {row.currency: round(float(row.avg_price), 2) for row in rows}
    except Exception as e:
        print(e)
        return {}


def add_amt(bidamt, askamt):
    url = f'http://ywydpapa.iptime.org:8000/restaddtradeamt/{bidamt}/{askamt}'
    response = requests.get(url)
    return response


def add_orderbook(datetag, idxrow, coinn, bidamt, askamt, totalamt, amtdiff):
    url = f'http://ywydpapa.iptime.org:8000/restaddorderbookamt/{datetag}/{idxrow}/{coinn}/{bidamt}/{askamt}/{totalamt}/{amtdiff}'
    response = requests.get(url)
    return response


async def get_new_orderbook_and_save():
    try:
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
                results.append({
                    'market': market,
                    'total_bid_amount': total_bid_amount,
                    'total_ask_amount': total_ask_amount,
                    'total_amount': total_amount,
                    'total_diff': total_diff
                })
                time.sleep(0.1)  # API rate limit 대응
            except Exception as e:
                print(f"Error for {market}: {e}")

        sorted_results = sorted(results, key=lambda x: x['total_amount'], reverse=True)

        total_bid_all = sum(r['total_bid_amount'] for r in results)
        total_ask_all = sum(r['total_ask_amount'] for r in results)
        add_amt(int(total_bid_all), int(total_ask_all))
        now = datetime.now()
        datetag = now.strftime("%Y%m%d%H%M%S")
        idxr = 0
        for r in sorted_results[:25]:
            idxr += 1
            add_orderbook(datetag, idxr, r['market'], int(r['total_bid_amount']), int(r['total_ask_amount']),
                          int(r['total_amount']), r['total_diff'])
        return True
    except Exception as e:
        print(e)
        return False


async def rest_get_avg_by_coin(uno, db: AsyncSession = Depends(get_db)):
    try:
        setkey = await get_seckey(uno, db=db)
        query = text(
            "SELECT currency,IFNULL(누적매수금액 / NULLIF(누적매수수량,0), 0) AS avg_price FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY currency ORDER BY regDate DESC, linkNo DESC) AS rn,SUM(CASE WHEN changeType LIKE 'BUY%' THEN unitPrice * inAmt ELSE 0 END) OVER (PARTITION BY currency, session_id ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS 누적매수금액, SUM(CASE WHEN changeType LIKE 'BUY%' THEN inAmt ELSE 0 END)                OVER (PARTITION BY currency, session_id ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS 누적매수수량    FROM (        SELECT *,               SUM(is_zero) OVER (PARTITION BY currency ORDER BY regDate, linkNo ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS session_id FROM ( SELECT *, CASE WHEN remainAmt = 0 THEN 1 ELSE 0 END AS is_zero FROM trWallet WHERE userNo = :uno and linkNo = :linkno ORDER BY regDate, linkNo ) t1 ) t2) t3 WHERE rn = 1")
        result = await db.execute(query, {"uno": uno, "linkno": setkey})
        rows = result.fetchall()
        return {row.currency: round(float(row.avg_price), 2) for row in rows}
    except Exception as e:
        print(e)
        return {}


def require_login(request: Request):
    user_no = request.session.get("user_No")
    if not user_no:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/"},
            detail="로그인이 필요합니다."
        )
    return user_no  # 필요하다면 user_Name, user_Role도 반환 가능


@app.get("/private")
async def private_page(request: Request, user_session: int = Depends(require_login)):
    return {"msg": f"로그인된 사용자 번호: {user_session}"}


@app.on_event("startup")
async def startup_event():
    # asyncio.create_task(get_new_orderbook_and_save())
    return True


@app.get("/")
async def login_form(request: Request):
    if request.session.get("user_No"):
        uno = request.session.get("user_No")
        return RedirectResponse(url=f"/balance/{uno}", status_code=303)
    return templates.TemplateResponse("login/login.html", {"request": request})


@app.get("/initTrade/{uno}")
async def initrade(request: Request, uno: int, user_session: int = Depends(require_login),
                   db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    usern = request.session.get("user_Name")
    return templates.TemplateResponse("trade/inittrade.html",
                                      {"request": request, "userNo": uno, "user_Name": usern})


@app.post("/loginchk")
async def login(request: Request, response: Response, uid: str = Form(...), upw: str = Form(...),
                db: AsyncSession = Depends(get_db)):
    query = text(
        "SELECT userNo, userName, userRole, setupKey FROM trUser WHERE userId = :username AND userPasswd = password(:password)")
    result = await db.execute(query, {"username": uid, "password": upw})
    user = result.fetchone()
    if user is None:
        return templates.TemplateResponse("login/login.html", {"request": request, "error": "Invalid credentials"})
    else:
        queryu = text("UPDATE trUser SET lastLogin = now() WHERE userId = :username")
        await db.execute(queryu, {"username": uid})
        await db.commit()
    # 서버 세션에 사용자 ID 저장
    request.session["user_No"] = user[0]
    request.session["user_Name"] = user[1]
    request.session["user_Role"] = user[2]
    request.session["setupKey"] = user[3]
    return RedirectResponse(url=f"/balance/{user[0]}", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()  # 세션 삭제
    return RedirectResponse(url="/")

@app.get("/privacy")
async def privacy(request: Request):
    return templates.TemplateResponse("/privacy/privacy.html", {"request": request})


@app.post("/balanceinit/{uno}/{iniamt}")
async def init_balance(request: Request, uno: int, iniamt: float, db: AsyncSession = Depends(get_db)):
    global seckey, mycoins
    mycoins = None
    result = {
        "success": False,
    }
    try:
        query0 = text(f"SELECT * FROM trWallet WHERE userNo = :uno and attrib not like :attxx")
        selres = await db.execute(query0, {"uno": uno, "attxx": "%XXX%"})
        if selres.rowcount > 0:
            query = text(f"UPDATE trWallet set attrib = :attset WHERE userNo = :uno")
            await db.execute(query, {"attset": "XXXUPXXXUP", "uno": uno})
        seckey = datetime.now().strftime("%Y%m%d%H%M%S")
        query2 = text(f"INSERT INTO trWallet (userNo,changeType,currency,unitPrice,inAmt, remainAmt, linkNo) "
                      "values (:uno, 'INITAMT','KRW', '1.0', :inamt, :inamt1, :seckey)")
        await db.execute(query2, {"uno": uno, "inamt": iniamt, "inamt1": iniamt, "seckey": seckey})
        await db.commit()
        query3 = text(f"UPDATE trUser set setupKey = :seckey WHERE userNo = :uno and attrib not like :attxx")
        await db.execute(query3, {"seckey": seckey, "uno": uno, "attxx": '%XXX%'})
        await db.commit()
        mycoins = await get_current_balance(uno, db)
        result = {
            "success": True,
            "setupKey": seckey,
            "userNo": uno,
            "user_Name": request.session.get("user_Name"),
        }
    except Exception as e:
        print("Init Error !!", e)
        mycoins = ([], {})
        result = {
            "success": False,
            "error": str(e)
        }
    request.session["setupKey"] = seckey
    return JSONResponse(content=result)


@app.get("/balance/{uno}")
async def my_balance(request: Request, uno: int, user_session: int = Depends(require_login),
                     db: AsyncSession = Depends(get_db)):
    global myavgp
    mycoins = None
    myavgp = None
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    try:
        mycoins = await get_current_balance(uno, db)
        myavgp = await get_avg_by_coin(uno, request.session.get("setupKey"), db)
        print(myavgp)
    except Exception as e:
        print("Init Error !!", e)
        mycoins = None
    usern = request.session.get("user_Name")
    return templates.TemplateResponse("wallet/mywallet.html",
                                      {"request": request, "userNo": uno, "user_Name": usern, "mycoins": mycoins[0],
                                       "myavgp": myavgp,
                                       "coinprice": mycoins[1]})


@app.get("/balancecrypto/{uno}/{coinn}")
async def my_balance(request: Request, uno: int, coinn: str, user_session: int = Depends(require_login),
                     db: AsyncSession = Depends(get_db)):
    mycoin = {}
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    try:
        mycoins = await get_current_balance(uno, db)
        myavgp = await get_avg_by_coin(uno, request.session.get("setupKey"), db)
        for coin in mycoins[0]:
            if coin[5] == coinn:
                mycoin[coin[5]] = coin[9]
                mycoin["avgPrice"] = myavgp.get(coin[5], 0)
    except Exception as e:
        print("Init Error !!", e)
        mycoin = None
    return mycoin


@app.get("/tradecenter/{uno}")
async def tradecenter(request: Request, uno: int, user_session: int = Depends(require_login),
                      db: AsyncSession = Depends(get_db)):
    global coinlist
    mycoins = None
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    try:
        mycoins = await get_current_balance(uno, db)
        coinlist = await get_krw_tickers()
    except Exception as e:
        print("Init Error !!", e)
    usern = request.session.get("user_Name")
    setkey = request.session.get("setupKey")
    return templates.TemplateResponse("trade/mytrade.html",
                                      {"request": request, "userNo": uno, "user_Name": usern, "mycoins": mycoins[0],
                                       "coinprice": mycoins[1], "setkey": setkey, "coinlist": coinlist})


@app.post("/tradebuymarket/{uno}/{coinn}/{cprice}/{volum}")
async def tradebuymarket(
        request: Request,
        uno: int,
        coinn: str,
        cprice: float,
        volum: float,
        user_session: int = Depends(require_login),
        db: AsyncSession = Depends(get_db)
):
    if uno != user_session:
        return JSONResponse({"success": False, "message": "권한이 없습니다.", "redirect": "/"})
    try:
        butm = await buy_crypto(request, uno, coinn, cprice, volum, db)
        if butm:
            # 거래 성공
            return JSONResponse({"success": True, "redirect": f"/balance/{uno}"})
        else:
            # 거래 실패
            return JSONResponse({"success": False, "message": "거래 실패", "redirect": "/tradecenter"})
    except Exception as e:
        print("Error!!", e)
        return JSONResponse({"success": False, "message": "서버 오류", "redirect": "/tradecenter"})


@app.get("/restaddtradeamt/{bidamt}/{askamt}")
async def restaddtradeamt(request: Request, bidamt: int, askamt: int, db: AsyncSession = Depends(get_db)):
    try:
        act = await rest_add_trade_amt(bidamt, askamt, db)
        return JSONResponse({"success": True})
    except Exception as e:
        print("Error!!", e)
        return JSONResponse({"success": False})


@app.get("/restaddorderbookamt/{datetag}/{idxrow}/{coinn}/{bidamt}/{askamt}/{totalamt}/{amtdiff}")
async def restaddorderbookamt(request: Request, datetag: str, idxrow: int, coinn: str, bidamt: int, askamt: int,
                              totalamt: int, amtdiff: float, db: AsyncSession = Depends(get_db)):
    try:
        act = await rest_add_orderbook_amt(datetag, idxrow, coinn, bidamt, askamt, totalamt, amtdiff, db)
        print(act)
        return JSONResponse({"success": True})
    except Exception as e:
        print("Error!!", e)
        return JSONResponse({"success": False})


@app.get("/restbuymarket/{uno}/{coinn}")
async def resttradebuymarket(request: Request, uno: int, coinn: str, db: AsyncSession = Depends(get_db)):
    try:
        mysets = await get_trsetups(uno, db)
        if mysets:
            for myset in mysets:
                if myset["coinName"] == coinn and myset["useYN"] == "Y":
                    amt = myset["stepAmt"]
                    cprice = await get_current_price(coinn)
                    price = cprice[0]['trade_price']
                    volum = float(amt) / float(price)
                    await rest_buy_crypto(request, uno, coinn, price, volum, db)
                    return JSONResponse({"success": True})
                else:
                    print("설정 없음")
        else:
            print("설정 없음")
    except Exception as e:
        print("Buy Error!!", e)
        return JSONResponse({"success": False})


@app.get("/restbuymarketadd/{uno}/{coinn}")
async def resttradebuymarketadd(request: Request, uno: int, coinn: str, db: AsyncSession = Depends(get_db)):
    try:
        mysets = await get_trsetups(uno, db)
        if mysets:
            for myset in mysets:
                if myset["coinName"] == coinn and myset["useYN"] == "Y":
                    amt = myset["stepAmt"] / 5
                    cprice = await get_current_price(coinn)
                    price = cprice[0]['trade_price']
                    volum = float(amt) / float(price)
                    await rest_buy_crypto(request, uno, coinn, price, volum, db)
                    return JSONResponse({"success": True})
                else:
                    print("설정 없음")
        else:
            print("설정 없음")
    except Exception as e:
        print("Buy Error!!", e)
        return JSONResponse({"success": False})


@app.get("/restsellmarket/{uno}/{coinn}")
async def resttradesellmarket(request: Request, uno: int, coinn: str, db: AsyncSession = Depends(get_db)):
    try:
        mycoins = await get_current_balance(uno, db)
        coin_list, coin_dict = mycoins
        for mycoin in coin_list:
            if mycoin[5] == coinn:
                amt = mycoin[9]
                cprice = await get_current_price(coinn)
                price = cprice[0]['trade_price']
                volum = float(amt)
                await rest_sell_crypto(request, uno, coinn, price, volum, db)
                return JSONResponse({"success": True})
    except Exception as e:
        print("Sell Error!!", e)
        return JSONResponse({"success": False})


@app.get("/restsellcut/{uno}/{coinn}")
async def resttradesellcut(request: Request, uno: int, coinn: str, db: AsyncSession = Depends(get_db)):
    try:
        mycoins = await get_current_balance(uno, db)
        coin_list, coin_dict = mycoins
        for mycoin in coin_list:
            if mycoin[5] == coinn:
                amt = mycoin[9]
                cprice = await get_current_price(coinn)
                price = cprice[0]['trade_price']
                volum = float(amt)
                await rest_cut_crypto(request, uno, coinn, price, volum, db)
                return JSONResponse({"success": True})
    except Exception as e:
        print("Sell Error!!", e)
        return JSONResponse({"success": False})


def tuple_to_list_with_datetime(t):
    result = []
    for item in t:
        if isinstance(item, datetime):
            result.append(item.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            result.append(item)
    return result


@app.get("/restwallet/{uno}")
async def restwallet(request: Request, uno: int, db: AsyncSession = Depends(get_db)):
    try:
        myassets = await get_current_balance(uno, db)
        wallet_list, wallet_dict = myassets
        wallet_list = [tuple_to_list_with_datetime(row) for row in wallet_list]

        return JSONResponse({
            "success": True,
            "data": {
                "wallet_list": wallet_list,
                "wallet_dict": wallet_dict
            }
        })
    except Exception as e:
        print("Get Balance Error!!", e)
        return JSONResponse({"success": False})


@app.get("/restsetup/{uno}")
async def restsetup(request: Request, uno: int, db: AsyncSession = Depends(get_db)):
    try:
        mysets = await get_trsetups(uno, db)
        if mysets:
            return JSONResponse({"success": True, "data": mysets})
        else:
            return JSONResponse({"success": False})
    except Exception as e:
        print("Get Setup Error!!", e)
        return JSONResponse({"success": False})


@app.post("/tradesellmarket/{uno}/{coinn}/{cprice}/{volum}")
async def tradesellmarket(request: Request, uno: int, coinn: str, cprice: float, volum: float,
                          user_session: int = Depends(require_login), db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return JSONResponse({"success": False, "message": "권한이 없습니다.", "redirect": "/"})
    try:
        butm = await sell_crypto(request, uno, coinn, cprice, volum, db)
        if butm:
            # 거래 성공
            return JSONResponse({"success": True, "redirect": f"/balance/{uno}"})
        else:
            # 거래 실패
            return JSONResponse({"success": False, "message": "거래 실패", "redirect": "/tradecenter"})
    except Exception as e:
        print("Error!!", e)
        return JSONResponse({"success": False, "message": "서버 오류", "redirect": "/tradecenter"})


@app.get("/tradelogbook/{uno}")
async def tradelogbook(request: Request, uno: int, user_session: int = Depends(require_login),
                       db: AsyncSession = Depends(get_db)):
    mycoins = None
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    try:
        mycoins = await get_current_balance(uno, db)
    except Exception as e:
        print("Init Error !!", e)
    usern = request.session.get("user_Name")
    setkey = request.session.get("setupKey")
    return templates.TemplateResponse("trade/tradelog.html",
                                      {"request": request, "userNo": uno, "user_Name": usern, "mycoins": mycoins[0],
                                       "coinprice": mycoins[1], "setkey": setkey})


@app.get("/gettradelog/{uno}/{coinn}")
async def gettradelog(request: Request, uno: int, coinn: str, user_session: int = Depends(require_login),
                      db: AsyncSession = Depends(get_db)):
    mylogs = None
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    try:
        mylogs = await get_logbook(request, uno, coinn, db)
        print(mylogs)
        return JSONResponse({"success": True, "data": mylogs})
    except Exception as e:
        print("Get Log Error !!", e)


@app.get("/hotcoin_list/{uno}")
async def hotcoinlist(request: Request, uno: int, user_session: int = Depends(require_login),
                      db: AsyncSession = Depends(get_db)):
    orderbooks = None
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    usern = request.session.get("user_Name")
    setkey = request.session.get("setupKey")
    try:
        orderbooks = await get_hotcoins(request, db)
        hotamt = await get_hotamt(request, db)
        gettime = orderbooks[0][8]
        nowtt = datetime.now()
        diff = nowtt - gettime
        days = diff.days
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60
        seconds = diff.seconds % 60
        time_diff = f"{days}일 {hours}시간 {minutes}분 {seconds}초 "
        is_reloadable = "Y" if diff.total_seconds() > 10800 else "N" # 3시간
        trsetups = await get_trsetups(uno, db)
        return templates.TemplateResponse(
            "/trade/hotcoinlist.html",
            {
                "request": request,
                "userNo": uno,
                "userName": usern,
                "setkey": setkey,
                "orderbooks": orderbooks,
                "time_diff": time_diff,
                "trsetups": trsetups,
                "reloadable": is_reloadable,
                "hotamt": hotamt,
            }
        )
    except Exception as e:
        print("Get Hotcoins Error !!", e)



@app.get("/tradestatus/{uno}")
async def tradestatus(request: Request, uno: int, user_session: int = Depends(require_login),
                      db: AsyncSession = Depends(get_db)):
    global coinlist
    mycoins = None
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    try:
        mycoins = await get_current_balance(uno, db)
        coinlist = await get_krw_tickers()
    except Exception as e:
        print("Init Error !!", e)
    usern = request.session.get("user_Name")
    setkey = request.session.get("setupKey")
    return templates.TemplateResponse("trade/tradestat.html",
                                      {"request": request, "userNo": uno, "user_Name": usern, "mycoins": mycoins[0],
                                       "coinprice": mycoins[1], "setkey": setkey, "coinlist": coinlist})


@app.get("/tradetrend")
async def get_tradetrend():
    return tradetrend


@app.get("/tradesignal")
async def get_tradesignal(request: Request):
    usern = request.session.get("user_Name")
    uno = request.session.get("user_No")
    return templates.TemplateResponse("trade/tradetrend.html",
                                      {"request": request, "userNo": uno, "user_Name": usern, })


@app.get("/tsignal/{coinn}", response_class=HTMLResponse)
async def tsignal(coinn: str):
    coin = coinn.upper()
    if coin not in tradetrend:
        raise HTTPException(status_code=404, detail="Coin not found")
    bulbs_html = make_signal_bulbs(tradetrend[coin])
    style = """
    <style>
    .signal-bulb {
    display: inline-block;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    margin: 0 2px;
    text-align: center;
    line-height: 15px;   /* 폰트와 동일하게 */
    font-weight: bold;
    font-size: 15px;     /* 폰트와 동일하게 */
    vertical-align: middle;
    position: relative;
    }
    .signal-bulb .tf-label {
    display: block;
    font-size: 9px;
    color: #333;
    font-weight: normal;
    line-height: 12px;
    margin-top: -4px;
    }
    .signal-bulb.black { background: #222; color: #fff;}
    .signal-bulb.red { background: #e7505a; }
    .signal-bulb.orange { background: #f7ca18; color: #333;}
    .signal-bulb.green { background: #26c281; }
    </style>
    """
    return style + bulbs_html


@app.websocket("/ws/coinprice/{coinn}")
async def coin_price_ws(websocket: WebSocket, coinn: str, db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    try:
        async for current_price in upbit_ws_price_stream(coinn):
            await websocket.send_json({"coinn": coinn, "current_price": current_price})
    except WebSocketDisconnect:
        print(f"WebSocket disconnected: coin {coinn}")
    except Exception as e:
        print("WebSocket Error:", e)


async def upbit_ws_price_stream(market: str):
    uri = "wss://api.upbit.com/websocket/v1"
    subscribe_data = [{
        "ticket": "test",
    }, {
        "type": "ticker",
        "codes": [market],
        "isOnlyRealtime": True
    }]
    async with websockets.connect(uri, ping_interval=60) as websocket:
        await websocket.send(json.dumps(subscribe_data))
        while True:
            data = await websocket.recv()
            parsed = json.loads(data)
            yield parsed['trade_price']  # 실시간 체결가


@app.get("/tradesetup/{uno}")
async def get_tradesetup(request: Request, uno: int, user_session: int = Depends(require_login),
                         db: AsyncSession = Depends(get_db)):
    global coinlist
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    try:
        setups = await get_trsetups(uno, db)
        mycoins = await get_current_balance(uno, db)
        coinlist = await get_krw_tickers()
        usern = request.session.get("user_Name")
        setkey = request.session.get("setupKey")

        return templates.TemplateResponse("trade/tradesetup.html", {
            "request": request,
            "userNo": uno,
            "user_Name": usern,
            "mycoins": mycoins[0],
            "coinprice": mycoins[1],
            "setkey": setkey,
            "coinlist": coinlist,
            "setups": setups
        })
    except Exception as e:
        print("Init Error !!", e)
        return templates.TemplateResponse(
            "trade/tradesetup.html",
            {
                "request": request,
                "userNo": uno,
                "user_Name": usern,
                "mycoins": [],
                "coinprice": {},
                "setkey": setkey,
                "coinlist": [],
                "setups": []
            })


@app.post("/setuponoff/{setupno}/{onoff}/{uno}")
async def update_setuponoff(uno: int, setupno: int, onoff: str, user_session: int = Depends(require_login),
                            db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    query = text("UPDATE polarisSets SET useYN = :onoff WHERE setupNo = :setupno")
    await db.execute(query, {"onoff": onoff, "setupno": setupno})
    await db.commit()
    return RedirectResponse(url=f"/tradesetup/{uno}", status_code=303)


@app.post("/setupdel/{setupno}/{uno}")
async def update_setupdel(uno: int, setupno: int, user_session: int = Depends(require_login),
                          db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    query = text(f"UPDATE polarisSets set attrib = :attx WHERE setupNo = {setupno}")
    await db.execute(query, {"attx": "XXXUPXXXUP"})
    await db.commit()
    return RedirectResponse(url=f"/tradesetup/{uno}", status_code=303)


@app.post("/insertsetup/{uno}/{coinn}/{setamont}")
async def insert_setup(uno: int, coinn: str, setamont: float, user_session: int = Depends(require_login),
                       db: AsyncSession = Depends(get_db)):
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    query = text(f"INSERT INTO polarisSets (userNo,coinName,stepAmt,maxAmt) "
                 "values (:uno, :coinn ,:setamt, :maxamt)")
    await db.execute(query, {"uno": uno, "coinn": coinn, "setamt": setamont, "maxamt": setamont})
    await db.commit()
    return RedirectResponse(url=f"/tradesetup/{uno}", status_code=303)


UPBIT_CANDLE_URL = "https://api.upbit.com/v1/candles/seconds"
MARKET = "KRW-WCT"


async def fetch_latest_candle():
    async with httpx.AsyncClient() as client:
        params = {"market": MARKET, "count": 1}
        response = await client.get(UPBIT_CANDLE_URL, params=params)
        response.raise_for_status()
        return response.json()[0]


async def fetch_latest_candle():
    async with httpx.AsyncClient() as client:
        params = {"market": MARKET, "count": 1}
        response = await client.get(UPBIT_CANDLE_URL, params=params)
        response.raise_for_status()
        return response.json()[0]


@app.get("/ws-chart", response_class=HTMLResponse)
async def get_chart(request: Request):
    return templates.TemplateResponse("chart.html", {"request": request})


@app.websocket("/ws/upbit-candle")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            candle = await fetch_latest_candle()
            data = {
                "timestamp": candle["candle_date_time_kst"],
                "open": candle["opening_price"],
                "high": candle["high_price"],
                "low": candle["low_price"],
                "close": candle["trade_price"],
                "volume": candle["candle_acc_trade_volume"]
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(5)
    except Exception as e:
        print("WebSocket 종료:", e)
    finally:
        await websocket.close()


@app.get("/hotcoin_reload/{uno}")
async def hotcoin_reload(uno: int, request: Request):
    # (로그인 체크 등 필요하다면 추가)
    result = await get_new_orderbook_and_save()
    if result:
        # 리로드 성공 시 리스트로 리다이렉트
        return RedirectResponse(url=f"/hotcoin_list/{uno}", status_code=303)
    else:
        # 실패 시 에러 페이지 혹은 메시지
        return RedirectResponse(url=f"/hotcoin_list/{uno}?error=reload_failed", status_code=303)


@app.get("/rest_add_predict/{dateTag}/{coinName}/{avgUprate}/{avgDownrate}/{currentPrice}/{predictA}/{predictB}/{predictC}/{predictD}/{rateA}/{rateB}/{rateC}/{rateD}/{intv}")
async def rest_add_predict(request:Request,dateTag:str,coinName:str,avgUprate:float,avgDownrate:float,currentPrice:float,predictA:float,predictB:float,predictC:float,predictD:float,rateA:float,rateB:float,rateC:float,rateD:float,intv:str, db: AsyncSession = Depends(get_db)):
    result = await rest_predict(dateTag,coinName,avgUprate,avgDownrate,currentPrice,predictA,predictB,predictC,predictD,rateA,rateB,rateC,rateD,intv, db)
    if result:
        return True
    else:
        return False


@app.get("/predict_list/{uno}")
async def predictlist(request: Request, uno: int, user_session: int = Depends(require_login),
                      db: AsyncSession = Depends(get_db)):
    predicts = None
    if uno != user_session:
        return RedirectResponse(url="/", status_code=303)
    usern = request.session.get("user_Name")
    setkey = request.session.get("setupKey")
    try:
        predicts = await get_predicts(request, db)
        gettime = predicts[0][15]
        nowtt = datetime.now()
        diff = nowtt - gettime
        days = diff.days
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60
        seconds = diff.seconds % 60
        time_diff = f"{days}일 {hours}시간 {minutes}분 {seconds}초 "
        return templates.TemplateResponse(
            "/trade/predictlist.html",
            {
                "request": request,
                "userNo": uno,
                "userName": usern,
                "setkey": setkey,
                "predicts": predicts,
                "time_diff": time_diff,
            }
        )
    except Exception as e:
        print("Get Hotcoins Error !!", e)

@app.get("/phapp/mlogin/{phoneno}/{passwd}")
async def mlogin(phoneno: str,passwd:str, db: AsyncSession = Depends(get_db)):
    try:
        query = text("SELECT userNo, userName,setupKey from trUser where userId = :phoneno and userPasswd = PASSWORD(:passwd)")
        result = await db.execute(query, {"phoneno": phoneno, "passwd": passwd})
        rows = result.fetchone()
        if rows is None:
            return {"error": "No data found for the given data."}
        result = {"userno": rows[0], "username": rows[1], "setupkey": rows[2]}
    except:
        print("mLogin error")
    finally:
        return result

@app.get("/phapp/hotcoinlist")
async def hotcoins(db: AsyncSession = Depends(get_db)):
    try:
        query = text("SELECT * FROM orderbookAmt where dateTag = (select max(dateTag) from orderbookAmt)")
        result = await db.execute(query)
        rows = result.fetchall()
        orderbooks = [
            {
                "dateTag":row[1],
                "idxRow":row[2],
                "coinName":row[3],
                "bidAmt":row[4],
                "askAmt":row[5],
                "totalAmt":row[6],
                "amtDiff":row[7]
            }
            for row in rows
        ]
        return orderbooks
    except Exception as e:
        print("Get Hotcoins Error !!", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/phapp/tradelog/{uno}")
async def tradelog(uno: int, db: AsyncSession = Depends(get_db)):
    mycoins = None
    try:
        query = text("SELECT changeType, currency,unitPrice,inAmt,outAmt,remainAmt,regDate FROM trWallet where linkNo = (select max(linkNo) from trWallet where userNo = :uno) order by regDate asc")
        result = await db.execute(query, {"uno": uno})
        rows = result.fetchall()
        mycoins = [{
            "changeType":row[0],
            "currency":row[1],
            "unitPrice":row[2],
            "inAmt":row[3],
            "outAmt":row[4],
            "remainAmt":row[5],
            "regDate":row[6]
        }
            for row in rows
        ]
    except Exception as e:
        print("Init Error !!", e)
    return mycoins

@app.get("/phapp/tradesetup/{uno}")
async def phapp_tradesetup(uno: int, db: AsyncSession = Depends(get_db)):
    setups = None
    try:
        query = text("SELECT * FROM polarisSets where userNo = :uno and attrib not like :attxx")
        result = await db.execute(query, {"uno": uno, "attxx": "%XXX%"})
        rows = result.fetchall()
        setups = [
            {
                "coinName":row[2],
                "stepAmt":row[3],
                "tradeType":row[4],
                "maxAmt":row[5],
                "useYN":row[6]
            } for row in rows
        ]
        query2 = text("SELECT changeType, currency,unitPrice,inAmt,outAmt,remainAmt,regDate FROM trWallet where userNo = :uno and attrib not like :attxx order by currency ")
        result2 = await db.execute(query2, {"uno": uno, "attxx": "%XXX%"})
        rows2 = result2.fetchall()
        mycoins = [{
            "changeType": row2[0],
            "currency": row2[1],
            "unitPrice": row2[2],
            "inAmt": row2[3],
            "outAmt": row2[4],
            "remainAmt": row2[5],
            "regDate": row2[6]
        } for row2 in rows2]
        cprices = await get_current_prices()
        return setups, mycoins, cprices
    except Exception as e:
        print("Init Error !!", e)