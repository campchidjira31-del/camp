"""
EMA 145 + ATR Filter Bot — Binance USDT-M Futures (Mainnet)

โหมด: FILTERED
- Timeframe: 4H
- Symbol: ETHUSDT
- Filter: ATR(14) > MA(ATR, 60) * 0.9
- ADX: ปิด (ไม่ใช้)

สิ่งที่สคริปต์นี้ทำ:
  - ต่อ Binance Futures (mainnet) ด้วย python-binance
  - ดึงแท่งเทียน 4H ล่าสุดมาคำนวณ EMA/ATR เหมือนไฟล์ backtest
  - ทุกครั้งที่แท่ง 4H ปิด จะเช็คสัญญาณ FILTERED:
      * มีสัญญาณ LONG/SHORT ใหม่ → เปิด/กลับฝั่ง position
      * ถ้า filter ไม่ผ่านตอนที่ EMA เปลี่ยนทิศ และมี position อยู่ → ปิด position (FLAT)

ก่อนรัน:
  1) pip install python-binance pandas numpy
  2) ใส่ API key/secret ของคุณด้านล่าง
  3) ตรวจสอบ Leverage/ขนาด position ให้เหมาะกับบัญชีจริง
"""

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException


# ╔═════════════════════════════════════════════════════════════╗
# ║                       CONFIG                               ║
# ╚═════════════════════════════════════════════════════════════╝

# TODO: ใส่ API key/secret ของคุณเองตรงนี้
API_KEY = "fKCmy4M1llg1J3YHcmNQvvic8dn6GjHZkt17vpM9jeZsypAbpFlsTSmwwpgI0l0W"
API_SECRET = "p43NUAWs1GXta9U4WxHo773xTbhjVcgSfvTWLfjKY0mIEiuhcnGoNRtoSQ168xOH"

SYMBOL = "ETHUSDT"
FUTURE_SYMBOL = "ETHUSDT"        # binance ใช้รูปแบบเดียวกันสำหรับ USDT-M futures
TIMEFRAME = Client.KLINE_INTERVAL_4HOUR

# Indicator settings (ให้ตรงกับไฟล์ backtest)
EMA_PERIOD = 145
ATR_PERIOD = 14
ATR_MA_PERIOD = 60
ATR_MULTIPLIER = 0.9

# Filter flags (ตามที่คุณระบุ)
ADX_ENABLED = False   # ปิด ADX
ATR_ENABLED = True    # เปิด ATR filter

# Trading / risk config
LEVERAGE = 3

# เลือกวิธีคำนวณขนาด position:
#   MODE = "fixed_usdt" → ใช้ notional คงที่เป็น USDT
#   MODE = "wallet_pct" → ใช้ % ของ wallet balance
POSITION_SIZE_MODE = "wallet_pct"
FIXED_USDT_NOTIONAL = 100.0      # ถ้าใช้ fixed_usdt
WALLET_RISK_PCT = 50.0            # % ของ USDT wallet ที่ใช้ต่อ 1 เทรด

POLL_INTERVAL_SEC = 60           # ตรวจเช็คทุก ๆ 60 วินาที
HEARTBEAT_INTERVAL_SEC = 30 * 60  # พิมพ์ heartbeat log ทุก 30 นาที
MIN_KLINES = EMA_PERIOD + ATR_MA_PERIOD + 10


# ╔═════════════════════════════════════════════════════════════╗
# ║                    INDICATOR HELPERS                       ║
# ╚═════════════════════════════════════════════════════════════╝

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ให้ตรงกับเวอร์ชันในไฟล์ backtest)"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr


def apply_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """คำนวณ EMA และ ATR + ATR_MA ให้ DataFrame แท่งเทียน"""
    df = df.copy()
    df["EMA"] = df["Close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["ATR"] = calc_atr(df, ATR_PERIOD)
    df["ATR_MA"] = df["ATR"].rolling(ATR_MA_PERIOD).mean()
    return df


# ╔═════════════════════════════════════════════════════════════╗
# ║                SIGNAL LOGIC (FILTERED MODE)                ║
# ╚═════════════════════════════════════════════════════════════╝

def evaluate_filtered_signal(df: pd.DataFrame, last_dir: str, in_position: bool):
    """
    จำลอง logic FILTERED จาก generate_signals() ในไฟล์ backtest
    แต่ประเมินเฉพาะแท่งสุดท้ายที่เพิ่งปิด

    Parameters
    ----------
    df : DataFrame พร้อมคอลัมน์ Open/High/Low/Close/EMA/ATR/ATR_MA
    last_dir : ทิศทาง EMA เดิมที่ใช้เป็น current_dir ('LONG'/'SHORT'/None)
    in_position : ตอนนี้มี position อยู่หรือไม่

    Returns
    -------
    signal : 'LONG' | 'SHORT' | 'FLAT' | None
    new_dir : ค่า ema_dir ปัจจุบัน (ใช้เก็บไว้เป็น last_dir รอบถัดไป)
    """
    if len(df) < 3:
        return None, last_dir

    row = df.iloc[-2]  # ใช้แท่งที่ "ปิดแล้ว" (ก่อนแท่งล่าสุด)
    next_bar = df.iloc[-1]  # แท่งถัดไป ใช้เป็นราคา open สำหรับเข้า/ออก

    close = row["Close"]
    ema = row["EMA"]

    if pd.isna(ema):
        return None, last_dir

    # EMA direction
    if close > ema:
        ema_dir = "LONG"
    elif close < ema:
        ema_dir = "SHORT"
    else:
        ema_dir = last_dir

    # ATR filter
    adx_ok = True  # ADX ปิดอยู่แล้ว
    atr_ok = True

    if ATR_ENABLED:
        atr_val = row.get("ATR", np.nan)
        atr_ma = row.get("ATR_MA", np.nan)
        if pd.isna(atr_val) or pd.isna(atr_ma) or atr_ma <= 0:
            atr_ok = False
        elif atr_val < atr_ma * ATR_MULTIPLIER:
            atr_ok = False

    filters_pass = adx_ok and atr_ok

    signal = None

    if ema_dir != last_dir:
        # Direction change
        if filters_pass:
            # filter ผ่าน → เข้า/flip position
            signal = ema_dir
        else:
            # filter ไม่ผ่าน → ถ้ามี position อยู่ให้ exit to flat
            if in_position:
                signal = "FLAT"

    # ถ้า ema_dir เท่าเดิม หรือไม่มีอะไรเกิดขึ้น → ไม่มีสัญญาณ
    return signal, ema_dir


# ╔═════════════════════════════════════════════════════════════╗
# ║                   BINANCE FUTURES HELPERS                  ║
# ╚═════════════════════════════════════════════════════════════╝

def init_client() -> Client:
    if API_KEY == "YOUR_API_KEY_HERE" or API_SECRET == "YOUR_API_SECRET_HERE":
        raise RuntimeError("กรุณาใส่ API_KEY / API_SECRET ในไฟล์ ema_145_ATR_bot.py ก่อนรัน")
    client = Client(API_KEY, API_SECRET)
    return client


def set_leverage(client: Client):
    try:
        client.futures_change_leverage(symbol=FUTURE_SYMBOL, leverage=LEVERAGE)
    except BinanceAPIException as e:
        print(f"[WARN] ตั้งค่า leverage ไม่สำเร็จ: {e}")


def fetch_klines_df(client: Client) -> pd.DataFrame:
    """
    ดึง klines ล่าสุดจาก futures (USDT-M)
    ใช้ limit 1000 แท่ง (มากพอสำหรับ EMA/ATR)
    """
    raw = client.futures_klines(
        symbol=FUTURE_SYMBOL,
        interval=TIMEFRAME,
        limit=1000,
    )
    cols = [
        "open_time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "close_time",
        "quote_vol",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.rename_axis("Date", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


def get_last_closed_kline_open_time(df: pd.DataFrame):
    """คืนค่าเวลา open ของแท่งที่ปิดแล้วล่าสุด (index อันรองสุดท้าย)"""
    if len(df) < 2:
        return None
    return df.index[-2]


def get_futures_position(client: Client):
    """
    คืนค่า (side, qty, entry_price) ของ position ปัจจุบันบน SYMBOL
    side: 'LONG' | 'SHORT' | None
    """
    positions = client.futures_position_information(symbol=FUTURE_SYMBOL)
    if not positions:
        return None, 0.0, 0.0
    pos = positions[0]
    qty = float(pos["positionAmt"])
    entry_price = float(pos["entryPrice"])

    if qty > 0:
        return "LONG", qty, entry_price
    elif qty < 0:
        return "SHORT", abs(qty), entry_price
    else:
        return None, 0.0, 0.0


def get_wallet_balance(client: Client) -> float:
    acc = client.futures_account_balance()
    for a in acc:
        if a["asset"] == "USDT":
            return float(a["balance"])
    return 0.0


def calc_order_quantity(client: Client, price: float) -> float:
    """คำนวณจำนวน ETH ที่จะเทรด จากโหมดที่เลือก"""
    if POSITION_SIZE_MODE == "fixed_usdt":
        notional = FIXED_USDT_NOTIONAL
    else:
        wallet = get_wallet_balance(client)
        notional = wallet * (WALLET_RISK_PCT / 100.0)

    if notional <= 0 or price <= 0:
        return 0.0

    qty = notional / price * LEVERAGE

    # ปัดขนาด qty ตาม step size (ที่นี่สมมุติ 0.001 ETH ถ้าต้องการแม่นให้ดึง exchangeInfo)
    qty = float(f"{qty:.3f}")
    return qty


def place_market_order(client: Client, side: str, qty: float):
    """
    side: 'BUY' หรือ 'SELL'
    qty: จำนวน ETHUSDT
    """
    if qty <= 0:
        print("[WARN] qty <= 0, ไม่ส่งคำสั่ง")
        return

    try:
        print(f"[ORDER] {side} {qty} {FUTURE_SYMBOL} (MARKET)")
        res = client.futures_create_order(
            symbol=FUTURE_SYMBOL,
            side=side,
            type="MARKET",
            quantity=qty,
        )
        print(f"       orderId={res.get('orderId')}")
    except BinanceAPIException as e:
        print(f"[ERROR] ส่งคำสั่งล้มเหลว: {e}")


def execute_signal(client: Client, signal: str, df: pd.DataFrame,
                   last_dir: str, in_position: bool):
    """
    แปลง signal ('LONG'/'SHORT'/'FLAT') → คำสั่งซื้อขายจริง
    """
    if signal not in ("LONG", "SHORT", "FLAT"):
        return last_dir, in_position

    # ใช้ open ของแท่งปัจจุบัน (แท่งล่าสุด) เป็น reference price
    ref_price = float(df["Open"].iloc[-1])

    current_side, current_qty, entry_price = get_futures_position(client)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{now}] SIGNAL: {signal} | current_pos={current_side} qty={current_qty}")

    if signal == "FLAT":
        if current_side is None:
            print("   - ไม่มี position อยู่แล้ว → ไม่ต้องทำอะไร")
            return last_dir, False
        # ปิด position ทั้งหมด
        side = "SELL" if current_side == "LONG" else "BUY"
        place_market_order(client, side, current_qty)
        return last_dir, False

    # LONG / SHORT
    desired_side = signal
    qty = calc_order_quantity(client, ref_price)

    if qty <= 0:
        print("[WARN] คำนวณ qty ไม่ได้ (wallet น้อยเกินไป?) → ไม่ส่งคำสั่ง")
        return last_dir, in_position

    if current_side is None:
        # ไม่มี position → เปิดใหม่ตามทิศทาง
        side = "BUY" if desired_side == "LONG" else "SELL"
        place_market_order(client, side, qty)
    elif current_side == desired_side:
        # มี position ทิศเดียวกันอยู่แล้ว → ข้าม (ไม่เพิ่มขนาด)
        print("   - มี position ทิศเดียวกันอยู่แล้ว → ข้าม")
        return desired_side, True
    else:
        # มี position ฝั่งตรงข้าม → ปิดเก่า + เปิดใหม่ (flip)
        close_side = "SELL" if current_side == "LONG" else "BUY"
        place_market_order(client, close_side, current_qty)
        open_side = "BUY" if desired_side == "LONG" else "SELL"
        place_market_order(client, open_side, qty)

    return desired_side, True


# ╔═════════════════════════════════════════════════════════════╗
# ║                       MAIN LOOP                             ║
# ╚═════════════════════════════════════════════════════════════╝

def main_loop():
    client = init_client()
    set_leverage(client)

    print("=== EMA 145 + ATR FILTER BOT — Binance USDT-M Futures (MAINNET) ===")
    print(f"Symbol: {FUTURE_SYMBOL} | Timeframe: 4H | Leverage: {LEVERAGE}x")
    print(f"Filters: ATR_ENABLED={ATR_ENABLED}, ATR_PERIOD={ATR_PERIOD}, "
          f"ATR_MA_PERIOD={ATR_MA_PERIOD}, ATR_MULTIPLIER={ATR_MULTIPLIER}")
    print(f"Position sizing mode: {POSITION_SIZE_MODE}")

    last_checked_kline_open = None
    last_dir = None
    in_position = False
    last_heartbeat_ts = 0.0

    while True:
        try:
            now_ts = time.time()

            df = fetch_klines_df(client)
            if len(df) < MIN_KLINES:
                print(f"[INFO] klines น้อยเกินไป ({len(df)}) รอข้อมูลเพิ่ม...")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            df = apply_indicators(df)

            last_closed_open = get_last_closed_kline_open_time(df)
            if last_closed_open is None:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # ถ้าแท่งปิดล่าสุดเคยประมวลผลแล้ว → ยังไม่มีแท่งใหม่
            if last_checked_kline_open is not None and last_closed_open <= last_checked_kline_open:
                # heartbeat log ทุก 30 นาที
                if now_ts - last_heartbeat_ts >= HEARTBEAT_INTERVAL_SEC:
                    side, qty, entry_price = get_futures_position(client)
                    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"[HEARTBEAT {now_str}] waiting for new 4H close | "
                        f"last_closed_bar={last_closed_open} | pos={side} qty={qty}"
                    )
                    last_heartbeat_ts = now_ts

                time.sleep(POLL_INTERVAL_SEC)
                continue

            # มีแท่ง 4H ใหม่ปิด → ประเมินสัญญาณ
            print(f"\n[NEW 4H BAR CLOSED] {last_closed_open}")
            signal, new_dir = evaluate_filtered_signal(df, last_dir, in_position)

            if signal is None:
                print("  - ไม่มีสัญญาณใหม่จาก FILTERED logic")
                last_dir = new_dir
                last_checked_kline_open = last_closed_open
                continue

            last_dir, in_position = execute_signal(
                client, signal, df, last_dir, in_position
            )
            last_checked_kline_open = last_closed_open

        except BinanceAPIException as e:
            print(f"[BinanceAPIException] {e}")
            time.sleep(POLL_INTERVAL_SEC)
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main_loop()

