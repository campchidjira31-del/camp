"""
================================================================
Trend-Following Mean Reversion — EMA + RSI + ATR
================================================================
Trend Filter:
  - ใช้ EMA เพื่อกำหนดโหมดตลาด (Bull / Bear)
  - ถ้า price > EMA → Bull regime → เล่นเฉพาะฝั่ง LONG
  - ถ้า price < EMA → Bear regime → เล่นเฉพาะฝั่ง SHORT

Entry:
  - LONG:  price > EMA  และ  RSI > RSI_LONG_TH
  - SHORT: price < EMA  และ  RSI < RSI_SHORT_TH

Exit (per trade):
  - ตั้ง Stop Loss จาก close ของแท่ง signal ด้วย ATR:
      * LONG:  SL = signal_close - ATR * ATR_MULTIPLIER
      * SHORT: SL = signal_close + ATR * ATR_MULTIPLIER
  - Take Profit: RR (Risk:Reward) เท่ากับ RISK_REWARD (เช่น 1:2)
  - ถ้าในแท่งเดียวกันโดนทั้ง SL และ TP จะถือว่าโดน SL ก่อน (conservative)

TF: ปรับได้ เช่น '1h', '4h', '15m'

Costs (Binance Futures):
  - Commission: taker fee per side
  - Funding Fee: ทุก N ชม.
  - Interest: 0.03%/day (ทั้ง 2 ฝั่งจ่าย)

Risk Grid: Leverage 1x 2x 3x 4x 5x
Metrics: Sharpe, Sortino, Calmar, CAGR, Max/Avg DD

Data: Binance Public API
วิธีรัน:  python3 rsi_mean_revert.py
================================================================
"""

import os, time, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    print("❌ ติดตั้งก่อน: pip3 install requests pandas numpy matplotlib")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

DESKTOP = os.path.expanduser("~/Desktop")


# ╔═════════════════════════════════════════════════════════════╗
# ║                  SETTINGS ปรับตรงนี้                        ║
# ╚═════════════════════════════════════════════════════════════╝

SYMBOL          = "DOTUSDT"
TIMEFRAME       = "1h"                # ปรับได้: '1h', '4h', '15m', '1d'
START_DATE      = "2023-08-01"
END_DATE        = "2024-08-01"        # "" = วันนี้
INITIAL_CAPITAL = 2500

# ── Trend / EMA Settings ─────────────────────
EMA_PERIOD      = 200                 # ใช้กำหนด bull/bear regime

# ── RSI Settings ──────────────────────────────
RSI_PERIOD      = 14
RSI_LONG_TH     = 60                  # Bull regime: RSI > TH → LONG
RSI_SHORT_TH    = 40                  # Bear regime: RSI < TH → SHORT

# ── ATR / Risk Settings ───────────────────────
ATR_PERIOD      = 14
ATR_MULTIPLIER  = 3.5                 # ระยะ SL = ATR * multiplier
RISK_REWARD     = 4.0                 # RR 1:RISK_REWARD (เช่น 2 = 1:2)

# ── Leverage Grid ─────────────────────────────
LEVERAGES       = [1, 2, 3, 4, 5]

# ── Costs (Binance Futures) ───────────────────
FEE_PER_SIDE    = 0.05                # 0.05% per side (taker)
FUNDING_RATE    = 0.01                # % per interval
FUNDING_INTERVAL_HOURS = 4            # ETH=4h, BTC=8h
INTEREST_RATE_DAILY = 0.00            # % per day (Binance fixed)

# ── Monte Carlo ───────────────────────────────
MC_ENABLED      = True
MC_RUNS         = 5000

# ── Warmup ────────────────────────────────────
WARMUP_DAYS     = 200                 # ให้พอสำหรับ EMA/ATR warmup

# ╔═════════════════════════════════════════════════════════════╗
# ║                     จบ SETTINGS                             ║
# ╚═════════════════════════════════════════════════════════════╝


if not END_DATE:
    END_DATE = datetime.now().strftime("%Y-%m-%d")
today_str = datetime.now().strftime("%Y-%m-%d")
if END_DATE > today_str:
    END_DATE = today_str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Binance Data Download
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def binance_klines(symbol, interval, start_str, end_str, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end_str,   "%Y-%m-%d").timestamp() * 1000)
    all_data = []
    cursor   = start_ms

    while cursor < end_ms:
        params = {
            'symbol': symbol, 'interval': interval,
            'startTime': cursor, 'endTime': end_ms, 'limit': limit,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ❌ API error: {e}"); break
        if not data:
            break
        all_data.extend(data)
        cursor = data[-1][6] + 1
        if len(data) < limit:
            break
        time.sleep(0.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['open_time'], unit='ms')
    df.index.name = 'Date'
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[~df.index.duplicated(keep='first')]
    return df.sort_index()


def safe_download(symbol, tf, start_str, end_str):
    print(f"\n📥 {tf.upper()} data ({start_str} → {end_str}) from Binance...")
    df = binance_klines(symbol, tf, start_str, end_str)
    if df.empty:
        print(f"   ❌ ไม่มีข้อมูล")
        return pd.DataFrame()
    print(f"   ✅ {len(df)} แท่ง ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def tz_strip(df):
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RSI / ATR Calculation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_rsi(prices, period=14):
    d  = prices.diff()
    ag = d.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    al = (-d.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


def calc_atr(df, period=14):
    """Average True Range สำหรับ SL distance"""
    high  = df['High']
    low   = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Funding & Interest Calculation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_holding_costs(direction, notional, holding_hours):
    n_funding = holding_hours / FUNDING_INTERVAL_HOURS
    funding_rate_decimal = FUNDING_RATE / 100

    if direction == 'LONG':
        funding_cost = notional * funding_rate_decimal * n_funding
    else:
        funding_cost = -notional * funding_rate_decimal * n_funding

    interest_rate_per_hour = (INTEREST_RATE_DAILY / 100) / 24
    interest_cost = notional * interest_rate_per_hour * holding_hours

    return funding_cost, interest_cost


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Backtest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_backtest():
    funding_per_day = 24 / FUNDING_INTERVAL_HOURS * FUNDING_RATE
    interest_per_day = INTEREST_RATE_DAILY

    print("=" * 70)
    print(f"  {SYMBOL} Trend-Following Mean Reversion | {TIMEFRAME.upper()}")
    print(f"  Period   : {START_DATE} → {END_DATE}")
    print(f"  Trend    : EMA({EMA_PERIOD}) → Bull/Bear filter")
    print(f"  Entry    : Bull: RSI>{RSI_LONG_TH} LONG  |  Bear: RSI<{RSI_SHORT_TH} SHORT")
    print(f"  SL/TP    : SL = close±ATR*{ATR_MULTIPLIER}  |  RR 1:{RISK_REWARD}")
    print(f"  Leverage : {' / '.join([f'{l}x' for l in LEVERAGES])}")
    print(f"  Fee      : {FEE_PER_SIDE}% per side")
    print(f"  Funding  : {FUNDING_RATE}% per {FUNDING_INTERVAL_HOURS}h "
          f"({funding_per_day:.3f}%/day)")
    print(f"  Interest : {INTEREST_RATE_DAILY}%/day")
    print("=" * 70)

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    warmup_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    # ---- Download ----
    df = safe_download(SYMBOL, TIMEFRAME, warmup_start, END_DATE)
    if df.empty:
        print("❌ ไม่สามารถดาวน์โหลดได้"); return
    df = tz_strip(df)

    # ---- Indicators: EMA / RSI / ATR ----
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['RSI'] = calc_rsi(df['Close'], RSI_PERIOD)
    df['ATR'] = calc_atr(df, ATR_PERIOD)

    warmup_bars = len(df[df.index < pd.Timestamp(START_DATE)])
    print(f"\n📐 RSI({RSI_PERIOD}) warmup: {warmup_bars} แท่ง ({TIMEFRAME.upper()}) "
          f"({'✅' if warmup_bars >= RSI_PERIOD*3 else '⚠️ น้อย'})")

    bt_start = max(pd.Timestamp(START_DATE), df.index[0])
    df_bt = df[df.index >= bt_start].copy()
    if df_bt.empty:
        print("❌ ไม่มี data ในช่วง backtest"); return

    start_idx = df.index.searchsorted(bt_start)

    # ---- Buy & Hold ----
    bnh_start = float(df_bt['Close'].iloc[0])
    bnh_end   = float(df_bt['Close'].iloc[-1])
    bnh_final = INITIAL_CAPITAL / bnh_start * bnh_end
    bnh_ret   = (bnh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # ========== DETECT TRADES (Trend + RSI + ATR SL/TP) ==========
    signals = []          # list of trade dicts
    position = None       # None, 'LONG', 'SHORT'
    entry_info = None     # dict สำหรับ trade ที่เปิดอยู่

    # เดินไปตามแท่งตั้งแต่เริ่ม backtest
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        close = row['Close']
        ema   = row['EMA']
        rsi   = row['RSI']
        atr   = row['ATR']

        if pd.isna(ema) or pd.isna(rsi) or pd.isna(atr):
            continue

        high = row['High']
        low  = row['Low']
        t    = df.index[i]

        # 1) ถ้ามี position อยู่ → เช็ค SL / TP บนแท่งนี้ก่อน
        if position is not None and entry_info is not None:
            sl = entry_info['sl']
            tp = entry_info['tp']
            exit_price = None
            outcome = None

            if position == 'LONG':
                # conservative: SL hit ก่อน TP ถ้าโดนทั้งคู่ในแท่งเดียว
                if low <= sl:
                    exit_price = sl
                    outcome = 'SL'
                elif high >= tp:
                    exit_price = tp
                    outcome = 'TP'
            else:  # SHORT
                if high >= sl:
                    exit_price = sl
                    outcome = 'SL'
                elif low <= tp:
                    exit_price = tp
                    outcome = 'TP'

            if exit_price is not None:
                signals.append({
                    **entry_info,
                    'exit_price': float(exit_price),
                    'exit_time': t,
                    'rsi_exit': float(rsi),
                    'outcome': outcome,
                })
                position = None
                entry_info = None
                # หลังจากปิดแล้ว ไม่เปิด trade ใหม่ในแท่งเดียวกัน
                continue

        # 2) ถ้าไม่มี position และยังมีแท่งถัดไปให้เข้าได้ → หา entry ใหม่
        if position is None and i < len(df) - 1:
            # Trend regime จาก EMA
            if close > ema:
                regime = 'BULL'
            elif close < ema:
                regime = 'BEAR'
            else:
                regime = None

            direction = None
            if regime == 'BULL' and close > ema and rsi > RSI_LONG_TH:
                direction = 'LONG'
            elif regime == 'BEAR' and close < ema and rsi < RSI_SHORT_TH:
                direction = 'SHORT'

            if direction is not None:
                next_bar = df.iloc[i + 1]
                entry_price = float(next_bar['Open'])
                entry_time  = df.index[i + 1]

                # คำนวณ SL/TP จาก close ของแท่ง signal + ATR
                if direction == 'LONG':
                    sl = float(close - ATR_MULTIPLIER * atr)
                    risk = max(entry_price - sl, 0)
                    tp = float(entry_price + RISK_REWARD * risk) if risk > 0 else entry_price
                else:
                    sl = float(close + ATR_MULTIPLIER * atr)
                    risk = max(sl - entry_price, 0)
                    tp = float(entry_price - RISK_REWARD * risk) if risk > 0 else entry_price

                entry_info = {
                    'i': i,
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'rsi_entry': float(rsi),
                    'regime': regime,
                    'signal_close': float(close),
                    'atr': float(atr),
                    'sl': sl,
                    'tp': tp,
                }
                position = direction

    # ปิด position ที่ค้างไว้ตอนจบ backtest ที่ราคา close สุดท้าย
    if position is not None and entry_info is not None:
        exit_price = float(df['Close'].iloc[-1])
        exit_time  = df.index[-1]
        signals.append({
            **entry_info,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'rsi_exit': float(df.iloc[-1]['RSI']),
            'outcome': 'OPEN',
        })

    print(f"\n   📊 พบ {len(signals)} trades (entry→exit pairs)")
    if len(signals) < 2:
        print("⚠️  ไม่พอ trades — ลองขยาย period หรือปรับ RSI thresholds")
        return

    # Count time in market
    total_bt_hours = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).total_seconds() / 3600
    in_market_hours = sum(
        (s['exit_time'] - s['entry_time']).total_seconds() / 3600
        for s in signals
    )
    in_market_pct = in_market_hours / total_bt_hours * 100 if total_bt_hours > 0 else 0
    print(f"   ⏱️  Time in market: {in_market_pct:.1f}%")

    # ========== RUN LEVERAGE GRID ==========
    lev_results = {}
    years = max((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 365.25, 0.5)

    for lev in LEVERAGES:
        trades   = []
        eq_curve = []
        capital  = INITIAL_CAPITAL

        for sig in signals:
            direction   = sig['direction']
            entry_price = sig['entry_price']
            entry_time  = sig['entry_time']
            exit_price  = sig['exit_price']
            exit_time   = sig['exit_time']
            outcome     = sig['outcome']

            position_value = capital * lev
            qty = position_value / entry_price

            holding_td = exit_time - entry_time
            holding_hours = holding_td.total_seconds() / 3600

            # PnL
            if direction == 'LONG':
                pnl_raw = (exit_price - entry_price) * qty
            else:
                pnl_raw = (entry_price - exit_price) * qty

            # Commission
            cost_entry = (entry_price * qty) * (FEE_PER_SIDE / 100)
            cost_exit  = (exit_price * qty) * (FEE_PER_SIDE / 100)
            commission = cost_entry + cost_exit

            # Funding + Interest
            avg_price = (entry_price + exit_price) / 2
            notional  = avg_price * qty
            funding_cost, interest_cost = calc_holding_costs(
                direction, notional, holding_hours)

            total_cost = commission + funding_cost + interest_cost
            pnl_net    = pnl_raw - total_cost
            pnl_pct    = pnl_net / capital * 100 if capital > 0 else 0

            capital += pnl_net
            if capital <= 0:
                capital = 0

            oc = 'WIN' if pnl_net > 0 else ('LOSS' if pnl_net < 0 else 'BE')

            eq_curve.append({
                'time': exit_time, 'equity': capital, 'pnl': pnl_net
            })
            trades.append({
                'trade_no': len(trades) + 1,
                'entry_time': entry_time, 'exit_time': exit_time,
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'rsi_entry': round(sig['rsi_entry'], 2),
                'rsi_exit': round(sig.get('rsi_exit', 0), 2),
                'outcome': outcome,
                'holding_hours': round(holding_hours, 1),
                'pnl_raw': round(pnl_raw, 2),
                'commission': round(commission, 2),
                'funding_fee': round(funding_cost, 2),
                'interest_fee': round(interest_cost, 2),
                'total_cost': round(total_cost, 2),
                'pnl_net': round(pnl_net, 2),
                'pnl_pct': round(pnl_pct, 2),
                'result': oc,
                'capital': round(capital, 2),
                'leverage': lev,
            })

            if capital <= 0:
                break

        df_t = pd.DataFrame(trades)
        df_e = pd.DataFrame(eq_curve)

        total = len(df_t)
        if total == 0:
            continue

        wins   = len(df_t[df_t['result'] == 'WIN'])
        losses = len(df_t[df_t['result'] == 'LOSS'])
        wr     = wins / total * 100
        fin    = capital
        ret    = (fin - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        eq_arr = np.array([INITIAL_CAPITAL] + list(df_e['equity']))
        peak   = np.maximum.accumulate(eq_arr)
        dd_arr = (eq_arr - peak) / peak * 100
        max_dd = dd_arr.min()
        avg_dd = dd_arr[dd_arr < 0].mean() if len(dd_arr[dd_arr < 0]) > 0 else 0

        gw = df_t[df_t['pnl_net'] > 0]['pnl_net'].sum()
        gl = abs(df_t[df_t['pnl_net'] < 0]['pnl_net'].sum())
        pf = gw / gl if gl > 0 else float('inf')

        avg_w = df_t[df_t['result'] == 'WIN']['pnl_net'].mean()  if wins   else 0
        avg_l = df_t[df_t['result'] == 'LOSS']['pnl_net'].mean() if losses else 0
        exp   = (wr / 100 * avg_w) + ((1 - wr / 100) * avg_l)

        # CAGR
        if fin > 0 and years > 0:
            cagr = ((fin / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
        else:
            cagr = -100

        # Sharpe
        pnl_arr = df_t['pnl_pct'].values
        trades_per_year = total / years
        if len(pnl_arr) > 1 and pnl_arr.std() > 0:
            sharpe = (pnl_arr.mean() / pnl_arr.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0

        # Sortino
        downside = pnl_arr[pnl_arr < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (pnl_arr.mean() / downside.std()) * np.sqrt(trades_per_year)
        else:
            sortino = 0

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Cost breakdown
        comm_total     = df_t['commission'].sum()
        funding_total  = df_t['funding_fee'].sum()
        interest_total = df_t['interest_fee'].sum()
        all_cost_total = df_t['total_cost'].sum()

        # Avg holding time
        avg_hold = df_t['holding_hours'].mean()

        lev_results[lev] = {
            'df_t': df_t, 'df_e': df_e,
            'total': total, 'wins': wins, 'losses': losses, 'wr': wr,
            'fin': fin, 'ret': ret, 'max_dd': max_dd, 'avg_dd': avg_dd,
            'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l, 'exp': exp,
            'eq_arr': eq_arr, 'dd_arr': dd_arr,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'cagr': cagr,
            'comm_total': comm_total,
            'funding_total': funding_total,
            'interest_total': interest_total,
            'all_cost_total': all_cost_total,
            'avg_hold': avg_hold,
            'in_market_pct': in_market_pct,
        }

    # ========== PRINT RISK GRID ==========
    levs = list(lev_results.keys())
    if not levs:
        print("⚠️  ไม่มีผลลัพธ์"); return

    col_w = 12
    total_w = 26 + col_w * len(levs) + col_w

    print(f"\n{'='*total_w}")
    print(f"  {SYMBOL} Trend-Following Mean Reversion | {TIMEFRAME.upper()}")
    print(f"  Trend: EMA({EMA_PERIOD})  |  Bull: RSI>{RSI_LONG_TH} LONG  |  Bear: RSI<{RSI_SHORT_TH} SHORT")
    print(f"  Fund: {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h | "
          f"Int: {INTEREST_RATE_DAILY}%/day | Fee: {FEE_PER_SIDE}%/side")
    print(f"  Time in market: {in_market_pct:.1f}%")
    print(f"{'='*total_w}")

    hdr = f"  {'':24s}"
    for l in levs:
        hdr += f" {f'{l}x':>{col_w}}"
    hdr += f" {'Buy&Hold':>{col_w}}"
    print(hdr)
    print(f"  {'-'*(total_w-2)}")

    def row(label, vals, fmt, bnh=None):
        s = f"  {label:<24}"
        for v in vals:
            s += f" {fmt.format(v):>{col_w}}"
        if bnh is not None:
            s += f" {fmt.format(bnh):>{col_w}}"
        print(s)

    row("เงินเริ่มต้น",  [INITIAL_CAPITAL]*len(levs), "${:>9,.0f}", INITIAL_CAPITAL)
    row("เงินสุดท้าย",   [lev_results[l]['fin'] for l in levs], "${:>9,.0f}", bnh_final)
    row("ผลตอบแทนรวม",   [lev_results[l]['ret'] for l in levs], "{:>9.1f}%", bnh_ret)
    row("CAGR",          [lev_results[l]['cagr'] for l in levs], "{:>9.1f}%")
    print(f"  {'-'*(total_w-2)}")
    row("Max Drawdown",  [lev_results[l]['max_dd'] for l in levs], "{:>9.1f}%")
    row("Avg Drawdown",  [lev_results[l]['avg_dd'] for l in levs], "{:>9.1f}%")
    row("Profit Factor", [lev_results[l]['pf'] for l in levs], "{:>10.2f}")
    print(f"  {'-'*(total_w-2)}")
    row("Sharpe Ratio",  [lev_results[l]['sharpe'] for l in levs], "{:>10.2f}")
    row("Sortino Ratio", [lev_results[l]['sortino'] for l in levs], "{:>10.2f}")
    row("Calmar Ratio",  [lev_results[l]['calmar'] for l in levs], "{:>10.2f}")
    print(f"  {'-'*(total_w-2)}")
    row("จำนวนเทรด",     [lev_results[l]['total'] for l in levs], "{:>10}")
    row("Win Rate",      [lev_results[l]['wr'] for l in levs], "{:>9.1f}%")
    row("Avg Win",       [lev_results[l]['avg_w'] for l in levs], "${:>9,.0f}")
    row("Avg Loss",      [lev_results[l]['avg_l'] for l in levs], "${:>9,.0f}")
    row("Expectancy",    [lev_results[l]['exp'] for l in levs], "${:>9,.0f}")
    row("Avg Hold (hrs)",[lev_results[l]['avg_hold'] for l in levs], "{:>9.0f}")

    # Cost Breakdown
    print(f"  {'-'*(total_w-2)}")
    print(f"  {'💸 COST BREAKDOWN':}")
    row("  Commission",  [lev_results[l]['comm_total'] for l in levs], "${:>9,.0f}")
    row("  Funding Fee", [lev_results[l]['funding_total'] for l in levs], "${:>9,.0f}")
    row("  Interest",    [lev_results[l]['interest_total'] for l in levs], "${:>9,.0f}")
    row("  TOTAL COST",  [lev_results[l]['all_cost_total'] for l in levs], "${:>9,.0f}")
    print(f"{'='*total_w}")

    # Cost as % of capital
    print(f"\n  📊 Cost เป็น % ของทุน:")
    for l in levs:
        r = lev_results[l]
        pct_comm = r['comm_total'] / INITIAL_CAPITAL * 100
        pct_fund = r['funding_total'] / INITIAL_CAPITAL * 100
        pct_int  = r['interest_total'] / INITIAL_CAPITAL * 100
        pct_all  = r['all_cost_total'] / INITIAL_CAPITAL * 100
        print(f"     {l:>2}x: Comm {pct_comm:>6.1f}%  "
              f"Fund {pct_fund:>+7.1f}%  "
              f"Int {pct_int:>6.1f}%  "
              f"→ Total {pct_all:>7.1f}%")

    # Direction breakdown
    base_lev = levs[0]
    df_base = lev_results[base_lev]['df_t']
    print(f"\n  📈 แยกตาม Direction ({base_lev}x):")
    for d in ['LONG', 'SHORT']:
        sub = df_base[df_base['direction'] == d]
        if not sub.empty:
            w   = len(sub[sub['result'] == 'WIN'])
            wr2 = w / len(sub) * 100
            p   = sub['pnl_net'].sum()
            avg_hold = sub['holding_hours'].mean()
            fund_d  = sub['funding_fee'].sum()
            int_d   = sub['interest_fee'].sum()
            print(f"     {d:<7}: {len(sub):>3} trades | WR: {wr2:.1f}% | "
                  f"PnL: ${p:,.0f} | Avg Hold: {avg_hold:.0f}h | "
                  f"Fund: ${fund_d:,.0f} | Int: ${int_d:,.0f}")

    # Save CSV
    best_lev = max(levs, key=lambda l: lev_results[l]['calmar'])
    csv_path = os.path.join(DESKTOP, "rsi_mean_revert_trades.csv")
    try:
        lev_results[best_lev]['df_t'].to_csv(csv_path, index=False)
        print(f"\n  💾 CSV ({best_lev}x, best Calmar): {csv_path}")
    except Exception as e:
        print(f"\n  ⚠️  CSV save failed: {e}")

    # ========== CHART ==========
    if not HAS_PLOT:
        print("\n✅ Backtest เสร็จสิ้น!")
        if MC_ENABLED:
            run_monte_carlo(lev_results[best_lev]['df_t'], best_lev)
        return

    try:
        colors_lev = ['#00BFFF', '#00FF88', '#FFD700', '#FF6BFF', '#FF4444',
                      '#7B68EE', '#FF8C00', '#00CED1']

        fig, axes = plt.subplots(4, 1, figsize=(18, 22),
                                 gridspec_kw={'height_ratios': [3, 1.2, 1.8, 1.5]})
        fig.patch.set_facecolor('#0f0f0f')
        for ax in axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            ax.xaxis.label.set_color('#cccccc')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')

        fig.suptitle(
            f'{SYMBOL}  RSI({RSI_PERIOD}) Mean Reversion — {TIMEFRAME.upper()}\n'
            f'{START_DATE} → {END_DATE}  |  '
            f'OB>{RSI_OB}→SHORT  OS<{RSI_OS}→LONG  |  '
            f'Fee {FEE_PER_SIDE}%  |  Fund {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h\n'
            f'Time in market: {in_market_pct:.1f}%  |  '
            f'Leverage: {" / ".join([f"{l}x" for l in levs])}',
            fontsize=12, fontweight='bold', color='white'
        )

        # --- Panel 1: Equity curves ---
        ax1 = axes[0]
        t0 = df_bt.index[0]
        for idx, lev in enumerate(levs):
            res = lev_results[lev]
            times = [t0] + list(res['df_e']['time'])
            vals  = [INITIAL_CAPITAL] + list(res['df_e']['equity'])
            c = colors_lev[idx % len(colors_lev)]
            ax1.plot(times, vals, color=c, lw=2,
                     label=f'{lev}x  {res["ret"]:+.0f}%  '
                           f'SR:{res["sharpe"]:.1f}  '
                           f'Cal:{res["calmar"]:.1f}  '
                           f'CAGR:{res["cagr"]:.0f}%',
                     zorder=3)

        bnh_equity = df_bt['Close'] / bnh_start * INITIAL_CAPITAL
        ax1.plot(df_bt.index, bnh_equity, color='#FFA500', lw=1.5,
                 ls='--', label=f'Buy & Hold  {bnh_ret:+.1f}%', alpha=0.7)
        ax1.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.4)
        ax1.set_ylabel('Portfolio Value (USD)', fontsize=11)
        ax1.set_yscale('log')
        ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                   loc='upper left')
        ax1.grid(True, alpha=0.15, color='white')
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # --- Panel 2: Drawdown ---
        ax2 = axes[1]
        for idx, lev in enumerate(levs):
            res = lev_results[lev]
            times = [t0] + list(res['df_e']['time'])
            c = colors_lev[idx % len(colors_lev)]
            ax2.plot(times, res['dd_arr'], color=c, lw=1.2, alpha=0.8,
                     label=f'{lev}x  MaxDD:{res["max_dd"]:.1f}%')
        ax2.axhline(0, color='white', lw=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white',
                   loc='lower left', ncol=2)
        ax2.grid(True, alpha=0.15, color='white')

        # --- Panel 3: Risk Grid Bar Chart ---
        ax3 = axes[2]
        x_pos = np.arange(len(levs))
        bar_w = 0.12

        cagrs     = [lev_results[l]['cagr'] for l in levs]
        max_dds   = [abs(lev_results[l]['max_dd']) for l in levs]
        sharpes   = [lev_results[l]['sharpe'] * 10 for l in levs]
        calmars   = [lev_results[l]['calmar'] * 10 for l in levs]
        costs_pct = [lev_results[l]['all_cost_total'] / INITIAL_CAPITAL * 100
                     for l in levs]

        ax3.bar(x_pos - 2*bar_w, cagrs, bar_w,
                label='CAGR %', color='#00FF88', alpha=0.85)
        ax3.bar(x_pos - bar_w, [-d for d in max_dds], bar_w,
                label='Max DD %', color='#FF4444', alpha=0.7)
        ax3.bar(x_pos, sharpes, bar_w,
                label='Sharpe ×10', color='#00BFFF', alpha=0.7)
        ax3.bar(x_pos + bar_w, calmars, bar_w,
                label='Calmar ×10', color='#FFD700', alpha=0.7)
        ax3.bar(x_pos + 2*bar_w, [-c for c in costs_pct], bar_w,
                label='Total Cost %', color='#FF8888', alpha=0.5)

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{l}x' for l in levs], color='white')
        ax3.axhline(0, color='white', lw=0.5)
        ax3.set_ylabel('Value', fontsize=11)
        ax3.set_xlabel('Leverage', fontsize=11)
        ax3.set_title('Risk Grid + Cost Impact', fontsize=11, color='white')
        ax3.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white', ncol=5)
        ax3.grid(True, alpha=0.15, color='white')

        # --- Panel 4: Trade P/L (best leverage) ---
        ax4 = axes[3]
        df_best = lev_results[best_lev]['df_t']
        trade_pnls = df_best['pnl_net'].values
        total_best = len(trade_pnls)
        colors_bar = ['#00FF88' if p > 0 else '#FF4444' for p in trade_pnls]

        ax4.bar(range(1, total_best + 1), trade_pnls, color=colors_bar,
                alpha=0.8, width=0.8)
        ax4.axhline(0, color='white', lw=0.5)
        ax4.set_ylabel(f'PnL per Trade $ ({best_lev}x)', fontsize=11)
        ax4.set_xlabel('Trade #', fontsize=10)

        wins_best   = len(df_best[df_best['result'] == 'WIN'])
        losses_best = len(df_best[df_best['result'] == 'LOSS'])
        ax4.set_title(
            f'Best Calmar: {best_lev}x  |  Trades: {total_best}  |  '
            f'Wins: {wins_best}  Losses: {losses_best}',
            fontsize=10, color='#00BFFF')
        ax4.grid(True, alpha=0.15, color='white')

        plt.tight_layout()
        chart_path = os.path.join(DESKTOP, "rsi_mean_revert_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  📊 กราฟ: {chart_path}")
        plt.show()

    except Exception as e:
        print(f"  ⚠️  Chart error: {e}")

    print("\n✅ Backtest เสร็จสิ้น!")

    # ========== MONTE CARLO ==========
    if MC_ENABLED:
        print(f"\n  🎲 Monte Carlo ใช้ {best_lev}x (best Calmar)")
        run_monte_carlo(lev_results[best_lev]['df_t'], best_lev)

    print(f"\n💡 Tips:")
    print(f"   - TIMEFRAME = '{TIMEFRAME}' (ลอง '1h', '4h', '1d')")
    print(f"   - EMA_PERIOD = {EMA_PERIOD} (ลอง 100, 150, 200)")
    print(f"   - RSI_LONG_TH / RSI_SHORT_TH = {RSI_LONG_TH} / {RSI_SHORT_TH}")
    print(f"   - ATR_PERIOD = {ATR_PERIOD}, ATR_MULTIPLIER = {ATR_MULTIPLIER}")
    print(f"   - RISK_REWARD = {RISK_REWARD} (เช่น 1.5, 2.0, 3.0)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Monte Carlo Simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_monte_carlo(df_t, leverage):
    print(f"\n{'='*65}")
    print(f"  🎲 Monte Carlo Simulation ({MC_RUNS:,} รอบ) | {leverage}x")
    print(f"{'='*65}")

    pnl_pcts = df_t['pnl_pct'].values.copy() / 100
    n_trades = len(pnl_pcts)

    print(f"  จำนวนเทรด: {n_trades}  |  Leverage: {leverage}x")
    print(f"  กำลังรัน...", end="", flush=True)

    np.random.seed(42)

    final_returns = np.zeros(MC_RUNS)
    max_drawdowns = np.zeros(MC_RUNS)
    all_equity    = np.zeros((MC_RUNS, n_trades + 1))

    for run in range(MC_RUNS):
        shuffled = np.random.permutation(pnl_pcts)
        equity = np.ones(n_trades + 1)
        for t in range(n_trades):
            equity[t + 1] = equity[t] * (1 + shuffled[t])
            if equity[t + 1] <= 0:
                equity[t + 1:] = 0
                break

        all_equity[run] = equity
        final_returns[run] = (equity[-1] - 1) * 100

        peak = np.maximum.accumulate(equity)
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = np.where(peak > 0, (equity - peak) / peak * 100, 0)
        max_drawdowns[run] = dd.min()

    print(" ✅")

    pcts = [5, 25, 50, 75, 95]
    ret_p = np.percentile(final_returns, pcts)
    dd_p  = np.percentile(max_drawdowns, pcts)
    prob_profit = np.mean(final_returns > 0) * 100
    prob_loss   = np.mean(final_returns < 0) * 100
    prob_blow   = np.mean(final_returns <= -99) * 100
    fin_p = INITIAL_CAPITAL * (1 + ret_p / 100)

    print(f"\n  📊 ผลลัพธ์ Monte Carlo ({MC_RUNS:,} รอบ)")
    print(f"  {'-'*55}")
    print(f"  {'Percentile':<18} {'Return':>10} {'Final $':>12} {'Max DD':>10}")
    print(f"  {'-'*55}")
    for i, p in enumerate(pcts):
        tag = ""
        if p == 5:  tag = " ← Worst"
        if p == 50: tag = " ← Median"
        if p == 95: tag = " ← Best"
        print(f"  {p:>3}th percentile   {ret_p[i]:>+9.1f}%  ${fin_p[i]:>10,.0f}  {dd_p[i]:>9.1f}%{tag}")
    print(f"  {'-'*55}")
    print(f"  {'โอกาสกำไร (>0%)':<30} {prob_profit:>6.1f}%")
    print(f"  {'โอกาสขาดทุน (<0%)':<30} {prob_loss:>6.1f}%")
    print(f"  {'โอกาส Blow-up (≤-99%)':<30} {prob_blow:>6.1f}%")
    print(f"  {'Average Return':<30} {np.mean(final_returns):>+6.1f}%")
    print(f"  {'Std Dev Return':<30} {np.std(final_returns):>6.1f}%")
    print(f"  {'Average Max DD':<30} {np.mean(max_drawdowns):>6.1f}%")
    print(f"{'='*65}")

    if not HAS_PLOT:
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor('#0f0f0f')
        for ax in axes.flat:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            ax.xaxis.label.set_color('#cccccc')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')

        fig.suptitle(
            f'Monte Carlo  |  {MC_RUNS:,} runs  |  RSI({RSI_PERIOD}) {SYMBOL} {TIMEFRAME.upper()}  |  {leverage}x\n'
            f'OB>{RSI_OB} OS<{RSI_OS}  |  '
            f'Fund {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h  |  Int {INTEREST_RATE_DAILY}%/day\n'
            f'Prob Profit: {prob_profit:.1f}%  |  Blow-up: {prob_blow:.1f}%  |  '
            f'Median: {ret_p[2]:+.1f}%',
            fontsize=11, fontweight='bold', color='white')

        # Equity Paths
        ax1 = axes[0, 0]
        x = np.arange(n_trades + 1)
        sample_idx = np.random.choice(MC_RUNS, min(200, MC_RUNS), replace=False)
        for idx in sample_idx:
            eq_line = all_equity[idx] * INITIAL_CAPITAL
            color = '#00FF88' if eq_line[-1] > INITIAL_CAPITAL else '#FF4444'
            ax1.plot(x, eq_line, color=color, alpha=0.04, lw=0.5)
        eq_50 = np.percentile(all_equity, 50, axis=0) * INITIAL_CAPITAL
        eq_5  = np.percentile(all_equity, 5,  axis=0) * INITIAL_CAPITAL
        eq_95 = np.percentile(all_equity, 95, axis=0) * INITIAL_CAPITAL
        ax1.fill_between(x, eq_5, eq_95, alpha=0.15, color='#00BFFF')
        ax1.plot(x, eq_50, color='#00BFFF', lw=2, label='Median')
        ax1.plot(x, eq_5,  color='#FF4444', lw=1, ls='--', label='5th')
        ax1.plot(x, eq_95, color='#00FF88', lw=1, ls='--', label='95th')
        ax1.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.5)
        ax1.set_title('Equity Paths', fontsize=11)
        ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax1.grid(True, alpha=0.15, color='white')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))

        # Final Returns
        ax2 = axes[0, 1]
        ax2.hist(final_returns, bins=80, color='#00BFFF', alpha=0.7, edgecolor='none')
        ax2.axvline(0, color='white', lw=1, alpha=0.5)
        ax2.axvline(np.median(final_returns), color='#00FF88', lw=2, ls='--',
                    label=f'Median: {np.median(final_returns):+.1f}%')
        ax2.set_title('Final Returns', fontsize=11)
        ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax2.grid(True, alpha=0.15, color='white')

        # Max Drawdown
        ax3 = axes[1, 0]
        ax3.hist(max_drawdowns, bins=80, color='#FF4444', alpha=0.7, edgecolor='none')
        ax3.axvline(np.median(max_drawdowns), color='#FFA500', lw=2, ls='--',
                    label=f'Median: {np.median(max_drawdowns):.1f}%')
        ax3.set_title('Max Drawdown', fontsize=11)
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax3.grid(True, alpha=0.15, color='white')

        # Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = (
            f"Monte Carlo Summary\n{'─'*35}\n"
            f"Runs: {MC_RUNS:,}  Trades: {n_trades}\n"
            f"Leverage: {leverage}x\n"
            f"RSI({RSI_PERIOD}) OB>{RSI_OB} OS<{RSI_OS}\n"
            f"TF: {TIMEFRAME.upper()}\n{'─'*35}\n"
            f"Prob Profit: {prob_profit:.1f}%\n"
            f"Prob Blow-up: {prob_blow:.1f}%\n"
            f"{'─'*35}\n"
            f"       Return   Final$   MaxDD\n"
        )
        for i, p in enumerate(pcts):
            summary += f"{p:>3}th {ret_p[i]:>+7.1f}% ${fin_p[i]:>7,.0f} {dd_p[i]:>6.1f}%\n"
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                 fontsize=10, va='top', color='white', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#0a0a1a', alpha=0.9))

        plt.tight_layout()
        mc_path = os.path.join(DESKTOP, "rsi_mean_revert_montecarlo.png")
        plt.savefig(mc_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  📊 Monte Carlo: {mc_path}")
        plt.show()
    except Exception as e:
        print(f"  ⚠️  MC chart error: {e}")


if __name__ == "__main__":
    run_backtest()
