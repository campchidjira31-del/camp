"""
====================================================================
EMA 145 + ATR Filter — Binance COIN-M Futures (Inverse Contract)
====================================================================
COIN-M vs USDT-M ต่างกันยังไง:

  1) Margin & PnL เป็น crypto (ETH/BTC) ไม่ใช่ USDT
     - ฝาก ETH เป็น margin → กำไร/ขาดทุนเป็น ETH
     - มูลค่าพอร์ตจึง "ผันผวน 2 ชั้น" (PnL + ราคา coin)

  2) Inverse Contract PnL:
     LONG:  PnL = qty × contract_size × (1/entry - 1/exit)  [in coin]
     SHORT: PnL = qty × contract_size × (1/exit - 1/entry)  [in coin]
     → LONG ได้กำไร "น้อยลง" เมื่อราคาขึ้น (diminishing returns)
     → SHORT ได้กำไร "มากขึ้น" เมื่อราคาลง (convex payoff)

  3) Contract Size: ETHUSD = $10/contract, BTCUSD = $100/contract

  4) Fee: Maker 0.01%, Taker 0.05% (จ่ายเป็น coin)
     ❌ ไม่มี BNB discount สำหรับ COIN-M

  5) Funding: ทุก 8h (ไม่ใช่ 4h เหมือน USDT-M ETH)
     Interest: 0.01% per 8h (Binance default)

  6) Data: Binance COIN-M Futures API (dapi)
     Symbol: ETHUSD_PERP (ไม่ใช่ ETHUSDT)

Strategy: EMA 145 + ATR(14) > SMA(ATR,60) × 0.9
No look-ahead bias: ใช้ข้อมูลจาก closed bar เท่านั้น

วิธีรัน:  python3 ema145_atr_coinm.py
====================================================================
"""

import os, time, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    print("❌ pip3 install requests pandas numpy matplotlib")
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

# ── Contract ───────────────────────────────────
SYMBOL          = "ETHUSD_PERP"      # COIN-M perpetual
PAIR            = "ETHUSD"           # pair name for display
CONTRACT_SIZE   = 10                 # USD per contract (ETH=10, BTC=100)
COIN_NAME       = "ETH"

# ── Period ─────────────────────────────────────
START_DATE      = "2023-01-01"
END_DATE        = "2026-01-01"

# ── Capital (in COIN, not USD) ─────────────────
# COIN-M ใช้ coin เป็น margin
INITIAL_CAPITAL_COIN = 1.0           # 1 ETH เป็นทุนเริ่มต้น

# ── EMA ────────────────────────────────────────
EMA_PERIOD      = 145

# ── ATR Filter ─────────────────────────────────
ATR_ENABLED     = True
ATR_PERIOD      = 14
ATR_MA_PERIOD   = 60
ATR_MULTIPLIER  = 0.9

# ── Leverage Grid ──────────────────────────────
LEVERAGES       = [1, 2, 3, 4, 5]

# ── Costs (COIN-M Futures) ────────────────────
# Fee: จ่ายเป็น coin (ไม่มี BNB discount)
FEE_PER_SIDE    = 0.05              # 0.05% taker (เหมือน USDT-M)

# Funding: ทุก 8h สำหรับ COIN-M (ไม่ใช่ 4h เหมือน USDT-M ETH)
FUNDING_RATE    = 0.01              # % per 8h interval
FUNDING_INTERVAL_HOURS = 8          # COIN-M = 8h

# Interest: 0.03%/day = 0.01% per 8h (เหมือนกัน)
INTEREST_RATE_DAILY = 0.03

# ── Monte Carlo ───────────────────────────────
MC_ENABLED      = True
MC_RUNS         = 5000

# ── Warmup ────────────────────────────────────
WARMUP_DAYS     = 300

# ╔═════════════════════════════════════════════════════════════╗
# ║                     จบ SETTINGS                             ║
# ╚═════════════════════════════════════════════════════════════╝


if not END_DATE:
    END_DATE = datetime.now().strftime("%Y-%m-%d")
today_str = datetime.now().strftime("%Y-%m-%d")
if END_DATE > today_str:
    END_DATE = today_str

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Binance COIN-M Futures Data Download
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def binance_coinm_klines(symbol, interval, start_str, end_str, limit=1000):
    """
    ดึงข้อมูลจาก Binance COIN-M Futures API
    Endpoint: https://dapi.binance.com/dapi/v1/klines
    ⚠️ ต่างจาก USDT-M ที่ใช้ fapi หรือ api
    """
    url = "https://dapi.binance.com/dapi/v1/klines"
    start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end_str,   "%Y-%m-%d").timestamp() * 1000)
    all_data = []
    cursor   = start_ms

    while cursor < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': cursor,
            #'endTime': end_ms,
            'limit': limit,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ❌ COIN-M API error: {e}")
            break
        if not data:
            break
        all_data.extend(data)
        cursor = data[-1][6] + 1  # close_time + 1
        if len(data) < limit:
            break
        time.sleep(0.15)

    if not all_data:
        return pd.DataFrame()

    # COIN-M klines format เหมือน spot/USDT-M
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'base_vol', 'trades', 'taker_buy_vol',
        'taker_buy_base_vol', 'ignore'
    ])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['open_time'], unit='ms')
    df.index.name = 'Date'
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[~df.index.duplicated(keep='first')]
    return df.sort_index()


def safe_download(symbol, tf, start_str, end_str):
    print(f"\n📥 COIN-M {tf.upper()} ({start_str} → {end_str})...")
    print(f"   Symbol: {symbol} | Contract: ${CONTRACT_SIZE}/cont")
    df = binance_coinm_klines(symbol, tf, start_str, end_str)
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
#  Indicators (no look-ahead bias)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ทุก indicator คำนวณจาก closed bar เท่านั้น
# EMA, ATR ใช้ Close price → available at bar close
# Signal ประเมินที่ bar close → entry ที่ next bar Open

def calc_atr(df, period=14):
    """Average True Range — ใช้ close ของ bar ก่อนหน้า"""
    high  = df['High']
    low   = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COIN-M Inverse Contract PnL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_coinm_pnl(direction, n_contracts, entry_price, exit_price):
    """
    COIN-M Inverse Contract PnL (in coin units)

    LONG:  PnL = n_contracts × contract_size × (1/entry - 1/exit)
    SHORT: PnL = n_contracts × contract_size × (1/exit - 1/entry)

    ⚠️ ต่างจาก USDT-M ที่ PnL = qty × (exit - entry)
       COIN-M เป็น inverse → convex payoff สำหรับ SHORT
    """
    if direction == 'LONG':
        pnl_coin = n_contracts * CONTRACT_SIZE * (1.0/entry_price - 1.0/exit_price)
    else:  # SHORT
        pnl_coin = n_contracts * CONTRACT_SIZE * (1.0/exit_price - 1.0/entry_price)
    return pnl_coin


def calc_coinm_fee(n_contracts, price):
    """
    Commission in coin = notional_coin × fee_rate
    Notional (coin) = n_contracts × contract_size / price
    """
    notional_coin = n_contracts * CONTRACT_SIZE / price
    return notional_coin * (FEE_PER_SIDE / 100)


def calc_coinm_holding_costs(direction, n_contracts, avg_price, holding_hours):
    """
    Funding Fee + Interest for COIN-M (in coin units)

    Notional (coin) = n_contracts × contract_size / avg_price
    Funding: LONG จ่าย, SHORT รับ (same logic)
    Interest: ทั้ง 2 ฝั่งจ่าย
    """
    notional_coin = n_contracts * CONTRACT_SIZE / avg_price

    # Funding
    n_funding = holding_hours / FUNDING_INTERVAL_HOURS
    funding_rate_decimal = FUNDING_RATE / 100
    if direction == 'LONG':
        funding_cost = notional_coin * funding_rate_decimal * n_funding
    else:
        funding_cost = -notional_coin * funding_rate_decimal * n_funding

    # Interest
    interest_rate_per_hour = (INTEREST_RATE_DAILY / 100) / 24
    interest_cost = notional_coin * interest_rate_per_hour * holding_hours

    return funding_cost, interest_cost


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Backtest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_backtest():
    funding_per_day = 24 / FUNDING_INTERVAL_HOURS * FUNDING_RATE

    print("=" * 75)
    print(f"  {PAIR} COIN-M | EMA {EMA_PERIOD} + ATR Filter | 4H")
    print(f"  ┌─ Contract: ${CONTRACT_SIZE}/cont | Inverse | PnL in {COIN_NAME}")
    print(f"  ├─ Capital : {INITIAL_CAPITAL_COIN} {COIN_NAME}")
    print(f"  ├─ Period  : {START_DATE} → {END_DATE}")
    print(f"  ├─ Leverage: {' / '.join([f'{l}x' for l in LEVERAGES])}")
    print(f"  ├─ Fee     : {FEE_PER_SIDE}% per side (paid in {COIN_NAME})")
    print(f"  ├─ Funding : {FUNDING_RATE}% per {FUNDING_INTERVAL_HOURS}h ({funding_per_day:.3f}%/day)")
    print(f"  ├─ Interest: {INTEREST_RATE_DAILY}%/day")
    print(f"  └─ ATR     : {'ON' if ATR_ENABLED else 'OFF'} "
          f"({ATR_PERIOD}/{ATR_MA_PERIOD}/×{ATR_MULTIPLIER})")
    print("=" * 75)

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    warmup_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    # ---- Download COIN-M 4H ----
    df = safe_download(SYMBOL, '4h', warmup_start, END_DATE)
    if df.empty:
        print("❌ Download failed"); return
    df = tz_strip(df)

    # ---- Indicators (no look-ahead bias) ----
    # EMA, ATR คำนวณจาก Close ของแต่ละแท่ง
    # Signal ดูที่แท่ง i → entry ที่ Open ของแท่ง i+1
    df['EMA']    = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['ATR']    = calc_atr(df, ATR_PERIOD)
    df['ATR_MA'] = df['ATR'].rolling(ATR_MA_PERIOD).mean()

    warmup_bars = len(df[df.index < pd.Timestamp(START_DATE)])
    print(f"\n📐 Warmup: {warmup_bars} แท่ง = {warmup_bars/EMA_PERIOD:.1f}x EMA period "
          f"({'✅' if warmup_bars >= EMA_PERIOD*2 else '⚠️'})")

    bt_start = max(pd.Timestamp(START_DATE), df.index[0])
    df_bt = df[df.index >= bt_start].copy()
    if df_bt.empty:
        print("❌ No data in backtest range"); return

    start_idx = df.index.searchsorted(bt_start)

    # ---- Buy & Hold (in coin terms) ----
    # COIN-M: ถือ coin เท่าเดิม → USD value เปลี่ยนตามราคา
    # แต่เราวัดเป็น coin → B&H = ยังเป็น 1 ETH เท่าเดิม
    # วัดเป็น USD: initial_coin × price_end vs initial_coin × price_start
    bnh_price_start = float(df_bt['Close'].iloc[0])
    bnh_price_end   = float(df_bt['Close'].iloc[-1])
    bnh_usd_start   = INITIAL_CAPITAL_COIN * bnh_price_start
    bnh_usd_end     = INITIAL_CAPITAL_COIN * bnh_price_end
    bnh_ret_usd     = (bnh_usd_end - bnh_usd_start) / bnh_usd_start * 100

    # ========== GENERATE SIGNALS ==========
    # No look-ahead bias:
    # - bar[i] Close, EMA[i], ATR[i] → ทั้งหมดคำนวณจากข้อมูล ≤ bar[i]
    # - Signal ตัดสินใจที่ bar[i] close
    # - Entry ที่ bar[i+1] Open (next bar)
    signals = []
    current_dir = None
    in_position = False

    for i in range(start_idx, len(df) - 1):
        row   = df.iloc[i]
        close = row['Close']
        ema   = row['EMA']

        if pd.isna(ema):
            continue

        # EMA direction (from closed bar)
        if close > ema:
            ema_dir = 'LONG'
        elif close < ema:
            ema_dir = 'SHORT'
        else:
            continue

        # ATR filter (from closed bar — no look-ahead)
        atr_pass = True
        if ATR_ENABLED:
            atr_val = row['ATR']
            atr_ma  = row['ATR_MA']
            if pd.isna(atr_val) or pd.isna(atr_ma) or atr_ma <= 0:
                atr_pass = False
            elif atr_val < atr_ma * ATR_MULTIPLIER:
                atr_pass = False

        if ema_dir != current_dir:
            next_bar = df.iloc[i + 1]
            entry_price = float(next_bar['Open'])  # entry at next bar open

            if atr_pass:
                signals.append({
                    'direction': ema_dir,
                    'entry_price': entry_price,
                    'entry_time': df.index[i + 1],
                    'atr': row['ATR'],
                    'atr_ma': row['ATR_MA'],
                    'filter_pass': True,
                })
                current_dir = ema_dir
                in_position = True
            else:
                # Filter fail → exit to flat
                if in_position:
                    signals.append({
                        'direction': 'FLAT',
                        'entry_price': entry_price,
                        'entry_time': df.index[i + 1],
                        'atr': row['ATR'],
                        'atr_ma': row['ATR_MA'],
                        'filter_pass': False,
                    })
                    in_position = False
                current_dir = ema_dir

    # Convert signals → trades
    trades_raw = []
    i = 0
    while i < len(signals):
        sig = signals[i]
        if sig['direction'] == 'FLAT':
            i += 1; continue

        direction   = sig['direction']
        entry_price = sig['entry_price']
        entry_time  = sig['entry_time']

        if i + 1 < len(signals):
            next_sig = signals[i + 1]
            exit_price = next_sig['entry_price']
            exit_time  = next_sig['entry_time']
            outcome = 'FLAT_EXIT' if next_sig['direction'] == 'FLAT' else 'FLIP'
        else:
            exit_price = float(df['Close'].iloc[-1])
            exit_time  = df.index[-1]
            outcome = 'OPEN'

        trades_raw.append({
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'outcome': outcome,
        })
        i += 1

    print(f"\n   📊 Signals: {len(signals)} | Trades: {len(trades_raw)}")
    if len(trades_raw) < 2:
        print("⚠️  ไม่พอ trades"); return

    # ========== RUN LEVERAGE GRID ==========
    lev_results = {}
    years = max((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 365.25, 0.5)

    for lev in LEVERAGES:
        trades   = []
        eq_curve = []
        capital_coin = INITIAL_CAPITAL_COIN

        for tr in trades_raw:
            direction   = tr['direction']
            entry_price = tr['entry_price']
            exit_price  = tr['exit_price']
            entry_time  = tr['entry_time']
            exit_time   = tr['exit_time']

            holding_hours = (exit_time - entry_time).total_seconds() / 3600

            # ── Position sizing (COIN-M) ──
            # Margin (coin) = capital × leverage → notional USD
            margin_coin    = capital_coin
            notional_usd   = margin_coin * entry_price * lev
            n_contracts    = notional_usd / CONTRACT_SIZE  # จำนวน contracts

            # ── PnL (inverse contract, in coin) ──
            pnl_coin = calc_coinm_pnl(direction, n_contracts, entry_price, exit_price)

            # ── Commission (in coin) ──
            fee_entry = calc_coinm_fee(n_contracts, entry_price)
            fee_exit  = calc_coinm_fee(n_contracts, exit_price)
            commission_coin = fee_entry + fee_exit

            # ── Funding + Interest (in coin) ──
            avg_price = (entry_price + exit_price) / 2
            funding_coin, interest_coin = calc_coinm_holding_costs(
                direction, n_contracts, avg_price, holding_hours)

            # ── Total cost & net PnL (coin) ──
            total_cost_coin = commission_coin + funding_coin + interest_coin
            pnl_net_coin    = pnl_coin - total_cost_coin

            # ── PnL % (on coin capital) ──
            pnl_pct = pnl_net_coin / capital_coin * 100 if capital_coin > 0 else 0

            capital_coin += pnl_net_coin
            if capital_coin <= 0:
                capital_coin = 0

            # ── USD equivalent (for reference) ──
            capital_usd = capital_coin * exit_price

            oc = 'WIN' if pnl_net_coin > 0 else ('LOSS' if pnl_net_coin < 0 else 'BE')

            eq_curve.append({
                'time': exit_time,
                'equity_coin': capital_coin,
                'equity_usd': capital_usd,
                'price': exit_price,
            })
            trades.append({
                'trade_no': len(trades) + 1,
                'entry_time': entry_time, 'exit_time': exit_time,
                'direction': direction,
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'outcome': tr['outcome'],
                'n_contracts': round(n_contracts, 1),
                'holding_hours': round(holding_hours, 1),
                'pnl_coin': round(pnl_coin, 6),
                'commission_coin': round(commission_coin, 6),
                'funding_coin': round(funding_coin, 6),
                'interest_coin': round(interest_coin, 6),
                'total_cost_coin': round(total_cost_coin, 6),
                'pnl_net_coin': round(pnl_net_coin, 6),
                'pnl_pct': round(pnl_pct, 2),
                'capital_coin': round(capital_coin, 6),
                'capital_usd': round(capital_usd, 2),
                'result': oc,
                'leverage': lev,
            })

            if capital_coin <= 0:
                break

        df_t = pd.DataFrame(trades)
        df_e = pd.DataFrame(eq_curve)
        total = len(df_t)
        if total == 0:
            continue

        wins = len(df_t[df_t['result'] == 'WIN'])
        losses = len(df_t[df_t['result'] == 'LOSS'])
        wr = wins / total * 100
        fin_coin = capital_coin
        fin_usd  = fin_coin * float(df['Close'].iloc[-1])
        ret_coin = (fin_coin - INITIAL_CAPITAL_COIN) / INITIAL_CAPITAL_COIN * 100

        init_usd = INITIAL_CAPITAL_COIN * bnh_price_start
        ret_usd  = (fin_usd - init_usd) / init_usd * 100

        # Equity & DD (coin-based)
        eq_arr = np.array([INITIAL_CAPITAL_COIN] + list(df_e['equity_coin']))
        peak   = np.maximum.accumulate(eq_arr)
        dd_arr = (eq_arr - peak) / peak * 100
        max_dd = dd_arr.min()
        avg_dd = dd_arr[dd_arr < 0].mean() if len(dd_arr[dd_arr < 0]) > 0 else 0

        gw = df_t[df_t['pnl_net_coin'] > 0]['pnl_net_coin'].sum()
        gl = abs(df_t[df_t['pnl_net_coin'] < 0]['pnl_net_coin'].sum())
        pf = gw / gl if gl > 0 else float('inf')

        avg_w = df_t[df_t['result'] == 'WIN']['pnl_net_coin'].mean()  if wins   else 0
        avg_l = df_t[df_t['result'] == 'LOSS']['pnl_net_coin'].mean() if losses else 0
        exp   = (wr/100 * avg_w) + ((1 - wr/100) * avg_l)

        if fin_coin > 0 and years > 0:
            cagr_coin = ((fin_coin / INITIAL_CAPITAL_COIN) ** (1/years) - 1) * 100
        else:
            cagr_coin = -100

        pnl_arr = df_t['pnl_pct'].values
        tpy = total / years
        sharpe = (pnl_arr.mean() / pnl_arr.std()) * np.sqrt(tpy) if len(pnl_arr)>1 and pnl_arr.std()>0 else 0
        ds = pnl_arr[pnl_arr < 0]
        sortino = (pnl_arr.mean() / ds.std()) * np.sqrt(tpy) if len(ds)>1 and ds.std()>0 else 0
        calmar = cagr_coin / abs(max_dd) if max_dd != 0 else 0

        lev_results[lev] = {
            'df_t': df_t, 'df_e': df_e,
            'total': total, 'wins': wins, 'losses': losses, 'wr': wr,
            'fin_coin': fin_coin, 'fin_usd': fin_usd,
            'ret_coin': ret_coin, 'ret_usd': ret_usd,
            'max_dd': max_dd, 'avg_dd': avg_dd,
            'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l, 'exp': exp,
            'eq_arr': eq_arr, 'dd_arr': dd_arr,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
            'cagr_coin': cagr_coin,
            'comm_total': df_t['commission_coin'].sum(),
            'funding_total': df_t['funding_coin'].sum(),
            'interest_total': df_t['interest_coin'].sum(),
            'cost_total': df_t['total_cost_coin'].sum(),
            'avg_hold': df_t['holding_hours'].mean(),
        }

    # ========== PRINT RISK GRID ==========
    levs = list(lev_results.keys())
    if not levs:
        print("⚠️  No results"); return

    col_w = 14
    total_w = 28 + col_w * len(levs) + col_w

    print(f"\n{'='*total_w}")
    print(f"  {PAIR} COIN-M | EMA{EMA_PERIOD} + ATR | 4H | Risk Grid")
    print(f"  PnL & Capital in {COIN_NAME} (inverse contract)")
    print(f"  Fund: {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h | "
          f"Int: {INTEREST_RATE_DAILY}%/day | Fee: {FEE_PER_SIDE}%/side")
    print(f"{'='*total_w}")

    hdr = f"  {'':26s}"
    for l in levs:
        hdr += f" {f'{l}x':>{col_w}}"
    hdr += f" {'Buy&Hold':>{col_w}}"
    print(hdr)
    print(f"  {'-'*(total_w-2)}")

    def row(label, vals, fmt, bnh=None):
        s = f"  {label:<26}"
        for v in vals:
            s += f" {fmt.format(v):>{col_w}}"
        if bnh is not None:
            s += f" {fmt.format(bnh):>{col_w}}"
        print(s)

    row(f"ทุนเริ่ม ({COIN_NAME})",
        [INITIAL_CAPITAL_COIN]*len(levs), "{:>10.4f}", INITIAL_CAPITAL_COIN)
    row(f"ทุนสุดท้าย ({COIN_NAME})",
        [lev_results[l]['fin_coin'] for l in levs], "{:>10.4f}", INITIAL_CAPITAL_COIN)
    row(f"ผลตอบแทน ({COIN_NAME})",
        [lev_results[l]['ret_coin'] for l in levs], "{:>9.1f}%", 0)
    row(f"ทุนสุดท้าย (USD)",
        [lev_results[l]['fin_usd'] for l in levs], "${:>9,.0f}", bnh_usd_end)
    row(f"ผลตอบแทน (USD)",
        [lev_results[l]['ret_usd'] for l in levs], "{:>9.1f}%", bnh_ret_usd)
    row("CAGR (coin)",
        [lev_results[l]['cagr_coin'] for l in levs], "{:>9.1f}%")
    print(f"  {'-'*(total_w-2)}")
    row("Max Drawdown (coin)",
        [lev_results[l]['max_dd'] for l in levs], "{:>9.1f}%")
    row("Avg Drawdown",
        [lev_results[l]['avg_dd'] for l in levs], "{:>9.1f}%")
    row("Profit Factor",
        [lev_results[l]['pf'] for l in levs], "{:>10.2f}")
    print(f"  {'-'*(total_w-2)}")
    row("Sharpe Ratio",
        [lev_results[l]['sharpe'] for l in levs], "{:>10.2f}")
    row("Sortino Ratio",
        [lev_results[l]['sortino'] for l in levs], "{:>10.2f}")
    row("Calmar Ratio",
        [lev_results[l]['calmar'] for l in levs], "{:>10.2f}")
    print(f"  {'-'*(total_w-2)}")
    row("จำนวนเทรด",
        [lev_results[l]['total'] for l in levs], "{:>10}")
    row("Win Rate",
        [lev_results[l]['wr'] for l in levs], "{:>9.1f}%")
    row(f"Avg Win ({COIN_NAME})",
        [lev_results[l]['avg_w'] for l in levs], "{:>12.6f}")
    row(f"Avg Loss ({COIN_NAME})",
        [lev_results[l]['avg_l'] for l in levs], "{:>12.6f}")
    row("Avg Hold (hrs)",
        [lev_results[l]['avg_hold'] for l in levs], "{:>9.0f}")

    # Cost breakdown
    print(f"  {'-'*(total_w-2)}")
    print(f"  {'💸 COST (in '+COIN_NAME+')':}")
    row("  Commission",
        [lev_results[l]['comm_total'] for l in levs], "{:>12.6f}")
    row("  Funding Fee",
        [lev_results[l]['funding_total'] for l in levs], "{:>+12.6f}")
    row("  Interest",
        [lev_results[l]['interest_total'] for l in levs], "{:>12.6f}")
    row("  TOTAL COST",
        [lev_results[l]['cost_total'] for l in levs], "{:>12.6f}")
    print(f"{'='*total_w}")

    # Cost as % of initial capital
    print(f"\n  📊 Cost เป็น % ของทุน ({COIN_NAME}):")
    for l in levs:
        r = lev_results[l]
        pct = r['cost_total'] / INITIAL_CAPITAL_COIN * 100
        print(f"     {l}x: {pct:>7.2f}% of initial capital")

    # Direction breakdown
    base_lev = levs[0]
    df_base = lev_results[base_lev]['df_t']
    print(f"\n  📈 แยกตาม Direction ({base_lev}x):")
    for d in ['LONG', 'SHORT']:
        sub = df_base[df_base['direction'] == d]
        if not sub.empty:
            w = len(sub[sub['result'] == 'WIN'])
            wr2 = w / len(sub) * 100
            p = sub['pnl_net_coin'].sum()
            avg_h = sub['holding_hours'].mean()
            print(f"     {d:<7}: {len(sub):>3} trades | WR: {wr2:.1f}% | "
                  f"PnL: {p:+.6f} {COIN_NAME} | Avg Hold: {avg_h:.0f}h")

    # ⚠️ COIN-M specific note
    print(f"\n  ⚠️  COIN-M Note:")
    print(f"     - PnL วัดเป็น {COIN_NAME} (inverse contract)")
    print(f"     - USD value = {COIN_NAME} balance × ราคาปัจจุบัน")
    print(f"     - LONG: diminishing returns เมื่อราคาขึ้นเยอะ")
    print(f"     - SHORT: convex payoff เมื่อราคาลงเยอะ")
    print(f"     - B&H coin = ถือ {INITIAL_CAPITAL_COIN} {COIN_NAME} ตลอด")

    # Save CSV
    best_lev = max(levs, key=lambda l: lev_results[l]['calmar'])
    csv_path = os.path.join(DESKTOP, "ema145_atr_coinm_trades.csv")
    try:
        lev_results[best_lev]['df_t'].to_csv(csv_path, index=False)
        print(f"\n  💾 CSV ({best_lev}x): {csv_path}")
    except Exception as e:
        print(f"\n  ⚠️  CSV: {e}")

    # ========== CHART ==========
    if not HAS_PLOT:
        print("\n✅ COIN-M Backtest เสร็จ!")
        if MC_ENABLED:
            run_monte_carlo(lev_results[best_lev]['df_t'], best_lev)
        return

    try:
        colors_lev = ['#00BFFF', '#00FF88', '#FFD700', '#FF6BFF', '#FF4444']

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
            f'{PAIR} COIN-M — EMA {EMA_PERIOD} + ATR — 4H | Inverse Contract\n'
            f'{START_DATE} → {END_DATE}  |  PnL in {COIN_NAME}  |  '
            f'Fee {FEE_PER_SIDE}%  Fund {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h\n'
            f'${CONTRACT_SIZE}/cont  |  '
            f'Leverage: {" / ".join([f"{l}x" for l in levs])}',
            fontsize=12, fontweight='bold', color='white'
        )

        # Panel 1: Equity (coin)
        ax1 = axes[0]
        t0 = df_bt.index[0]
        for idx, lev in enumerate(levs):
            res = lev_results[lev]
            times = [t0] + list(res['df_e']['time'])
            vals  = [INITIAL_CAPITAL_COIN] + list(res['df_e']['equity_coin'])
            c = colors_lev[idx % len(colors_lev)]
            ax1.plot(times, vals, color=c, lw=2,
                     label=f'{lev}x  {res["ret_coin"]:+.0f}% {COIN_NAME}  '
                           f'SR:{res["sharpe"]:.1f}  Cal:{res["calmar"]:.1f}',
                     zorder=3)

        ax1.axhline(INITIAL_CAPITAL_COIN, color='#FFA500', ls='--', lw=1.5,
                    alpha=0.6, label=f'Buy & Hold {INITIAL_CAPITAL_COIN} {COIN_NAME}')
        ax1.set_ylabel(f'Portfolio ({COIN_NAME})', fontsize=11)
        ax1.set_yscale('log')
        ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                   loc='upper left')
        ax1.grid(True, alpha=0.15, color='white')

        # Panel 2: Drawdown (coin)
        ax2 = axes[1]
        for idx, lev in enumerate(levs):
            res = lev_results[lev]
            times = [t0] + list(res['df_e']['time'])
            c = colors_lev[idx % len(colors_lev)]
            ax2.plot(times, res['dd_arr'], color=c, lw=1.2, alpha=0.8,
                     label=f'{lev}x MaxDD:{res["max_dd"]:.1f}%')
        ax2.axhline(0, color='white', lw=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white',
                   loc='lower left', ncol=2)
        ax2.grid(True, alpha=0.15, color='white')

        # Panel 3: Risk Grid
        ax3 = axes[2]
        x_pos = np.arange(len(levs))
        bar_w = 0.15
        cagrs   = [lev_results[l]['cagr_coin'] for l in levs]
        max_dds = [abs(lev_results[l]['max_dd']) for l in levs]
        sharpes = [lev_results[l]['sharpe'] * 10 for l in levs]
        calmars = [lev_results[l]['calmar'] * 10 for l in levs]

        ax3.bar(x_pos - 1.5*bar_w, cagrs, bar_w,
                label='CAGR %', color='#00FF88', alpha=0.85)
        ax3.bar(x_pos - 0.5*bar_w, [-d for d in max_dds], bar_w,
                label='Max DD %', color='#FF4444', alpha=0.7)
        ax3.bar(x_pos + 0.5*bar_w, sharpes, bar_w,
                label='Sharpe ×10', color='#00BFFF', alpha=0.7)
        ax3.bar(x_pos + 1.5*bar_w, calmars, bar_w,
                label='Calmar ×10', color='#FFD700', alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{l}x' for l in levs], color='white')
        ax3.axhline(0, color='white', lw=0.5)
        ax3.set_title('Risk Grid (coin-based metrics)', fontsize=11, color='white')
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', ncol=4)
        ax3.grid(True, alpha=0.15, color='white')

        # Panel 4: Trade PnL
        ax4 = axes[3]
        df_best = lev_results[best_lev]['df_t']
        pnls = df_best['pnl_net_coin'].values
        colors_bar = ['#00FF88' if p > 0 else '#FF4444' for p in pnls]
        ax4.bar(range(1, len(pnls)+1), pnls, color=colors_bar, alpha=0.8, width=0.8)
        ax4.axhline(0, color='white', lw=0.5)
        ax4.set_ylabel(f'PnL ({COIN_NAME}) per trade', fontsize=11)
        ax4.set_xlabel('Trade #', fontsize=10)
        ax4.set_title(
            f'Best Calmar: {best_lev}x  |  {len(pnls)} trades  |  '
            f'W:{lev_results[best_lev]["wins"]}  L:{lev_results[best_lev]["losses"]}',
            fontsize=10, color='#00BFFF')
        ax4.grid(True, alpha=0.15, color='white')

        plt.tight_layout()
        chart_path = os.path.join(DESKTOP, "ema145_atr_coinm_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  📊 Chart: {chart_path}")
        plt.show()
    except Exception as e:
        print(f"  ⚠️  Chart error: {e}")

    print("\n✅ COIN-M Backtest เสร็จ!")

    if MC_ENABLED:
        run_monte_carlo(lev_results[best_lev]['df_t'], best_lev)

    print(f"\n💡 Tips:")
    print(f"   - COIN-M funding = {FUNDING_INTERVAL_HOURS}h (vs USDT-M ETH = 4h)")
    print(f"   - Inverse PnL: LONG convexity ต่ำ, SHORT convexity สูง")
    print(f"   - เปรียบเทียบ ret_coin vs ret_usd เพื่อเห็น coin exposure")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Monte Carlo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_monte_carlo(df_t, leverage):
    print(f"\n{'='*65}")
    print(f"  🎲 Monte Carlo ({MC_RUNS:,} รอบ) | {leverage}x | COIN-M")
    print(f"{'='*65}")

    pnl_pcts = df_t['pnl_pct'].values.copy() / 100
    n_trades = len(pnl_pcts)
    print(f"  Trades: {n_trades} | Leverage: {leverage}x")
    print(f"  Running...", end="", flush=True)

    np.random.seed(42)
    final_returns = np.zeros(MC_RUNS)
    max_drawdowns = np.zeros(MC_RUNS)
    all_equity    = np.zeros((MC_RUNS, n_trades + 1))

    for run in range(MC_RUNS):
        shuffled = np.random.permutation(pnl_pcts)
        equity = np.ones(n_trades + 1)
        for t in range(n_trades):
            equity[t+1] = equity[t] * (1 + shuffled[t])
            if equity[t+1] <= 0:
                equity[t+1:] = 0; break
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
    prob_blow   = np.mean(final_returns <= -99) * 100
    fin_p = INITIAL_CAPITAL_COIN * (1 + ret_p / 100)

    print(f"\n  {'Pctl':<8} {'Return':>10} {f'Final {COIN_NAME}':>14} {'Max DD':>10}")
    print(f"  {'-'*45}")
    for i, p in enumerate(pcts):
        tag = " ← Worst" if p == 5 else (" ← Median" if p == 50 else (" ← Best" if p == 95 else ""))
        print(f"  {p:>3}th    {ret_p[i]:>+9.1f}%  {fin_p[i]:>12.6f}  {dd_p[i]:>9.1f}%{tag}")
    print(f"  {'-'*45}")
    print(f"  Prob Profit:  {prob_profit:.1f}%")
    print(f"  Prob Blow-up: {prob_blow:.1f}%")
    print(f"  Avg Return:   {np.mean(final_returns):+.1f}%")
    print(f"  Avg Max DD:   {np.mean(max_drawdowns):.1f}%")
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
            f'Monte Carlo | {MC_RUNS:,} runs | {PAIR} COIN-M EMA{EMA_PERIOD}+ATR | {leverage}x\n'
            f'PnL in {COIN_NAME} | Prob Profit: {prob_profit:.1f}% | '
            f'Median: {ret_p[2]:+.1f}%',
            fontsize=11, fontweight='bold', color='white')

        x = np.arange(n_trades + 1)
        sample = np.random.choice(MC_RUNS, min(200, MC_RUNS), replace=False)

        ax1 = axes[0, 0]
        for idx in sample:
            eq = all_equity[idx] * INITIAL_CAPITAL_COIN
            c = '#00FF88' if eq[-1] > INITIAL_CAPITAL_COIN else '#FF4444'
            ax1.plot(x, eq, color=c, alpha=0.04, lw=0.5)
        eq_50 = np.percentile(all_equity, 50, axis=0) * INITIAL_CAPITAL_COIN
        eq_5  = np.percentile(all_equity, 5,  axis=0) * INITIAL_CAPITAL_COIN
        eq_95 = np.percentile(all_equity, 95, axis=0) * INITIAL_CAPITAL_COIN
        ax1.fill_between(x, eq_5, eq_95, alpha=0.15, color='#00BFFF')
        ax1.plot(x, eq_50, color='#00BFFF', lw=2, label='Median')
        ax1.axhline(INITIAL_CAPITAL_COIN, color='gray', ls=':', alpha=0.5)
        ax1.set_title(f'Equity ({COIN_NAME})', fontsize=11)
        ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax1.grid(True, alpha=0.15, color='white')

        ax2 = axes[0, 1]
        ax2.hist(final_returns, bins=80, color='#00BFFF', alpha=0.7, edgecolor='none')
        ax2.axvline(0, color='white', lw=1, alpha=0.5)
        ax2.axvline(ret_p[2], color='#00FF88', lw=2, ls='--',
                    label=f'Median: {ret_p[2]:+.1f}%')
        ax2.set_title('Final Returns %', fontsize=11)
        ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax2.grid(True, alpha=0.15, color='white')

        ax3 = axes[1, 0]
        ax3.hist(max_drawdowns, bins=80, color='#FF4444', alpha=0.7, edgecolor='none')
        ax3.axvline(dd_p[2], color='#FFA500', lw=2, ls='--',
                    label=f'Median: {dd_p[2]:.1f}%')
        ax3.set_title('Max Drawdown', fontsize=11)
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax3.grid(True, alpha=0.15, color='white')

        ax4 = axes[1, 1]
        ax4.axis('off')
        txt = (
            f"COIN-M Monte Carlo\n{'─'*30}\n"
            f"{PAIR} | {leverage}x | {n_trades} trades\n"
            f"PnL in {COIN_NAME}\n{'─'*30}\n"
            f"Prob Profit: {prob_profit:.1f}%\n"
            f"Prob Blow:   {prob_blow:.1f}%\n{'─'*30}\n"
            f"      Return   {COIN_NAME}     MaxDD\n"
        )
        for i, p in enumerate(pcts):
            txt += f"{p:>3}th {ret_p[i]:>+7.1f}%  {fin_p[i]:>8.4f}  {dd_p[i]:>6.1f}%\n"
        ax4.text(0.05, 0.95, txt, transform=ax4.transAxes,
                 fontsize=10, va='top', color='white', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#0a0a1a', alpha=0.9))

        plt.tight_layout()
        mc_path = os.path.join(DESKTOP, "ema145_atr_coinm_mc.png")
        plt.savefig(mc_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  📊 MC: {mc_path}")
        plt.show()
    except Exception as e:
        print(f"  ⚠️  MC chart error: {e}")


if __name__ == "__main__":
    run_backtest()
