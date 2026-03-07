"""
================================================================
Multi-TF Strategy — SuperTrend + RSI(4H) + ADX + Bollinger Bands
================================================================
Higher TF (4H):
  - SuperTrend uptrend  → LONG  |  SuperTrend downtrend → SHORT
  - RSI(4H) > 70        → LONG  |  RSI(4H) < 30         → SHORT
  - ADX(4H) > 40        → strong trend required

Current TF (TIMEFRAME):
  - ADX > 20            → local trend confirmation
  - Entry LONG : price touches lower Bollinger Band → enter at lower BB
  - Entry SHORT: price touches upper Bollinger Band → enter at upper BB
  - SL   LONG : entry - 6 x ATR  (fixed)
  - SL   SHORT: entry + 6 x ATR  (fixed)
  - TP   LONG : upper Bollinger Band (dynamic each bar)
  - TP   SHORT: lower Bollinger Band (dynamic each bar)

Risk: 2% x leverage of account per trade

Data: Binance Public API
Run:  python3 rsi_mean_revert.py
================================================================
"""

import os, time, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    print("Install first: pip3 install requests pandas numpy matplotlib")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

DESKTOP = os.path.expanduser("~/Desktop")


# ============================================================
#  SETTINGS
# ============================================================

SYMBOL          = "ETHUSDT"
TIMEFRAME       = "1h"        # Current / lower trading TF
HTF             = "4h"        # Higher TF for RSI, SuperTrend, ADX
START_DATE      = "2023-01-01"
END_DATE        = "2023-08-01"   # "" = today
INITIAL_CAPITAL = 2500

# -- Higher TF (4H) indicators --------------------------------
USE_HTF_FILTER      = True    # Toggle all 4H filters on/off
RSI_PERIOD_HTF      = 14
RSI_LONG_TH         = 51      # 4H RSI > 70  -> LONG condition
RSI_SHORT_TH        = 49      # 4H RSI < 30  -> SHORT condition

SUPERTREND_PERIOD   = 10
SUPERTREND_MULT     = 1.0

ADX_PERIOD_HTF      = 25
ADX_TH_HTF          = 5      # 4H ADX must exceed this

# -- Current TF indicators ------------------------------------
ADX_PERIOD          = 14
ADX_TH              = 20      # Current TF ADX must be BELOW this (ranging market)

RSI_PERIOD_LTF      = 2
RSI_LTF_LONG_MAX    = 10     # LTF RSI must be below this to confirm long pullback
RSI_LTF_SHORT_MIN   = 90     # LTF RSI must be above this to confirm short pullback

TIME_STOP_BARS      = 120      # Close trade after N bars if no SL/TP hit (~2 days)

BB_PERIOD           = 20
BB_STD              = 3.0     # Wider bands = fewer but higher-quality entries

ATR_PERIOD          = 14
ATR_SL_MULT         = 1.5    # SL = entry +/- ATR_SL_MULT * ATR

# -- Risk management ------------------------------------------
RISK_PCT            = 0.02   # 2% of capital per trade (scaled by leverage)

# -- Leverage grid --------------------------------------------
LEVERAGES           = [1, 3, 4, 5, 10]

# -- Costs (Binance Futures) ----------------------------------
FEE_PER_SIDE            = 0.05   # % taker per side
FUNDING_RATE            = 0.01   # % per interval
FUNDING_INTERVAL_HOURS  = 8      # Binance USDT-M BTC/ETH fund every 8h (not 4h)
INTEREST_RATE_DAILY     = 0.00   # Already included in funding rate on Binance USDT-M

# -- Monte Carlo ----------------------------------------------
MC_ENABLED  = True
MC_RUNS     = 5000

# -- Warmup ---------------------------------------------------
WARMUP_DAYS = 60

# ============================================================

if not END_DATE:
    END_DATE = datetime.now().strftime("%Y-%m-%d")
today_str = datetime.now().strftime("%Y-%m-%d")
if END_DATE > today_str:
    END_DATE = today_str


# ============================================================
#  Binance Data Download
# ============================================================

def binance_klines(symbol, interval, start_str, end_str, limit=1000):
    url      = "https://api.binance.com/api/v3/klines"
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
            print(f"  API error: {e}"); break
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
    print(f"\n  Downloading {symbol} {tf.upper()} ({start_str} -> {end_str})...")
    df = binance_klines(symbol, tf, start_str, end_str)
    if df.empty:
        print(f"   No data returned")
        return pd.DataFrame()
    print(f"   {len(df)} bars  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def tz_strip(df):
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


# ============================================================
#  Indicator Functions
# ============================================================

def calc_rsi(prices, period=14):
    d  = prices.diff()
    ag = d.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    al = (-d.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


def calc_atr(df, period=14):
    high  = df['High']
    low   = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def calc_supertrend(df, period=10, multiplier=3.0):
    """
    Returns (supertrend_line, trend_series).
    trend: 1 = uptrend, -1 = downtrend.
    """
    atr  = calc_atr(df, period)
    hl2  = (df['High'] + df['Low']) / 2

    bu = (hl2 + multiplier * atr).values
    bl = (hl2 - multiplier * atr).values
    close = df['Close'].values
    n = len(df)

    fu = bu.copy()   # final upper band
    fl = bl.copy()   # final lower band
    st = np.full(n, np.nan)
    tr = np.zeros(n, dtype=int)  # 0=uninit, 1=up, -1=down

    for i in range(1, n):
        if np.isnan(bu[i]) or np.isnan(bl[i]):
            continue

        # Carry-forward upper band
        if np.isnan(fu[i - 1]):
            fu[i] = bu[i]
        else:
            fu[i] = bu[i] if (bu[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]

        # Carry-forward lower band
        if np.isnan(fl[i - 1]):
            fl[i] = bl[i]
        else:
            fl[i] = bl[i] if (bl[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]

        if tr[i - 1] == 0:          # first valid bar -> start bearish
            st[i] = fu[i]
            tr[i] = -1
        elif tr[i - 1] == -1:       # was bearish
            if close[i] > fu[i]:
                st[i] = fl[i]; tr[i] = 1
            else:
                st[i] = fu[i]; tr[i] = -1
        else:                        # was bullish
            if close[i] < fl[i]:
                st[i] = fu[i]; tr[i] = -1
            else:
                st[i] = fl[i]; tr[i] = 1

    return pd.Series(st, index=df.index), pd.Series(tr, index=df.index)


def calc_adx(df, period=14):
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up   = high.diff()
    down = -low.diff()
    plus_dm  = pd.Series(np.where((up > down) & (up > 0),   up,   0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    alpha    = 1 / period
    atr_s    = tr.ewm(alpha=alpha, min_periods=period).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=alpha,  min_periods=period).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=alpha, min_periods=period).mean() / atr_s

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx    = 100 * (plus_di - minus_di).abs() / denom
    adx   = dx.ewm(alpha=alpha, min_periods=period).mean()
    return adx


def calc_bollinger(df, period=20, n_std=2.0):
    close = df['Close']
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std(ddof=1)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


# ============================================================
#  Holding Costs
# ============================================================

def calc_holding_costs(direction, notional, holding_hours):
    n_funding        = holding_hours / FUNDING_INTERVAL_HOURS
    funding_decimal  = FUNDING_RATE / 100
    if direction == 'LONG':
        funding_cost = notional * funding_decimal * n_funding
    else:
        funding_cost = -notional * funding_decimal * n_funding
    interest_ph   = (INTEREST_RATE_DAILY / 100) / 24
    interest_cost = notional * interest_ph * holding_hours
    return funding_cost, interest_cost


# ============================================================
#  Main Backtest
# ============================================================

def run_backtest():
    funding_per_day = 24 / FUNDING_INTERVAL_HOURS * FUNDING_RATE

    print("=" * 72)
    print(f"  {SYMBOL}  |  SuperTrend + RSI({HTF.upper()}) + ADX + Bollinger Bands  |  {TIMEFRAME.upper()}")
    print(f"  Period   : {START_DATE} -> {END_DATE}  |  Capital: ${INITIAL_CAPITAL:,.0f}")
    if USE_HTF_FILTER:
        print(f"  HTF ({HTF.upper()}) : SuperTrend({SUPERTREND_PERIOD},{SUPERTREND_MULT})  "
              f"RSI>{RSI_LONG_TH} LONG / <{RSI_SHORT_TH} SHORT  ADX>{ADX_TH_HTF}")
    else:
        print(f"  HTF ({HTF.upper()}) : OFF")
    print(f"  LTF ({TIMEFRAME.upper()}) : ADX<{ADX_TH}  |  BB({BB_PERIOD},{BB_STD})  |  "
          f"SL = entry +/- {ATR_SL_MULT}x ATR({ATR_PERIOD})")
    print(f"  Entry    : LONG @ Lower BB  /  SHORT @ Upper BB")
    print(f"  TP       : LONG @ Upper BB  /  SHORT @ Lower BB  (dynamic)")
    print(f"  Risk     : {RISK_PCT*100:.0f}% x leverage per trade")
    print(f"  Leverage : {' / '.join(f'{l}x' for l in LEVERAGES)}")
    print(f"  Fee      : {FEE_PER_SIDE}%/side  |  Funding: {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h "
          f"({funding_per_day:.3f}%/day)")
    print("=" * 72)

    start_dt     = datetime.strptime(START_DATE, "%Y-%m-%d")
    warmup_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    # ---- Download both timeframes ----
    df_ltf = safe_download(SYMBOL, TIMEFRAME, warmup_start, END_DATE)
    df_htf = safe_download(SYMBOL, HTF,       warmup_start, END_DATE)

    if df_ltf.empty or df_htf.empty:
        print("Download failed"); return

    df_ltf = tz_strip(df_ltf)
    df_htf = tz_strip(df_htf)

    # ---- HTF indicators ----
    df_htf['RSI'] = calc_rsi(df_htf['Close'], RSI_PERIOD_HTF)
    df_htf['ADX'] = calc_adx(df_htf, ADX_PERIOD_HTF)
    st_vals, st_trend    = calc_supertrend(df_htf, SUPERTREND_PERIOD, SUPERTREND_MULT)
    df_htf['ST_val']   = st_vals
    df_htf['ST_trend'] = st_trend   # 1=uptrend, -1=downtrend

    # ---- Current TF indicators ----
    df_ltf['ATR']     = calc_atr(df_ltf, ATR_PERIOD)
    df_ltf['ADX']     = calc_adx(df_ltf, ADX_PERIOD)
    df_ltf['RSI_ltf'] = calc_rsi(df_ltf['Close'], RSI_PERIOD_LTF)
    bb_up, bb_mid, bb_lo = calc_bollinger(df_ltf, BB_PERIOD, BB_STD)
    df_ltf['BB_upper'] = bb_up
    df_ltf['BB_mid']   = bb_mid
    df_ltf['BB_lower'] = bb_lo

    # ---- Merge HTF indicators into LTF via forward-fill ----
    if USE_HTF_FILTER:
        df_ltf['HTF_RSI']      = df_htf['RSI'].shift(1).reindex(df_ltf.index, method='ffill')
        df_ltf['HTF_ADX']      = df_htf['ADX'].shift(1).reindex(df_ltf.index, method='ffill')
        df_ltf['HTF_ST_trend'] = df_htf['ST_trend'].shift(1).reindex(df_ltf.index, method='ffill')
    else:
        df_ltf['HTF_RSI']      = 50
        df_ltf['HTF_ADX']      = 100
        df_ltf['HTF_ST_trend'] = 1  # default to allowing longs/shorts depending on logic

    # ---- Shift current-TF signals by 1 bar to fix look-ahead bias ----
    df_ltf['BB_upper_s']  = df_ltf['BB_upper'].shift(1)
    df_ltf['BB_mid_s']    = df_ltf['BB_mid'].shift(1)
    df_ltf['BB_lower_s']  = df_ltf['BB_lower'].shift(1)
    df_ltf['ADX_s']       = df_ltf['ADX'].shift(1)
    df_ltf['ATR_s']       = df_ltf['ATR'].shift(1)
    df_ltf['RSI_ltf_s']   = df_ltf['RSI_ltf'].shift(1)

    # ---- Clip to backtest window ----
    bt_start  = max(pd.Timestamp(START_DATE), df_ltf.index[0])
    df_bt     = df_ltf[df_ltf.index >= bt_start].copy()
    if df_bt.empty:
        print("No data in backtest window"); return

    start_idx = df_ltf.index.searchsorted(bt_start)

    # ---- Buy & Hold benchmark ----
    bnh_start = float(df_bt['Close'].iloc[0])
    bnh_end   = float(df_bt['Close'].iloc[-1])
    bnh_final = INITIAL_CAPITAL / bnh_start * bnh_end
    bnh_ret   = (bnh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # ============================================================
    #  Signal Detection
    # ============================================================
    signals    = []
    position   = None
    entry_info = None

    required_cols = ['ATR_s', 'ADX_s', 'BB_upper_s', 'BB_mid_s', 'BB_lower_s',
                     'HTF_RSI', 'HTF_ADX', 'HTF_ST_trend', 'RSI_ltf_s']

    for i in range(start_idx, len(df_ltf)):
        row = df_ltf.iloc[i]

        if any(pd.isna(row[c]) for c in required_cols):
            continue

        high     = row['High']
        low      = row['Low']
        t        = df_ltf.index[i]
        atr      = row['ATR_s']       # previous bar's ATR for SL sizing
        bb_upper = row['BB_upper_s']  # previous bar's upper BB
        bb_mid   = row['BB_mid_s']    # previous bar's middle BB
        bb_lower = row['BB_lower_s']  # previous bar's lower BB
        adx      = row['ADX_s']       # previous bar's ADX

        htf_trend = int(row['HTF_ST_trend'])
        htf_rsi   = row['HTF_RSI']
        htf_adx   = row['HTF_ADX']
        rsi_ltf   = row['RSI_ltf_s']

        # ---- 1. Check open position: SL / TP / SuperTrend flip / time stop ----
        if position is not None and entry_info is not None:
            sl         = entry_info['sl']
            exit_price = None
            outcome    = None

            bars_held = i - entry_info['entry_bar']

            # SuperTrend flip: HTF trend reversed against our position
            if USE_HTF_FILTER:
                st_flip = (position == 'LONG'  and htf_trend == -1) or \
                          (position == 'SHORT' and htf_trend ==  1)
            else:
                st_flip = False

            # Time stop
            time_stop = bars_held >= TIME_STOP_BARS

            if position == 'LONG':
                tp = bb_upper          # upper BB is TP, updated each bar
                if low <= sl:
                    exit_price, outcome = sl, 'SL'
                elif high >= tp:
                    exit_price, outcome = tp, 'TP'
                elif st_flip:
                    exit_price, outcome = float(row['Open']), 'ST_FLIP'
                elif time_stop:
                    exit_price, outcome = float(row['Open']), 'TIME'
            else:                       # SHORT
                tp = bb_lower          # lower BB is TP, updated each bar
                if high >= sl:
                    exit_price, outcome = sl, 'SL'
                elif low <= tp:
                    exit_price, outcome = tp, 'TP'
                elif st_flip:
                    exit_price, outcome = float(row['Open']), 'ST_FLIP'
                elif time_stop:
                    exit_price, outcome = float(row['Open']), 'TIME'

            if exit_price is not None:
                signals.append({
                    **entry_info,
                    'exit_price': float(exit_price),
                    'exit_time':  t,
                    'outcome':    outcome,
                })
                position   = None
                entry_info = None
                continue

        # ---- 2. Look for new entry ----
        if position is None:
            if USE_HTF_FILTER:
                bull_htf = (htf_trend == 1  and htf_rsi > RSI_LONG_TH  and htf_adx > ADX_TH_HTF)
                bear_htf = (htf_trend == -1 and htf_rsi < RSI_SHORT_TH and htf_adx > ADX_TH_HTF)
            else:
                bull_htf = True
                bear_htf = True

            adx_ok   = adx < ADX_TH

            direction   = None
            entry_price = None

            # LTF RSI must confirm pullback (oversold for LONG, overbought for SHORT)
            if bull_htf and adx_ok and rsi_ltf < RSI_LTF_LONG_MAX and low <= bb_lower:
                direction   = 'LONG'
                # Limit buy at lower BB; if open gaps below, fill at open
                entry_price = min(float(bb_lower), float(row['Open']))

            elif bear_htf and adx_ok and rsi_ltf > RSI_LTF_SHORT_MIN and high >= bb_upper:
                direction   = 'SHORT'
                # Limit sell at upper BB; if open gaps above, fill at open
                entry_price = max(float(bb_upper), float(row['Open']))

            if direction is not None:
                if direction == 'LONG':
                    sl = entry_price - ATR_SL_MULT * atr
                else:
                    sl = entry_price + ATR_SL_MULT * atr

                entry_info = {
                    'direction':   direction,
                    'entry_price': entry_price,
                    'entry_time':  t,
                    'entry_bar':   i,
                    'atr':         float(atr),
                    'sl':          float(sl),
                }
                position = direction

    # Close any trade still open at last bar
    if position is not None and entry_info is not None:
        last = df_ltf.iloc[-1]
        signals.append({
            **entry_info,
            'exit_price': float(last['Close']),
            'exit_time':  df_ltf.index[-1],
            'outcome':    'OPEN',
        })

    print(f"\n  Trades found: {len(signals)}")
    if len(signals) < 2:
        print("  Too few trades. Try adjusting thresholds or expanding the date range.")
        return

    total_bt_hours  = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).total_seconds() / 3600
    in_market_hours = sum(
        (s['exit_time'] - s['entry_time']).total_seconds() / 3600 for s in signals
    )
    in_market_pct = in_market_hours / total_bt_hours * 100 if total_bt_hours > 0 else 0
    print(f"  Time in market: {in_market_pct:.1f}%")

    # ============================================================
    #  Leverage Grid
    # ============================================================
    lev_results = {}
    years = max((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 365.25, 0.5)

    for lev in LEVERAGES:
        trades   = []
        eq_curve = []
        capital  = INITIAL_CAPITAL

        for sig in signals:
            direction   = sig['direction']
            entry_price = sig['entry_price']
            exit_price  = sig['exit_price']
            entry_time  = sig['entry_time']
            exit_time   = sig['exit_time']
            outcome     = sig['outcome']
            atr_entry   = sig['atr']

            # Position sizing: risk = RISK_PCT * lev * capital
            risk_amount   = capital * RISK_PCT * lev
            risk_per_unit = ATR_SL_MULT * atr_entry
            if risk_per_unit <= 0 or capital <= 0:
                continue
            qty = risk_amount / risk_per_unit

            holding_hours = max(
                (exit_time - entry_time).total_seconds() / 3600, 0
            )

            if direction == 'LONG':
                pnl_raw = (exit_price - entry_price) * qty
            else:
                pnl_raw = (entry_price - exit_price) * qty

            notional   = entry_price * qty
            commission = notional * (FEE_PER_SIDE / 100) * 2

            avg_notional = ((entry_price + exit_price) / 2) * qty
            funding_cost, interest_cost = calc_holding_costs(
                direction, avg_notional, holding_hours
            )

            total_cost = commission + funding_cost + interest_cost
            pnl_net    = pnl_raw - total_cost
            pnl_pct    = pnl_net / capital * 100 if capital > 0 else 0

            capital += pnl_net
            if capital <= 0:
                capital = 0

            oc = 'WIN' if pnl_net > 0 else ('LOSS' if pnl_net < 0 else 'BE')

            eq_curve.append({'time': exit_time, 'equity': capital, 'pnl': pnl_net})
            trades.append({
                'trade_no':     len(trades) + 1,
                'entry_time':   entry_time,
                'exit_time':    exit_time,
                'direction':    direction,
                'entry_price':  round(entry_price, 4),
                'exit_price':   round(exit_price,  4),
                'sl':           round(sig['sl'],    4),
                'outcome':      outcome,
                'holding_hours':round(holding_hours, 1),
                'qty':          round(qty, 4),
                'pnl_raw':      round(pnl_raw,      2),
                'commission':   round(commission,   2),
                'funding_fee':  round(funding_cost, 2),
                'interest_fee': round(interest_cost,2),
                'total_cost':   round(total_cost,   2),
                'pnl_net':      round(pnl_net,      2),
                'pnl_pct':      round(pnl_pct,      2),
                'result':       oc,
                'capital':      round(capital,      2),
                'leverage':     lev,
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
        avg_dd = dd_arr[dd_arr < 0].mean() if (dd_arr < 0).any() else 0

        gw = df_t[df_t['pnl_net'] > 0]['pnl_net'].sum()
        gl = abs(df_t[df_t['pnl_net'] < 0]['pnl_net'].sum())
        pf = gw / gl if gl > 0 else float('inf')

        avg_w = df_t[df_t['result'] == 'WIN']['pnl_net'].mean()  if wins   else 0
        avg_l = df_t[df_t['result'] == 'LOSS']['pnl_net'].mean() if losses else 0
        exp   = (wr / 100 * avg_w) + ((1 - wr / 100) * avg_l)

        cagr = ((fin / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if fin > 0 else -100

        pnl_arr         = df_t['pnl_pct'].values
        trades_per_year = total / years
        sharpe  = ((pnl_arr.mean() / pnl_arr.std()) * np.sqrt(trades_per_year)
                   if len(pnl_arr) > 1 and pnl_arr.std() > 0 else 0)
        down    = pnl_arr[pnl_arr < 0]
        sortino = ((pnl_arr.mean() / down.std()) * np.sqrt(trades_per_year)
                   if len(down) > 1 and down.std() > 0 else 0)
        calmar  = cagr / abs(max_dd) if max_dd != 0 else 0

        comm_total     = df_t['commission'].sum()
        funding_total  = df_t['funding_fee'].sum()
        interest_total = df_t['interest_fee'].sum()
        all_cost_total = df_t['total_cost'].sum()
        avg_hold       = df_t['holding_hours'].mean()

        lev_results[lev] = {
            'df_t': df_t, 'df_e': df_e,
            'total': total, 'wins': wins, 'losses': losses, 'wr': wr,
            'fin': fin, 'ret': ret, 'max_dd': max_dd, 'avg_dd': avg_dd,
            'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l, 'exp': exp,
            'eq_arr': eq_arr, 'dd_arr': dd_arr,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'cagr': cagr,
            'comm_total': comm_total, 'funding_total': funding_total,
            'interest_total': interest_total, 'all_cost_total': all_cost_total,
            'avg_hold': avg_hold, 'in_market_pct': in_market_pct,
        }

    # ============================================================
    #  Print Results Table
    # ============================================================
    levs = list(lev_results.keys())
    if not levs:
        print("No results"); return

    col_w   = 12
    total_w = 26 + col_w * len(levs) + col_w

    print(f"\n{'='*total_w}")
    print(f"  {SYMBOL}  |  SuperTrend+RSI({HTF.upper()})+ADX+BB  |  {TIMEFRAME.upper()}")
    print(f"  HTF: ST({SUPERTREND_PERIOD},{SUPERTREND_MULT})  RSI>{RSI_LONG_TH}/<{RSI_SHORT_TH}  ADX>{ADX_TH_HTF}")
    print(f"  LTF: ADX>{ADX_TH}  BB({BB_PERIOD})  SL=entry+/-{ATR_SL_MULT}xATR  Risk={RISK_PCT*100:.0f}%xlev")
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

    row("Initial Capital",   [INITIAL_CAPITAL]*len(levs), "${:>9,.0f}", INITIAL_CAPITAL)
    row("Final Capital",     [lev_results[l]['fin']  for l in levs], "${:>9,.0f}", bnh_final)
    row("Total Return",      [lev_results[l]['ret']  for l in levs], "{:>9.1f}%",  bnh_ret)
    row("CAGR",              [lev_results[l]['cagr'] for l in levs], "{:>9.1f}%")
    print(f"  {'-'*(total_w-2)}")
    row("Max Drawdown",      [lev_results[l]['max_dd']  for l in levs], "{:>9.1f}%")
    row("Avg Drawdown",      [lev_results[l]['avg_dd']  for l in levs], "{:>9.1f}%")
    row("Profit Factor",     [lev_results[l]['pf']      for l in levs], "{:>10.2f}")
    print(f"  {'-'*(total_w-2)}")
    row("Sharpe Ratio",      [lev_results[l]['sharpe']  for l in levs], "{:>10.2f}")
    row("Sortino Ratio",     [lev_results[l]['sortino'] for l in levs], "{:>10.2f}")
    row("Calmar Ratio",      [lev_results[l]['calmar']  for l in levs], "{:>10.2f}")
    print(f"  {'-'*(total_w-2)}")
    row("Total Trades",      [lev_results[l]['total']   for l in levs], "{:>10}")
    row("Win Rate",          [lev_results[l]['wr']      for l in levs], "{:>9.1f}%")
    row("Avg Win",           [lev_results[l]['avg_w']   for l in levs], "${:>9,.0f}")
    row("Avg Loss",          [lev_results[l]['avg_l']   for l in levs], "${:>9,.0f}")
    row("Expectancy",        [lev_results[l]['exp']     for l in levs], "${:>9,.0f}")
    row("Avg Hold (hrs)",    [lev_results[l]['avg_hold']for l in levs], "{:>9.0f}")
    print(f"  {'-'*(total_w-2)}")
    print(f"  COST BREAKDOWN")
    row("  Commission",      [lev_results[l]['comm_total']     for l in levs], "${:>9,.0f}")
    row("  Funding Fee",     [lev_results[l]['funding_total']  for l in levs], "${:>9,.0f}")
    row("  Interest",        [lev_results[l]['interest_total'] for l in levs], "${:>9,.0f}")
    row("  TOTAL COST",      [lev_results[l]['all_cost_total'] for l in levs], "${:>9,.0f}")
    print(f"{'='*total_w}")

    print(f"\n  Cost as % of capital:")
    for l in levs:
        r  = lev_results[l]
        pc = r['comm_total']     / INITIAL_CAPITAL * 100
        pf = r['funding_total']  / INITIAL_CAPITAL * 100
        pi = r['interest_total'] / INITIAL_CAPITAL * 100
        pa = r['all_cost_total'] / INITIAL_CAPITAL * 100
        print(f"     {l:>2}x: Comm {pc:>6.1f}%  Fund {pf:>+7.1f}%  Int {pi:>6.1f}%  -> Total {pa:>7.1f}%")

    base_lev = levs[0]
    df_base  = lev_results[base_lev]['df_t']
    print(f"\n  Direction breakdown ({base_lev}x):")
    for d in ['LONG', 'SHORT']:
        sub = df_base[df_base['direction'] == d]
        if not sub.empty:
            w   = len(sub[sub['result'] == 'WIN'])
            wr2 = w / len(sub) * 100
            p   = sub['pnl_net'].sum()
            ah  = sub['holding_hours'].mean()
            fd  = sub['funding_fee'].sum()
            print(f"     {d:<7}: {len(sub):>3} trades | WR: {wr2:.1f}% | "
                  f"PnL: ${p:,.0f} | Avg Hold: {ah:.0f}h | Fund: ${fd:,.0f}")

    # Save CSV
    best_lev = max(levs, key=lambda l: lev_results[l]['calmar'])
    csv_path = os.path.join(DESKTOP, "rsi_mean_revert_trades.csv")
    try:
        lev_results[best_lev]['df_t'].to_csv(csv_path, index=False)
        print(f"\n  CSV ({best_lev}x, best Calmar): {csv_path}")
    except Exception as e:
        print(f"\n  CSV save failed: {e}")

    # ============================================================
    #  Chart
    # ============================================================
    if not HAS_PLOT:
        print("\nBacktest complete!")
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
            f'{SYMBOL}  SuperTrend + RSI({RSI_PERIOD_HTF},{HTF.upper()}) + ADX + Bollinger Bands  |  {TIMEFRAME.upper()}\n'
            f'{START_DATE} -> {END_DATE}  |  '
            f'ST({SUPERTREND_PERIOD},{SUPERTREND_MULT})  RSI>{RSI_LONG_TH}/<{RSI_SHORT_TH}  '
            f'ADX4H>{ADX_TH_HTF}  ADX>{ADX_TH}\n'
            f'Entry: Lower/Upper BB  |  SL: +/-{ATR_SL_MULT}xATR  |  '
            f'Risk: {RISK_PCT*100:.0f}%xlev  |  Time in market: {in_market_pct:.1f}%',
            fontsize=12, fontweight='bold', color='white'
        )

        df_bt_plot = df_ltf[df_ltf.index >= bt_start]
        t0 = df_bt_plot.index[0]

        # Panel 1: Equity curves
        ax1 = axes[0]
        for idx, lev in enumerate(levs):
            res   = lev_results[lev]
            times = [t0] + list(res['df_e']['time'])
            vals  = [INITIAL_CAPITAL] + list(res['df_e']['equity'])
            c     = colors_lev[idx % len(colors_lev)]
            ax1.plot(times, vals, color=c, lw=2,
                     label=f'{lev}x  {res["ret"]:+.0f}%  SR:{res["sharpe"]:.1f}  '
                           f'Cal:{res["calmar"]:.1f}  CAGR:{res["cagr"]:.0f}%',
                     zorder=3)

        bnh_equity = df_bt_plot['Close'] / bnh_start * INITIAL_CAPITAL
        ax1.plot(df_bt_plot.index, bnh_equity, color='#FFA500', lw=1.5,
                 ls='--', label=f'Buy & Hold  {bnh_ret:+.1f}%', alpha=0.7)
        ax1.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.4)
        ax1.set_ylabel('Portfolio Value (USD)', fontsize=11)
        ax1.set_yscale('log')
        ax1.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', loc='upper left')
        ax1.grid(True, alpha=0.15, color='white')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Panel 2: Drawdown
        ax2 = axes[1]
        for idx, lev in enumerate(levs):
            res   = lev_results[lev]
            times = [t0] + list(res['df_e']['time'])
            c     = colors_lev[idx % len(colors_lev)]
            ax2.plot(times, res['dd_arr'], color=c, lw=1.2, alpha=0.8,
                     label=f'{lev}x  MaxDD:{res["max_dd"]:.1f}%')
        ax2.axhline(0, color='white', lw=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white',
                   loc='lower left', ncol=2)
        ax2.grid(True, alpha=0.15, color='white')

        # Panel 3: Risk grid bar chart
        ax3 = axes[2]
        x_pos = np.arange(len(levs))
        bar_w = 0.12
        cagrs     = [lev_results[l]['cagr'] for l in levs]
        max_dds   = [abs(lev_results[l]['max_dd']) for l in levs]
        sharpes   = [lev_results[l]['sharpe'] * 10 for l in levs]
        calmars   = [lev_results[l]['calmar'] * 10 for l in levs]
        costs_pct = [lev_results[l]['all_cost_total'] / INITIAL_CAPITAL * 100 for l in levs]

        ax3.bar(x_pos - 2*bar_w, cagrs,                 bar_w, label='CAGR %',      color='#00FF88', alpha=0.85)
        ax3.bar(x_pos -   bar_w, [-d for d in max_dds], bar_w, label='Max DD %',    color='#FF4444', alpha=0.7)
        ax3.bar(x_pos,           sharpes,                bar_w, label='Sharpe x10',  color='#00BFFF', alpha=0.7)
        ax3.bar(x_pos +   bar_w, calmars,                bar_w, label='Calmar x10',  color='#FFD700', alpha=0.7)
        ax3.bar(x_pos + 2*bar_w, [-c for c in costs_pct],bar_w, label='Total Cost %',color='#FF8888', alpha=0.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{l}x' for l in levs], color='white')
        ax3.axhline(0, color='white', lw=0.5)
        ax3.set_ylabel('Value', fontsize=11)
        ax3.set_xlabel('Leverage', fontsize=11)
        ax3.set_title('Risk Grid + Cost Impact', fontsize=11, color='white')
        ax3.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white', ncol=5)
        ax3.grid(True, alpha=0.15, color='white')

        # Panel 4: Trade P/L (best leverage)
        ax4 = axes[3]
        df_best    = lev_results[best_lev]['df_t']
        trade_pnls = df_best['pnl_net'].values
        total_best = len(trade_pnls)
        colors_bar = ['#00FF88' if p > 0 else '#FF4444' for p in trade_pnls]
        ax4.bar(range(1, total_best + 1), trade_pnls, color=colors_bar, alpha=0.8, width=0.8)
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
        print(f"\n  Chart saved: {chart_path}")
        plt.show()

    except Exception as e:
        print(f"  Chart error: {e}")

    print("\nBacktest complete!")

    if MC_ENABLED:
        print(f"\n  Monte Carlo using {best_lev}x (best Calmar)")
        run_monte_carlo(lev_results[best_lev]['df_t'], best_lev)

    print(f"\nTips:")
    print(f"   TIMEFRAME        = '{TIMEFRAME}'  (try '15m', '1h')")
    print(f"   SUPERTREND_MULT  = {SUPERTREND_MULT}  (try 2.0, 3.0, 4.0)")
    print(f"   RSI thresholds   = {RSI_LONG_TH} / {RSI_SHORT_TH}")
    print(f"   ADX_TH_HTF       = {ADX_TH_HTF}   ADX_TH = {ADX_TH}")
    print(f"   ATR_SL_MULT      = {ATR_SL_MULT}")
    print(f"   BB_PERIOD        = {BB_PERIOD}  BB_STD = {BB_STD}")


# ============================================================
#  Monte Carlo Simulation
# ============================================================

def run_monte_carlo(df_t, leverage):
    print(f"\n{'='*65}")
    print(f"  Monte Carlo Simulation ({MC_RUNS:,} runs) | {leverage}x")
    print(f"{'='*65}")

    pnl_pcts = df_t['pnl_pct'].values.copy() / 100
    n_trades = len(pnl_pcts)
    print(f"  Trades: {n_trades}  |  Leverage: {leverage}x")
    print(f"  Running...", end="", flush=True)

    np.random.seed(42)
    final_returns = np.zeros(MC_RUNS)
    max_drawdowns = np.zeros(MC_RUNS)
    all_equity    = np.zeros((MC_RUNS, n_trades + 1))

    for run in range(MC_RUNS):
        shuffled = np.random.permutation(pnl_pcts)
        equity   = np.ones(n_trades + 1)
        for t in range(n_trades):
            equity[t + 1] = equity[t] * (1 + shuffled[t])
            if equity[t + 1] <= 0:
                equity[t + 1:] = 0
                break
        all_equity[run]    = equity
        final_returns[run] = (equity[-1] - 1) * 100
        peak = np.maximum.accumulate(equity)
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = np.where(peak > 0, (equity - peak) / peak * 100, 0)
        max_drawdowns[run] = dd.min()

    print(" done")

    pcts        = [5, 25, 50, 75, 95]
    ret_p       = np.percentile(final_returns, pcts)
    dd_p        = np.percentile(max_drawdowns, pcts)
    prob_profit = np.mean(final_returns > 0)  * 100
    prob_loss   = np.mean(final_returns < 0)  * 100
    prob_blow   = np.mean(final_returns <= -99) * 100
    fin_p       = INITIAL_CAPITAL * (1 + ret_p / 100)

    print(f"\n  Results ({MC_RUNS:,} runs)")
    print(f"  {'-'*55}")
    print(f"  {'Percentile':<18} {'Return':>10} {'Final $':>12} {'Max DD':>10}")
    print(f"  {'-'*55}")
    for i, p in enumerate(pcts):
        tag = ""
        if p == 5:  tag = " <- Worst"
        if p == 50: tag = " <- Median"
        if p == 95: tag = " <- Best"
        print(f"  {p:>3}th percentile   {ret_p[i]:>+9.1f}%  "
              f"${fin_p[i]:>10,.0f}  {dd_p[i]:>9.1f}%{tag}")
    print(f"  {'-'*55}")
    print(f"  {'Prob Profit (>0%)':<30} {prob_profit:>6.1f}%")
    print(f"  {'Prob Loss (<0%)':<30} {prob_loss:>6.1f}%")
    print(f"  {'Prob Blow-up (<=-99%)':<30} {prob_blow:>6.1f}%")
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
            f'Monte Carlo  |  {MC_RUNS:,} runs  |  {SYMBOL} {TIMEFRAME.upper()}  |  {leverage}x\n'
            f'ST({SUPERTREND_PERIOD},{SUPERTREND_MULT})  RSI({RSI_PERIOD_HTF},{HTF.upper()}) '
            f'>{RSI_LONG_TH}/<{RSI_SHORT_TH}  ADX4H>{ADX_TH_HTF}\n'
            f'Prob Profit: {prob_profit:.1f}%  |  Blow-up: {prob_blow:.1f}%  |  '
            f'Median: {ret_p[2]:+.1f}%',
            fontsize=11, fontweight='bold', color='white')

        # Equity paths
        ax1 = axes[0, 0]
        x   = np.arange(n_trades + 1)
        sample_idx = np.random.choice(MC_RUNS, min(200, MC_RUNS), replace=False)
        for idx in sample_idx:
            eq_line = all_equity[idx] * INITIAL_CAPITAL
            color   = '#00FF88' if eq_line[-1] > INITIAL_CAPITAL else '#FF4444'
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

        # Final returns histogram
        ax2 = axes[0, 1]
        ax2.hist(final_returns, bins=80, color='#00BFFF', alpha=0.7, edgecolor='none')
        ax2.axvline(0, color='white', lw=1, alpha=0.5)
        ax2.axvline(np.median(final_returns), color='#00FF88', lw=2, ls='--',
                    label=f'Median: {np.median(final_returns):+.1f}%')
        ax2.set_title('Final Returns Distribution', fontsize=11)
        ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax2.grid(True, alpha=0.15, color='white')

        # Max drawdown histogram
        ax3 = axes[1, 0]
        ax3.hist(max_drawdowns, bins=80, color='#FF4444', alpha=0.7, edgecolor='none')
        ax3.axvline(np.median(max_drawdowns), color='#FFA500', lw=2, ls='--',
                    label=f'Median: {np.median(max_drawdowns):.1f}%')
        ax3.set_title('Max Drawdown Distribution', fontsize=11)
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax3.grid(True, alpha=0.15, color='white')

        # Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = (
            f"Monte Carlo Summary\n{'─'*35}\n"
            f"Runs: {MC_RUNS:,}  Trades: {n_trades}\n"
            f"Leverage: {leverage}x\n"
            f"ST({SUPERTREND_PERIOD},{SUPERTREND_MULT})  RSI({RSI_PERIOD_HTF},{HTF.upper()}) >{RSI_LONG_TH}/<{RSI_SHORT_TH}\n"
            f"TF: {TIMEFRAME.upper()}\n{'─'*35}\n"
            f"Prob Profit: {prob_profit:.1f}%\n"
            f"Prob Blow-up: {prob_blow:.1f}%\n"
            f"{'─'*35}\n"
            f"       Return    Final$   MaxDD\n"
        )
        for i, p in enumerate(pcts):
            summary += f"{p:>3}th {ret_p[i]:>+7.1f}% ${fin_p[i]:>7,.0f} {dd_p[i]:>6.1f}%\n"
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                 fontsize=10, va='top', color='white', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#0a0a1a', alpha=0.9))

        plt.tight_layout()
        mc_path = os.path.join(DESKTOP, "rsi_mean_revert_montecarlo.png")
        plt.savefig(mc_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  Monte Carlo chart: {mc_path}")
        plt.show()

    except Exception as e:
        print(f"  MC chart error: {e}")


if __name__ == "__main__":
    run_backtest()
