"""
================================================================
MACD + EMA Trend Strategy — Futures Backtest
================================================================
Strategy:
  Trend Filter : Price vs EMA(EMA_PERIOD) on 1H
  Entry Signal : MACD line crosses signal line in trend direction
                 LONG  — price > EMA  AND  MACD crosses UP   above signal
                 SHORT — price < EMA  AND  MACD crosses DOWN  below signal

  Stop Loss    : LONG  — below last swing low (SWING_LOOKBACK bars) − ATR×ATR_SL_MULT
                 SHORT — above last swing high (SWING_LOOKBACK bars) + ATR×ATR_SL_MULT

  Take Profit  : Fixed RR (entry ± RR_TARGET × risk distance)

  Signal detected at bar i close → entry fills at bar i+1 open.
  One position at a time (no pyramiding).

Data: Binance Futures API (fapi.binance.com)
Run:  python3 MACD.py
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

SYMBOL          = "BTCUSDT"
TIMEFRAME       = "1h"
START_DATE      = "2023-01-01"
END_DATE        = "2025-01-01"
INITIAL_CAPITAL = 2500

# -- EMA (trend direction) ------------------------------------
EMA_PERIOD      = 200        # price > EMA → bullish trend

# -- MACD (entry signal) --------------------------------------
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL_P   = 9

# -- Stop Loss ------------------------------------------------
ATR_PERIOD      = 14
ATR_SL_MULT     = 1.5        # SL = swing_low  − ATR_SL_MULT × ATR  (long)
                              #      swing_high + ATR_SL_MULT × ATR  (short)
SWING_LOOKBACK  = 20         # bars to look back for swing high/low

# -- Take Profit ----------------------------------------------
RR_TARGET       = 2.0        # TP = entry ± RR_TARGET × risk

# -- Risk management ------------------------------------------
RISK_PCT        = 0.02       # 2% of capital per trade (scaled by leverage)
LEVERAGES       = [1, 3, 5, 10]

# -- Costs (Binance USDT-M Futures) ---------------------------
FEE_PER_SIDE            = 0.05
FUNDING_RATE            = 0.01
FUNDING_INTERVAL_HOURS  = 8
INTEREST_RATE_DAILY     = 0.00

# -- Monte Carlo ----------------------------------------------
MC_ENABLED  = True
MC_RUNS     = 5000

# -- Warmup ---------------------------------------------------
WARMUP_DAYS = 60    # needs MACD_SLOW + EMA_PERIOD bars to warm up

# ============================================================

if not END_DATE:
    END_DATE = datetime.now().strftime("%Y-%m-%d")
today_str = datetime.now().strftime("%Y-%m-%d")
if END_DATE > today_str:
    END_DATE = today_str


# ============================================================
#  Binance Futures Data Download
# ============================================================

def binance_klines(symbol, interval, start_str, end_str, limit=1000):
    url      = "https://fapi.binance.com/fapi/v1/klines"
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
        print("   No data returned"); return pd.DataFrame()
    print(f"   {len(df)} bars  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


def tz_strip(df):
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


# ============================================================
#  Indicators
# ============================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast   = calc_ema(close, fast)
    ema_slow   = calc_ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_atr(df, period=14):
    high  = df['High']
    low   = df['Low']
    close = df['Close']
    tr1   = high - low
    tr2   = (high - close.shift(1)).abs()
    tr3   = (low  - close.shift(1)).abs()
    tr    = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


# ============================================================
#  Holding Costs
# ============================================================

def calc_holding_costs(direction, notional, holding_hours):
    n_funding       = holding_hours / FUNDING_INTERVAL_HOURS
    funding_decimal = FUNDING_RATE / 100
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
    print(f"  {SYMBOL}  |  MACD + EMA Strategy  |  {TIMEFRAME.upper()}")
    print(f"  Period   : {START_DATE} -> {END_DATE}  |  Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Trend    : EMA({EMA_PERIOD})  —  price > EMA = LONG bias, < EMA = SHORT bias")
    print(f"  Signal   : MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL_P}) cross signal line")
    print(f"  SL       : Swing HL({SWING_LOOKBACK}) ± {ATR_SL_MULT}×ATR({ATR_PERIOD})")
    print(f"  TP       : Fixed {RR_TARGET}:1 RR")
    print(f"  Risk     : {RISK_PCT*100:.0f}% × leverage per trade")
    print(f"  Leverage : {' / '.join(f'{l}x' for l in LEVERAGES)}")
    print(f"  Fee      : {FEE_PER_SIDE}%/side  |  Funding: {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h "
          f"({funding_per_day:.3f}%/day)")
    print("=" * 72)

    start_dt     = datetime.strptime(START_DATE, "%Y-%m-%d")
    warmup_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    df = safe_download(SYMBOL, TIMEFRAME, warmup_start, END_DATE)
    if df.empty:
        print("Download failed"); return
    df = tz_strip(df)

    # ---- Indicators ----
    df['EMA']        = calc_ema(df['Close'], EMA_PERIOD)
    macd, sig, hist  = calc_macd(df['Close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL_P)
    df['MACD']       = macd
    df['Signal']     = sig
    df['Hist']       = hist
    df['ATR']        = calc_atr(df, ATR_PERIOD)

    # Rolling swing high / low (look back only, include current bar)
    df['swing_low']  = df['Low'].rolling(SWING_LOOKBACK).min()
    df['swing_high'] = df['High'].rolling(SWING_LOOKBACK).max()

    # MACD crossover flags (bar i)
    df['cross_up'] = (df['MACD'] >  df['Signal']) & (df['MACD'].shift(1) <= df['Signal'].shift(1))
    df['cross_dn'] = (df['MACD'] <  df['Signal']) & (df['MACD'].shift(1) >= df['Signal'].shift(1))

    # ---- Clip to backtest window ----
    bt_start  = max(pd.Timestamp(START_DATE), df.index[0])
    df_bt     = df[df.index >= bt_start].copy()
    if df_bt.empty:
        print("No data in backtest window"); return

    start_idx = df.index.searchsorted(bt_start)

    # ---- Buy & Hold benchmark ----
    bnh_start = float(df_bt['Close'].iloc[0])
    bnh_end   = float(df_bt['Close'].iloc[-1])
    bnh_final = INITIAL_CAPITAL / bnh_start * bnh_end
    bnh_ret   = (bnh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # ============================================================
    #  Signal Detection Loop
    # ============================================================
    signals    = []
    position   = None
    entry_info = None

    required_cols = ['EMA', 'MACD', 'Signal', 'ATR', 'swing_low', 'swing_high',
                     'cross_up', 'cross_dn']

    # Loop to start_idx+1 so we can always read i+1 for entry fill
    for i in range(start_idx, len(df) - 1):
        row      = df.iloc[i]
        next_bar = df.iloc[i + 1]      # entry fills at next bar's open
        t        = df.index[i]

        if any(pd.isna(row[c]) for c in required_cols):
            continue

        close      = float(row['Close'])
        ema        = float(row['EMA'])
        atr        = float(row['ATR'])
        swing_lo   = float(row['swing_low'])
        swing_hi   = float(row['swing_high'])
        cross_up   = bool(row['cross_up'])
        cross_dn   = bool(row['cross_dn'])

        # ---- 1. Manage open position ----
        if position is not None and entry_info is not None:
            sl = entry_info['sl']
            tp = entry_info['tp']

            h = float(row['High'])
            l = float(row['Low'])
            exit_price = None
            outcome    = None

            if position == 'LONG':
                if l <= sl:
                    exit_price, outcome = sl, 'SL'
                elif h >= tp:
                    exit_price, outcome = tp, 'TP'
            else:
                if h >= sl:
                    exit_price, outcome = sl, 'SL'
                elif l <= tp:
                    exit_price, outcome = tp, 'TP'

            if exit_price is not None:
                signals.append({
                    **entry_info,
                    'exit_price': float(exit_price),
                    'exit_time':  t,
                    'outcome':    outcome,
                })
                position   = None
                entry_info = None
                # fall through — can open new trade same bar if signal fires

        # ---- 2. Look for new entry (only one position at a time) ----
        if position is None:
            direction   = None
            entry_price = float(next_bar['Open'])  # fill at next bar open
            sl          = None
            tp          = None

            # LONG: price > EMA + MACD crosses up
            if close > ema and cross_up:
                sl = swing_lo - ATR_SL_MULT * atr
                if entry_price > sl:
                    risk = entry_price - sl
                    tp   = entry_price + RR_TARGET * risk
                    direction = 'LONG'

            # SHORT: price < EMA + MACD crosses down
            elif close < ema and cross_dn:
                sl = swing_hi + ATR_SL_MULT * atr
                if entry_price < sl:
                    risk = sl - entry_price
                    tp   = entry_price - RR_TARGET * risk
                    direction = 'SHORT'

            if direction is not None:
                entry_info = {
                    'direction':   direction,
                    'entry_price': entry_price,
                    'entry_time':  df.index[i + 1],
                    'signal_time': t,
                    'sl':          float(sl),
                    'tp':          float(tp),
                    'atr':         float(atr),
                    'swing_lo':    float(swing_lo),
                    'swing_hi':    float(swing_hi),
                    'ema_at_entry': float(ema),
                }
                position = direction

    # Close any open trade at last bar
    if position is not None and entry_info is not None:
        last = df.iloc[-1]
        signals.append({
            **entry_info,
            'exit_price': float(last['Close']),
            'exit_time':  df.index[-1],
            'outcome':    'OPEN',
        })

    print(f"\n  Trades found: {len(signals)}")
    if len(signals) < 2:
        print("  Too few trades. Try expanding the date range or loosening parameters.")
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
            sl          = sig['sl']

            risk_per_unit = abs(entry_price - sl)
            if risk_per_unit <= 0 or capital <= 0:
                continue

            risk_amount = capital * RISK_PCT * lev
            qty         = risk_amount / risk_per_unit

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
                'signal_time':  sig.get('signal_time', entry_time),
                'entry_time':   entry_time,
                'exit_time':    exit_time,
                'direction':    direction,
                'entry_price':  round(entry_price, 2),
                'exit_price':   round(exit_price,  2),
                'sl':           round(sl,           2),
                'tp':           round(sig['tp'],    2),
                'outcome':      outcome,
                'holding_hours':round(holding_hours, 1),
                'qty':          round(qty, 6),
                'atr':          round(sig['atr'],   2),
                'swing_lo':     round(sig['swing_lo'], 2),
                'swing_hi':     round(sig['swing_hi'], 2),
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

        comm_total    = df_t['commission'].sum()
        funding_total = df_t['funding_fee'].sum()
        int_total     = df_t['interest_fee'].sum()
        cost_total    = df_t['total_cost'].sum()
        avg_hold      = df_t['holding_hours'].mean()

        lev_results[lev] = {
            'df_t': df_t, 'df_e': df_e,
            'total': total, 'wins': wins, 'losses': losses, 'wr': wr,
            'fin': fin, 'ret': ret, 'max_dd': max_dd, 'avg_dd': avg_dd,
            'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l, 'exp': exp,
            'eq_arr': eq_arr, 'dd_arr': dd_arr,
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'cagr': cagr,
            'comm_total': comm_total, 'funding_total': funding_total,
            'int_total': int_total, 'cost_total': cost_total,
            'avg_hold': avg_hold, 'in_market_pct': in_market_pct,
        }

    # ============================================================
    #  Print Results
    # ============================================================
    levs = list(lev_results.keys())
    if not levs:
        print("No results"); return

    col_w   = 12
    total_w = 26 + col_w * len(levs) + col_w

    print(f"\n{'='*total_w}")
    print(f"  {SYMBOL}  |  MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL_P}) + EMA({EMA_PERIOD})  |  {TIMEFRAME.upper()}")
    print(f"  Trend: price vs EMA({EMA_PERIOD})  |  Signal: MACD cross")
    print(f"  SL: Swing({SWING_LOOKBACK}) ± {ATR_SL_MULT}×ATR  |  TP: {RR_TARGET}:1 RR  |  Risk: {RISK_PCT*100:.0f}%×lev")
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
    row("  Commission",      [lev_results[l]['comm_total']    for l in levs], "${:>9,.0f}")
    row("  Funding Fee",     [lev_results[l]['funding_total'] for l in levs], "${:>9,.0f}")
    row("  Interest",        [lev_results[l]['int_total']     for l in levs], "${:>9,.0f}")
    row("  TOTAL COST",      [lev_results[l]['cost_total']    for l in levs], "${:>9,.0f}")
    print(f"{'='*total_w}")

    print(f"\n  Cost as % of capital:")
    for l in levs:
        r  = lev_results[l]
        pc = r['comm_total']    / INITIAL_CAPITAL * 100
        pf = r['funding_total'] / INITIAL_CAPITAL * 100
        pa = r['cost_total']    / INITIAL_CAPITAL * 100
        print(f"     {l:>2}x: Comm {pc:>6.1f}%  Fund {pf:>+7.1f}%  -> Total {pa:>7.1f}%")

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
            print(f"     {d:<7}: {len(sub):>3} trades | WR: {wr2:.1f}% | "
                  f"PnL: ${p:,.0f} | Avg Hold: {ah:.0f}h")

    print(f"\n  Outcome breakdown ({base_lev}x):")
    for o in ['TP', 'SL', 'OPEN']:
        sub = df_base[df_base['outcome'] == o]
        if not sub.empty:
            print(f"     {o:<8}: {len(sub):>3} trades")

    # Save CSV
    best_lev = max(levs, key=lambda l: lev_results[l]['calmar'])
    csv_path = os.path.join(DESKTOP, "macd_trades.csv")
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
        colors_lev = ['#00BFFF', '#00FF88', '#FFD700', '#FF6BFF',
                      '#FF4444', '#7B68EE', '#FF8C00', '#00CED1']

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
            f'{SYMBOL}  MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL_P}) + EMA({EMA_PERIOD})  |  {TIMEFRAME.upper()}\n'
            f'{START_DATE} -> {END_DATE}  |  '
            f'SL: Swing({SWING_LOOKBACK}) ± {ATR_SL_MULT}×ATR  |  '
            f'TP: {RR_TARGET}:1  |  Risk: {RISK_PCT*100:.0f}%×lev\n'
            f'Time in market: {in_market_pct:.1f}%',
            fontsize=12, fontweight='bold', color='white'
        )

        df_bt_plot = df[df.index >= bt_start]
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

        # Panel 3: Risk grid
        ax3 = axes[2]
        x_pos = np.arange(len(levs))
        bar_w = 0.12
        cagrs     = [lev_results[l]['cagr']    for l in levs]
        max_dds   = [abs(lev_results[l]['max_dd']) for l in levs]
        sharpes   = [lev_results[l]['sharpe'] * 10 for l in levs]
        calmars   = [lev_results[l]['calmar'] * 10 for l in levs]
        costs_pct = [lev_results[l]['cost_total'] / INITIAL_CAPITAL * 100 for l in levs]

        ax3.bar(x_pos - 2*bar_w, cagrs,                   bar_w, label='CAGR %',       color='#00FF88', alpha=0.85)
        ax3.bar(x_pos -   bar_w, [-d for d in max_dds],   bar_w, label='Max DD %',     color='#FF4444', alpha=0.7)
        ax3.bar(x_pos,           sharpes,                  bar_w, label='Sharpe x10',   color='#00BFFF', alpha=0.7)
        ax3.bar(x_pos +   bar_w, calmars,                  bar_w, label='Calmar x10',   color='#FFD700', alpha=0.7)
        ax3.bar(x_pos + 2*bar_w, [-c for c in costs_pct], bar_w, label='Total Cost %', color='#FF8888', alpha=0.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{l}x' for l in levs], color='white')
        ax3.axhline(0, color='white', lw=0.5)
        ax3.set_ylabel('Value', fontsize=11)
        ax3.set_title('Risk Grid + Cost Impact', fontsize=11, color='white')
        ax3.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white', ncol=5)
        ax3.grid(True, alpha=0.15, color='white')

        # Panel 4: Trade PnL (best leverage)
        ax4 = axes[3]
        df_best    = lev_results[best_lev]['df_t']
        trade_pnls = df_best['pnl_net'].values
        colors_bar = ['#00FF88' if p > 0 else '#FF4444' for p in trade_pnls]
        ax4.bar(range(1, len(trade_pnls) + 1), trade_pnls,
                color=colors_bar, alpha=0.8, width=0.8)
        ax4.axhline(0, color='white', lw=0.5)
        ax4.set_ylabel(f'PnL per Trade $ ({best_lev}x)', fontsize=11)
        ax4.set_xlabel('Trade #', fontsize=10)
        wins_best   = len(df_best[df_best['result'] == 'WIN'])
        losses_best = len(df_best[df_best['result'] == 'LOSS'])
        ax4.set_title(
            f'Best Calmar: {best_lev}x  |  Trades: {len(trade_pnls)}  |  '
            f'Wins: {wins_best}  Losses: {losses_best}  |  WR: {wins_best/len(trade_pnls)*100:.1f}%',
            fontsize=10, color='#00BFFF')
        ax4.grid(True, alpha=0.15, color='white')

        plt.tight_layout()
        chart_path = os.path.join(DESKTOP, "macd_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  Chart saved: {chart_path}")
        plt.show()

    except Exception as e:
        print(f"  Chart error: {e}")

    print("\nBacktest complete!")

    if MC_ENABLED:
        print(f"\n  Monte Carlo using {best_lev}x (best Calmar)")
        run_monte_carlo(lev_results[best_lev]['df_t'], best_lev)

    print(f"\nTips for tuning:")
    print(f"   EMA_PERIOD     = {EMA_PERIOD}   (try 50, 100, 200)")
    print(f"   MACD params    = ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL_P})  (try 12,26,9 or 8,21,5)")
    print(f"   SWING_LOOKBACK = {SWING_LOOKBACK}    (try 10, 20, 30)")
    print(f"   ATR_SL_MULT    = {ATR_SL_MULT}   (try 1.0, 1.5, 2.0)")
    print(f"   RR_TARGET      = {RR_TARGET}   (try 1.5, 2.0, 3.0)")
    print(f"   SYMBOL         = '{SYMBOL}'  (try 'ETHUSDT')")


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


if __name__ == "__main__":
    run_backtest()
