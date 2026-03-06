"""
================================================================
SMC (Smart Money Concepts) — Futures Backtest
================================================================
Strategy Flow (per day):
  [1] Daily Bias   — prev day candle direction (bull/bear)
  [2] Asian Range  — build high/low during 00:00-08:00 UTC
  [3] Manipulation — London open sweeps Asian range (Judas Swing)
                     Bull bias → SSL sweep (dip below Asian low)
                     Bear bias → BSL sweep (spike above Asian high)
  [4] MSS          — Market Structure Shift: close above/below
                     running reference after the sweep
  [5] OB           — Last counter-trend candle before MSS impulse
  [6] FVG          — 3-bar imbalance gap formed during impulse
  [7] Entry        — Price retraces into OB or FVG (limit order)
  [8] SL           — Beyond sweep extreme + ATR buffer
  [9] TP           — Previous day opposing level (min RR enforced)

Sessions (UTC):
  Asian   00:00 – 08:00  (range building / consolidation)
  London  07:00 – 12:00  (manipulation + primary entries)
  NY      13:00 – 17:00  (continuation entries)

One setup per day. Trade can run past day boundaries until SL/TP.

Data: Binance Futures API (fapi.binance.com)
Run:  python3 smc_backtest.py
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
TIMEFRAME       = "5m"       # 5m or 15m recommended
START_DATE      = "2023-01-01"
END_DATE        = "2023-04-01"
INITIAL_CAPITAL = 1000

# -- Risk management ------------------------------------------
RISK_PCT        = 0.02        # 1% of capital per trade
LEVERAGES       = [1, 3, 5, 10]

# -- Sessions (UTC hours) -------------------------------------
ASIAN_START_H   = 0
ASIAN_END_H     = 8
LONDON_START_H  = 7           # Sweep window starts here
LONDON_END_H    = 12
NY_START_H      = 13
NY_END_H        = 17

# -- SMC Parameters -------------------------------------------
MIN_RR          = 1.5         # Minimum risk:reward ratio
ATR_PERIOD      = 14
ATR_SL_BUFFER   = 0.0         # SL = sweep_extreme ± ATR_SL_BUFFER × ATR
FVG_MIN_PCT     = 0.02        # Min FVG height as % of price (0.05%)
MAX_SETUP_BARS  = 1000          # Abandon setup (no entry) after N bars
ONE_TRADE_PER_DAY = True      # Only first valid setup per day
DAILY_EMA_PERIOD  = 20        # Strengthen daily bias: only take longs when price > daily EMA

# -- Costs (Binance USDT-M Futures) ---------------------------
FEE_PER_SIDE            = 0.05
FUNDING_RATE            = 0.01
FUNDING_INTERVAL_HOURS  = 8    # ETH/BTC fund every 8h on Binance
INTEREST_RATE_DAILY     = 0.00

# -- Monte Carlo ----------------------------------------------
MC_ENABLED  = True
MC_RUNS     = 5000

# -- Warmup ---------------------------------------------------
WARMUP_DAYS = 15

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
#  Indicator Helpers
# ============================================================

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
#  SMC Helper Functions
# ============================================================

def find_bullish_ob(df, start_i, end_i):
    """
    Bullish Order Block: the last BEARISH candle (Close < Open)
    in the range [start_i, end_i]. This is the candle just before
    the impulse that caused the bullish MSS.
    Returns (low, high) of that candle, or None.
    """
    for j in range(end_i, start_i - 1, -1):
        row = df.iloc[j]
        if float(row['Close']) < float(row['Open']):  # bearish candle
            return (float(row['Low']), float(row['High']))
    return None


def find_bearish_ob(df, start_i, end_i):
    """
    Bearish Order Block: the last BULLISH candle (Close > Open)
    in the range [start_i, end_i].
    Returns (low, high) of that candle, or None.
    """
    for j in range(end_i, start_i - 1, -1):
        row = df.iloc[j]
        if float(row['Close']) > float(row['Open']):  # bullish candle
            return (float(row['Low']), float(row['High']))
    return None


def find_bullish_fvgs(df, start_i, end_i):
    """
    Bullish FVG: 3-bar pattern where High[i-2] < Low[i].
    Price gap between the high of 2 bars ago and the low of current bar.
    Returns list of (gap_low, gap_high) tuples.
    """
    fvgs = []
    for j in range(start_i + 2, min(end_i + 1, len(df))):
        gap_low  = float(df.iloc[j - 2]['High'])
        gap_high = float(df.iloc[j]['Low'])
        if gap_high > gap_low:
            size_pct = (gap_high - gap_low) / float(df.iloc[j]['Close']) * 100
            if size_pct >= FVG_MIN_PCT:
                fvgs.append((gap_low, gap_high))
    return fvgs


def find_bearish_fvgs(df, start_i, end_i):
    """
    Bearish FVG: 3-bar pattern where Low[i-2] > High[i].
    Returns list of (gap_low, gap_high) tuples.
    """
    fvgs = []
    for j in range(start_i + 2, min(end_i + 1, len(df))):
        gap_high = float(df.iloc[j - 2]['Low'])
        gap_low  = float(df.iloc[j]['High'])
        if gap_high > gap_low:
            size_pct = (gap_high - gap_low) / float(df.iloc[j]['Close']) * 100
            if size_pct >= FVG_MIN_PCT:
                fvgs.append((gap_low, gap_high))
    return fvgs


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
    print(f"  {SYMBOL}  |  SMC Strategy  |  {TIMEFRAME.upper()}")
    print(f"  Period   : {START_DATE} -> {END_DATE}  |  Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Sessions : Asian 00-08  London 07-12  NY 13-17  (UTC)")
    print(f"  Setup    : Daily Bias -> Asian Range -> Sweep -> MSS -> OB/FVG -> Entry")
    print(f"  Risk     : {RISK_PCT*100:.0f}% per trade  |  Min RR: {MIN_RR:.1f}:1")
    print(f"  SL       : Sweep extreme +/- {ATR_SL_BUFFER}x ATR({ATR_PERIOD})")
    print(f"  TP       : Prev day opposing level (min {MIN_RR:.1f}:1 enforced)")
    print(f"  Fee      : {FEE_PER_SIDE}%/side  |  Funding: {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h "
          f"({funding_per_day:.3f}%/day)")
    print("=" * 72)

    start_dt     = datetime.strptime(START_DATE, "%Y-%m-%d")
    warmup_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    df = safe_download(SYMBOL, TIMEFRAME, warmup_start, END_DATE)
    if df.empty:
        print("Download failed"); return
    df = tz_strip(df)

    # ATR for SL buffer
    df['ATR'] = calc_atr(df, ATR_PERIOD)

    # Derive daily levels from LTF data (no extra API call)
    df_daily = df.resample('D').agg(
        Open=('Open', 'first'),
        High=('High', 'max'),
        Low=('Low', 'min'),
        Close=('Close', 'last')
    ).dropna()
    df_daily['prev_high']  = df_daily['High'].shift(1)
    df_daily['prev_low']   = df_daily['Low'].shift(1)
    df_daily['prev_open']  = df_daily['Open'].shift(1)
    df_daily['prev_close'] = df_daily['Close'].shift(1)
    # Daily bias: previous day's candle direction
    df_daily['daily_bias'] = np.where(
        df_daily['prev_close'] > df_daily['prev_open'], 1, -1
    )

    # Daily EMA for higher-timeframe bias (use previous day's EMA value intraday)
    df_daily['daily_ema']      = df_daily['Close'].ewm(span=DAILY_EMA_PERIOD, adjust=False).mean()
    df_daily['prev_daily_ema'] = df_daily['daily_ema'].shift(1)

    # Merge daily levels into LTF via date
    df['_date']     = df.index.date
    daily_map = df_daily[['prev_high', 'prev_low', 'daily_bias', 'prev_daily_ema']].copy()
    daily_map.index = daily_map.index.date
    df['prev_day_high'] = df['_date'].map(daily_map['prev_high'])
    df['prev_day_low']  = df['_date'].map(daily_map['prev_low'])
    df['daily_bias']    = df['_date'].map(daily_map['daily_bias'])
    df['daily_ema']     = df['_date'].map(daily_map['prev_daily_ema'])
    df.drop(columns=['_date'], inplace=True)

    # Clip to backtest window
    bt_start  = max(pd.Timestamp(START_DATE), df.index[0])
    start_idx = df.index.searchsorted(bt_start)

    # Buy & Hold benchmark
    df_bt     = df[df.index >= bt_start]
    bnh_start = float(df_bt['Close'].iloc[0])
    bnh_end   = float(df_bt['Close'].iloc[-1])
    bnh_final = INITIAL_CAPITAL / bnh_start * bnh_end
    bnh_ret   = (bnh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # ============================================================
    #  State Machine
    # ============================================================
    # Day state (reset at 00:00 UTC each day)
    day_date        = None
    asian_high      = -np.inf
    asian_low       =  np.inf
    asian_locked    = False

    # Setup state
    sweep_detected  = False
    bull_sweep      = None      # True=bullish setup, False=bearish
    sweep_extreme   = None      # lowest low (bull) or highest high (bear)
    sweep_bar_i     = None
    post_sweep_ref  = None      # running high (bull) or low (bear) for MSS

    mss_confirmed   = False
    mss_bar_i       = None
    ob_zone         = None      # (low, high)
    fvg_zones       = []
    tp_level        = None
    sl_level        = None
    setup_bar_i     = None      # when entry zones were identified

    traded_today    = False
    position        = None
    entry_info      = None
    signals         = []        # completed trades

    # Counters for diagnostics
    stat_days           = 0
    stat_sweeps         = 0
    stat_mss            = 0
    stat_setups         = 0
    stat_rr_skipped     = 0

    print(f"\n  Running SMC state machine...")

    for i in range(start_idx, len(df)):
        row    = df.iloc[i]
        ts     = df.index[i]
        hour   = ts.hour
        c_date = ts.date()

        # ---- New day: reset state ----
        if c_date != day_date:
            day_date     = c_date
            asian_high   = -np.inf
            asian_low    =  np.inf
            asian_locked = False
            sweep_detected = False
            bull_sweep   = None
            sweep_extreme = None
            sweep_bar_i  = None
            post_sweep_ref = None
            mss_confirmed = False
            mss_bar_i    = None
            ob_zone      = None
            fvg_zones    = []
            tp_level     = None
            sl_level     = None
            setup_bar_i  = None
            if ONE_TRADE_PER_DAY:
                traded_today = False
            stat_days += 1

        # ---- Check open position (SL / TP) ----
        if position is not None and entry_info is not None:
            ep  = entry_info['entry_price']
            sl  = entry_info['sl']
            tp  = entry_info['tp']
            dir = entry_info['direction']
            exit_price = None
            outcome    = None

            if dir == 'LONG':
                if float(row['Low']) <= sl:
                    exit_price, outcome = sl, 'SL'
                elif float(row['High']) >= tp:
                    exit_price, outcome = tp, 'TP'
            else:  # SHORT
                if float(row['High']) >= sl:
                    exit_price, outcome = sl, 'SL'
                elif float(row['Low']) <= tp:
                    exit_price, outcome = tp, 'TP'

            if exit_price is not None:
                signals.append({
                    **entry_info,
                    'exit_price': float(exit_price),
                    'exit_time':  ts,
                    'outcome':    outcome,
                    'session_entry': entry_info.get('session_entry', 'unknown'),
                })
                position   = None
                entry_info = None
                continue

        # ---- Skip if already traded today and no open position ----
        if ONE_TRADE_PER_DAY and traded_today and position is None:
            continue

        daily_bias   = row.get('daily_bias', 0)
        prev_d_high  = row.get('prev_day_high', np.nan)
        prev_d_low   = row.get('prev_day_low',  np.nan)
        daily_ema    = row.get('daily_ema', np.nan)
        atr          = float(row['ATR']) if not pd.isna(row['ATR']) else 0

        if pd.isna(daily_bias) or daily_bias == 0:
            continue

        # ---- [1] Asian session: build range ----
        is_asian = (ASIAN_START_H <= hour < ASIAN_END_H)
        if is_asian:
            asian_high = max(asian_high, float(row['High']))
            asian_low  = min(asian_low,  float(row['Low']))
            continue

        # ---- Lock Asian range when session ends ----
        if not asian_locked:
            if asian_high == -np.inf or asian_low == np.inf:
                continue  # no Asian data yet
            asian_locked = True

        # ---- Active trading sessions only ----
        is_london = (LONDON_START_H <= hour < LONDON_END_H)
        is_ny     = (NY_START_H     <= hour < NY_END_H)
        if not (is_london or is_ny):
            continue

        session_label = 'London' if is_london else 'NY'

        # ---- [3] Manipulation: Detect sweep of Asian range ----
        if not sweep_detected and not traded_today:
            if daily_bias == 1:   # Bullish bias → expect SSL sweep (dip below Asian low)
                if float(row['Low']) < asian_low:
                    sweep_detected  = True
                    bull_sweep      = True
                    sweep_extreme   = float(row['Low'])
                    sweep_bar_i     = i
                    post_sweep_ref  = float(row['High'])  # track highs for MSS
                    stat_sweeps    += 1

            elif daily_bias == -1:  # Bearish bias → expect BSL sweep (spike above Asian high)
                if float(row['High']) > asian_high:
                    sweep_detected  = True
                    bull_sweep      = False
                    sweep_extreme   = float(row['High'])
                    sweep_bar_i     = i
                    post_sweep_ref  = float(row['Low'])   # track lows for MSS
                    stat_sweeps    += 1

        # ---- [4] MSS: Market Structure Shift after sweep ----
        elif sweep_detected and not mss_confirmed and not traded_today:
            # Abandon setup if too many bars passed
            if (i - sweep_bar_i) > MAX_SETUP_BARS:
                sweep_detected = False
                bull_sweep     = None
                continue

            if bull_sweep:
                # Check MSS: close above running high since sweep
                if float(row['Close']) > post_sweep_ref:
                    mss_confirmed = True
                    mss_bar_i     = i
                    stat_mss     += 1

                    # [5] OB: last bearish candle before MSS impulse
                    ob_zone = find_bullish_ob(df, sweep_bar_i, i - 1)

                    # [6] FVG: 3-bar gaps in the impulse move
                    fvg_zones = find_bullish_fvgs(df, sweep_bar_i, i)

                    # SL: below sweep extreme with ATR buffer
                    sl_level = sweep_extreme - ATR_SL_BUFFER * atr

                    # TP: previous day high (buy-side liquidity target)
                    if not pd.isna(prev_d_high) and prev_d_high > 0:
                        tp_level = prev_d_high
                    else:
                        tp_level = asian_high

                    setup_bar_i = i
                    stat_setups += 1 if (ob_zone or fvg_zones) else 0

                else:
                    # Update running reference (only with confirmed past bar)
                    post_sweep_ref = max(post_sweep_ref, float(row['High']))

            else:  # bearish
                if float(row['Close']) < post_sweep_ref:
                    mss_confirmed = True
                    mss_bar_i     = i
                    stat_mss     += 1

                    ob_zone   = find_bearish_ob(df, sweep_bar_i, i - 1)
                    fvg_zones = find_bearish_fvgs(df, sweep_bar_i, i)

                    sl_level = sweep_extreme + ATR_SL_BUFFER * atr

                    if not pd.isna(prev_d_low) and prev_d_low > 0:
                        tp_level = prev_d_low
                    else:
                        tp_level = asian_low

                    setup_bar_i = i
                    stat_setups += 1 if (ob_zone or fvg_zones) else 0

                else:
                    post_sweep_ref = min(post_sweep_ref, float(row['Low']))

        # ---- [7] Entry: Retest of OB or FVG ----
        elif mss_confirmed and not traded_today and position is None:
            if ob_zone is None and not fvg_zones:
                continue  # no entry zone defined

            # Abandon if setup is too old
            if (i - setup_bar_i) > MAX_SETUP_BARS:
                mss_confirmed  = False
                sweep_detected = False
                continue

            entry_price = None

            if bull_sweep:
                # Look for price to retrace into OB (from above) or FVG
                if ob_zone is not None:
                    ob_low, ob_high = ob_zone
                    if float(row['Low']) <= ob_high and float(row['High']) >= ob_low:
                        # Price has entered OB — limit buy at OB high
                        entry_price = ob_high

                if entry_price is None and fvg_zones:
                    for (fvg_low, fvg_high) in fvg_zones:
                        if float(row['Low']) <= fvg_high and float(row['High']) >= fvg_low:
                            entry_price = fvg_high
                            break

                if entry_price is not None:
                    # Ensure entry > sl
                    if entry_price <= sl_level:
                        entry_price = None

            if entry_price is not None:
                    # Daily EMA filter: only take longs when current close > daily EMA
                    if not pd.isna(daily_ema) and float(row['Close']) <= float(daily_ema):
                        entry_price = None

            if entry_price is not None:
                    # Validate TP: must be above entry
                    if tp_level <= entry_price:
                        tp_level = entry_price + MIN_RR * (entry_price - sl_level)

                    risk   = entry_price - sl_level
                    reward = tp_level - entry_price
                    rr     = reward / risk if risk > 0 else 0

                    if rr < MIN_RR:
                        # Extend TP to meet minimum RR
                        tp_level = entry_price + MIN_RR * risk
                        rr       = MIN_RR

                    direction = 'LONG'

            else:  # bearish
                if ob_zone is not None:
                    ob_low, ob_high = ob_zone
                    if float(row['High']) >= ob_low and float(row['Low']) <= ob_high:
                        entry_price = ob_low

                if entry_price is None and fvg_zones:
                    for (fvg_low, fvg_high) in fvg_zones:
                        if float(row['High']) >= fvg_low and float(row['Low']) <= fvg_high:
                            entry_price = fvg_low
                            break

                if entry_price is not None:
                    if entry_price >= sl_level:
                        entry_price = None

                if entry_price is not None:
                    # Daily EMA filter: only take shorts when current close < daily EMA
                    if not pd.isna(daily_ema) and float(row['Close']) >= float(daily_ema):
                        entry_price = None

                if entry_price is not None:
                    if tp_level >= entry_price:
                        tp_level = entry_price - MIN_RR * (sl_level - entry_price)

                    risk   = sl_level - entry_price
                    reward = entry_price - tp_level
                    rr     = reward / risk if risk > 0 else 0

                    if rr < MIN_RR:
                        tp_level = entry_price - MIN_RR * risk
                        rr       = MIN_RR

                    direction = 'SHORT'

            if entry_price is not None and risk > 0:
                entry_info = {
                    'direction':     direction,
                    'entry_price':   float(entry_price),
                    'entry_time':    ts,
                    'sl':            float(sl_level),
                    'tp':            float(tp_level),
                    'atr':           float(atr),
                    'daily_bias':    int(daily_bias),
                    'asian_high':    float(asian_high),
                    'asian_low':     float(asian_low),
                    'sweep_extreme': float(sweep_extreme),
                    'session_entry': session_label,
                }
                position     = direction
                traded_today = True

    # Close any still-open trade at last bar
    if position is not None and entry_info is not None:
        last = df.iloc[-1]
        signals.append({
            **entry_info,
            'exit_price': float(last['Close']),
            'exit_time':  df.index[-1],
            'outcome':    'OPEN',
        })

    print(f"\n  SMC Diagnostics:")
    print(f"     Days analysed : {stat_days}")
    print(f"     Sweeps found  : {stat_sweeps}  ({stat_sweeps/stat_days*100:.1f}% of days)")
    print(f"     MSS confirmed : {stat_mss}")
    print(f"     Setups with OB/FVG: {stat_setups}")
    print(f"     Trades taken  : {len(signals)}")

    if len(signals) < 2:
        print("  Too few trades. Try expanding the date range or loosening parameters.")
        return

    total_bt_hours  = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).total_seconds() / 3600
    in_market_hours = sum(
        (s['exit_time'] - s['entry_time']).total_seconds() / 3600 for s in signals
    )
    in_market_pct = in_market_hours / total_bt_hours * 100 if total_bt_hours > 0 else 0

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
                'trade_no':      len(trades) + 1,
                'entry_time':    entry_time,
                'exit_time':     exit_time,
                'direction':     direction,
                'entry_price':   round(entry_price, 2),
                'exit_price':    round(exit_price,  2),
                'sl':            round(sl, 2),
                'tp':            round(sig['tp'], 2),
                'outcome':       outcome,
                'session_entry': sig.get('session_entry', ''),
                'holding_hours': round(holding_hours, 1),
                'qty':           round(qty, 6),
                'pnl_raw':       round(pnl_raw,      2),
                'commission':    round(commission,   2),
                'funding_fee':   round(funding_cost, 2),
                'interest_fee':  round(interest_cost,2),
                'total_cost':    round(total_cost,   2),
                'pnl_net':       round(pnl_net,      2),
                'pnl_pct':       round(pnl_pct,      2),
                'result':        oc,
                'capital':       round(capital,      2),
                'leverage':      lev,
                'daily_bias':    sig.get('daily_bias', 0),
                'sweep_ext':     round(sig.get('sweep_extreme', 0), 2),
                'asian_h':       round(sig.get('asian_high', 0), 2),
                'asian_l':       round(sig.get('asian_low', 0), 2),
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
    print(f"  {SYMBOL}  |  SMC Strategy  |  {TIMEFRAME.upper()}")
    print(f"  Sessions: London 07-12 UTC + NY 13-17 UTC")
    print(f"  Setup: Bias -> Asian Range -> Sweep -> MSS -> OB/FVG -> Entry")
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
    row("  TOTAL COST",      [lev_results[l]['cost_total']    for l in levs], "${:>9,.0f}")
    print(f"{'='*total_w}")

    # Direction + Session breakdown (1x)
    base_lev = levs[0]
    df_base  = lev_results[base_lev]['df_t']

    print(f"\n  Direction breakdown ({base_lev}x):")
    for d in ['LONG', 'SHORT']:
        sub = df_base[df_base['direction'] == d]
        if not sub.empty:
            w   = len(sub[sub['result'] == 'WIN'])
            wr2 = w / len(sub) * 100
            p   = sub['pnl_net'].sum()
            print(f"     {d:<7}: {len(sub):>3} trades | WR: {wr2:.1f}% | PnL: ${p:,.0f}")

    print(f"\n  Session breakdown ({base_lev}x):")
    for s in ['London', 'NY']:
        sub = df_base[df_base['session_entry'] == s]
        if not sub.empty:
            w   = len(sub[sub['result'] == 'WIN'])
            wr2 = w / len(sub) * 100
            p   = sub['pnl_net'].sum()
            print(f"     {s:<8}: {len(sub):>3} trades | WR: {wr2:.1f}% | PnL: ${p:,.0f}")

    print(f"\n  Outcome breakdown ({base_lev}x):")
    for o in ['TP', 'SL', 'OPEN']:
        sub = df_base[df_base['outcome'] == o]
        if not sub.empty:
            print(f"     {o:<8}: {len(sub):>3}")

    # Save CSV
    best_lev = max(levs, key=lambda l: lev_results[l]['calmar'])
    csv_path = os.path.join(DESKTOP, "smc_trades.csv")
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
            f'{SYMBOL}  SMC Strategy  |  {TIMEFRAME.upper()}\n'
            f'{START_DATE} -> {END_DATE}  |  '
            f'Bias + Asian Range + Sweep + MSS + OB/FVG\n'
            f'Min RR: {MIN_RR}:1  |  Risk: {RISK_PCT*100:.0f}%/trade  |  '
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

        bnh_eq = df_bt_plot['Close'] / bnh_start * INITIAL_CAPITAL
        ax1.plot(df_bt_plot.index, bnh_eq, color='#FFA500', lw=1.5,
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
        cagrs     = [lev_results[l]['cagr']    for l in levs]
        max_dds   = [abs(lev_results[l]['max_dd']) for l in levs]
        sharpes   = [lev_results[l]['sharpe'] * 10 for l in levs]
        calmars   = [lev_results[l]['calmar'] * 10 for l in levs]
        costs_pct = [lev_results[l]['cost_total'] / INITIAL_CAPITAL * 100 for l in levs]

        ax3.bar(x_pos - 2*bar_w, cagrs,                  bar_w, label='CAGR %',       color='#00FF88', alpha=0.85)
        ax3.bar(x_pos -   bar_w, [-d for d in max_dds],  bar_w, label='Max DD %',     color='#FF4444', alpha=0.7)
        ax3.bar(x_pos,           sharpes,                 bar_w, label='Sharpe x10',   color='#00BFFF', alpha=0.7)
        ax3.bar(x_pos +   bar_w, calmars,                 bar_w, label='Calmar x10',   color='#FFD700', alpha=0.7)
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
        chart_path = os.path.join(DESKTOP, "smc_chart.png")
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
    print(f"   MIN_RR          = {MIN_RR}    (try 1.5, 2.0, 3.0)")
    print(f"   ATR_SL_BUFFER   = {ATR_SL_BUFFER}    (try 0.3, 0.5, 1.0)")
    print(f"   FVG_MIN_PCT     = {FVG_MIN_PCT}  (try 0.02, 0.05, 0.1)")
    print(f"   MAX_SETUP_BARS  = {MAX_SETUP_BARS}   (try 24, 48, 96)")
    print(f"   TIMEFRAME       = '{TIMEFRAME}'   (try '5m', '15m')")
    print(f"   SYMBOL          = '{SYMBOL}'  (try 'ETHUSDT')")


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
    print(f"{'='*65}")


if __name__ == "__main__":
    run_backtest()
