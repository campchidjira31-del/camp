"""
====================================================================
EMA 195 ETH — A/B Comparison: Original vs Filtered
====================================================================
รัน 2 versions พร้อมกัน:

  [A] ORIGINAL — EMA crossover เดิม (ไม่มี filter)
  [B] FILTERED — เพิ่ม ADX + ATR filter กรอง whipsaw

Filters:
  1) ADX Trend Strength
     - ADX(14) > threshold → trend แรงพอ → เทรดได้
     - ADX ต่ำ = sideways → skip signal (ลด whipsaw)

  2) ATR Volatility Gate
     - ATR(14) > ATR_MA(50) × multiplier → volatility พอ
     - ATR ต่ำ = ตลาดนิ่ง → signal ไม่น่าเชื่อถือ → skip

Strategy: EMA crossover (always-in-market for original,
          selective entry for filtered)

Output: ตาราง A/B เทียบทุก metric + กราฟ overlay

Data: Binance Public API
วิธีรัน:  python3 ema_145_ATR.py
====================================================================
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

SYMBOL          = "ETHUSDT"
START_DATE      = "2023-07-01"
END_DATE        = "2023-12-04"
INITIAL_CAPITAL = 2500

# ── EMA ────────────────────────────────────────
EMA_PERIOD      = 145

# ── Leverage (ใช้ตัวเดียวสำหรับ A/B comparison) ──
LEVERAGE        = 3                  # เทียบ 1x ก่อน (เห็น edge ชัด)

# ── Filter 1: ADX Trend Strength ──────────────
ADX_ENABLED     = False               # เปิด/ปิด ADX filter
ADX_PERIOD      = 14
ADX_THRESHOLD   = 20                 # ADX > 20 = trending, < 20 = sideways
                                     # ลอง 15, 20, 25

# ── Filter 2: ATR Volatility Gate ─────────────
ATR_ENABLED     = True               # เปิด/ปิด ATR filter
ATR_PERIOD      = 14
ATR_MA_PERIOD   = 60                 # MA ของ ATR (baseline)
ATR_MULTIPLIER  = 0.9                # ATR ปัจจุบัน > ATR_MA × multiplier
                                     # 0.8 = ATR ≥ 80% ของ average
                                     # ลอง 0.5, 0.8, 1.0, 1.2

# ── Costs (Binance Futures) ───────────────────
FEE_PER_SIDE    = 0.05
FUNDING_RATE    = 0.01
FUNDING_INTERVAL_HOURS = 8      # Binance USDT-M ETH/BTC funds every 8h (not 4h)
INTEREST_RATE_DAILY = 0.00

# ── Monte Carlo ───────────────────────────────
MC_ENABLED      = False
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
#  Binance Data Download
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def binance_klines(symbol, interval, start_str, end_str, limit=1000):
    url = "https://fapi.binance.com/fapi/v1/klines"
    start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end_str,   "%Y-%m-%d").timestamp() * 1000)
    all_data = []
    cursor   = start_ms
    while cursor < end_ms:
        params = {
            'symbol': symbol, 'interval': interval,
            'startTime': cursor,
            'endTime': end_ms,
            'limit': limit,
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
        print(f"   ❌ ไม่มีข้อมูล"); return pd.DataFrame()
    print(f"   ✅ {len(df)} แท่ง ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def tz_strip(df):
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Indicators
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_adx(df, period=14):
    """
    Average Directional Index (ADX)
    วัดความแรงของ trend (ไม่สนทิศทาง)
    ADX > 25 = strong trend, < 20 = sideways/weak
    """
    high  = df['High']
    low   = df['Low']
    close = df['Close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # +DM, -DM
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed (Wilder's EMA)
    atr    = pd.Series(tr, index=df.index).ewm(alpha=1/period, min_periods=period).mean()
    plus_di  = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1/period, min_periods=period).mean() / atr

    # DX → ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], 0).fillna(0)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx


def calc_atr(df, period=14):
    """Average True Range"""
    high  = df['High']
    low   = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Funding & Interest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_holding_costs(direction, notional, holding_hours):
    n_funding = holding_hours / FUNDING_INTERVAL_HOURS
    funding_rate_decimal = FUNDING_RATE / 100
    # ใช้ historic average funding: LONG จ่าย, SHORT รับ
    if direction == 'LONG':
        funding_cost = notional * funding_rate_decimal * n_funding
    else:
        funding_cost = -notional * funding_rate_decimal * n_funding
    interest_rate_per_hour = (INTEREST_RATE_DAILY / 100) / 24
    interest_cost = notional * interest_rate_per_hour * holding_hours
    return funding_cost, interest_cost


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Signal Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_signals(df, start_idx, use_filters=False):
    """
    สร้าง signals สำหรับ EMA crossover

    use_filters=False → Original (always-in-market)
    use_filters=True  → Filtered (ADX + ATR gate)

    Filtered version:
      - เมื่อ filter ไม่ผ่าน → ปิด position ปัจจุบัน (exit to flat)
      - เมื่อ filter ผ่าน + EMA signal → เข้า position
      - ลดเทรดที่แพ้ช่วง sideways/low-vol
    """
    signals = []
    current_dir = None
    in_position = False  # สำหรับ filtered mode

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        close = row['Close']
        ema   = row['EMA']

        if pd.isna(ema):
            continue

        # EMA direction
        if close > ema:
            ema_dir = 'LONG'
        elif close < ema:
            ema_dir = 'SHORT'
        else:
            continue

        if not use_filters:
            # ──── [A] ORIGINAL: always-in-market ────
            if ema_dir != current_dir:
                next_bar = df.iloc[i + 1]
                signals.append({
                    'i': i,
                    'direction': ema_dir,
                    'entry_price': float(next_bar['Open']),
                    'entry_time': df.index[i + 1],
                    'signal_time': df.index[i],
                    'adx': row.get('ADX', 0),
                    'atr': row.get('ATR', 0),
                    'atr_ma': row.get('ATR_MA', 0),
                    'filter_pass': True,
                })
                current_dir = ema_dir
        else:
            # ──── [B] FILTERED: selective entry ────
            # Check filters
            adx_ok = True
            atr_ok = True

            if ADX_ENABLED:
                adx_val = row.get('ADX', 0)
                if pd.isna(adx_val) or adx_val < ADX_THRESHOLD:
                    adx_ok = False

            if ATR_ENABLED:
                atr_val = row.get('ATR', 0)
                atr_ma  = row.get('ATR_MA', 0)
                if pd.isna(atr_val) or pd.isna(atr_ma) or atr_ma <= 0:
                    atr_ok = False
                elif atr_val < atr_ma * ATR_MULTIPLIER:
                    atr_ok = False

            filters_pass = adx_ok and atr_ok

            if ema_dir != current_dir:
                # Direction change
                if filters_pass:
                    # Filter ผ่าน → เข้า/flip position
                    next_bar = df.iloc[i + 1]
                    signals.append({
                        'i': i,
                        'direction': ema_dir,
                        'entry_price': float(next_bar['Open']),
                        'entry_time': df.index[i + 1],
                        'signal_time': df.index[i],
                        'adx': row.get('ADX', 0),
                        'atr': row.get('ATR', 0),
                        'atr_ma': row.get('ATR_MA', 0),
                        'filter_pass': True,
                    })
                    current_dir = ema_dir
                    in_position = True
                else:
                    # Filter ไม่ผ่าน → ถ้ามี position ก็ปิด (exit to flat)
                    if in_position:
                        next_bar = df.iloc[i + 1]
                        signals.append({
                            'i': i,
                            'direction': 'FLAT',  # special: exit only
                            'entry_price': float(next_bar['Open']),
                            'entry_time': df.index[i + 1],
                            'signal_time': df.index[i],
                            'adx': row.get('ADX', 0),
                            'atr': row.get('ATR', 0),
                            'atr_ma': row.get('ATR_MA', 0),
                            'filter_pass': False,
                        })
                        in_position = False
                    current_dir = ema_dir

            elif in_position and not filters_pass:
                # Direction ไม่เปลี่ยนแต่ filter หลุด → ยังถือต่อ (ไม่ออก)
                # เราออกเฉพาะตอน direction เปลี่ยน
                pass

    return signals


def signals_to_trades(signals, df):
    """
    แปลง signals เป็น trades (entry→exit pairs)
    รองรับ FLAT signal สำหรับ filtered version
    """
    trades_raw = []

    i = 0
    while i < len(signals):
        sig = signals[i]

        if sig['direction'] == 'FLAT':
            # FLAT ใช้เป็น exit ของ trade ก่อนหน้า (ไม่เปิดใหม่)
            i += 1
            continue

        direction   = sig['direction']
        entry_price = sig['entry_price']
        entry_time  = sig['entry_time']

        # หา exit: signal ถัดไป (ไม่ว่าจะเป็น flip หรือ FLAT)
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
            'entry_time': entry_time,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'outcome': outcome,
            'adx': sig.get('adx', 0),
            'atr': sig.get('atr', 0),
            'atr_ma': sig.get('atr_ma', 0),
            'filter_pass': sig.get('filter_pass', True),
        })
        i += 1

    return trades_raw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Execute Trades → Stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def execute_trades(trades_raw, lev, label=""):
    """Execute trades with costs, return stats dict"""
    trades   = []
    eq_curve = []
    capital  = INITIAL_CAPITAL

    for tr in trades_raw:
        direction   = tr['direction']
        entry_price = tr['entry_price']
        entry_time  = tr['entry_time']
        exit_price  = tr['exit_price']
        exit_time   = tr['exit_time']

        position_value = capital * lev
        qty = position_value / entry_price

        holding_hours = (exit_time - entry_time).total_seconds() / 3600

        if direction == 'LONG':
            pnl_raw = (exit_price - entry_price) * qty
        else:
            pnl_raw = (entry_price - exit_price) * qty

        commission = ((entry_price * qty) + (exit_price * qty)) * (FEE_PER_SIDE / 100)
        avg_price = (entry_price + exit_price) / 2
        notional  = avg_price * qty
        funding_cost, interest_cost = calc_holding_costs(direction, notional, holding_hours)

        total_cost = commission + funding_cost + interest_cost
        pnl_net = pnl_raw - total_cost
        pnl_pct = pnl_net / capital * 100 if capital > 0 else 0

        capital += pnl_net
        if capital <= 0:
            capital = 0

        oc = 'WIN' if pnl_net > 0 else ('LOSS' if pnl_net < 0 else 'BE')

        eq_curve.append({'time': exit_time, 'equity': capital, 'pnl': pnl_net})
        trades.append({
            'trade_no': len(trades) + 1,
            'entry_time': entry_time, 'exit_time': exit_time,
            'direction': direction,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'outcome': tr['outcome'],
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
            'adx': round(tr.get('adx', 0), 1),
            'atr': round(tr.get('atr', 0), 2),
            'version': label,
        })

        if capital <= 0:
            break

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(eq_curve)

    return calc_stats(df_t, df_e, capital, label)


def calc_stats(df_t, df_e, capital, label):
    """Calculate all stats from trade results"""
    total = len(df_t)
    if total == 0:
        return None

    years = max((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 365.25, 0.5)

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

    if fin > 0 and years > 0:
        cagr = ((fin / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    else:
        cagr = -100

    pnl_arr = df_t['pnl_pct'].values
    trades_per_year = total / years
    if len(pnl_arr) > 1 and pnl_arr.std() > 0:
        sharpe = (pnl_arr.mean() / pnl_arr.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    downside = pnl_arr[pnl_arr < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (pnl_arr.mean() / downside.std()) * np.sqrt(trades_per_year)
    else:
        sortino = 0

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    avg_hold = df_t['holding_hours'].mean()

    return {
        'label': label,
        'df_t': df_t, 'df_e': df_e,
        'eq_arr': eq_arr, 'dd_arr': dd_arr,
        'total': total, 'wins': wins, 'losses': losses, 'wr': wr,
        'fin': fin, 'ret': ret, 'max_dd': max_dd, 'avg_dd': avg_dd,
        'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l, 'exp': exp,
        'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'cagr': cagr,
        'comm_total': df_t['commission'].sum(),
        'funding_total': df_t['funding_fee'].sum(),
        'interest_total': df_t['interest_fee'].sum(),
        'all_cost_total': df_t['total_cost'].sum(),
        'avg_hold': avg_hold,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ab_comparison():
    filter_desc = []
    if ADX_ENABLED:
        filter_desc.append(f"ADX({ADX_PERIOD})>{ADX_THRESHOLD}")
    if ATR_ENABLED:
        filter_desc.append(f"ATR({ATR_PERIOD})>MA({ATR_MA_PERIOD})×{ATR_MULTIPLIER}")
    filter_str = " + ".join(filter_desc) if filter_desc else "None"

    print("=" * 75)
    print(f"  {SYMBOL} EMA {EMA_PERIOD} — A/B Comparison | 4H | {LEVERAGE}x")
    print(f"  [A] ORIGINAL: EMA crossover เดิม")
    print(f"  [B] FILTERED: {filter_str}")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"  Costs: Fee {FEE_PER_SIDE}%  |  Fund {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h  |  "
          f"Int {INTEREST_RATE_DAILY}%/day")
    print("=" * 75)

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    warmup_start = (start_dt - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    # ---- Download ----
    df = safe_download(SYMBOL, '4h', warmup_start, END_DATE)
    if df.empty:
        print("❌ Download failed"); return
    df = tz_strip(df)

    # ---- Indicators ----
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['ADX'] = calc_adx(df, ADX_PERIOD)
    df['ATR'] = calc_atr(df, ATR_PERIOD)
    df['ATR_MA'] = df['ATR'].rolling(ATR_MA_PERIOD).mean()

    warmup_bars = len(df[df.index < pd.Timestamp(START_DATE)])
    print(f"\n📐 Warmup: {warmup_bars} แท่ง = {warmup_bars/EMA_PERIOD:.1f}x EMA period")

    bt_start = max(pd.Timestamp(START_DATE), df.index[0])
    df_bt = df[df.index >= bt_start].copy()
    if df_bt.empty:
        print("❌ No data"); return

    start_idx = df.index.searchsorted(bt_start)

    # ---- Buy & Hold ----
    bnh_start = float(df_bt['Close'].iloc[0])
    bnh_end   = float(df_bt['Close'].iloc[-1])
    bnh_final = INITIAL_CAPITAL / bnh_start * bnh_end
    bnh_ret   = (bnh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # ========== [A] ORIGINAL ==========
    print(f"\n🔵 Running [A] ORIGINAL...")
    signals_a = generate_signals(df, start_idx, use_filters=False)
    trades_a  = signals_to_trades(signals_a, df)
    print(f"   Signals: {len(signals_a)}  |  Trades: {len(trades_a)}")
    result_a  = execute_trades(trades_a, LEVERAGE, "ORIGINAL")

    # ========== [B] FILTERED ==========
    print(f"\n🟢 Running [B] FILTERED ({filter_str})...")
    signals_b = generate_signals(df, start_idx, use_filters=True)
    trades_b  = signals_to_trades(signals_b, df)
    print(f"   Signals: {len(signals_b)}  |  Trades: {len(trades_b)}")
    result_b  = execute_trades(trades_b, LEVERAGE, "FILTERED")

    if result_a is None or result_b is None:
        print("⚠️  ไม่พอ trades"); return

    # ========== A/B COMPARISON TABLE ==========
    print(f"\n{'='*75}")
    print(f"  📊 A/B COMPARISON — {SYMBOL} EMA{EMA_PERIOD} | {LEVERAGE}x")
    print(f"  [B] Filters: {filter_str}")
    print(f"{'='*75}")

    col_a = 18
    col_b = 18
    col_d = 16

    hdr = f"  {'Metric':<22} {'[A] ORIGINAL':>{col_a}} {'[B] FILTERED':>{col_b}} {'Δ Change':>{col_d}}"
    print(hdr)
    print(f"  {'-'*72}")

    def ab_row(label, va, vb, fmt, delta_fmt=None, higher_better=True):
        sa = fmt.format(va)
        sb = fmt.format(vb)
        delta = vb - va
        if delta_fmt:
            sd = delta_fmt.format(delta)
        else:
            sd = f"{delta:+.1f}"
        # Arrow indicator
        if higher_better:
            arrow = "✅" if delta > 0 else ("⚠️" if delta < 0 else "  ")
        else:
            arrow = "✅" if delta < 0 else ("⚠️" if delta > 0 else "  ")

        print(f"  {label:<22} {sa:>{col_a}} {sb:>{col_b}} {sd:>{col_d-2}} {arrow}")

    a, b = result_a, result_b

    ab_row("จำนวนเทรด",     a['total'], b['total'], "{:>10}", "{:>+10}", False)
    trades_reduced = (1 - b['total'] / a['total']) * 100 if a['total'] > 0 else 0
    print(f"  {'  (ลดลง)':<22} {'':>{col_a}} {'':>{col_b}} {f'{trades_reduced:.0f}% ลดลง':>{col_d}}")
    print(f"  {'-'*72}")

    ab_row("ผลตอบแทนรวม",   a['ret'], b['ret'], "{:>9.1f}%", "{:>+9.1f}%")
    ab_row("CAGR",          a['cagr'], b['cagr'], "{:>9.1f}%", "{:>+9.1f}%")
    ab_row("เงินสุดท้าย",   a['fin'], b['fin'], "${:>9,.0f}", "${:>+9,.0f}")
    print(f"  {'-'*72}")

    ab_row("Win Rate",      a['wr'], b['wr'], "{:>9.1f}%", "{:>+9.1f}%")
    ab_row("Avg Win",       a['avg_w'], b['avg_w'], "${:>9,.0f}", "${:>+9,.0f}")
    ab_row("Avg Loss",      a['avg_l'], b['avg_l'], "${:>9,.0f}", "${:>+9,.0f}", False)
    ab_row("Expectancy",    a['exp'], b['exp'], "${:>9,.0f}", "${:>+9,.0f}")
    ab_row("Profit Factor", a['pf'], b['pf'], "{:>10.2f}", "{:>+10.2f}")
    print(f"  {'-'*72}")

    ab_row("Max Drawdown",  a['max_dd'], b['max_dd'], "{:>9.1f}%", "{:>+9.1f}%", False)
    ab_row("Avg Drawdown",  a['avg_dd'], b['avg_dd'], "{:>9.1f}%", "{:>+9.1f}%", False)
    print(f"  {'-'*72}")

    ab_row("Sharpe Ratio",  a['sharpe'], b['sharpe'], "{:>10.2f}", "{:>+10.2f}")
    ab_row("Sortino Ratio", a['sortino'], b['sortino'], "{:>10.2f}", "{:>+10.2f}")
    ab_row("Calmar Ratio",  a['calmar'], b['calmar'], "{:>10.2f}", "{:>+10.2f}")
    print(f"  {'-'*72}")

    ab_row("Avg Hold (hrs)",a['avg_hold'], b['avg_hold'], "{:>9.0f}", "{:>+9.0f}")
    print(f"  {'-'*72}")

    print(f"  {'💸 COST':}")
    ab_row("  Commission",  a['comm_total'], b['comm_total'], "${:>9,.0f}", "${:>+9,.0f}", False)
    ab_row("  Funding Fee", a['funding_total'], b['funding_total'], "${:>9,.0f}", "${:>+9,.0f}", False)
    ab_row("  Interest",    a['interest_total'], b['interest_total'], "${:>9,.0f}", "${:>+9,.0f}", False)
    ab_row("  TOTAL COST",  a['all_cost_total'], b['all_cost_total'], "${:>9,.0f}", "${:>+9,.0f}", False)
    print(f"{'='*75}")

    # ---- B&H reference ----
    print(f"\n  📈 Buy & Hold: ${bnh_final:,.0f} ({bnh_ret:+.1f}%)")

    # ---- Filtered trades that were SKIPPED ----
    skipped = a['total'] - b['total']
    if skipped > 0:
        # Estimate PnL of skipped trades
        print(f"\n  🔍 Filter กรองออก {skipped} trades ({trades_reduced:.0f}%)")
        # Check if losses were primarily filtered
        orig_losses = a['losses']
        filt_losses = b['losses']
        losses_removed = orig_losses - filt_losses
        if losses_removed > 0:
            print(f"     ลด Losses: {orig_losses} → {filt_losses} "
                  f"(กรอง {losses_removed} losing trades)")

    # ---- Direction breakdown ----
    for label, res in [("[A] ORIGINAL", a), ("[B] FILTERED", b)]:
        print(f"\n  {label}:")
        for d in ['LONG', 'SHORT']:
            sub = res['df_t'][res['df_t']['direction'] == d]
            if not sub.empty:
                w = len(sub[sub['result'] == 'WIN'])
                wr2 = w / len(sub) * 100
                p = sub['pnl_net'].sum()
                print(f"     {d:<7}: {len(sub):>3} trades | "
                      f"WR: {wr2:.1f}% | PnL: ${p:,.0f}")

    # ---- Save CSV ----
    csv_path = os.path.join(DESKTOP, "ema_195_ab_trades.csv")
    try:
        df_combined = pd.concat([a['df_t'], b['df_t']], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"\n  💾 CSV (A+B): {csv_path}")
    except Exception as e:
        print(f"\n  ⚠️  CSV: {e}")

    # ========== CHART — A/B Overlay ==========
    if not HAS_PLOT:
        print("\n✅ A/B Comparison เสร็จสิ้น!")
        if MC_ENABLED:
            run_monte_carlo_ab(a, b)
        return

    try:
        fig, axes = plt.subplots(3, 1, figsize=(18, 18),
                                 gridspec_kw={'height_ratios': [3.5, 1.2, 2]})
        fig.patch.set_facecolor('#0f0f0f')
        for ax in axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            ax.xaxis.label.set_color('#cccccc')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')

        fig.suptitle(
            f'{SYMBOL}  EMA {EMA_PERIOD} — A/B Comparison — 4H | {LEVERAGE}x\n'
            f'[A] Original: {a["total"]} trades  {a["ret"]:+.0f}%  |  '
            f'[B] Filtered ({filter_str}): {b["total"]} trades  {b["ret"]:+.0f}%\n'
            f'{START_DATE} → {END_DATE}  |  '
            f'Fee {FEE_PER_SIDE}%  |  Fund {FUNDING_RATE}%/{FUNDING_INTERVAL_HOURS}h',
            fontsize=12, fontweight='bold', color='white'
        )

        # ---- Panel 1: Equity Curves ----
        ax1 = axes[0]
        t0 = df_bt.index[0]

        # [A] Original
        times_a = [t0] + list(a['df_e']['time'])
        vals_a  = [INITIAL_CAPITAL] + list(a['df_e']['equity'])
        ax1.plot(times_a, vals_a, color='#FF6B6B', lw=2.5,
                 label=f'[A] Original  {a["ret"]:+.0f}%  WR:{a["wr"]:.0f}%  '
                       f'SR:{a["sharpe"]:.1f}  Cal:{a["calmar"]:.1f}',
                 zorder=3)

        # [B] Filtered
        times_b = [t0] + list(b['df_e']['time'])
        vals_b  = [INITIAL_CAPITAL] + list(b['df_e']['equity'])
        ax1.plot(times_b, vals_b, color='#00FF88', lw=2.5,
                 label=f'[B] Filtered  {b["ret"]:+.0f}%  WR:{b["wr"]:.0f}%  '
                       f'SR:{b["sharpe"]:.1f}  Cal:{b["calmar"]:.1f}',
                 zorder=4)

        # B&H
        bnh_equity = df_bt['Close'] / bnh_start * INITIAL_CAPITAL
        ax1.plot(df_bt.index, bnh_equity, color='#FFA500', lw=1.5,
                 ls='--', label=f'Buy & Hold  {bnh_ret:+.1f}%', alpha=0.6)

        ax1.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.4)
        ax1.set_ylabel('Portfolio Value (USD)', fontsize=11)
        ax1.set_yscale('log')
        ax1.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white',
                   loc='upper left')
        ax1.grid(True, alpha=0.15, color='white')
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # ---- Panel 2: Drawdown Overlay ----
        ax2 = axes[1]
        times_dd_a = [t0] + list(a['df_e']['time'])
        times_dd_b = [t0] + list(b['df_e']['time'])
        ax2.fill_between(times_dd_a, a['dd_arr'], 0,
                         color='#FF6B6B', alpha=0.3, label=f'[A] MaxDD: {a["max_dd"]:.1f}%')
        ax2.fill_between(times_dd_b, b['dd_arr'], 0,
                         color='#00FF88', alpha=0.3, label=f'[B] MaxDD: {b["max_dd"]:.1f}%')
        ax2.plot(times_dd_a, a['dd_arr'], color='#FF6B6B', lw=1, alpha=0.7)
        ax2.plot(times_dd_b, b['dd_arr'], color='#00FF88', lw=1, alpha=0.7)
        ax2.axhline(0, color='white', lw=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                   loc='lower left')
        ax2.grid(True, alpha=0.15, color='white')

        # ---- Panel 3: A/B Comparison Summary ----
        ax3 = axes[2]
        ax3.axis('off')

        # Build comparison text block
        metrics = [
            ("Trades",        f"{a['total']}",           f"{b['total']}",          f"{b['total']-a['total']:+d}"),
            ("Return",        f"{a['ret']:+.1f}%",       f"{b['ret']:+.1f}%",      f"{b['ret']-a['ret']:+.1f}%"),
            ("CAGR",          f"{a['cagr']:.1f}%",       f"{b['cagr']:.1f}%",      f"{b['cagr']-a['cagr']:+.1f}%"),
            ("Win Rate",      f"{a['wr']:.1f}%",         f"{b['wr']:.1f}%",        f"{b['wr']-a['wr']:+.1f}%"),
            ("Profit Factor", f"{a['pf']:.2f}",          f"{b['pf']:.2f}",         f"{b['pf']-a['pf']:+.2f}"),
            ("Max DD",        f"{a['max_dd']:.1f}%",     f"{b['max_dd']:.1f}%",    f"{b['max_dd']-a['max_dd']:+.1f}%"),
            ("Sharpe",        f"{a['sharpe']:.2f}",      f"{b['sharpe']:.2f}",     f"{b['sharpe']-a['sharpe']:+.2f}"),
            ("Sortino",       f"{a['sortino']:.2f}",     f"{b['sortino']:.2f}",    f"{b['sortino']-a['sortino']:+.2f}"),
            ("Calmar",        f"{a['calmar']:.2f}",      f"{b['calmar']:.2f}",     f"{b['calmar']-a['calmar']:+.2f}"),
            ("Total Cost",    f"${a['all_cost_total']:,.0f}", f"${b['all_cost_total']:,.0f}",
             f"${b['all_cost_total']-a['all_cost_total']:+,.0f}"),
        ]

        summary_text = f"{'A/B COMPARISON SUMMARY':^60}\n"
        summary_text += f"{'─'*60}\n"
        summary_text += f"{'Metric':<16} {'[A] Original':>14} {'[B] Filtered':>14} {'Δ Delta':>14}\n"
        summary_text += f"{'─'*60}\n"
        for name, va, vb, vd in metrics:
            summary_text += f"{name:<16} {va:>14} {vb:>14} {vd:>14}\n"
        summary_text += f"{'─'*60}\n"
        summary_text += f"\nFilters: {filter_str}\n"
        summary_text += f"Buy & Hold: {bnh_ret:+.1f}%"

        ax3.text(0.5, 0.95, summary_text, transform=ax3.transAxes,
                 fontsize=11, va='top', ha='center', color='white',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#0a0a1a', alpha=0.95))

        plt.tight_layout()
        chart_path = os.path.join(DESKTOP, "ema_195_ab_chart.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  📊 A/B Chart: {chart_path}")
        plt.show()

    except Exception as e:
        print(f"  ⚠️  Chart error: {e}")

    print("\n✅ A/B Comparison เสร็จสิ้น!")

    # ========== MONTE CARLO (both) ==========
    if MC_ENABLED:
        run_monte_carlo_ab(a, b)

    print(f"\n💡 Tips:")
    print(f"   - ADX_THRESHOLD = {ADX_THRESHOLD} (ลอง 15, 20, 25, 30)")
    print(f"   - ATR_MULTIPLIER = {ATR_MULTIPLIER} (ลอง 0.5, 0.8, 1.0, 1.2)")
    print(f"   - ปิด filter: ADX_ENABLED=False / ATR_ENABLED=False")
    print(f"   - ลอง LEVERAGE = 2 หรือ 3 เพื่อดู amplified effect")
    print(f"ข้อมูลเริ่มตั้งแต่วันที่: {df.index.min()}")
    print(f"ข้อมูลสิ้นสุดวันที่: {df.index.max()}") 


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Monte Carlo — A/B Side-by-Side
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_monte_carlo_ab(result_a, result_b):
    print(f"\n{'='*70}")
    print(f"  🎲 Monte Carlo A/B ({MC_RUNS:,} รอบ) | {LEVERAGE}x")
    print(f"{'='*70}")

    results_mc = {}
    for label, res in [("[A] ORIGINAL", result_a), ("[B] FILTERED", result_b)]:
        pnl_pcts = res['df_t']['pnl_pct'].values.copy() / 100
        n_trades = len(pnl_pcts)

        print(f"\n  {label}: {n_trades} trades...", end="", flush=True)
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
                    equity[t + 1:] = 0; break
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

        results_mc[label] = {
            'n_trades': n_trades,
            'final_returns': final_returns,
            'max_drawdowns': max_drawdowns,
            'all_equity': all_equity,
            'ret_p': ret_p, 'dd_p': dd_p,
            'prob_profit': prob_profit, 'prob_blow': prob_blow,
        }

    # Print comparison
    pcts = [5, 25, 50, 75, 95]
    print(f"\n  {'Percentile':<14} {'[A] Return':>12} {'[B] Return':>12} {'[A] MaxDD':>10} {'[B] MaxDD':>10}")
    print(f"  {'-'*60}")
    for i, p in enumerate(pcts):
        ra = results_mc["[A] ORIGINAL"]['ret_p'][i]
        rb = results_mc["[B] FILTERED"]['ret_p'][i]
        da = results_mc["[A] ORIGINAL"]['dd_p'][i]
        db = results_mc["[B] FILTERED"]['dd_p'][i]
        print(f"  {p:>3}th          {ra:>+10.1f}%  {rb:>+10.1f}%  {da:>9.1f}%  {db:>9.1f}%")

    for label, mc in results_mc.items():
        print(f"\n  {label}:")
        print(f"     Prob Profit: {mc['prob_profit']:.1f}%  |  "
              f"Prob Blow-up: {mc['prob_blow']:.1f}%  |  "
              f"Median Return: {mc['ret_p'][2]:+.1f}%")

    if not HAS_PLOT:
        return

    # ---- MC Chart: 2×2 for A, 2×2 for B → 2×4 layout ----
    try:
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
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
            f'Monte Carlo A/B  |  {MC_RUNS:,} runs  |  {SYMBOL} EMA{EMA_PERIOD} 4H | {LEVERAGE}x',
            fontsize=13, fontweight='bold', color='white')

        for col_offset, (label, mc, color) in enumerate([
            ("[A] ORIGINAL", results_mc["[A] ORIGINAL"], '#FF6B6B'),
            ("[B] FILTERED", results_mc["[B] FILTERED"], '#00FF88'),
        ]):
            n = mc['n_trades']
            x = np.arange(n + 1)

            # Equity paths
            ax_eq = axes[0, col_offset * 2]
            sample = np.random.choice(MC_RUNS, min(150, MC_RUNS), replace=False)
            for idx in sample:
                eq = mc['all_equity'][idx] * INITIAL_CAPITAL
                c = '#00FF88' if eq[-1] > INITIAL_CAPITAL else '#FF4444'
                ax_eq.plot(x, eq, color=c, alpha=0.04, lw=0.5)
            eq_50 = np.percentile(mc['all_equity'], 50, axis=0) * INITIAL_CAPITAL
            eq_5  = np.percentile(mc['all_equity'], 5,  axis=0) * INITIAL_CAPITAL
            eq_95 = np.percentile(mc['all_equity'], 95, axis=0) * INITIAL_CAPITAL
            ax_eq.fill_between(x, eq_5, eq_95, alpha=0.15, color=color)
            ax_eq.plot(x, eq_50, color=color, lw=2)
            ax_eq.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.5)
            ax_eq.set_title(f'{label}\nEquity Paths', fontsize=9)
            ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
            ax_eq.grid(True, alpha=0.15, color='white')

            # Returns distribution
            ax_ret = axes[0, col_offset * 2 + 1]
            ax_ret.hist(mc['final_returns'], bins=60, color=color, alpha=0.7, edgecolor='none')
            ax_ret.axvline(0, color='white', lw=1, alpha=0.5)
            ax_ret.axvline(mc['ret_p'][2], color='white', lw=2, ls='--',
                          label=f'Median: {mc["ret_p"][2]:+.0f}%')
            ax_ret.set_title(f'Returns\nProfit: {mc["prob_profit"]:.0f}%', fontsize=9)
            ax_ret.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white')
            ax_ret.grid(True, alpha=0.15, color='white')

            # Max DD distribution
            ax_dd = axes[1, col_offset * 2]
            ax_dd.hist(mc['max_drawdowns'], bins=60, color='#FF4444', alpha=0.7, edgecolor='none')
            ax_dd.axvline(mc['dd_p'][2], color='#FFA500', lw=2, ls='--',
                         label=f'Median: {mc["dd_p"][2]:.0f}%')
            ax_dd.set_title(f'Max Drawdown', fontsize=9)
            ax_dd.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white')
            ax_dd.grid(True, alpha=0.15, color='white')

            # Summary box
            ax_s = axes[1, col_offset * 2 + 1]
            ax_s.axis('off')
            txt = (
                f"{label}\n{'─'*28}\n"
                f"Trades: {n}\n"
                f"Prob Profit: {mc['prob_profit']:.1f}%\n"
                f"Prob Blow: {mc['prob_blow']:.1f}%\n"
                f"{'─'*28}\n"
                f"     Ret     MaxDD\n"
            )
            for i, p in enumerate(pcts):
                txt += f"{p:>3}th {mc['ret_p'][i]:>+7.0f}%  {mc['dd_p'][i]:>6.1f}%\n"
            ax_s.text(0.05, 0.95, txt, transform=ax_s.transAxes,
                     fontsize=9, va='top', color='white', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#0a0a1a', alpha=0.9))

        plt.tight_layout()
        mc_path = os.path.join(DESKTOP, "ema_195_ab_montecarlo.png")
        plt.savefig(mc_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  📊 MC A/B: {mc_path}")
        plt.show()

    except Exception as e:
        print(f"  ⚠️  MC chart error: {e}")


if __name__ == "__main__":
    run_ab_comparison()
