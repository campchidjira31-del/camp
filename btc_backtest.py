"""
================================================
Backtest v10 — RSI Reversal (4H + RSI-MA filter)
================================================
Strategy:
  - 1D EMA: Trend filter (ปรับ period ได้)
    * Close > EMA = Bullish → LONG only
    * Close < EMA = Bearish → SHORT only

  - 4H RSI(14) + RSI-MA: Signal
    * LONG: RSI ทำ V-shape (ลงแล้วกลับขึ้น), ≤ 75
            + RSI อยู่ใต้เส้น RSI-MA (เปิด/ปิดได้)
    * SHORT: RSI ทำ peak (ขึ้นแล้วกลับลง), ≥ 25
             + RSI อยู่เหนือเส้น RSI-MA (เปิด/ปิดได้)

  - 4H ไม่ดู EMA trend (ดูแค่ 1D)
  - Entry: Open แท่งถัดไปหลัง signal
  - SL: Body แท่ง signal
  - TP: Fixed R:R

Data: Binance Public API
วิธีรัน:  python3 ~/Desktop/btc_backtest.py
================================================
"""

import os, time, sys, json
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
# ║            SETTINGS ปรับตรงนี้                               ║
# ╚═════════════════════════════════════════════════════════════╝

SYMBOL          = "ETHUSDT"
START_DATE      = "2024-01-01"
END_DATE        = "2026-01-01"
INITIAL_CAPITAL = 2500
RISK_PER_TRADE  = 0.02              # 2% ต่อเทรด
LEVERAGE        = 6                 # 1 = Spot, 6 = Futures x6
RR_RATIOS       = [4, 8, 10, 12, 15]

# Direction filter
LONG_ONLY       = False             # True = LONG only, False = LONG + SHORT

# TF เล็ก — ใช้ 4H (ไม่ดู EMA ใน TF นี้)
SMALL_TF        = '1h'              # '1h' หรือ '4h'

# RSI
RSI_PERIOD      = 14
RSI_BULL_MAX    = 75                # LONG: RSI ≤ ค่านี้
RSI_BEAR_MIN    = 25                # SHORT: RSI ≥ ค่านี้

# RSI-MA (เส้นสีเหลือง)
RSI_MA_PERIOD   = 14                # MA period ของ RSI
RSI_MA_TYPE     = "SMA"             # "SMA" หรือ "EMA"

# RSI-MA Zone Filter (เปิด/ปิดได้)
RSI_MA_FILTER   = True              # True = เปิดใช้ RSI-MA zone filter
                                    # LONG:  RSI ต้องอยู่ใต้ RSI-MA (reversal จาก oversold zone)
                                    # SHORT: RSI ต้องอยู่เหนือ RSI-MA (reversal จาก overbought zone)
                                    # False = ไม่สน RSI-MA zone (เหมือน v9 เดิม)

# EMA (1D trend) — ปรับ period ได้
EMA_PERIOD      = 65                # ลอง 20, 50, 100, 200

# Trade management
MAX_BARS        = 10000

# Costs — Aggressive Limit Order
FEE_PER_SIDE    = 0.05              # 0.03% per side (maker 0.02% + limit slip 0.01%)

# MIN SL filter
MIN_RISK_PCT    = 0.15              # SL ต้องห่างอย่างน้อย x%

# Monte Carlo
MC_ENABLED      = True
MC_RUNS         = 5000
MC_USE_NET      = True

# Warmup
WARMUP_1D_DAYS  = 365
WARMUP_1H_DAYS  = 90                # เพิ่มสำหรับ 4H (ต้อง warmup RSI+MA)

# ╔═════════════════════════════════════════════════════════════╗
# ║          จบ SETTINGS                                        ║
# ╚═════════════════════════════════════════════════════════════╝


# ถ้า END_DATE ว่าง ใช้วันนี้
if not END_DATE:
    END_DATE = datetime.now().strftime("%Y-%m-%d")

# ป้องกัน END_DATE อยู่ในอนาคต
today_str = datetime.now().strftime("%Y-%m-%d")
if END_DATE > today_str:
    END_DATE = today_str


def binance_klines(symbol, interval, start_str, end_str, limit=1000):
    """
    ดึง klines จาก Binance public API (ไม่ต้อง API key)
    interval: '1h', '1d', '4h', '15m', etc.
    ไม่มีจำกัดวัน — ดึงย้อนหลังได้เท่าที่ Binance มี
    """
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end_str,   "%Y-%m-%d").timestamp() * 1000)

    all_data = []
    cursor   = start_ms

    while cursor < end_ms:
        params = {
            'symbol':    symbol,
            'interval':  interval,
            'startTime': cursor,
            'endTime':   end_ms,
            'limit':     limit,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ❌ API error: {e}")
            break

        if not data:
            break

        all_data.extend(data)
        # เลื่อน cursor ไปหลัง candle สุดท้าย
        cursor = data[-1][6] + 1  # closeTime + 1ms

        # ถ้าได้น้อยกว่า limit = หมดแล้ว
        if len(data) < limit:
            break

        time.sleep(0.1)  # rate limit friendly

    if not all_data:
        return pd.DataFrame()

    # แปลงเป็น DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_vol', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    df['Open']  = df['Open'].astype(float)
    df['High']  = df['High'].astype(float)
    df['Low']   = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(float)

    df.index = pd.to_datetime(df['open_time'], unit='ms')
    df.index.name = 'Date'
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df[~df.index.duplicated(keep='first')]
    return df.sort_index()


def safe_download_1d(symbol, start, end):
    """ดาวน์โหลด 1D จาก Binance"""
    print(f"\n📥 1D data ({start} → {end}) from Binance...")
    df = binance_klines(symbol, '1d', start, end)
    if df.empty:
        print("   ❌ ไม่มีข้อมูล 1D")
        return pd.DataFrame()
    print(f"   ✅ {len(df)} แท่ง ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def safe_download_small(symbol, start_str, end_str):
    """ดาวน์โหลด small TF จาก Binance"""
    print(f"📥 {SMALL_TF.upper()} data ({start_str} → {end_str}) from Binance...")
    df = binance_klines(symbol, SMALL_TF, start_str, end_str)
    if df.empty:
        print(f"   ❌ ไม่มีข้อมูล {SMALL_TF.upper()}")
        return pd.DataFrame()
    print(f"   ✅ {len(df)} แท่ง ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def tz_strip(df):
    """Strip timezone if present (Binance data is UTC without tz)"""
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


def calc_rsi(prices, period=14):
    d  = prices.diff()
    ag = d.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    al = (-d.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)

def simulate_trade(df_1h, signal_idx, direction, entry_price, sl_price, rr_ratio):
    """Fixed TP/SL — entry bar = signal_idx + 1"""
    risk = abs(entry_price - sl_price)
    if direction == 'LONG':
        tp = entry_price + risk * rr_ratio
    else:
        tp = entry_price - risk * rr_ratio

    entry_idx = signal_idx + 1
    for j in range(entry_idx + 1, min(entry_idx + MAX_BARS, len(df_1h))):
        fh = float(df_1h.iloc[j]['High'])
        fl = float(df_1h.iloc[j]['Low'])

        if direction == 'LONG':
            if fl <= sl_price:
                return 'SL', sl_price, df_1h.index[j], tp, j
            if fh >= tp:
                return 'TP', tp, df_1h.index[j], tp, j
        else:
            if fh >= sl_price:
                return 'SL', sl_price, df_1h.index[j], tp, j
            if fl <= tp:
                return 'TP', tp, df_1h.index[j], tp, j

    last_idx = min(entry_idx + MAX_BARS - 1, len(df_1h) - 1)
    return 'TIMEOUT', float(df_1h.iloc[last_idx]['Close']), \
           df_1h.index[last_idx], tp, last_idx


def run_backtest():
    rr_str = ' / '.join([f'1:{r}' for r in RR_RATIOS])
    print("=" * 70)
    print(f"  {SYMBOL} Backtest v10 | RSI Reversal + RSI-MA | {SMALL_TF.upper()}")
    print(f"  Period : {START_DATE} → {END_DATE}")
    print(f"  R:R    : {rr_str}  |  Leverage: x{LEVERAGE}")
    print(f"  Risk   : {RISK_PER_TRADE*100:.0f}% per trade  |  {'LONG ONLY' if LONG_ONLY else 'LONG + SHORT'}")
    print(f"  Fee    : {FEE_PER_SIDE}% per side (aggressive limit)")
    print(f"  1D EMA : {EMA_PERIOD}  |  RSI-MA: {RSI_MA_TYPE}({RSI_MA_PERIOD})")
    print(f"  RSI-MA filter: {'ON' if RSI_MA_FILTER else 'OFF'}")
    print(f"  MIN SL : {MIN_RISK_PCT}%")
    print("=" * 70)

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    w1d = (start_dt - timedelta(days=WARMUP_1D_DAYS)).strftime("%Y-%m-%d")
    w_small = (start_dt - timedelta(days=WARMUP_1H_DAYS)).strftime("%Y-%m-%d")

    # ---- Download ----
    df_1d = safe_download_1d(SYMBOL, w1d, END_DATE)
    if df_1d.empty:
        print("❌ ไม่สามารถดาวน์โหลด 1D ได้"); return
    df_1d = tz_strip(df_1d)

    df_small = safe_download_small(SYMBOL, w_small, END_DATE)
    if df_small.empty:
        print(f"❌ ไม่สามารถดาวน์โหลด {SMALL_TF.upper()} ได้"); return
    df_small = tz_strip(df_small)
    print(f"   ✅ {SMALL_TF.upper()} รวม: {len(df_small)} แท่ง "
          f"({df_small.index[0].date()} → {df_small.index[-1].date()})")

    # ---- Indicators ----
    # 1D: EMA for trend only
    df_1d['EMA']   = df_1d['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_1d['trend'] = np.where(df_1d['Close'] > df_1d['EMA'], 'bullish', 'bearish')

    # Small TF: RSI + RSI-MA (ไม่ดู EMA ใน TF นี้)
    df_small['RSI'] = calc_rsi(df_small['Close'], RSI_PERIOD)
    if RSI_MA_TYPE == "EMA":
        df_small['RSI_MA'] = df_small['RSI'].ewm(span=RSI_MA_PERIOD, adjust=False).mean()
    else:
        df_small['RSI_MA'] = df_small['RSI'].rolling(RSI_MA_PERIOD).mean()

    warmup_bars = len(df_1d[df_1d.index < pd.Timestamp(START_DATE)])
    print(f"\n📐 EMA{EMA_PERIOD} warmup: {warmup_bars} แท่ง = "
          f"{warmup_bars/EMA_PERIOD:.1f}x period "
          f"({'✅' if warmup_bars >= EMA_PERIOD*3 else '⚠️ น้อยเกินไป'})")

    # ---- BT start ----
    actual_small_start = df_small.index[0]
    bt_start = max(pd.Timestamp(START_DATE), actual_small_start)

    # ---- Buy & Hold ----
    df_1d_bt   = df_1d[df_1d.index >= bt_start].copy()
    if df_1d_bt.empty:
        print("❌ ไม่มี 1D data ในช่วง backtest"); return
    bnh_start  = float(df_1d_bt['Close'].iloc[0])
    bnh_end    = float(df_1d_bt['Close'].iloc[-1])
    bnh_final  = INITIAL_CAPITAL / bnh_start * bnh_end
    bnh_ret    = (bnh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    bnh_equity = df_1d_bt['Close'] / bnh_start * INITIAL_CAPITAL

    # ---- Lookup ----
    d1_dates  = df_1d.index.normalize().values.astype('datetime64[ns]')
    d1_trend  = df_1d['trend'].values
    start_idx = df_small.index.searchsorted(bt_start)

    print(f"\n🔄 Backtest จาก {df_small.index[min(start_idx, len(df_small)-1)].date()}...")

    # ========== COLLECT SIGNALS — RSI Reversal + RSI-MA zone ==========
    signals = []
    for i in range(2, len(df_small) - 1):
        if i < start_idx:
            continue

        row   = df_small.iloc[i]
        prev  = df_small.iloc[i - 1]
        prev2 = df_small.iloc[i - 2]

        rsi_now = row['RSI']
        rsi_p1  = prev['RSI']
        rsi_p2  = prev2['RSI']
        rsi_ma  = row['RSI_MA']

        if pd.isna(rsi_now) or pd.isna(rsi_p1) or pd.isna(rsi_p2) or pd.isna(rsi_ma):
            continue

        # 1D trend (ใช้ yesterday's)
        cur_date = np.datetime64(df_small.index[i].normalize(), 'ns')
        idx_1d   = np.searchsorted(d1_dates, cur_date, side='right') - 1
        if idx_1d < 1:
            continue
        trend_1d = d1_trend[idx_1d - 1]

        direction = None

        # LONG: 1D bullish + RSI V-shape + RSI ≤ 75 + (optional: RSI ใต้ RSI-MA)
        if trend_1d == 'bullish':
            if (rsi_p1 < rsi_p2 and rsi_now > rsi_p1
                    and rsi_now <= RSI_BULL_MAX):
                # RSI-MA zone filter
                if RSI_MA_FILTER:
                    if rsi_now < rsi_ma:    # RSI อยู่ใต้ RSI-MA → oversold zone
                        direction = 'LONG'
                else:
                    direction = 'LONG'

        # SHORT: 1D bearish + RSI peak + RSI ≥ 25 + (optional: RSI เหนือ RSI-MA)
        elif trend_1d == 'bearish':
            if (rsi_p1 > rsi_p2 and rsi_now < rsi_p1
                    and rsi_now >= RSI_BEAR_MIN):
                # RSI-MA zone filter
                if RSI_MA_FILTER:
                    if rsi_now > rsi_ma:    # RSI อยู่เหนือ RSI-MA → overbought zone
                        direction = 'SHORT'
                else:
                    direction = 'SHORT'

        if direction is None:
            continue

        # LONG_ONLY filter
        if LONG_ONLY and direction == 'SHORT':
            continue

        next_bar    = df_small.iloc[i + 1]
        entry_price = float(next_bar['Open'])
        entry_time  = df_small.index[i + 1]

        # SL = body ของแท่ง i (signal bar)
        if direction == 'LONG':
            sl = min(float(row['Open']), float(row['Close']))
        else:
            sl = max(float(row['Open']), float(row['Close']))

        risk = abs(entry_price - sl)
        risk_pct = risk / entry_price * 100
        if risk_pct < MIN_RISK_PCT:
            continue    # ข้ามเทรดที่ SL แคบเกินไป
        if risk <= 0:
            continue

        signals.append({
            'i': i, 'direction': direction,
            'entry_price': entry_price, 'entry_time': entry_time,
            'sl': sl, 'risk': risk,
            'trend_1d': trend_1d,
            'rsi': rsi_now, 'rsi_ma': rsi_ma,
        })

    print(f"   📊 พบ {len(signals)} signals")
    if not signals:
        print("⚠️  ไม่พบสัญญาณ — ลองขยาย START_DATE / END_DATE")
        return

    # ========== RUN EACH RR RATIO ==========
    rr_results = {}  # rr -> { trades, equity_curve, stats }

    for rr in RR_RATIOS:
        trades = []
        eq_curve = []
        capital  = INITIAL_CAPITAL
        skip_until = 0

        for sig in signals:
            i = sig['i']
            if i <= skip_until:
                continue

            entry_price = sig['entry_price']
            sl          = sig['sl']
            risk        = sig['risk']
            direction   = sig['direction']

            qty = (capital * RISK_PER_TRADE) / risk

            # Leverage margin cap (Binance Futures)
            margin_required = (entry_price * qty) / LEVERAGE
            if margin_required > capital:
                max_qty = (capital * LEVERAGE) / entry_price
                qty = max_qty * 0.95  # 5% buffer

            outcome, exit_p, exit_time, tp_val, exit_idx = simulate_trade(
                df_small, i, direction, entry_price, sl, rr)

            pnl = (exit_p - entry_price) * qty if direction == 'LONG' \
                  else (entry_price - exit_p) * qty

            # Cost = fee per side × notional ทั้งขา entry + exit
            cost_entry = (entry_price * qty) * (FEE_PER_SIDE / 100)
            cost_exit  = (exit_p * qty) * (FEE_PER_SIDE / 100)
            cost       = cost_entry + cost_exit
            pnl_net   = pnl - cost

            capital  += pnl_net
            skip_until = exit_idx

            rr_actual = pnl_net / (risk * qty) if qty > 0 else 0
            oc = 'WIN' if pnl_net > 0 else ('LOSS' if pnl_net < 0 else 'BE')

            eq_curve.append({'time': exit_time, 'equity': capital, 'pnl': pnl_net})
            trades.append({
                'entry_time': sig['entry_time'], 'exit_time': exit_time,
                'direction': direction, 'entry_price': round(entry_price, 2),
                'sl': round(sl, 2), 'tp': round(tp_val, 2),
                'exit_price': round(exit_p, 2), 'outcome': outcome,
                'pnl_gross': round(pnl, 2), 'cost': round(cost, 2),
                'pnl_net': round(pnl_net, 2), 'rr_actual': round(rr_actual, 2),
                'result': oc, 'capital': round(capital, 2),
                'trend_1d': sig['trend_1d'],
                'rsi': round(sig['rsi'], 2),
                'rsi_ma': round(sig.get('rsi_ma', 0), 2),
            })

        df_t = pd.DataFrame(trades)
        df_e = pd.DataFrame(eq_curve)

        # Stats
        total  = len(df_t)
        if total == 0:
            continue

        wins   = len(df_t[df_t['result'] == 'WIN'])
        losses = len(df_t[df_t['result'] == 'LOSS'])
        wr     = wins / total * 100
        fin    = capital
        ret    = (fin - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        eq_arr  = np.array([INITIAL_CAPITAL] + list(df_e['equity']))
        peak    = np.maximum.accumulate(eq_arr)
        dd_arr  = (eq_arr - peak) / peak * 100
        max_dd  = dd_arr.min()
        avg_dd  = dd_arr[dd_arr < 0].mean() if len(dd_arr[dd_arr < 0]) > 0 else 0

        gw = df_t[df_t['pnl_net'] > 0]['pnl_net'].sum()
        gl = abs(df_t[df_t['pnl_net'] < 0]['pnl_net'].sum())
        pf = gw / gl if gl > 0 else float('inf')

        avg_w = df_t[df_t['result'] == 'WIN']['pnl_net'].mean()  if wins   else 0
        avg_l = df_t[df_t['result'] == 'LOSS']['pnl_net'].mean() if losses else 0
        exp   = (wr / 100 * avg_w) + ((1 - wr / 100) * avg_l)

        # ---- Sharpe Ratio (annualized) ----
        pnl_arr = df_t['pnl_net'].values
        years = max((pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days / 365, 0.5)
        trades_per_year = total / years
        if len(pnl_arr) > 1 and pnl_arr.std() > 0:
            sharpe = (pnl_arr.mean() / pnl_arr.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0

        # ---- Sortino Ratio ----
        downside = pnl_arr[pnl_arr < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (pnl_arr.mean() / downside.std()) * np.sqrt(trades_per_year)
        else:
            sortino = 0

        # ---- Calmar Ratio ----
        ann_ret = ret / years
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        rr_results[rr] = {
            'df_t': df_t, 'df_e': df_e,
            'total': total, 'wins': wins, 'losses': losses, 'wr': wr,
            'fin': fin, 'ret': ret, 'max_dd': max_dd, 'avg_dd': avg_dd,
            'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l, 'exp': exp,
            'eq_arr': eq_arr, 'dd_arr': dd_arr,
            'cost_total': df_t['cost'].sum(),
            'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        }

    # ========== PRINT COMPARISON TABLE ==========
    rrs = list(rr_results.keys())
    print(f"\n{'='*90}")
    print(f"  RSI Reversal v10 ({SMALL_TF.upper()}) + RSI-MA  |  {SYMBOL}  |  "
          f"{'LONG ONLY' if LONG_ONLY else 'L+S'}  |  RSI-MA: {'ON' if RSI_MA_FILTER else 'OFF'}")
    print(f"{'='*90}")

    # Header
    hdr = f"  {'':24s}"
    for rr in rrs:
        hdr += f" {'1:'+str(rr):>10}"
    hdr += f" {'Buy&Hold':>10}"
    print(hdr)
    print(f"  {'-'*(24 + 11*len(rrs) + 11)}")

    # Rows
    def row(label, vals, fmt, bnh=None):
        s = f"  {label:<24}"
        for v in vals:
            s += f" {fmt.format(v):>10}"
        if bnh is not None:
            s += f" {fmt.format(bnh):>10}"
        print(s)

    row("เงินเริ่มต้น",  [INITIAL_CAPITAL]*len(rrs), "${:>8,.0f}", INITIAL_CAPITAL)
    row("เงินสุดท้าย",   [rr_results[r]['fin'] for r in rrs], "${:>8,.0f}", bnh_final)
    row("ผลตอบแทนรวม",   [rr_results[r]['ret'] for r in rrs], "{:>8.1f}%", bnh_ret)
    row("Max Drawdown",  [rr_results[r]['max_dd'] for r in rrs], "{:>8.1f}%")
    row("Avg Drawdown",  [rr_results[r]['avg_dd'] for r in rrs], "{:>8.1f}%")
    row("Profit Factor", [rr_results[r]['pf'] for r in rrs], "{:>9.2f}")

    print(f"  {'-'*(24 + 11*len(rrs) + 11)}")
    row("Sharpe Ratio",  [rr_results[r]['sharpe'] for r in rrs], "{:>9.2f}")
    row("Sortino Ratio", [rr_results[r]['sortino'] for r in rrs], "{:>9.2f}")
    row("Calmar Ratio",  [rr_results[r]['calmar'] for r in rrs], "{:>9.2f}")

    print(f"  {'-'*(24 + 11*len(rrs) + 11)}")
    row("จำนวนเทรด",     [rr_results[r]['total'] for r in rrs], "{:>9}")
    row("Win Rate",      [rr_results[r]['wr'] for r in rrs], "{:>8.1f}%")
    row("Avg Win",       [rr_results[r]['avg_w'] for r in rrs], "${:>8,.0f}")
    row("Avg Loss",      [rr_results[r]['avg_l'] for r in rrs], "${:>8,.0f}")
    row("Expectancy",    [rr_results[r]['exp'] for r in rrs], "${:>8,.0f}")
    row("💸 Total Cost",  [rr_results[r]['cost_total'] for r in rrs], "${:>8,.0f}")
    print(f"{'='*80}")

    # Direction breakdown
    print(f"\n  📈 แยกตาม Direction (RR ที่ดีสุด):")
    best_rr = max(rrs, key=lambda r: rr_results[r]['ret'])
    df_best = rr_results[best_rr]['df_t']
    for d in ['LONG', 'SHORT']:
        sub = df_best[df_best['direction'] == d]
        if not sub.empty:
            w   = len(sub[sub['result'] == 'WIN'])
            wr2 = w / len(sub) * 100
            p   = sub['pnl_net'].sum()
            print(f"     {d:<7}: {len(sub):>3} trades | WR: {wr2:.1f}% | PnL: ${p:,.0f}")

    # ---- Save CSV (best RR) ----
    csv_path = os.path.join(DESKTOP, "trades_result_v9.csv")
    try:
        df_best.to_csv(csv_path, index=False)
        print(f"\n  💾 CSV (1:{best_rr}): {csv_path}")
    except Exception as e:
        print(f"\n  ⚠️  CSV save failed: {e}")

    # ========== CHART — Multi-RR ==========
    if not HAS_PLOT:
        print("\n✅ เสร็จสิ้น!")
        return

    try:
        colors_rr = ['#00BFFF', '#00FF88', '#FFD700', '#FF6BFF', '#FF4444',
                     '#7B68EE', '#FF8C00', '#00CED1']

        fig, axes = plt.subplots(3, 1, figsize=(16, 13),
                                 gridspec_kw={'height_ratios': [3.5, 1.5, 1]})
        fig.patch.set_facecolor('#0f0f0f')
        for ax in axes:
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            ax.xaxis.label.set_color('#cccccc')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')

        fig.suptitle(
            f'{SYMBOL}  RSI Reversal v10 ({SMALL_TF.upper()}) + RSI-MA({"ON" if RSI_MA_FILTER else "OFF"})\n'
            f'{START_DATE} → {END_DATE}  |  SL = bar(i) body  |  '
            f'{"LONG ONLY" if LONG_ONLY else "L+S"}  |  '
            f'Fee {FEE_PER_SIDE}%/side  |  EMA{EMA_PERIOD}',
            fontsize=13, fontweight='bold', color='white'
        )

        # --- Panel 1: Equity curves ---
        ax1 = axes[0]
        t0  = df_1d_bt.index[0]

        for idx, rr in enumerate(rrs):
            res = rr_results[rr]
            times = [t0] + list(res['df_e']['time'])
            vals  = [INITIAL_CAPITAL] + list(res['df_e']['equity'])
            c = colors_rr[idx % len(colors_rr)]
            ax1.plot(times, vals, color=c, lw=2,
                     label=f'1:{rr}  {res["ret"]:+.1f}%  (WR:{res["wr"]:.0f}% SR:{res["sharpe"]:.1f})',
                     zorder=3)

        ax1.plot(df_1d_bt.index, bnh_equity, color='#FFA500', lw=1.5,
                 ls='--', label=f'Buy & Hold  {bnh_ret:+.1f}%', alpha=0.7)
        ax1.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.4)
        ax1.set_ylabel('Portfolio Value (USD)', fontsize=11)
        ax1.legend(fontsize=8.5, facecolor='#1a1a2e', labelcolor='white',
                   loc='upper left')
        ax1.grid(True, alpha=0.15, color='white')
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # --- Panel 2: Drawdown per RR ---
        ax2 = axes[1]
        for idx, rr in enumerate(rrs):
            res = rr_results[rr]
            times = [t0] + list(res['df_e']['time'])
            c = colors_rr[idx % len(colors_rr)]
            ax2.plot(times, res['dd_arr'], color=c, lw=1.2, alpha=0.8,
                     label=f'1:{rr}  MaxDD:{res["max_dd"]:.1f}%  AvgDD:{res["avg_dd"]:.1f}%')
        ax2.axhline(0, color='white', lw=0.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.legend(fontsize=7.5, facecolor='#1a1a2e', labelcolor='white',
                   loc='lower left', ncol=2)
        ax2.grid(True, alpha=0.15, color='white')

        # --- Panel 3: Bar chart comparison ---
        ax3 = axes[2]
        x_pos = np.arange(len(rrs))
        bar_w = 0.25
        returns = [rr_results[r]['ret'] for r in rrs]
        max_dds = [abs(rr_results[r]['max_dd']) for r in rrs]
        avg_dds = [abs(rr_results[r]['avg_dd']) for r in rrs]

        b1 = ax3.bar(x_pos - bar_w, returns, bar_w, label='Return %',
                     color=[colors_rr[i % len(colors_rr)] for i in range(len(rrs))],
                     alpha=0.85)
        b2 = ax3.bar(x_pos, [-d for d in max_dds], bar_w, label='Max DD %',
                     color='#FF4444', alpha=0.7)
        b3 = ax3.bar(x_pos + bar_w, [-d for d in avg_dds], bar_w, label='Avg DD %',
                     color='#FF8888', alpha=0.5)

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'1:{r}' for r in rrs], color='white')
        ax3.axhline(0, color='white', lw=0.5)
        ax3.axhline(bnh_ret, color='#FFA500', ls='--', lw=1,
                    label=f'B&H {bnh_ret:+.1f}%')
        ax3.set_ylabel('Return / DD (%)', fontsize=11)
        ax3.set_xlabel('Risk : Reward Ratio', fontsize=11)
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax3.grid(True, alpha=0.15, color='white')

        plt.tight_layout()
        chart_path = os.path.join(DESKTOP, "backtest_multi_rr.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"  📊 กราฟ: {chart_path}")
        plt.show()

    except Exception as e:
        print(f"  ⚠️  Chart error: {e}")

    print("\n✅ Backtest เสร็จสิ้น!")

    # ========== MONTE CARLO (ใช้ best RR) ==========
    if MC_ENABLED and best_rr in rr_results:
        print(f"\n  🎲 Monte Carlo ใช้ RR 1:{best_rr} (ผลตอบแทนสูงสุด)")
        run_monte_carlo(rr_results[best_rr]['df_t'])

    print(f"\n💡 Tips:")
    print(f"   - แก้ RR_RATIOS = [2, 2.5, 3, 5, 10] ได้ตามใจ")
    print(f"   - CSV บันทึกเฉพาะ RR ที่ดีที่สุด (1:{best_rr})")
    print(f"   - ตั้ง MC_ENABLED=False เพื่อข้าม Monte Carlo")


def run_monte_carlo(df_t):
    """Monte Carlo Simulation"""
    print(f"\n{'='*65}")
    print(f"  🎲 Monte Carlo Simulation ({MC_RUNS:,} รอบ)")
    print(f"{'='*65}")

    r_multiples = df_t['rr_actual'].values.copy()
    n_trades = len(r_multiples)
    risk_frac = RISK_PER_TRADE

    label = "Net (รวม cost)"
    print(f"  ใช้ PnL: {label}")
    print(f"  จำนวนเทรด: {n_trades}  |  Risk/trade: {risk_frac*100:.0f}%")
    print(f"  กำลังรัน...", end="", flush=True)

    np.random.seed(42)

    final_returns = np.zeros(MC_RUNS)
    max_drawdowns = np.zeros(MC_RUNS)
    all_equity    = np.zeros((MC_RUNS, n_trades + 1))

    for run in range(MC_RUNS):
        shuffled = np.random.permutation(r_multiples)
        equity = np.ones(n_trades + 1)
        for t in range(n_trades):
            pnl_pct = risk_frac * shuffled[t]
            equity[t + 1] = equity[t] * (1 + pnl_pct)

        all_equity[run] = equity
        final_returns[run] = (equity[-1] - 1) * 100

        peak = np.maximum.accumulate(equity)
        dd   = (equity - peak) / peak * 100
        max_drawdowns[run] = dd.min()

    print(" ✅")

    pcts = [5, 25, 50, 75, 95]
    ret_p = np.percentile(final_returns, pcts)
    dd_p  = np.percentile(max_drawdowns, pcts)
    prob_profit = np.mean(final_returns > 0) * 100
    prob_loss   = np.mean(final_returns < 0) * 100
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
            f'Monte Carlo Simulation  |  {MC_RUNS:,} runs  |  {label}\n'
            f'Prob Profit: {prob_profit:.1f}%  |  '
            f'Median Return: {ret_p[2]:+.1f}%  |  '
            f'Avg Max DD: {np.mean(max_drawdowns):.1f}%',
            fontsize=13, fontweight='bold', color='white')

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

        ax2 = axes[0, 1]
        ax2.hist(final_returns, bins=80, color='#00BFFF', alpha=0.7, edgecolor='none')
        ax2.axvline(0, color='white', lw=1, alpha=0.5)
        ax2.axvline(np.median(final_returns), color='#00FF88', lw=2, ls='--',
                    label=f'Median: {np.median(final_returns):+.1f}%')
        ax2.set_title('Final Returns', fontsize=11)
        ax2.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax2.grid(True, alpha=0.15, color='white')

        ax3 = axes[1, 0]
        ax3.hist(max_drawdowns, bins=80, color='#FF4444', alpha=0.7, edgecolor='none')
        ax3.axvline(np.median(max_drawdowns), color='#FFA500', lw=2, ls='--',
                    label=f'Median: {np.median(max_drawdowns):.1f}%')
        ax3.set_title('Max Drawdown', fontsize=11)
        ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax3.grid(True, alpha=0.15, color='white')

        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = (
            f"Monte Carlo Summary\n{'─'*32}\n"
            f"Runs: {MC_RUNS:,}  Trades: {n_trades}\n"
            f"Risk: {risk_frac*100:.0f}%\n{'─'*32}\n"
            f"Prob Profit: {prob_profit:.1f}%\n"
            f"{'─'*32}\n"
            f"       Return   Final$   MaxDD\n"
        )
        for i, p in enumerate(pcts):
            summary += f"{p:>3}th {ret_p[i]:>+7.1f}% ${fin_p[i]:>7,.0f} {dd_p[i]:>6.1f}%\n"
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
                 fontsize=10, va='top', color='white', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#0a0a1a', alpha=0.9))

        plt.tight_layout()
        mc_path = os.path.join(DESKTOP, "monte_carlo_v9.png")
        plt.savefig(mc_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  📊 Monte Carlo: {mc_path}")
        plt.show()
    except Exception as e:
        print(f"  ⚠️  MC chart error: {e}")


if __name__ == "__main__":
    run_backtest()
