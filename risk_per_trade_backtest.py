"""
================================================================
Risk Per Trade Optimization — Grid Search + Kelly Criterion
================================================================
Strategy: RSI Reversal | RR 1:4 (fixed)
Grid:     Risk Per Trade (1%-10%) × Leverage (2x-20x)

Features:
  - Kelly Criterion → optimal risk ทางทฤษฎี
  - Risk Heatmap: Return vs MaxDD vs Risk-Adjusted
  - Ruin probability per combination
  - Identifies sweet spot: best return with DD < threshold
  - Monte Carlo validation

Data: Binance Public API
วิธีรัน:  python3 risk_per_trade_backtest.py
================================================================
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
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

DESKTOP = os.path.expanduser("~/Desktop")


# ╔══════════════════════════════════════════════════╗
# ║            SETTINGS ปรับตรงนี้                     ║
# ╚══════════════════════════════════════════════════╝

SYMBOL          = "ETHUSDT"
START_DATE      = "2025-01-01"
END_DATE        = "2026-01-01"
INITIAL_CAPITAL = 2500

# Fixed strategy params
RR_RATIO        = 4
MIN_RISK_PCT    = 0.15

# Grid search ranges
RISK_RANGE      = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
LEVERAGE_RANGE  = [2, 3, 4, 6, 8, 10, 15, 20]

# DD tolerance (สำหรับหา sweet spot)
MAX_DD_TOLERANCE = -60              # % — ยอมรับ DD ได้มากสุดเท่าไหร่

# Indicators
RSI_PERIOD      = 14
RSI_BULL_MAX    = 75
RSI_BEAR_MIN    = 25
EMA_PERIOD      = 50

# Trade
MAX_BARS        = 10000
# Costs
FEE_PER_SIDE    = 0.05              # 0.05% per side → round trip 0.10%

# Monte Carlo (สำหรับ sweet spot)
MC_RUNS         = 5000

# Warmup
WARMUP_1D_DAYS  = 365
WARMUP_1H_DAYS  = 60

# ╔══════════════════════════════════════════════════╗
# ║          จบ SETTINGS                               ║
# ╚══════════════════════════════════════════════════╝

if not END_DATE:
    END_DATE = datetime.now().strftime("%Y-%m-%d")
today_str = datetime.now().strftime("%Y-%m-%d")
if END_DATE > today_str:
    END_DATE = today_str


# ============================================================
#  DATA & INDICATORS (เหมือน btc_backtest.py)
# ============================================================

def binance_klines(symbol, interval, start_str, end_str, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end_str,   "%Y-%m-%d").timestamp() * 1000)
    all_data, cursor = [], start_ms
    while cursor < end_ms:
        params = {'symbol': symbol, 'interval': interval,
                  'startTime': cursor, 'endTime': end_ms, 'limit': limit}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ❌ API error: {e}"); break
        if not data: break
        all_data.extend(data)
        cursor = data[-1][6] + 1
        if len(data) < limit: break
        time.sleep(0.1)
    if not all_data: return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=[
        'open_time','Open','High','Low','Close','Volume',
        'close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'])
    for c in ['Open','High','Low','Close','Volume']:
        df[c] = df[c].astype(float)
    df.index = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['Open','High','Low','Close','Volume']]
    return df[~df.index.duplicated(keep='first')].sort_index()


def calc_rsi(prices, period=14):
    d = prices.diff()
    ag = d.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    al = (-d.clip(upper=0)).ewm(com=period-1, min_periods=period).mean()
    return 100 - 100 / (1 + ag / al)


def simulate_trade(df_1h, signal_idx, direction, entry_price, sl_price):
    risk = abs(entry_price - sl_price)
    tp = entry_price + risk * RR_RATIO if direction == 'LONG' \
         else entry_price - risk * RR_RATIO
    entry_idx = signal_idx + 1
    for j in range(entry_idx + 1, min(entry_idx + MAX_BARS, len(df_1h))):
        fh, fl = float(df_1h.iloc[j]['High']), float(df_1h.iloc[j]['Low'])
        if direction == 'LONG':
            if fl <= sl_price: return 'SL', sl_price, df_1h.index[j], tp, j
            if fh >= tp:       return 'TP', tp, df_1h.index[j], tp, j
        else:
            if fh >= sl_price: return 'SL', sl_price, df_1h.index[j], tp, j
            if fl <= tp:       return 'TP', tp, df_1h.index[j], tp, j
    last_idx = min(entry_idx + MAX_BARS - 1, len(df_1h) - 1)
    return 'TIMEOUT', float(df_1h.iloc[last_idx]['Close']), \
           df_1h.index[last_idx], tp, last_idx


# ============================================================
#  RUN SINGLE COMBO (risk_pct × leverage)
# ============================================================

def run_combo(signals, df_1h, risk_pct, leverage):
    """Run backtest for one (risk%, leverage) combination"""
    capital = INITIAL_CAPITAL
    skip_until = 0
    peak = capital
    max_dd = 0
    trades_won = 0
    trades_lost = 0
    pnl_list = []
    rr_list = []

    for sig in signals:
        if sig['i'] <= skip_until:
            continue
        if capital <= 0:
            break

        entry_price = sig['entry_price']
        sl, risk = sig['sl'], sig['risk']
        direction = sig['direction']

        qty = (capital * risk_pct) / risk

        # Leverage margin cap
        margin_req = (entry_price * qty) / leverage
        if margin_req > capital:
            qty = (capital * leverage) / entry_price * 0.95

        outcome, exit_p, exit_time, tp_val, exit_idx = simulate_trade(
            df_1h, sig['i'], direction, entry_price, sl)

        pnl = (exit_p - entry_price) * qty if direction == 'LONG' \
              else (entry_price - exit_p) * qty
        cost_entry = (entry_price * qty) * (FEE_PER_SIDE / 100)
        cost_exit  = (exit_p * qty) * (FEE_PER_SIDE / 100)
        cost = cost_entry + cost_exit
        pnl_net = pnl - cost

        capital += pnl_net
        skip_until = exit_idx

        rr_actual = pnl_net / (risk * qty) if qty > 0 and risk > 0 else 0
        pnl_list.append(pnl_net)
        rr_list.append(rr_actual)

        if pnl_net > 0:
            trades_won += 1
        elif pnl_net < 0:
            trades_lost += 1

        peak = max(peak, capital)
        dd = (capital - peak) / peak * 100
        max_dd = min(max_dd, dd)

    total = trades_won + trades_lost
    wr = trades_won / total * 100 if total > 0 else 0
    ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Sharpe (annualized from trade returns)
    if len(pnl_list) > 1:
        pnl_arr = np.array(pnl_list)
        avg_pnl = pnl_arr.mean()
        std_pnl = pnl_arr.std()
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0
    else:
        sharpe = 0

    # Calmar
    calmar = (ret / 100) / abs(max_dd / 100) if max_dd != 0 else 0

    return {
        'risk_pct': risk_pct, 'leverage': leverage,
        'total': total, 'wins': trades_won, 'wr': wr,
        'ret': ret, 'final': capital, 'max_dd': max_dd,
        'sharpe': sharpe, 'calmar': calmar,
        'rr_list': rr_list, 'pnl_list': pnl_list,
        'blown': capital <= 0,
    }


# ============================================================
#  KELLY CRITERION
# ============================================================

def calc_kelly(win_rate, rr_ratio):
    """
    Kelly Criterion: f* = (W × R - L) / R
    W = win rate, L = loss rate, R = reward/risk ratio
    """
    w = win_rate / 100
    l = 1 - w
    r = rr_ratio
    kelly = (w * r - l) / r
    return max(kelly, 0)  # ไม่ติดลบ


# ============================================================
#  MAIN
# ============================================================

def run_backtest():
    print("=" * 70)
    print(f"  Risk Optimization — {SYMBOL} RSI Reversal 1:{RR_RATIO}")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"  Capital: ${INITIAL_CAPITAL:,}")
    print(f"  Grid: Risk {RISK_RANGE[0]*100:.0f}%-{RISK_RANGE[-1]*100:.0f}% × "
          f"Leverage {LEVERAGE_RANGE[0]}x-{LEVERAGE_RANGE[-1]}x")
    print("=" * 70)

    # ---- Download data (once) ----
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    w1d = (start_dt - timedelta(days=WARMUP_1D_DAYS)).strftime("%Y-%m-%d")
    w1h = (start_dt - timedelta(days=WARMUP_1H_DAYS)).strftime("%Y-%m-%d")

    print(f"\n📥 1D data...", end=" ", flush=True)
    df_1d = binance_klines(SYMBOL, '1d', w1d, END_DATE)
    if df_1d.empty: print("❌"); return
    print(f"✅ {len(df_1d)} bars")

    print(f"📥 1H data...", end=" ", flush=True)
    df_1h = binance_klines(SYMBOL, '1h', w1h, END_DATE)
    if df_1h.empty: print("❌"); return
    print(f"✅ {len(df_1h)} bars")

    # Indicators
    df_1d['EMA']   = df_1d['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_1d['trend'] = np.where(df_1d['Close'] > df_1d['EMA'], 'bullish', 'bearish')
    df_1h['RSI']   = calc_rsi(df_1h['Close'], RSI_PERIOD)
    df_1h['EMA']   = df_1h['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df_1h['trend'] = np.where(df_1h['Close'] > df_1h['EMA'], 'bullish', 'bearish')

    bt_start = max(pd.Timestamp(START_DATE), df_1h.index[0])
    d1_dates = df_1d.index.normalize().values.astype('datetime64[ns]')
    d1_trend = df_1d['trend'].values
    start_idx = df_1h.index.searchsorted(bt_start)

    # ---- Collect signals (once) ----
    signals = []
    for i in range(2, len(df_1h) - 1):
        if i < start_idx: continue
        row, prev, prev2 = df_1h.iloc[i], df_1h.iloc[i-1], df_1h.iloc[i-2]
        rsi_now, rsi_p1, rsi_p2 = row['RSI'], prev['RSI'], prev2['RSI']
        if pd.isna(rsi_now) or pd.isna(rsi_p1) or pd.isna(rsi_p2): continue

        cur_date = np.datetime64(df_1h.index[i].normalize(), 'ns')
        idx_1d = np.searchsorted(d1_dates, cur_date, side='right') - 1
        if idx_1d < 1: continue
        trend_1d, trend_1h = d1_trend[idx_1d - 1], row['trend']

        direction = None
        if (trend_1d == 'bullish' and trend_1h == 'bullish'
                and rsi_p1 < rsi_p2 and rsi_now > rsi_p1 and rsi_now <= RSI_BULL_MAX):
            direction = 'LONG'
        elif (trend_1d == 'bearish' and trend_1h == 'bearish'
                and rsi_p1 > rsi_p2 and rsi_now < rsi_p1 and rsi_now >= RSI_BEAR_MIN):
            direction = 'SHORT'
        if direction is None: continue

        entry_price = float(df_1h.iloc[i+1]['Open'])
        entry_time = df_1h.index[i+1]
        sl = min(float(row['Open']), float(row['Close'])) if direction == 'LONG' \
             else max(float(row['Open']), float(row['Close']))
        risk = abs(entry_price - sl)
        risk_pct_val = risk / entry_price * 100
        if risk_pct_val < MIN_RISK_PCT or risk <= 0: continue

        signals.append({'i': i, 'direction': direction,
                        'entry_price': entry_price, 'entry_time': entry_time,
                        'sl': sl, 'risk': risk})

    print(f"\n📊 {len(signals)} signals collected")
    if not signals:
        print("⚠️  No signals"); return

    # ---- Kelly Criterion (from a baseline run) ----
    # Run baseline to get win rate
    baseline = run_combo(signals, df_1h, 0.02, 6)
    wr_base = baseline['wr']
    kelly_full = calc_kelly(wr_base, RR_RATIO)
    kelly_half = kelly_full / 2
    kelly_quarter = kelly_full / 4

    print(f"\n{'='*60}")
    print(f"  📐 Kelly Criterion (WR: {wr_base:.1f}%, RR: 1:{RR_RATIO})")
    print(f"{'='*60}")
    print(f"  Full Kelly:    {kelly_full*100:.1f}% per trade")
    print(f"  Half Kelly:    {kelly_half*100:.1f}% per trade  ← recommended")
    print(f"  Quarter Kelly: {kelly_quarter*100:.1f}% per trade  ← conservative")
    print(f"\n  ⚠️  Kelly assumes no leverage cap — real optimal may differ")
    print(f"      Grid search below accounts for leverage cap")

    # ---- Grid Search ----
    print(f"\n🔄 Running grid search ({len(RISK_RANGE)} × {len(LEVERAGE_RANGE)} "
          f"= {len(RISK_RANGE)*len(LEVERAGE_RANGE)} combos)...")

    results = []
    total_combos = len(RISK_RANGE) * len(LEVERAGE_RANGE)
    done = 0

    for rpt in RISK_RANGE:
        for lev in LEVERAGE_RANGE:
            res = run_combo(signals, df_1h, rpt, lev)
            results.append(res)
            done += 1
            if done % 10 == 0:
                print(f"   {done}/{total_combos}...", end="\r", flush=True)

    print(f"   ✅ {total_combos} combos completed       ")

    # ---- Build matrices for heatmap ----
    n_risk = len(RISK_RANGE)
    n_lev  = len(LEVERAGE_RANGE)

    ret_matrix  = np.zeros((n_risk, n_lev))
    dd_matrix   = np.zeros((n_risk, n_lev))
    calmar_matrix = np.zeros((n_risk, n_lev))
    sharpe_matrix = np.zeros((n_risk, n_lev))
    blown_matrix  = np.zeros((n_risk, n_lev))

    for res in results:
        ri = RISK_RANGE.index(res['risk_pct'])
        li = LEVERAGE_RANGE.index(res['leverage'])
        ret_matrix[ri, li]    = res['ret']
        dd_matrix[ri, li]     = res['max_dd']
        calmar_matrix[ri, li] = res['calmar']
        sharpe_matrix[ri, li] = res['sharpe']
        blown_matrix[ri, li]  = 1 if res['blown'] else 0

    # ---- Print Table ----
    print(f"\n{'='*100}")
    print(f"  Risk × Leverage Grid — Return % (MaxDD%)")
    print(f"{'='*100}")

    hdr = f"  {'RPT':>6}"
    for lev in LEVERAGE_RANGE:
        hdr += f"  {'x'+str(lev):>12}"
    print(hdr)
    print(f"  {'-'*(6 + 14*n_lev)}")

    for ri, rpt in enumerate(RISK_RANGE):
        row_str = f"  {rpt*100:>5.1f}%"
        for li, lev in enumerate(LEVERAGE_RANGE):
            ret_v = ret_matrix[ri, li]
            dd_v  = dd_matrix[ri, li]
            if blown_matrix[ri, li]:
                row_str += f"  {'💀 BLOWN':>12}"
            else:
                row_str += f"  {ret_v:>+7.0f}({dd_v:.0f}%)"
        print(row_str)

    # ---- Find sweet spots ----
    print(f"\n{'='*80}")
    print(f"  🏆 Sweet Spot Analysis (DD tolerance: {MAX_DD_TOLERANCE}%)")
    print(f"{'='*80}")

    # Filter: not blown, DD within tolerance
    valid = [r for r in results if not r['blown'] and r['max_dd'] >= MAX_DD_TOLERANCE]

    if valid:
        # Best return within DD tolerance
        best_ret = max(valid, key=lambda x: x['ret'])
        # Best Calmar (return/DD ratio)
        best_calmar = max(valid, key=lambda x: x['calmar'])
        # Best Sharpe
        best_sharpe = max(valid, key=lambda x: x['sharpe'])
        # Closest to half-Kelly
        closest_kelly = min(valid, key=lambda x: abs(x['risk_pct'] - kelly_half))

        print(f"\n  {'Criteria':<25} {'Risk%':>7} {'Lev':>5} {'Return':>10} "
              f"{'MaxDD':>8} {'Calmar':>8} {'Sharpe':>8}")
        print(f"  {'-'*75}")

        for label, res in [
            ("📈 Best Return", best_ret),
            ("⚖️  Best Calmar", best_calmar),
            ("📊 Best Sharpe", best_sharpe),
            ("🎯 Nearest Half-Kelly", closest_kelly),
        ]:
            print(f"  {label:<25} {res['risk_pct']*100:>6.1f}% "
                  f"x{res['leverage']:>3} {res['ret']:>+9.0f}% "
                  f"{res['max_dd']:>7.1f}% {res['calmar']:>7.1f}x "
                  f"{res['sharpe']:>7.2f}")

        # Recommend
        # Score = Calmar × log(1 + ret/100) — balanced metric
        for r in valid:
            r['score'] = r['calmar'] * np.log1p(max(r['ret'], 0) / 100)

        recommended = max(valid, key=lambda x: x['score'])
        print(f"\n  ⭐ RECOMMENDED: Risk {recommended['risk_pct']*100:.1f}% × "
              f"Leverage x{recommended['leverage']}")
        print(f"     Return: {recommended['ret']:+,.0f}%  |  "
              f"MaxDD: {recommended['max_dd']:.1f}%  |  "
              f"Calmar: {recommended['calmar']:.1f}x  |  "
              f"Sharpe: {recommended['sharpe']:.2f}")
    else:
        print("  ⚠️  ไม่มี combo ที่ผ่าน DD tolerance — ลองเพิ่ม MAX_DD_TOLERANCE")
        recommended = max(results, key=lambda x: x['ret'] if not x['blown'] else -999)

    # ---- Kelly comparison table ----
    print(f"\n{'='*80}")
    print(f"  📐 Kelly vs Grid (at Leverage x6)")
    print(f"{'='*80}")

    lev6_results = [r for r in results if r['leverage'] == 6]
    if lev6_results:
        print(f"\n  {'RPT':>7} {'Return':>10} {'MaxDD':>8} {'Calmar':>8} {'Kelly?':>12}")
        print(f"  {'-'*50}")
        for r in sorted(lev6_results, key=lambda x: x['risk_pct']):
            kelly_mark = ""
            if abs(r['risk_pct'] - kelly_quarter) < 0.003:
                kelly_mark = "¼ Kelly"
            elif abs(r['risk_pct'] - kelly_half) < 0.003:
                kelly_mark = "½ Kelly"
            elif abs(r['risk_pct'] - kelly_full) < 0.003:
                kelly_mark = "Full Kelly"
            print(f"  {r['risk_pct']*100:>6.1f}% {r['ret']:>+9.0f}% "
                  f"{r['max_dd']:>7.1f}% {r['calmar']:>7.1f}x "
                  f"{kelly_mark:>12}")

    # ---- Save CSV ----
    csv_data = []
    for r in results:
        csv_data.append({
            'risk_pct': r['risk_pct']*100, 'leverage': r['leverage'],
            'trades': r['total'], 'win_rate': round(r['wr'], 1),
            'return_pct': round(r['ret'], 1), 'max_dd': round(r['max_dd'], 1),
            'calmar': round(r['calmar'], 2), 'sharpe': round(r['sharpe'], 2),
            'blown': r['blown'],
        })
    csv_path = os.path.join(DESKTOP, "risk_grid_results.csv")
    try:
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        print(f"\n  💾 CSV: {csv_path}")
    except: pass

    # ========== CHARTS ==========
    if not HAS_PLOT:
        print("\n✅ Done!")
        return

    try:
        fig = plt.figure(figsize=(20, 18))
        fig.patch.set_facecolor('#0f0f0f')
        gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

        risk_labels = [f'{r*100:.0f}%' if r*100 == int(r*100) else f'{r*100:.1f}%'
                       for r in RISK_RANGE]
        lev_labels  = [f'x{l}' for l in LEVERAGE_RANGE]

        def style_ax(ax, title=""):
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#cccccc')
            for sp in ax.spines.values():
                sp.set_edgecolor('#333355')
            if title:
                ax.set_title(title, fontsize=12, color='white', fontweight='bold')

        fig.suptitle(
            f'Risk Optimization — {SYMBOL} RSI Reversal 1:{RR_RATIO}\n'
            f'{START_DATE} → {END_DATE}  |  Kelly: {kelly_full*100:.1f}% '
            f'(½K: {kelly_half*100:.1f}%)  |  {len(signals)} signals',
            fontsize=14, fontweight='bold', color='white')

        # ---- Heatmap 1: Return ----
        ax1 = fig.add_subplot(gs[0, 0])
        style_ax(ax1, 'Return %')
        ret_display = ret_matrix.copy()
        ret_display[blown_matrix == 1] = np.nan
        im1 = ax1.imshow(ret_display, aspect='auto', cmap='RdYlGn',
                         interpolation='nearest')
        for i in range(n_risk):
            for j in range(n_lev):
                if blown_matrix[i, j]:
                    ax1.text(j, i, '💀', ha='center', va='center', fontsize=10)
                else:
                    v = ret_matrix[i, j]
                    color = 'white' if abs(v) > ret_matrix[~np.isnan(ret_display)].max()*0.5 else 'black'
                    ax1.text(j, i, f'{v:+.0f}%', ha='center', va='center',
                             fontsize=7, color=color, fontweight='bold')
        ax1.set_xticks(range(n_lev)); ax1.set_xticklabels(lev_labels, color='white', fontsize=8)
        ax1.set_yticks(range(n_risk)); ax1.set_yticklabels(risk_labels, color='white', fontsize=8)
        ax1.set_xlabel('Leverage', color='white'); ax1.set_ylabel('Risk Per Trade', color='white')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # ---- Heatmap 2: Max Drawdown ----
        ax2 = fig.add_subplot(gs[0, 1])
        style_ax(ax2, 'Max Drawdown %')
        dd_display = dd_matrix.copy()
        dd_display[blown_matrix == 1] = np.nan
        cmap_dd = LinearSegmentedColormap.from_list('dd', ['#00FF00', '#FFFF00', '#FF0000'])
        im2 = ax2.imshow(-dd_display, aspect='auto', cmap=cmap_dd, interpolation='nearest')
        for i in range(n_risk):
            for j in range(n_lev):
                if blown_matrix[i, j]:
                    ax2.text(j, i, '💀', ha='center', va='center', fontsize=10)
                else:
                    v = dd_matrix[i, j]
                    ax2.text(j, i, f'{v:.0f}%', ha='center', va='center',
                             fontsize=7, color='white', fontweight='bold')
        ax2.set_xticks(range(n_lev)); ax2.set_xticklabels(lev_labels, color='white', fontsize=8)
        ax2.set_yticks(range(n_risk)); ax2.set_yticklabels(risk_labels, color='white', fontsize=8)
        ax2.set_xlabel('Leverage', color='white'); ax2.set_ylabel('Risk Per Trade', color='white')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # ---- Heatmap 3: Calmar Ratio (Return/DD) ----
        ax3 = fig.add_subplot(gs[1, 0])
        style_ax(ax3, 'Calmar Ratio (Return / |MaxDD|)')
        calmar_display = calmar_matrix.copy()
        calmar_display[blown_matrix == 1] = np.nan
        im3 = ax3.imshow(calmar_display, aspect='auto', cmap='viridis',
                         interpolation='nearest')
        for i in range(n_risk):
            for j in range(n_lev):
                if blown_matrix[i, j]:
                    ax3.text(j, i, '💀', ha='center', va='center', fontsize=10)
                else:
                    v = calmar_matrix[i, j]
                    ax3.text(j, i, f'{v:.0f}', ha='center', va='center',
                             fontsize=7, color='white', fontweight='bold')
        ax3.set_xticks(range(n_lev)); ax3.set_xticklabels(lev_labels, color='white', fontsize=8)
        ax3.set_yticks(range(n_risk)); ax3.set_yticklabels(risk_labels, color='white', fontsize=8)
        ax3.set_xlabel('Leverage', color='white'); ax3.set_ylabel('Risk Per Trade', color='white')
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # ---- Plot 4: Risk curve at fixed leverage ----
        ax4 = fig.add_subplot(gs[1, 1])
        style_ax(ax4, 'Return vs Risk% (selected leverages)')
        for li, lev in enumerate([4, 6, 10, 15]):
            if lev not in LEVERAGE_RANGE: continue
            lev_idx = LEVERAGE_RANGE.index(lev)
            rets = [ret_matrix[ri, lev_idx] for ri in range(n_risk)]
            dds  = [abs(dd_matrix[ri, lev_idx]) for ri in range(n_risk)]
            risk_x = [r*100 for r in RISK_RANGE]
            colors = ['#F7931A', '#627EEA', '#00FFA3', '#FF6BFF']
            ax4.plot(risk_x, rets, 'o-', color=colors[li], lw=2,
                     label=f'x{lev} Return', markersize=5)
            ax4.plot(risk_x, [-d for d in dds], 's--', color=colors[li],
                     lw=1, alpha=0.5, label=f'x{lev} MaxDD', markersize=3)

        # Kelly lines
        ax4.axvline(kelly_full*100, color='#FF4444', ls=':', lw=2, alpha=0.8,
                    label=f'Full Kelly ({kelly_full*100:.1f}%)')
        ax4.axvline(kelly_half*100, color='#00FF88', ls=':', lw=2, alpha=0.8,
                    label=f'Half Kelly ({kelly_half*100:.1f}%)')
        ax4.axhline(0, color='white', lw=0.5)
        ax4.set_xlabel('Risk Per Trade (%)', color='white', fontsize=11)
        ax4.set_ylabel('Return / DD (%)', color='white', fontsize=11)
        ax4.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                   loc='upper left', ncol=2)
        ax4.grid(True, alpha=0.15, color='white')

        # ---- Plot 5: Equity curves for key combos ----
        ax5 = fig.add_subplot(gs[2, :])
        style_ax(ax5, 'Equity Curves — Key Combinations')

        key_combos = [
            (0.02, 6,  'Current (2%, x6)',    '#627EEA'),
            (0.03, 6,  '3%, x6',              '#F7931A'),
            (0.05, 6,  '5%, x6',              '#00FFA3'),
            (0.05, 10, '5%, x10',             '#FF6BFF'),
            (0.05, 15, '5%, x15',             '#FFD700'),
            (recommended['risk_pct'], recommended['leverage'],
             f"⭐ Recommended ({recommended['risk_pct']*100:.0f}%, x{recommended['leverage']})",
             '#FF4444'),
        ]

        # Remove duplicates
        seen = set()
        unique_combos = []
        for rpt, lev, label, color in key_combos:
            key = (rpt, lev)
            if key not in seen:
                seen.add(key)
                unique_combos.append((rpt, lev, label, color))

        for rpt, lev, label, color in unique_combos:
            # Re-run to get equity curve
            capital = INITIAL_CAPITAL
            skip_until = 0
            eq_times = [pd.Timestamp(START_DATE)]
            eq_vals  = [INITIAL_CAPITAL]

            for sig in signals:
                if sig['i'] <= skip_until: continue
                if capital <= 0: break
                entry_price, sl, risk = sig['entry_price'], sig['sl'], sig['risk']
                direction = sig['direction']
                qty = (capital * rpt) / risk
                margin_req = (entry_price * qty) / lev
                if margin_req > capital:
                    qty = (capital * lev) / entry_price * 0.95
                outcome, exit_p, exit_time, tp_val, exit_idx = simulate_trade(
                    df_1h, sig['i'], direction, entry_price, sl)
                pnl = (exit_p - entry_price) * qty if direction == 'LONG' \
                      else (entry_price - exit_p) * qty
                cost_entry = (entry_price * qty) * (FEE_PER_SIDE / 100)
                cost_exit  = (exit_p * qty) * (FEE_PER_SIDE / 100)
                cost = cost_entry + cost_exit
                capital += pnl - cost
                skip_until = exit_idx
                eq_times.append(exit_time)
                eq_vals.append(max(capital, 0))

            ret_v = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            ax5.plot(eq_times, eq_vals, color=color, lw=2,
                     label=f'{label}  {ret_v:+,.0f}%', zorder=3)

        ax5.axhline(INITIAL_CAPITAL, color='gray', ls=':', alpha=0.4)
        ax5.set_ylabel('Portfolio Value (USD)', color='white', fontsize=11)
        ax5.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white', loc='upper left')
        ax5.grid(True, alpha=0.15, color='white')
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

        # Mark recommended
        if recommended:
            ri = RISK_RANGE.index(recommended['risk_pct'])
            li = LEVERAGE_RANGE.index(recommended['leverage'])
            for ax_h in [ax1, ax2, ax3]:
                rect = plt.Rectangle((li-0.5, ri-0.5), 1, 1, linewidth=3,
                                     edgecolor='#FF4444', facecolor='none', zorder=10)
                ax_h.add_patch(rect)

        chart_path = os.path.join(DESKTOP, "risk_heatmap.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        print(f"\n  📊 กราฟ: {chart_path}")
        plt.show()

    except Exception as e:
        print(f"  ⚠️  Chart error: {e}")
        import traceback; traceback.print_exc()

    # ========== MONTE CARLO for recommended ==========
    if MC_RUNS > 0 and recommended:
        print(f"\n{'='*65}")
        print(f"  🎲 Monte Carlo — Recommended: "
              f"{recommended['risk_pct']*100:.1f}% × x{recommended['leverage']}")
        print(f"{'='*65}")

        rr_arr = np.array(recommended['rr_list'])
        n_trades = len(rr_arr)
        risk_frac = recommended['risk_pct']

        np.random.seed(42)
        final_returns = np.zeros(MC_RUNS)
        max_drawdowns = np.zeros(MC_RUNS)

        for run in range(MC_RUNS):
            shuffled = np.random.permutation(rr_arr)
            equity = np.ones(n_trades + 1)
            for t in range(n_trades):
                pnl_pct = risk_frac * shuffled[t]
                equity[t+1] = equity[t] * (1 + pnl_pct)
                if equity[t+1] <= 0:
                    equity[t+1:] = 0
                    break
            final_returns[run] = (equity[-1] - 1) * 100
            peak = np.maximum.accumulate(equity)
            mask = peak > 0
            dd = np.zeros_like(equity)
            dd[mask] = (equity[mask] - peak[mask]) / peak[mask] * 100
            max_drawdowns[run] = dd.min()

        pcts = [5, 25, 50, 75, 95]
        ret_p = np.percentile(final_returns, pcts)
        dd_p  = np.percentile(max_drawdowns, pcts)
        prob_profit = np.mean(final_returns > 0) * 100
        prob_ruin   = np.mean(final_returns <= -90) * 100
        fin_p = INITIAL_CAPITAL * (1 + ret_p / 100)

        print(f"\n  {'Percentile':<18} {'Return':>10} {'Final $':>12} {'Max DD':>10}")
        print(f"  {'-'*55}")
        for i, p in enumerate(pcts):
            print(f"  {p:>3}th percentile   {ret_p[i]:>+9.1f}%  ${fin_p[i]:>10,.0f}  {dd_p[i]:>9.1f}%")
        print(f"  {'-'*55}")
        print(f"  {'โอกาสกำไร':<30} {prob_profit:>6.1f}%")
        print(f"  {'โอกาสล้างพอร์ต (DD>90%)':<30} {prob_ruin:>6.1f}%")
        print(f"  {'Avg Return':<30} {np.mean(final_returns):>+6.1f}%")
        print(f"  {'Avg Max DD':<30} {np.mean(max_drawdowns):>6.1f}%")

    print(f"\n✅ Risk Optimization เสร็จสิ้น!")


if __name__ == "__main__":
    run_backtest()
