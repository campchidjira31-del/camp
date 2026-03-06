# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Scripts

All scripts are standalone — run directly with Python 3:

```bash
python3 rsi_mean_revert.py
python3 btc_backtest.py
python3 ema_145_ATR.py
python3 ema145_atr_coinm.py
python3 risk_per_trade_backtest.py
```

Dependencies: `pip3 install requests pandas numpy matplotlib`

Charts require a display (TkAgg backend). If running headless, charts are skipped automatically — results still print to terminal and CSVs are saved.

## Architecture

Each script is self-contained with no shared modules. They all follow the same pattern:

1. **SETTINGS block** at the top — all user-tunable parameters live here
2. **Data download** — Binance public REST API (`/api/v3/klines` for spot, `/fapi/v1/klines` for futures), paginated with 1000-bar chunks, no API key required
3. **Indicator calculation** — pandas/numpy only, no TA libraries
4. **Signal detection loop** — iterates bars, checks conditions, records entry/exit dicts
5. **Leverage/RR grid** — reruns the trade list with different parameters
6. **Output** — prints table to terminal, saves CSV to `~/Desktop/`, saves chart PNG to `~/Desktop/`

## Scripts Overview

| Script | Strategy | Key output |
|---|---|---|
| `rsi_mean_revert.py` | Multi-TF: SuperTrend + RSI(4H) + ADX + Bollinger Bands | Main active strategy |
| `btc_backtest.py` | RSI V-shape reversal + 1D EMA trend + RSI-MA zone filter | RR grid (1:4 to 1:15) |
| `ema_145_ATR.py` | EMA crossover A/B test: Original vs ADX+ATR filtered | A/B comparison table |
| `ema145_atr_coinm.py` | EMA crossover for COIN-M futures | Variant of ema_145_ATR |
| `risk_per_trade_backtest.py` | Kelly Criterion + Risk% × Leverage heatmap | Grid search optimization |

## Key Design Decisions

**Look-ahead bias prevention (`rsi_mean_revert.py`):** Current-TF indicators (BB, ADX, ATR) are shifted by 1 bar before use in signal detection. Columns `BB_upper_s`, `BB_lower_s`, `ADX_s`, `ATR_s` are the lag-corrected versions. HTF indicators are forward-filled from completed 4H bars and do not need additional shifting.

**Multi-TF alignment:** HTF (4H) indicators are computed on the 4H dataframe then merged into the LTF (1H) dataframe using `reindex(..., method='ffill')`. This correctly carries the last completed 4H bar's value forward.

**Position sizing in `rsi_mean_revert.py`:** Risk-based — `qty = (capital × RISK_PCT × leverage) / (ATR_SL_MULT × ATR)`. This means higher leverage multiplies risk proportionally.

**Position sizing in `btc_backtest.py` / `risk_per_trade_backtest.py`:** Also risk-based — `qty = (capital × risk_pct) / SL_distance`, with a leverage margin cap: if `(entry × qty) / leverage > capital`, qty is capped at `capital × leverage / entry_price × 0.95`.

**Funding cost direction:** In `rsi_mean_revert.py`, LONG pays funding, SHORT receives it (negative cost). In `ema_145_ATR.py`, funding is always a cost regardless of direction — note this discrepancy if comparing scripts.

**Funding rate:** Binance USDT-M perpetuals for BTC/ETH fund every **8 hours** (`FUNDING_INTERVAL_HOURS = 8`). `ema_145_ATR.py` still uses 4h — correct this if editing that file.

**Monte Carlo:** Shuffles trade `pnl_pct` values across 5000 runs. With fixed-fraction sizing, terminal wealth is path-independent (std dev ≈ 0%), so Monte Carlo is primarily useful for drawdown distribution, not return distribution.

## CSV Outputs (saved to ~/Desktop/)

- `rsi_mean_revert_trades.csv` — trades from best Calmar leverage
- `trades_result_v9.csv` — trades from best RR in btc_backtest.py
- `ema_195_eth_trades.csv` / `ema_195_ab_trades.csv` — EMA A/B trades
- `risk_grid_results.csv` — full grid search results
