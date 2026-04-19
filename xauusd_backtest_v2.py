"""
XAUUSD Heat Map Trading Bot — Backtester v2
Built with backtesting.py (https://kernc.github.io/backtesting.py/)

Period : Jan 1, 2024 → Apr 18, 2026
Pair   : XAUUSD (H1)
Signal : 3/5 indicator confluence + EMA200 trend filter + ADX>15
R:R    : 1:2  (SL=15 pips, TP=30 pips)
"""

import os
import warnings
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

warnings.filterwarnings("ignore")

IS_MAC = platform.system() == "Darwin"
if not IS_MAC:
    import MetaTrader5 as mt5

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════
SYMBOL = "XAUUSD"
START_DATE = datetime(2024,  1,  1, tzinfo=timezone.utc)
END_DATE = datetime(2026,  4, 18, tzinfo=timezone.utc)

INITIAL_CASH = 10_000
LOT_SIZE = 0.1          # contract size used for pip-value maths
SL_PIPS = 15
TP_PIPS = 30           # 1:2 R:R
PIP_SIZE = 0.10         # Gold: 1 pip = $0.10

CACHE_FOLDER = "cache_data"
os.makedirs(CACHE_FOLDER, exist_ok=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUMS = list(range(1, 13))


# ═══════════════════════════════════════════════════════
#  CACHE HELPERS
# ═══════════════════════════════════════════════════════
def _cache_path(symbol: str) -> str:
    s = START_DATE.strftime("%Y%m%d")
    e = END_DATE.strftime("%Y%m%d")
    return os.path.join(CACHE_FOLDER, f"cache_{symbol}_{s}_{e}.csv")


def load_from_cache(symbol: str) -> pd.DataFrame | None:
    p = _cache_path(symbol)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, index_col="time", parse_dates=True)
            print(f"  [CACHE] Loaded {symbol} ({len(df):,} candles)")
            return df
        except Exception as exc:
            print(f"  [WARN] Cache corrupted ({exc}), will re-fetch.")
    return None


def save_to_cache(df: pd.DataFrame, symbol: str) -> None:
    try:
        df.to_csv(_cache_path(symbol))
        print(f"  [CACHE] Saved {symbol} ({len(df):,} candles)")
    except Exception as exc:
        print(f"  [WARN] Could not save cache: {exc}")


# ═══════════════════════════════════════════════════════
#  DATA FETCH
# ═══════════════════════════════════════════════════════
def fetch_data(symbol: str) -> pd.DataFrame | None:
    cached = load_from_cache(symbol)
    if cached is not None:
        return cached

    if IS_MAC:
        print(f"  [ERROR] No cache for {symbol} and MT5 not available on macOS.")
        print(f"          Expected: {_cache_path(symbol)}")
        return None

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, START_DATE, END_DATE)
    if rates is None or len(rates) == 0:
        print(f"  [WARN] MT5 returned no data for {symbol}.")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    save_to_cache(df, symbol)
    return df


def prepare_for_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    backtesting.py expects a DataFrame with columns:
    Open, High, Low, Close, Volume  (title-case)
    and a DatetimeIndex.
    """
    out = df[["open", "high", "low", "close", "tick_volume"]].copy()
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="first")]
    out.sort_index(inplace=True)
    return out


# ═══════════════════════════════════════════════════════
#  INDICATOR HELPERS  (plain numpy/pandas — no ta-lib dep)
# ═══════════════════════════════════════════════════════
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, sig=9):
    e_fast = ema(series, fast)
    e_slow = ema(series, slow)
    line = e_fast - e_slow
    signal = ema(line, sig)
    hist = line - signal
    return line, signal, hist


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period=14, d_period=3):
    lo = low.rolling(k_period).min()
    hi = high.rolling(k_period).max()
    k = 100 * (close - lo) / (hi - lo + 1e-10)
    d = k.rolling(d_period).mean()
    return k, d


def bollinger(series: pd.Series, period=20, std_mult=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    pct = (series - lower) / (upper - lower + 1e-10)
    return mid, upper, lower, pct


def atr(high, low, close, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr14 = atr(high, low, close, period)
    plus_di = 100 * plus_dm.rolling(period).mean() / (tr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).mean() / (tr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean(), plus_di, minus_di


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all signal indicators to an OHLCV dataframe (raw lowercase columns)."""
    d = df.copy()
    d["ema9"] = ema(d["close"], 9)
    d["ema21"] = ema(d["close"], 21)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)

    d["rsi"] = rsi(d["close"])

    d["macd"], d["macd_sig"], d["macd_hist"] = macd(d["close"])

    d["stoch_k"], d["stoch_d"] = stochastic(d["high"], d["low"], d["close"])

    d["bb_mid"], d["bb_upper"], d["bb_lower"], d["bb_pct"] = bollinger(d["close"])

    d["atr"] = atr(d["high"], d["low"], d["close"])
    d["adx"], d["plus_di"], d["minus_di"] = adx(d["high"], d["low"], d["close"])

    vol_mean = d["tick_volume"].rolling(20).mean()
    d["vol_ratio"] = d["tick_volume"] / (vol_mean + 1e-10)

    d["momentum"] = d["close"].pct_change(10) * 100
    return d


# ═══════════════════════════════════════════════════════
#  STRATEGY  (backtesting.py Strategy subclass)
# ═══════════════════════════════════════════════════════
class HeatMapStrategy(Strategy):
    """
    5-condition confluence system:
      1. EMA trend alignment  (EMA9 > EMA21 > EMA50)
      2. RSI zone             (30-55 buy / 45-70 sell)
      3. MACD histogram       (bullish / bearish)
      4. Stochastic           (K>D & <80 buy / K<D & >20 sell)
      5. Bollinger %B         (<0.35 buy / >0.65 sell)
    Filters:
      - Volume ratio > 0.6
      - ADX > 15
      - EMA200 major trend direction
    Needs 3/5 conditions + all filters for a signal.
    SL = 15 pips, TP = 30 pips (1:2 R:R)
    """

    sl_pips = SL_PIPS
    tp_pips = TP_PIPS
    pip_sz = PIP_SIZE

    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High,  index=self.data.index)
        low = pd.Series(self.data.Low,   index=self.data.index)
        vol = pd.Series(self.data.Volume, index=self.data.index)

        # EMAs
        self.ema9 = self.I(lambda s: ema(pd.Series(s), 9).values,   close, name="EMA9")
        self.ema21 = self.I(lambda s: ema(pd.Series(s), 21).values,  close, name="EMA21")
        self.ema50 = self.I(lambda s: ema(pd.Series(s), 50).values,  close, name="EMA50")
        self.ema200 = self.I(lambda s: ema(pd.Series(s), 200).values, close, name="EMA200")

        # RSI
        self.rsi_ind = self.I(lambda s: rsi(pd.Series(s)).values, close, name="RSI")

        # MACD hist
        def _macd_hist(s):
            _, _, h = macd(pd.Series(s))
            return h.values
        self.macd_hist = self.I(_macd_hist, close, name="MACD_hist")

        def _macd_sig_diff(s):
            line, sig, _ = macd(pd.Series(s))
            return (line - sig).values
        self.macd_diff = self.I(_macd_sig_diff, close, name="MACD_diff")

        # Stochastic
        def _stoch_k(h, l, c):
            k, _ = stochastic(pd.Series(h), pd.Series(l), pd.Series(c))
            return k.values

        def _stoch_d(h, l, c):
            _, d = stochastic(pd.Series(h), pd.Series(l), pd.Series(c))
            return d.values
        self.stoch_k = self.I(_stoch_k, high, low, close, name="Stoch_K")
        self.stoch_d = self.I(_stoch_d, high, low, close, name="Stoch_D")

        # BB %B
        def _bb_pct(s):
            _, _, _, pct = bollinger(pd.Series(s))
            return pct.values
        self.bb_pct = self.I(_bb_pct, close, name="BB_pct")

        # ADX
        def _adx(h, l, c):
            a, _, _ = adx(pd.Series(h), pd.Series(l), pd.Series(c))
            return a.values
        self.adx_ind = self.I(_adx, high, low, close, name="ADX")

        # Volume ratio
        def _vol_ratio(v):
            s = pd.Series(v)
            return (s / (s.rolling(20).mean() + 1e-10)).values
        self.vol_ratio = self.I(_vol_ratio, vol, name="VolRatio")

    def next(self):
        # Require minimum bars for indicators to warm up
        if len(self.data) < 210:
            return

        # Only one trade at a time
        if self.position:
            return

        price = self.data.Close[-1]

        # ── Score signals ──────────────────────────
        buy_score = sell_score = 0

        # 1. EMA alignment
        if self.ema9[-1] > self.ema21[-1] > self.ema50[-1]:
            buy_score += 1
        elif self.ema9[-1] < self.ema21[-1] < self.ema50[-1]:
            sell_score += 1

        # 2. RSI zone
        r = self.rsi_ind[-1]
        if 30 < r < 55:
            buy_score += 1
        elif 45 < r < 70:
            sell_score += 1

        # 3. MACD
        if self.macd_hist[-1] > 0 and self.macd_diff[-1] > 0:
            buy_score += 1
        elif self.macd_hist[-1] < 0 and self.macd_diff[-1] < 0:
            sell_score += 1

        # 4. Stochastic
        sk, sd = self.stoch_k[-1], self.stoch_d[-1]
        if sk > sd and sk < 80:
            buy_score += 1
        elif sk < sd and sk > 20:
            sell_score += 1

        # 5. Bollinger %B
        bp = self.bb_pct[-1]
        if bp < 0.35:
            buy_score += 1
        elif bp > 0.65:
            sell_score += 1

        # ── Filters ────────────────────────────────
        if self.vol_ratio[-1] < 0.6:
            return
        if self.adx_ind[-1] < 15:
            return

        sl_dist = self.sl_pips * self.pip_sz
        tp_dist = self.tp_pips * self.pip_sz

        # ── Entry ──────────────────────────────────
        if buy_score >= 3 and price > self.ema200[-1]:
            sl = price - sl_dist
            tp = price + tp_dist
            self.buy(sl=sl, tp=tp)

        elif sell_score >= 3 and price < self.ema200[-1]:
            sl = price + sl_dist
            tp = price - tp_dist
            self.sell(sl=sl, tp=tp)


# ═══════════════════════════════════════════════════════
#  HEAT MAP PLOTS
# ═══════════════════════════════════════════════════════
def build_monthly_trade_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Extract month from the ReturnPct, EntryTime etc. columns."""
    df = trades_df.copy()
    df["month"] = pd.to_datetime(df["EntryTime"]).dt.month
    return df


def plot_heatmaps(raw_df: pd.DataFrame, stats, trades_df: pd.DataFrame):
    """
    12-panel heat-map dashboard using monthly trade statistics.
    raw_df   : original OHLCV + indicators dataframe (lowercase columns)
    stats    : backtesting.py stats object
    trades_df: stats._trades DataFrame
    """
    sym = SYMBOL
    tt = len(trades_df)
    wins = int((trades_df["ReturnPct"] > 0).sum())
    losses = tt - wins
    wr = round(wins / tt * 100, 1) if tt > 0 else 0
    net = round(stats["Equity Final [$]"] - INITIAL_CASH, 2)

    raw_ind = add_all_indicators(raw_df)

    fig = plt.figure(figsize=(26, 32))
    fig.suptitle(
        f"XAUUSD Heat Map Bot — Backtest 2024-2026\n"
        f"H1 | XAUUSD | SL:{SL_PIPS}p TP:{TP_PIPS}p (1:2) | "
        f"Trades:{tt}  WR:{wr}%  Net P&L:${net:+,.2f}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    tdf_m = build_monthly_trade_df(trades_df)

    def monthly_series(fn, default=0.0):
        mat = pd.DataFrame(index=[sym], columns=MONTHS, dtype=float)
        for mn, ml in zip(MONTH_NUMS, MONTHS):
            sub = tdf_m[tdf_m["month"] == mn]
            mat.loc[sym, ml] = fn(sub)
        return mat.fillna(default).astype(float)

    # ── 1. Price Momentum ────────────────────
    ax1 = fig.add_subplot(4, 3, 1)
    raw_ind["month"] = raw_ind.index.month
    mom_mat = pd.DataFrame(index=[sym], columns=MONTHS, dtype=float)
    for mn, ml in zip(MONTH_NUMS, MONTHS):
        sub = raw_ind[raw_ind["month"] == mn]["momentum"]
        mom_mat.loc[sym, ml] = round(sub.mean(), 2) if len(sub) else 0
    sns.heatmap(mom_mat.astype(float), ax=ax1, cmap="RdYlGn", center=0,
                annot=True, fmt=".1f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "Mom %", "shrink": 0.8})
    ax1.set_title("Price Momentum (%)", fontweight="bold", fontsize=10)
    ax1.tick_params(labelsize=8)

    # ── 2. Volume ────────────────────────────
    ax2 = fig.add_subplot(4, 3, 2)
    vol_mat = pd.DataFrame(index=[sym], columns=MONTHS, dtype=float)
    for mn, ml in zip(MONTH_NUMS, MONTHS):
        sub = raw_ind[raw_ind["month"] == mn]["tick_volume"]
        vol_mat.loc[sym, ml] = round(sub.mean(), 0) if len(sub) else 0
    sns.heatmap(vol_mat.astype(float), ax=ax2, cmap="Blues",
                annot=True, fmt=".0f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "Avg Volume", "shrink": 0.8})
    ax2.set_title("Volume Activity", fontweight="bold", fontsize=10)
    ax2.tick_params(labelsize=8)

    # ── 3. Win Rate ──────────────────────────
    ax3 = fig.add_subplot(4, 3, 3)

    def wrf(sub):
        return round((sub["ReturnPct"] > 0).mean() * 100, 1) if len(sub) else 0
    sns.heatmap(monthly_series(wrf), ax=ax3, cmap="RdYlGn", vmin=0, vmax=100,
                annot=True, fmt=".0f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "WR %", "shrink": 0.8})
    ax3.set_title("Win Rate per Month (%)", fontweight="bold", fontsize=10)
    ax3.tick_params(labelsize=8)

    # ── 4. P&L ───────────────────────────────
    ax4 = fig.add_subplot(4, 3, 4)

    def pnlf(sub):
        if len(sub) == 0:
            return 0
        # PnL approximation from ReturnPct * entry price  * pip_value scaling
        return round(sub["PnL"].sum(), 1) if "PnL" in sub.columns else round(
            (sub["ReturnPct"] * INITIAL_CASH / 100).sum(), 1)
    sns.heatmap(monthly_series(pnlf), ax=ax4, cmap="RdYlGn", center=0,
                annot=True, fmt=".0f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "P&L $", "shrink": 0.8})
    ax4.set_title("P&L per Month (USD)", fontweight="bold", fontsize=10)
    ax4.tick_params(labelsize=8)

    # ── 5. Trade Count ───────────────────────
    ax5 = fig.add_subplot(4, 3, 5)
    sns.heatmap(monthly_series(len), ax=ax5, cmap="YlOrBr",
                annot=True, fmt=".0f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "Trades", "shrink": 0.8})
    ax5.set_title("Trade Count per Month", fontweight="bold", fontsize=10)
    ax5.tick_params(labelsize=8)

    # ── 6. Max Drawdown (monthly) ────────────
    ax6 = fig.add_subplot(4, 3, 6)

    def ddf(sub):
        if len(sub) < 2 or "PnL" not in sub.columns:
            return 0
        cum = sub["PnL"].cumsum()
        pk = cum.cummax()
        return round(((pk - cum) / (pk.abs() + 1e-10) * 100).max(), 1)
    sns.heatmap(monthly_series(ddf), ax=ax6, cmap="YlOrRd",
                annot=True, fmt=".1f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "Max DD %", "shrink": 0.8})
    ax6.set_title("Max Drawdown per Month (%)", fontweight="bold", fontsize=10)
    ax6.tick_params(labelsize=8)

    # ── 7. Profit Factor ─────────────────────
    ax7 = fig.add_subplot(4, 3, 7)

    def pff(sub):
        if len(sub) == 0:
            return 0
        col = "PnL" if "PnL" in sub.columns else None
        if col is None:
            return 0
        gp = sub[sub[col] > 0][col].sum()
        gl = abs(sub[sub[col] < 0][col].sum())
        return round(gp / gl, 2) if gl > 0 else round(gp, 2)
    sns.heatmap(monthly_series(pff), ax=ax7, cmap="RdYlGn", center=1,
                annot=True, fmt=".2f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "PF", "shrink": 0.8})
    ax7.set_title("Profit Factor per Month", fontweight="bold", fontsize=10)
    ax7.tick_params(labelsize=8)

    # ── 8. ADX Trend Strength ────────────────
    ax8 = fig.add_subplot(4, 3, 8)
    adx_mat = pd.DataFrame(index=[sym], columns=MONTHS, dtype=float)
    for mn, ml in zip(MONTH_NUMS, MONTHS):
        sub = raw_ind[raw_ind["month"] == mn]["adx"]
        adx_mat.loc[sym, ml] = round(sub.mean(), 1) if len(sub) else 0
    sns.heatmap(adx_mat.astype(float), ax=ax8, cmap="Blues", vmin=0, vmax=50,
                annot=True, fmt=".0f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "ADX", "shrink": 0.8})
    ax8.set_title("Trend Strength (ADX)", fontweight="bold", fontsize=10)
    ax8.tick_params(labelsize=8)

    # ── 9. Correlation (single asset placeholder) ──
    ax9 = fig.add_subplot(4, 3, 9)
    # Monthly return correlation  year-over-year
    raw_ind["year"] = raw_ind.index.year
    years = sorted(raw_ind["year"].unique())
    corr_data = {}
    for yr in years:
        yr_sub = raw_ind[raw_ind["year"] == yr]
        yr_sub2 = yr_sub.set_index("month")["close"].resample if hasattr(yr_sub, "resample") else None
        monthly_ret = []
        for mn in MONTH_NUMS:
            ms = yr_sub[yr_sub["month"] == mn]["close"]
            if len(ms) > 1:
                monthly_ret.append(ms.pct_change().mean() * 100)
            else:
                monthly_ret.append(np.nan)
        corr_data[str(yr)] = monthly_ret
    corr_df = pd.DataFrame(corr_data, index=MONTHS)
    corr_matrix = corr_df.corr()
    sns.heatmap(corr_matrix, ax=ax9, cmap="RdYlGn", center=0,
                annot=True, fmt=".2f", linewidths=0.4, annot_kws={"size": 8},
                cbar_kws={"label": "Corr", "shrink": 0.8})
    ax9.set_title("Year-over-Year Monthly Return Corr", fontweight="bold", fontsize=10)
    ax9.tick_params(labelsize=8)

    # ── 10. RSI Heat Map ─────────────────────
    ax10 = fig.add_subplot(4, 3, 10)
    rsi_mat = pd.DataFrame(index=[sym], columns=MONTHS, dtype=float)
    for mn, ml in zip(MONTH_NUMS, MONTHS):
        sub = raw_ind[raw_ind["month"] == mn]["rsi"]
        rsi_mat.loc[sym, ml] = round(sub.mean(), 1) if len(sub) else 50
    sns.heatmap(rsi_mat.astype(float), ax=ax10, cmap="RdYlGn", center=50,
                vmin=30, vmax=70,
                annot=True, fmt=".0f", linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"label": "RSI", "shrink": 0.8})
    ax10.set_title("Avg RSI per Month", fontweight="bold", fontsize=10)
    ax10.tick_params(labelsize=8)

    # ── 11. Monthly P&L Bar Chart ────────────
    ax11 = fig.add_subplot(4, 3, 11)
    monthly_pnl = []
    for mn in MONTH_NUMS:
        sub = tdf_m[tdf_m["month"] == mn]
        if len(sub) == 0:
            monthly_pnl.append(0)
        elif "PnL" in sub.columns:
            monthly_pnl.append(sub["PnL"].sum())
        else:
            monthly_pnl.append((sub["ReturnPct"] * INITIAL_CASH / 100).sum())
    colors = ["#3b6d11" if v >= 0 else "#a32d2d" for v in monthly_pnl]
    ax11.bar(MONTHS, monthly_pnl, color=colors, edgecolor="white", linewidth=0.5)
    ax11.axhline(0, color="gray", lw=0.8, ls="--")
    ax11.set_title("Monthly P&L (USD)", fontweight="bold", fontsize=10)
    ax11.set_xlabel("Month", fontsize=8)
    ax11.set_ylabel("P&L (USD)", fontsize=8)
    ax11.tick_params(labelsize=8)
    ax11.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:+,.0f}"))
    ax11.grid(True, alpha=0.2, axis="y")
    for i, v in enumerate(monthly_pnl):
        offset = 2 if v >= 0 else -4
        ax11.text(i, v + offset, f"${v:+.0f}",
                  ha="center", va="bottom", fontsize=6.5,
                  color="#3b6d11" if v >= 0 else "#a32d2d")

    # ── 12. Summary Table ────────────────────
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis("off")
    pf_val = round(stats.get("Profit Factor", 0), 2) if hasattr(stats, "get") else 0
    dd_val = round(abs(stats["Max. Drawdown [%]"]), 2)
    rows = [[
        sym,
        str(tt),
        f"{wr}%",
        f"${net:+.0f}",
        str(pf_val),
        f"{dd_val}%",
    ]]
    tbl = ax12.table(
        cellText=rows,
        colLabels=["Pair", "Trades", "WR", "P&L", "PF", "MaxDD"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#185fa5")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            is_pos = net >= 0
            cell.set_facecolor("#eaf3de" if (is_pos and c == 3) else
                               "#fcebeb" if (not is_pos and c == 3) else "#f5f5f3")
    ax12.set_title(
        f"2024-2026 | Trades:{tt} | WR:{wr}% | ${net:+,.2f}",
        fontweight="bold", fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "backtest_heatmaps_2024_2026_xauusd.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out}")
    plt.show()


# ═══════════════════════════════════════════════════════
#  EQUITY CURVE PLOT
# ═══════════════════════════════════════════════════════
def plot_equity_curve(stats, trades_df: pd.DataFrame):
    equity = stats["_equity_curve"]["Equity"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("XAUUSD Equity Curve — 2024-2026", fontsize=14, fontweight="bold")

    # ── Equity curve ──
    ax = axes[0]
    color = "#3b6d11" if equity.iloc[-1] >= INITIAL_CASH else "#a32d2d"
    ax.plot(equity.index, equity.values, color=color, lw=1.5)
    ax.axhline(INITIAL_CASH, color="gray", lw=0.8, ls="--")
    ax.fill_between(equity.index, INITIAL_CASH, equity.values,
                    where=equity.values >= INITIAL_CASH,
                    alpha=0.15, color="#3b6d11")
    ax.fill_between(equity.index, INITIAL_CASH, equity.values,
                    where=equity.values < INITIAL_CASH,
                    alpha=0.15, color="#a32d2d")
    net = round(equity.iloc[-1] - INITIAL_CASH, 2)
    ret = round(net / INITIAL_CASH * 100, 2)
    wr = round(int((trades_df["ReturnPct"] > 0).sum()) / len(trades_df) * 100, 1) if len(trades_df) else 0
    ax.set_title(
        f"{SYMBOL} | Trades:{len(trades_df)} WR:{wr}% | ${net:+,.2f} ({ret:+.1f}%)",
        fontsize=10,
    )
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Equity ($)", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

    # ── Drawdown ──
    ax2 = axes[1]
    dd_curve = stats["_equity_curve"]["DrawdownPct"] * 100
    ax2.fill_between(dd_curve.index, 0, -dd_curve.values, color="#a32d2d", alpha=0.4)
    ax2.plot(dd_curve.index, -dd_curve.values, color="#a32d2d", lw=1)
    ax2.set_title("Drawdown (%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=9)
    ax2.set_ylabel("Drawdown %", fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=8)

    plt.tight_layout()
    out = "equity_curves_2024_2026_xauusd.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out}")
    plt.show()


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  XAUUSD HEAT MAP BOT — backtesting.py v2")
    print(f"  Period : Jan 1 2024 → Apr 18 2026  (H1)")
    print(f"  Capital: ${INITIAL_CASH:,} | SL:{SL_PIPS}p TP:{TP_PIPS}p (1:{TP_PIPS//SL_PIPS} R:R)")
    print(f"  Signal : 3/5 confluence + EMA200 + ADX>15")
    print("=" * 65)

    # ── MT5 connection (Windows only) ──────────────────
    if IS_MAC:
        print("[INFO] macOS — reading from cache only.")
    else:
        if not mt5.initialize():
            print(f"[ERROR] MT5 failed: {mt5.last_error()}")
            return
        info = mt5.terminal_info()
        print(f"[OK] Connected: {info.name} | Build: {info.build}")

    # ── Fetch data ─────────────────────────────────────
    print(f"\n[{SYMBOL}] Fetching data...")
    raw_df = fetch_data(SYMBOL)
    if raw_df is None:
        print("[ERROR] No data available. Exiting.")
        return
    print(f"  {len(raw_df):,} candles | {raw_df.index[0].date()} → {raw_df.index[-1].date()}")

    if not IS_MAC:
        mt5.shutdown()
        print("[OK] MT5 disconnected.")

    # ── Prepare OHLCV for backtesting.py ───────────────
    bt_df = prepare_for_backtest(raw_df)

    # ── Run backtest ────────────────────────────────────
    print("\n[BACKTEST] Running...")
    bt = Backtest(
        bt_df,
        HeatMapStrategy,
        cash=INITIAL_CASH,
        commission=0.0,        # spread/commission can be added here
        exclusive_orders=True,  # prevent stacking positions
    )
    stats = bt.run()

    # ── Results ─────────────────────────────────────────
    trades_df = stats["_trades"]
    tt = len(trades_df)
    wins = int((trades_df["ReturnPct"] > 0).sum()) if tt else 0
    losses = tt - wins
    wr = round(wins / tt * 100, 1) if tt else 0
    net = round(stats["Equity Final [$]"] - INITIAL_CASH, 2)
    ret_pct = round(net / INITIAL_CASH * 100, 2)
    max_dd = round(abs(stats["Max. Drawdown [%]"]), 2)

    # Profit Factor
    if tt > 0 and "PnL" in trades_df.columns:
        gp = trades_df[trades_df["PnL"] > 0]["PnL"].sum()
        gl = abs(trades_df[trades_df["PnL"] < 0]["PnL"].sum())
        pf = round(gp / gl, 2) if gl > 0 else round(gp, 2)
    else:
        pf = stats.get("Profit Factor", 0)

    print("\n" + "=" * 65)
    print("  FINAL RESULTS — backtesting.py")
    print("=" * 65)
    print(f"  Total Trades  : {tt:,}")
    print(f"  Wins / Losses : {wins} / {losses}")
    print(f"  Win Rate      : {wr}%")
    print(f"  Net P&L       : ${net:+,.2f}")
    print(f"  Return        : {ret_pct:+.2f}%")
    print(f"  Final Balance : ${INITIAL_CASH + net:,.2f}")
    print(f"  Max Drawdown  : {max_dd}%")
    print(f"  Profit Factor : {pf}")
    print("=" * 65)

    # Full backtesting.py stats
    print("\n[backtesting.py stats]\n")
    # Print key stats only to keep output clean
    key_stats = [
        "Start", "End", "Duration",
        "Equity Final [$]", "Equity Peak [$]",
        "Return [%]", "Buy & Hold Return [%]",
        "Return (Ann.) [%]", "Volatility (Ann.) [%]",
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Max. Drawdown [%]", "Avg. Drawdown [%]",
        "Max. Drawdown Duration",
        "# Trades", "Win Rate [%]",
        "Best Trade [%]", "Worst Trade [%]",
        "Avg. Trade [%]", "Max. Trade Duration",
        "Profit Factor", "Expectancy [%]", "SQN",
    ]
    for k in key_stats:
        if k in stats:
            print(f"  {k:<35} {stats[k]}")

    # ── Plots ───────────────────────────────────────────
    if tt > 0:
        print("\n[PLOTTING] Generating heat maps & equity curves...")
        plot_heatmaps(raw_df, stats, trades_df)
        plot_equity_curve(stats, trades_df)
        print("\n[COMPLETE] Charts saved:")
        print("  -> backtest_heatmaps_2024_2026_xauusd.png")
        print("  -> equity_curves_2024_2026_xauusd.png")
    else:
        print("\n[WARN] No trades were generated — check signal thresholds or data.")
        # Still show the built-in backtesting.py plot
        bt.plot()


if __name__ == "__main__":
    main()
