"""Forex + Metals Heat Map Trading Bot - MT5 Backtester v4 FINAL
Period: January 1, 2024 -> April 18, 2026
Pairs: XAUUSD (Gold) only
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timezone
import warnings
import os
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════
PAIRS = [
    # Metals
    "XAUUSD",   # Gold
]

TIMEFRAME       = mt5.TIMEFRAME_H1
START_DATE      = datetime(2024, 1,  1, tzinfo=timezone.utc)
END_DATE        = datetime(2026, 4, 18, tzinfo=timezone.utc)

INITIAL_BALANCE = 10_000
LOT_SIZE        = 1
SL_PIPS         = 15
TP_PIPS         = 30     # 1:2 R:R

# Pip values per 0.1 lot (USD account, approximate)
PIP_VALUE = {
    "XAUUSD": 1.00,   # Gold: pip = $0.01, 0.1 lot = 10 oz, so $0.01*10=$0.10 ... adjusted
}

# Pip sizes
PIP_SIZE = {
    "XAUUSD": 0.10,   # Gold price moves in 0.10 increments typically
}
DEFAULT_PIP_SIZE = 0.0001

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
MONTH_NUMS = list(range(1, 13))

# Cache folder for storing candle data
CACHE_FOLDER = 'cache_data'
if not os.path.exists(CACHE_FOLDER):
    os.makedirs(CACHE_FOLDER)

# ═══════════════════════════════════════════════════════
#  CACHE FUNCTIONS
# ═══════════════════════════════════════════════════════
def get_cache_filename(symbol):
    """Generate cache filename based on symbol and date range."""
    start_str = START_DATE.strftime('%Y%m%d')
    end_str   = END_DATE.strftime('%Y%m%d')
    return os.path.join(CACHE_FOLDER, f'cache_{symbol}_{start_str}_{end_str}.csv')

def load_from_cache(symbol):
    """Load cached OHLC data if it exists."""
    cache_file = get_cache_filename(symbol)
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col='time', parse_dates=True)
            print(f"  [CACHE] Loaded {symbol} from cache ({len(df)} candles)")
            return df
        except Exception as e:
            print(f"  [WARN] Cache file corrupted: {e}, fetching fresh data...")
    return None

def save_to_cache(df, symbol):
    """Save OHLC data to cache file."""
    cache_file = get_cache_filename(symbol)
    try:
        df.to_csv(cache_file)
        print(f"  [CACHE] Saved {symbol} to cache ({len(df)} candles)")
    except Exception as e:
        print(f"  [WARN] Failed to save cache: {e}")

# ═══════════════════════════════════════════════════════
#  MT5 CONNECTION
# ═══════════════════════════════════════════════════════
def connect_mt5():
    if not mt5.initialize():
        print(f"[ERROR] MT5 failed: {mt5.last_error()}")
        print("        Make sure MT5 is open and logged in.")
        return False
    info = mt5.terminal_info()
    print(f"[OK] Connected: {info.name} | Build: {info.build} | Connected: {info.connected}")
    return True

# ═══════════════════════════════════════════════════════
#  FETCH (with caching)
# ═══════════════════════════════════════════════════════
def fetch_data(symbol):
    """Fetch data: load from cache first, then MT5 if needed."""
    # Try loading from cache first
    cached_df = load_from_cache(symbol)
    if cached_df is not None:
        return cached_df
    
    # If not in cache, fetch from MT5
    print(f"  [MT5] Fetching fresh data from MetaTrader5...")
    rates = mt5.copy_rates_range(symbol, TIMEFRAME, START_DATE, END_DATE)
    if rates is None or len(rates) == 0:
        print(f"  [WARN] No data for {symbol} — check broker symbol name")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Save to cache for next time
    save_to_cache(df, symbol)
    return df

# ═══════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════
def add_indicators(df):
    df = df.copy()

    # Triple EMA system
    df['ema9']   = df['close'].ewm(span=9,   adjust=False).mean()
    df['ema21']  = df['close'].ewm(span=21,  adjust=False).mean()
    df['ema50']  = df['close'].ewm(span=50,  adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    # RSI 14
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']      = ema12 - ema26
    df['macd_sig']  = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']

    # Stochastic (14,3,3)
    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-10)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # Bollinger Bands (20,2)
    bb_mid        = df['close'].rolling(20).mean()
    bb_std        = df['close'].rolling(20).std()
    df['bb_mid']   = bb_mid
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_pct']   = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # ATR 14
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # ADX 14 (trend strength)
    plus_dm  = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    tr14     = tr.rolling(14).mean()
    plus_di  = 100 * plus_dm.rolling(14).mean()  / (tr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (tr14 + 1e-10)
    dx       = 100 * (plus_di - minus_di).abs()  / (plus_di + minus_di + 1e-10)
    df['adx']      = dx.rolling(14).mean()
    df['plus_di']  = plus_di
    df['minus_di'] = minus_di

    # Volume ratio
    df['vol_ratio'] = df['tick_volume'] / (df['tick_volume'].rolling(20).mean() + 1e-10)

    # Momentum & returns
    df['momentum']  = df['close'].pct_change(10) * 100
    df['ret1']      = df['close'].pct_change(1)

    return df

# ═══════════════════════════════════════════════════════
#  SIGNAL — 5 CONDITION SYSTEM (needs 3/5)
#  EMA trend filter (EMA200) added for quality
# ═══════════════════════════════════════════════════════
def generate_signal(row):
    buy  = 0
    sell = 0

    # 1. EMA trend alignment (fast > slow)
    if row['ema9'] > row['ema21'] > row['ema50']:
        buy += 1
    elif row['ema9'] < row['ema21'] < row['ema50']:
        sell += 1

    # 2. RSI zone (not overbought/oversold extremes)
    if 30 < row['rsi'] < 55:
        buy += 1
    elif 45 < row['rsi'] < 70:
        sell += 1

    # 3. MACD histogram direction
    if row['macd_hist'] > 0 and row['macd'] > row['macd_sig']:
        buy += 1
    elif row['macd_hist'] < 0 and row['macd'] < row['macd_sig']:
        sell += 1

    # 4. Stochastic (momentum confirmation)
    if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 80:
        buy += 1
    elif row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 20:
        sell += 1

    # 5. Bollinger position
    if row['bb_pct'] < 0.35:   # near lower band = potential buy
        buy += 1
    elif row['bb_pct'] > 0.65: # near upper band = potential sell
        sell += 1

    # Filters
    if row['vol_ratio'] < 0.6:   return 0  # dead market
    if row['adx'] < 15:          return 0  # no trend (range market)

    # EMA200 major trend filter — only trade with the big trend
    if buy >= 3  and row['close'] > row['ema200']: return 1
    if sell >= 3 and row['close'] < row['ema200']: return -1
    return 0

# ═══════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════
def backtest(df, symbol):
    df = add_indicators(df)
    df.dropna(inplace=True)

    pip_size  = PIP_SIZE.get(symbol, DEFAULT_PIP_SIZE)
    pip_value = PIP_VALUE.get(symbol, 1.0)

    balance  = float(INITIAL_BALANCE)
    trades   = []
    in_trade = False
    entry    = direction = sl = tp = 0.0
    cooldown = 0

    for ts, row in df.iterrows():
        if cooldown > 0:
            cooldown -= 1
            continue

        if in_trade:
            hit_tp = hit_sl = False
            if direction == 1:
                if row['high'] >= tp: hit_tp = True
                elif row['low'] <= sl: hit_sl = True
            else:
                if row['low']  <= tp: hit_tp = True
                elif row['high'] >= sl: hit_sl = True

            if hit_tp:
                pnl = TP_PIPS * pip_value
                balance += pnl
                trades.append({'time': ts, 'symbol': symbol,
                    'type': 'BUY' if direction==1 else 'SELL',
                    'result': 'WIN', 'pips': TP_PIPS,
                    'pnl': round(pnl,2), 'balance': round(balance,2)})
                in_trade = False; cooldown = 3

            elif hit_sl:
                pnl = -SL_PIPS * pip_value
                balance += pnl
                trades.append({'time': ts, 'symbol': symbol,
                    'type': 'BUY' if direction==1 else 'SELL',
                    'result': 'LOSS', 'pips': -SL_PIPS,
                    'pnl': round(pnl,2), 'balance': round(balance,2)})
                in_trade = False; cooldown = 3

            if balance < INITIAL_BALANCE * 0.70:
                break

        if not in_trade and cooldown == 0:
            sig = generate_signal(row)
            if sig != 0:
                entry     = float(row['close'])
                direction = sig
                sl = entry - SL_PIPS * pip_size if sig==1 else entry + SL_PIPS * pip_size
                tp = entry + TP_PIPS * pip_size if sig==1 else entry - TP_PIPS * pip_size
                in_trade  = True

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {'symbol':symbol,'total_trades':0,'wins':0,'losses':0,
                'win_rate':0,'net_pnl':0,'profit_factor':0,
                'final_balance':INITIAL_BALANCE,'max_drawdown':0,'trades':tdf}

    wins   = int((tdf['result']=='WIN').sum())
    losses = int((tdf['result']=='LOSS').sum())
    wr     = round(wins/len(tdf)*100, 1)
    net    = round(tdf['pnl'].sum(), 2)
    gp     = tdf[tdf['pnl']>0]['pnl'].sum()
    gl     = abs(tdf[tdf['pnl']<0]['pnl'].sum())
    pf     = round(gp/gl, 2) if gl > 0 else round(gp, 2)

    curve = [INITIAL_BALANCE] + list(tdf['balance'].values)
    peak  = np.maximum.accumulate(curve)
    dd    = round(((peak-curve)/peak*100).max(), 2)

    return {'symbol':symbol,'total_trades':len(tdf),'wins':wins,'losses':losses,
            'win_rate':wr,'net_pnl':net,'profit_factor':pf,
            'final_balance':round(INITIAL_BALANCE+net,2),'max_drawdown':dd,'trades':tdf}

# ═══════════════════════════════════════════════════════
#  PLOTTING — 12 CHARTS
# ═══════════════════════════════════════════════════════
def plot_all(all_data, all_results):
    syms = [r['symbol'] for r in all_results]
    tt   = sum(r['total_trades'] for r in all_results)
    aw   = sum(r['wins'] for r in all_results)
    net  = sum(r['net_pnl'] for r in all_results)
    ow   = round(aw/tt*100,1) if tt>0 else 0

    fig = plt.figure(figsize=(26, 32))
    fig.suptitle(
        f'XAUUSD Heat Map Bot — Backtest 2024-2026\n'
        f'H1 | 1 Pair (XAUUSD) | 0.1 Lot | SL:{SL_PIPS}p TP:{TP_PIPS}p (1:2) | '
        f'Trades:{tt}  WR:{ow}%  Net P&L:${net:+,.2f}',
        fontsize=14, fontweight='bold', y=0.995
    )

    def mt(fill_fn, default=0.0):
        mat = pd.DataFrame(index=syms, columns=MONTHS, dtype=float)
        for res in all_results:
            for mn,ml in zip(MONTH_NUMS, MONTHS):
                mat.loc[res['symbol'], ml] = fill_fn(res, mn)
        return mat.fillna(default).astype(float)

    def mtr(res, mn):
        if res['trades'].empty: return pd.DataFrame()
        df_t = res['trades'].copy()
        df_t['month'] = pd.to_datetime(df_t['time']).dt.month
        return df_t[df_t['month']==mn]

    # ── 1. Momentum ─────────────────────────
    ax1 = fig.add_subplot(4,3,1)
    mom = pd.DataFrame(index=syms, columns=MONTHS, dtype=float)
    for sym, df in all_data.items():
        df2 = add_indicators(df); df2['month'] = df2.index.month
        for mn,ml in zip(MONTH_NUMS,MONTHS):
            sub = df2[df2['month']==mn]['momentum']
            mom.loc[sym,ml] = round(sub.mean(),2) if len(sub)>0 else 0
    sns.heatmap(mom.astype(float), ax=ax1, cmap='RdYlGn', center=0,
                annot=True, fmt='.1f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'Mom %', 'shrink':0.8})
    ax1.set_title('Price Momentum (%)', fontweight='bold', fontsize=10)
    ax1.tick_params(axis='both', labelsize=8)

    # ── 2. Volume ────────────────────────────
    ax2 = fig.add_subplot(4,3,2)
    def vf(res,mn):
        df = all_data.get(res['symbol'])
        if df is None: return 0
        df2=df.copy(); df2['month']=df2.index.month
        sub=df2[df2['month']==mn]['tick_volume']
        return round(sub.mean(),0) if len(sub)>0 else 0
    sns.heatmap(mt(vf), ax=ax2, cmap='Blues',
                annot=True, fmt='.0f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'Avg Volume','shrink':0.8})
    ax2.set_title('Volume Activity', fontweight='bold', fontsize=10)
    ax2.tick_params(axis='both', labelsize=8)

    # ── 3. Win Rate ──────────────────────────
    ax3 = fig.add_subplot(4,3,3)
    def wrf(res,mn):
        sub=mtr(res,mn)
        return round((sub['result']=='WIN').mean()*100,1) if len(sub)>0 else 0
    sns.heatmap(mt(wrf), ax=ax3, cmap='RdYlGn', vmin=0, vmax=100,
                annot=True, fmt='.0f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'WR %','shrink':0.8})
    ax3.set_title('Win Rate per Month (%)', fontweight='bold', fontsize=10)
    ax3.tick_params(axis='both', labelsize=8)

    # ── 4. P&L ───────────────────────────────
    ax4 = fig.add_subplot(4,3,4)
    def pnlf(res,mn):
        sub=mtr(res,mn)
        return round(sub['pnl'].sum(),1) if len(sub)>0 else 0
    sns.heatmap(mt(pnlf), ax=ax4, cmap='RdYlGn', center=0,
                annot=True, fmt='.0f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'P&L $','shrink':0.8})
    ax4.set_title('P&L per Month (USD)', fontweight='bold', fontsize=10)
    ax4.tick_params(axis='both', labelsize=8)

    # ── 5. Trade Count ───────────────────────
    ax5 = fig.add_subplot(4,3,5)
    def tcf(res,mn): return len(mtr(res,mn))
    sns.heatmap(mt(tcf), ax=ax5, cmap='YlOrBr',
                annot=True, fmt='.0f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'Trades','shrink':0.8})
    ax5.set_title('Trade Count per Month', fontweight='bold', fontsize=10)
    ax5.tick_params(axis='both', labelsize=8)

    # ── 6. Max Drawdown ──────────────────────
    ax6 = fig.add_subplot(4,3,6)
    def ddf(res,mn):
        sub=mtr(res,mn)
        if len(sub)<2: return 0
        pk=sub['balance'].cummax()
        return round(((pk-sub['balance'])/pk*100).max(),1)
    sns.heatmap(mt(ddf), ax=ax6, cmap='YlOrRd',
                annot=True, fmt='.1f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'Max DD %','shrink':0.8})
    ax6.set_title('Max Drawdown per Month (%)', fontweight='bold', fontsize=10)
    ax6.tick_params(axis='both', labelsize=8)

    # ── 7. Profit Factor ─────────────────────
    ax7 = fig.add_subplot(4,3,7)
    def pff(res,mn):
        sub=mtr(res,mn)
        if len(sub)==0: return 0
        gp=sub[sub['pnl']>0]['pnl'].sum()
        gl=abs(sub[sub['pnl']<0]['pnl'].sum())
        return round(gp/gl,2) if gl>0 else round(gp,2)
    sns.heatmap(mt(pff), ax=ax7, cmap='RdYlGn', center=1,
                annot=True, fmt='.2f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'PF','shrink':0.8})
    ax7.set_title('Profit Factor per Month', fontweight='bold', fontsize=10)
    ax7.tick_params(axis='both', labelsize=8)

    # ── 8. ADX / Trend Strength ──────────────
    ax8 = fig.add_subplot(4,3,8)
    adx_mat = pd.DataFrame(index=syms, columns=MONTHS, dtype=float)
    for sym, df in all_data.items():
        df2 = add_indicators(df); df2['month'] = df2.index.month
        for mn,ml in zip(MONTH_NUMS,MONTHS):
            sub = df2[df2['month']==mn]['adx']
            adx_mat.loc[sym,ml] = round(sub.mean(),1) if len(sub)>0 else 0
    sns.heatmap(adx_mat.astype(float), ax=ax8, cmap='Blues', vmin=0, vmax=50,
                annot=True, fmt='.0f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'ADX','shrink':0.8})
    ax8.set_title('Trend Strength (ADX)', fontweight='bold', fontsize=10)
    ax8.tick_params(axis='both', labelsize=8)

    # ── 9. Correlation Matrix ────────────────
    ax9 = fig.add_subplot(4,3,9)
    closes = pd.DataFrame()
    for sym, df in all_data.items():
        closes[sym] = df['close'].resample('D').last()
    closes.dropna(how='all', inplace=True)
    corr = closes.pct_change().corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax9, cmap='RdYlGn', center=0, mask=mask,
                annot=True, fmt='.2f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'Corr','shrink':0.8})
    ax9.set_title('Asset Correlation Matrix', fontweight='bold', fontsize=10)
    ax9.tick_params(axis='both', labelsize=8)

    # ── 10. RSI Heat Map ─────────────────────
    ax10 = fig.add_subplot(4,3,10)
    rsi_mat = pd.DataFrame(index=syms, columns=MONTHS, dtype=float)
    for sym, df in all_data.items():
        df2 = add_indicators(df); df2['month'] = df2.index.month
        for mn,ml in zip(MONTH_NUMS,MONTHS):
            sub = df2[df2['month']==mn]['rsi']
            rsi_mat.loc[sym,ml] = round(sub.mean(),1) if len(sub)>0 else 50
    sns.heatmap(rsi_mat.astype(float), ax=ax10, cmap='RdYlGn', center=50,
                vmin=30, vmax=70,
                annot=True, fmt='.0f', linewidths=0.4, annot_kws={'size':7},
                cbar_kws={'label':'RSI','shrink':0.8})
    ax10.set_title('Avg RSI per Month', fontweight='bold', fontsize=10)
    ax10.tick_params(axis='both', labelsize=8)

    # ── 11. Monthly Cumulative P&L ───────────
    ax11 = fig.add_subplot(4,3,11)
    monthly_pnl = pd.DataFrame(index=MONTHS, columns=syms, dtype=float)
    for res in all_results:
        if res['trades'].empty: continue
        df_t = res['trades'].copy()
        df_t['month'] = pd.to_datetime(df_t['time']).dt.month
        for mn,ml in zip(MONTH_NUMS,MONTHS):
            sub = df_t[df_t['month']==mn]
            monthly_pnl.loc[ml, res['symbol']] = sub['pnl'].sum() if len(sub)>0 else 0
    monthly_pnl = monthly_pnl.fillna(0).astype(float)
    monthly_total = monthly_pnl.sum(axis=1)
    colors = ['#3b6d11' if v>=0 else '#a32d2d' for v in monthly_total]
    ax11.bar(MONTHS, monthly_total.values, color=colors, edgecolor='white', linewidth=0.5)
    ax11.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax11.set_title('Monthly P&L — All Pairs Combined', fontweight='bold', fontsize=10)
    ax11.set_xlabel('Month', fontsize=8)
    ax11.set_ylabel('P&L (USD)', fontsize=8)
    ax11.tick_params(axis='both', labelsize=8)
    ax11.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:+,.0f}'))
    ax11.grid(True, alpha=0.2, axis='y')
    for i,(m,v) in enumerate(zip(MONTHS, monthly_total)):
        ax11.text(i, v + (8 if v>=0 else -12), f'${v:+.0f}',
                  ha='center', va='bottom', fontsize=7,
                  color='#3b6d11' if v>=0 else '#a32d2d')

    # ── 12. Summary Table ────────────────────
    ax12 = fig.add_subplot(4,3,12)
    ax12.axis('off')
    rows = []
    for r in sorted(all_results, key=lambda x: x['net_pnl'], reverse=True):
        rows.append([
            r['symbol'],
            str(r['total_trades']),
            f"{r['win_rate']}%",
            f"${r['net_pnl']:+.0f}",
            str(r['profit_factor']),
            f"{r['max_drawdown']}%"
        ])
    tbl = ax12.table(cellText=rows,
                     colLabels=['Pair','Trades','WR','P&L','PF','MaxDD'],
                     loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.65)
    for (r,c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#185fa5')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            pnl_str = rows[r-1][3] if r<=len(rows) else '0'
            is_pos  = '+' in pnl_str and '-' not in pnl_str
            if c == 3:
                cell.set_facecolor('#eaf3de' if is_pos else '#fcebeb')
            else:
                cell.set_facecolor('#f5f5f3' if r%2==0 else 'white')
    ax12.set_title(
        f'2024-2026 | Total:{tt} trades | WR:{ow}% | ${net:+,.2f}',
        fontweight='bold', fontsize=9
    )

    plt.tight_layout(rect=[0,0,1,0.97])
    out = 'backtest_heatmaps_2024_2026_xauusd.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {out}")
    plt.show()


# ═══════════════════════════════════════════════════════
#  EQUITY CURVES
# ═══════════════════════════════════════════════════════
def plot_equity(all_results):
    n    = len(all_results)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(25, rows*5))
    fig.suptitle('Equity Curves — XAUUSD | 2024-2026', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for i, res in enumerate(all_results):
        ax = axes[i]
        if res['total_trades'] == 0:
            ax.text(0.5,0.5,'No Trades\n(Signal filtered)',
                    ha='center',va='center',transform=ax.transAxes,color='gray',fontsize=10)
            ax.set_title(res['symbol'])
            continue
        curve = [INITIAL_BALANCE] + list(res['trades']['balance'])
        col   = '#3b6d11' if curve[-1]>=INITIAL_BALANCE else '#a32d2d'
        ax.plot(curve, color=col, lw=1.5)
        ax.axhline(INITIAL_BALANCE, color='gray', lw=0.8, ls='--')
        ax.fill_between(range(len(curve)), INITIAL_BALANCE, curve,
                        where=[c>=INITIAL_BALANCE for c in curve], alpha=0.15, color='#3b6d11')
        ax.fill_between(range(len(curve)), INITIAL_BALANCE, curve,
                        where=[c<INITIAL_BALANCE  for c in curve], alpha=0.15, color='#a32d2d')
        ret = round((curve[-1]-INITIAL_BALANCE)/INITIAL_BALANCE*100, 1)
        ax.set_title(
            f"{res['symbol']} | T:{res['total_trades']} WR:{res['win_rate']}% "
            f"PF:{res['profit_factor']} | ${res['net_pnl']:+,.0f} ({ret:+.1f}%)",
            fontsize=8.5
        )
        ax.set_xlabel('Trades', fontsize=8)
        ax.set_ylabel('Balance $', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    out = 'equity_curves_2024_2026_xauusd.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[SAVED] {out}")
    plt.show()


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  XAUUSD HEAT MAP BOT — BACKTESTER v4 FINAL")
    print(f"  Period  : Jan 1, 2024 → Apr 18, 2026  (Full 2024-2026, H1)")
    print(f"  Pairs   : {len(PAIRS)} — Gold (XAU) only")
    print(f"  Capital : ${INITIAL_BALANCE:,} | Lot: {LOT_SIZE}")
    print(f"  SL: {SL_PIPS}p  TP: {TP_PIPS}p  (1:{TP_PIPS//SL_PIPS} R:R)")
    print(f"  Signal  : 3/5 indicators + EMA200 trend filter + ADX>15")
    print("=" * 65)

    if not connect_mt5():
        return

    all_data, all_results = {}, []

    for sym in PAIRS:
        print(f"\n[{sym}] Fetching data...")
        df = fetch_data(sym)
        if df is None:
            continue
        print(f"  {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")
        all_data[sym] = df
        print(f"  Running backtest...")
        res = backtest(df, sym)
        all_results.append(res)
        status = "PROFIT" if res['net_pnl'] >= 0 else "LOSS  "
        print(f"  [{status}] Trades:{res['total_trades']:>4}  WR:{res['win_rate']:>5}%  "
              f"PF:{res['profit_factor']:>5}  "
              f"P&L:${res['net_pnl']:>+8,.2f}  "
              f"MaxDD:{res['max_drawdown']:>5}%")

    mt5.shutdown()
    print("\n[OK] MT5 disconnected.")

    if not all_results:
        print("[ERROR] No results — check MT5 connection and symbol names.")
        return

    tt  = sum(r['total_trades'] for r in all_results)
    aw  = sum(r['wins']         for r in all_results)
    al  = sum(r['losses']       for r in all_results)
    net = sum(r['net_pnl']      for r in all_results)
    ow  = round(aw/tt*100, 1) if tt>0 else 0
    best  = max(all_results, key=lambda x: x['net_pnl'])
    worst = min(all_results, key=lambda x: x['net_pnl'])

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY — 2024-2026")
    print("=" * 65)
    print(f"  Total Trades  : {tt:,}")
    print(f"  Total Wins    : {aw:,}  |  Losses: {al:,}")
    print(f"  Overall WR    : {ow}%")
    print(f"  Net P&L       : ${net:+,.2f}")
    print(f"  Final Balance : ${INITIAL_BALANCE+net:,.2f}")
    print(f"  Return        : {net/INITIAL_BALANCE*100:+.2f}%")
    print(f"  Best Pair     : {best['symbol']}  (${best['net_pnl']:+,.2f})")
    print(f"  Worst Pair    : {worst['symbol']} (${worst['net_pnl']:+,.2f})")
    print("=" * 65)

    print("\n  Per-Pair Summary (sorted by P&L):")
    print(f"  {'Pair':<10} {'Trades':>7} {'WR':>7} {'PF':>6} {'P&L':>10} {'MaxDD':>8}")
    print(f"  {'-'*55}")
    for r in sorted(all_results, key=lambda x: x['net_pnl'], reverse=True):
        flag = " <-- BEST" if r['symbol']==best['symbol'] else (
               " <-- AVOID" if r['symbol']==worst['symbol'] else "")
        print(f"  {r['symbol']:<10} {r['total_trades']:>7} {r['win_rate']:>6}% "
              f"{r['profit_factor']:>6} ${r['net_pnl']:>+9,.2f} {r['max_drawdown']:>7}%{flag}")

    if all_data:
        print("\n[PLOTTING] 12 heat maps + equity curves generating...")
        plot_all(all_data, all_results)
        plot_equity(all_results)
        print("\n[COMPLETE] Tamam charts save ho gaye!")

        print("  -> backtest_heatmaps_2026_xauusd.png")
        print("  -> equity_curves_2026_xauusd.png")

if __name__ == "__main__":
    main()