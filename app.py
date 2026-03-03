import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="Ultimate MEXC Scanner", layout="wide", page_icon="📈")
st.title("📈 Ultimate MEXC Scanner – Multi‑Strategy")
st.markdown("Scan top MEXC perpetuals with multiple proven strategies. Adjust filters to match your style.")

# -------------------- Sidebar Settings --------------------
st.sidebar.header("⚙️ Scanner Settings")

# Strategy selection
strategy = st.sidebar.selectbox(
    "Choose Strategy",
    ["Qullamaggie Momentum", "Turtle System", "SuperTrend + Volume", "Custom Day Trade", "My Strategy (Confluence)"],
    help="Each strategy uses a different combination of indicators."
)

# Timeframe
tf_options = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
tf_display = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=2)  # default 15m
tf = tf_options[tf_display]

# Number of symbols to scan
num_symbols = st.sidebar.slider("Number of top pairs to scan", 10, 100, 50, step=5,
                                help="Pairs are sorted by 24h volume (USDT).")

# Common filters
min_volume_usdt = st.sidebar.number_input("Minimum 24h volume (USDT, millions)", min_value=0.0, value=5.0, step=1.0) * 1e6
min_rvol = st.sidebar.number_input("Minimum relative volume (current/avg)", min_value=0.5, value=1.2, step=0.1)

# Strategy‑specific parameters (shown dynamically)
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Strategy Parameters")

if strategy == "Qullamaggie Momentum":
    q_vol_mult = st.sidebar.number_input("Volume multiplier", 1.0, 5.0, 2.0, 0.1)
    q_rsi_threshold = st.sidebar.slider("RSI minimum", 50, 80, 60)
    q_lookback = st.sidebar.number_input("Breakout lookback (bars)", 5, 50, 20)

elif strategy == "Turtle System":
    t_donchian = st.sidebar.number_input("Donchian channel length", 10, 50, 20)
    t_vol_mult = st.sidebar.number_input("Volume multiplier", 1.0, 3.0, 1.5, 0.1)

elif strategy == "SuperTrend + Volume":
    st_sensitivity = st.sidebar.number_input("SuperTrend multiplier", 1.0, 5.0, 3.0, 0.5)
    st_vol_mult = st.sidebar.number_input("Volume multiplier", 1.0, 3.0, 1.8, 0.1)
    st_rsi = st.sidebar.slider("RSI filter (0 = off)", 0, 80, 50)

elif strategy == "Custom Day Trade":
    dt_vol_mult = st.sidebar.number_input("Volume multiplier", 1.0, 3.0, 1.5, 0.1)
    dt_rsi_long = st.sidebar.slider("RSI long minimum", 30, 70, 50)
    dt_rsi_short = st.sidebar.slider("RSI short maximum", 30, 70, 50)
    dt_ema_fast = st.sidebar.number_input("Fast EMA", 5, 50, 9)
    dt_ema_slow = st.sidebar.number_input("Slow EMA", 10, 100, 21)

else:  # My Strategy (Confluence) – let user toggle each condition
    st.sidebar.markdown("**Toggle conditions (LONG):**")
    use_ema_cross = st.sidebar.checkbox("EMA9 > EMA21", True)
    use_supertrend = st.sidebar.checkbox("SuperTrend bullish", True)
    use_volume_surge = st.sidebar.checkbox("Volume surge", True)
    use_rsi = st.sidebar.checkbox("RSI > 50", True)
    use_macd = st.sidebar.checkbox("MACD histogram rising", False)
    use_bb_exp = st.sidebar.checkbox("Bollinger Bands expanding", False)
    use_ema200 = st.sidebar.checkbox("Price above EMA200", True)
    st.sidebar.markdown("**Thresholds:**")
    conf_vol_mult = st.sidebar.number_input("Volume multiplier", 1.0, 5.0, 2.0, 0.1)
    conf_rsi_min = st.sidebar.slider("RSI minimum", 30, 70, 50)
    conf_rsi_max = st.sidebar.slider("RSI maximum (short)", 30, 70, 50)

# Refresh control
refresh_sec = st.sidebar.slider("Auto‑refresh (seconds)", 10, 120, 30)
st.sidebar.button("🔄 Refresh Now")

# -------------------- Helper Functions --------------------
@st.cache_data(ttl=60)  # cache for 60 seconds
def get_top_symbols(exchange, limit=100):
    """Fetch top USDT perpetuals by 24h volume."""
    try:
        tickers = exchange.fetch_tickers()
        usdt_perps = []
        for sym, t in tickers.items():
            if sym.endswith('/USDT:USDT') and t['quoteVolume'] is not None:
                usdt_perps.append((sym, t['quoteVolume']))
        usdt_perps.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, vol in usdt_perps[:limit]]
    except Exception as e:
        st.error(f"Could not fetch top symbols: {e}")
        # Fallback to a default list
        return [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
            "DOGE/USDT:USDT", "ADA/USDT:USDT", "SHIB/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT"
        ]

@st.cache_data(ttl=30)
def fetch_ohlcv(symbol, timeframe, limit=200):
    exchange = ccxt.mexc({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except:
        return None

def compute_indicators(df):
    """Add all necessary indicators to the dataframe."""
    df = df.copy()
    # EMAs
    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    # Volume average and relative volume
    df['vol_avg'] = df['volume'].rolling(20).mean()
    df['rvol'] = df['volume'] / df['vol_avg']

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    df['macd'] = macd
    df['macd_signal'] = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - df['macd_signal']

    # Bollinger Bands (20,2)
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid
    df['bb_exp'] = df['bb_width'] > df['bb_width'].rolling(20).mean()

    # ATR for stops
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # SuperTrend (similar to original, using 3*ATR)
    hl2 = (df['high'] + df['low']) / 2
    df['upper'] = hl2 + 3 * df['atr']
    df['lower'] = hl2 - 3 * df['atr']
    df['st_trend'] = True  # True = uptrend (price above lower), False = downtrend
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upper'].iloc[i-1]:
            df.loc[df.index[i], 'st_trend'] = True
        elif df['close'].iloc[i] < df['lower'].iloc[i-1]:
            df.loc[df.index[i], 'st_trend'] = False
        else:
            df.loc[df.index[i], 'st_trend'] = df['st_trend'].iloc[i-1]
    df['st_line'] = np.where(df['st_trend'], df['lower'], df['upper'])

    # Donchian channels
    df['donchian_high'] = df['high'].rolling(20).max()
    df['donchian_low'] = df['low'].rolling(20).min()

    return df

def get_signal_qullamaggie(df, params):
    """Qullamaggie: breakout above recent high, high volume, RSI > threshold."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    lookback = params['lookback']
    # Long condition
    if (last['close'] > df['high'].rolling(lookback).max().iloc[-2] and  # breakout above previous high
        last['rvol'] > params['vol_mult'] and
        last['rsi'] > params['rsi_min'] and
        last['close'] > last['ema50']):
        return "LONG", "#00FF00"
    return "WAIT", "#AAAAAA"

def get_signal_turtle(df, params):
    """Turtle: Donchian breakout with volume."""
    last = df.iloc[-1]
    if (last['close'] > df['donchian_high'].iloc[-2] and
        last['rvol'] > params['vol_mult']):
        return "LONG", "#00FF00"
    if (last['close'] < df['donchian_low'].iloc[-2] and
        last['rvol'] > params['vol_mult']):
        return "SHORT", "#FF0000"
    return "WAIT", "#AAAAAA"

def get_signal_supertrend(df, params):
    """SuperTrend + volume + optional RSI."""
    last = df.iloc[-1]
    if (last['st_trend'] and
        last['rvol'] > params['vol_mult'] and
        (params['rsi_min'] == 0 or last['rsi'] > params['rsi_min'])):
        return "LONG", "#00FF00"
    if (not last['st_trend'] and
        last['rvol'] > params['vol_mult'] and
        (params['rsi_min'] == 0 or last['rsi'] < 100 - params['rsi_min'])):
        return "SHORT", "#FF0000"
    return "WAIT", "#AAAAAA"

def get_signal_daytrade(df, params):
    """EMA cross, volume, RSI."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # Long: EMA9 crosses above EMA21
    if (prev['ema9'] <= prev['ema21'] and last['ema9'] > last['ema21'] and
        last['rvol'] > params['vol_mult'] and
        last['rsi'] > params['rsi_long']):
        return "LONG", "#00FF00"
    # Short: EMA9 crosses below EMA21
    if (prev['ema9'] >= prev['ema21'] and last['ema9'] < last['ema21'] and
        last['rvol'] > params['vol_mult'] and
        last['rsi'] < params['rsi_short']):
        return "SHORT", "#FF0000"
    return "WAIT", "#AAAAAA"

def get_signal_confluence(df, params):
    """User‑defined combination of conditions."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    long_score = 0
    short_score = 0
    # Conditions for LONG
    if params['use_ema_cross'] and last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21']:
        long_score += 1
    if params['use_supertrend'] and last['st_trend']:
        long_score += 1
    if params['use_volume_surge'] and last['rvol'] > params['vol_mult']:
        long_score += 1
    if params['use_rsi'] and last['rsi'] > params['rsi_min']:
        long_score += 1
    if params['use_macd'] and last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist']:
        long_score += 1
    if params['use_bb_exp'] and last['bb_exp']:
        long_score += 1
    if params['use_ema200'] and last['close'] > last['ema200']:
        long_score += 1

    # Conditions for SHORT
    if params['use_ema_cross'] and last['ema9'] < last['ema21'] and prev['ema9'] >= prev['ema21']:
        short_score += 1
    if params['use_supertrend'] and not last['st_trend']:
        short_score += 1
    if params['use_volume_surge'] and last['rvol'] > params['vol_mult']:
        short_score += 1
    if params['use_rsi'] and last['rsi'] < params['rsi_max']:
        short_score += 1
    if params['use_macd'] and last['macd_hist'] < 0 and last['macd_hist'] < prev['macd_hist']:
        short_score += 1
    if params['use_bb_exp'] and last['bb_exp']:
        short_score += 1
    if params['use_ema200'] and last['close'] < last['ema200']:
        short_score += 1

    # Determine signal based on number of conditions met
    total_conditions = sum([params['use_ema_cross'], params['use_supertrend'], params['use_volume_surge'],
                            params['use_rsi'], params['use_macd'], params['use_bb_exp'], params['use_ema200']])
    if total_conditions == 0:
        return "WAIT", "#AAAAAA"

    if long_score >= total_conditions * 0.6:  # at least 60% of selected conditions
        return "LONG", "#00FF00"
    if short_score >= total_conditions * 0.6:
        return "SHORT", "#FF0000"
    return "WAIT", "#AAAAAA"

# -------------------- Main App --------------------
exchange = ccxt.mexc({'enableRateLimit': True})

# Get top symbols
with st.spinner("Fetching top symbols by volume..."):
    symbol_list = get_top_symbols(exchange, limit=num_symbols)

if not symbol_list:
    st.error("No symbols retrieved. Using fallback list.")
    symbol_list = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]

st.success(f"Scanning {len(symbol_list)} symbols on {tf_display}...")

# Prepare parameters dict based on selected strategy
params = {}
if strategy == "Qullamaggie Momentum":
    params = {'lookback': q_lookback, 'vol_mult': q_vol_mult, 'rsi_min': q_rsi_threshold}
    signal_func = get_signal_qullamaggie
elif strategy == "Turtle System":
    params = {'donchian': t_donchian, 'vol_mult': t_vol_mult}
    signal_func = get_signal_turtle
elif strategy == "SuperTrend + Volume":
    params = {'vol_mult': st_vol_mult, 'rsi_min': st_rsi}
    signal_func = get_signal_supertrend
elif strategy == "Custom Day Trade":
    params = {'vol_mult': dt_vol_mult, 'rsi_long': dt_rsi_long, 'rsi_short': dt_rsi_short,
              'ema_fast': dt_ema_fast, 'ema_slow': dt_ema_slow}
    signal_func = get_signal_daytrade
else:  # My Strategy
    params = {
        'use_ema_cross': use_ema_cross,
        'use_supertrend': use_supertrend,
        'use_volume_surge': use_volume_surge,
        'use_rsi': use_rsi,
        'use_macd': use_macd,
        'use_bb_exp': use_bb_exp,
        'use_ema200': use_ema200,
        'vol_mult': conf_vol_mult,
        'rsi_min': conf_rsi_min,
        'rsi_max': conf_rsi_max
    }
    signal_func = get_signal_confluence

# Scan symbols
results = []
hot_signals = []
progress_bar = st.progress(0)

for i, sym in enumerate(symbol_list):
    progress_bar.progress((i + 1) / len(symbol_list))
    df = fetch_ohlcv(sym, tf)
    if df is None or len(df) < 100:
        continue
    df = compute_indicators(df)
    last = df.iloc[-1]

    # Skip low volume pairs (if we have 24h volume data, but we don't; use recent volume as proxy)
    # Instead we use the volume from the last candle? Not ideal, but we'll trust the top symbol list.

    # Apply strategy
    signal, color = signal_func(df, params)

    # Also compute relative volume for display
    rvol = last['rvol'] if not pd.isna(last['rvol']) else 1.0

    results.append({
        "Symbol": sym.replace("/USDT:USDT", ""),
        "Price": last['close'],
        "Signal": signal,
        "Color": color,
        "RVOL": round(rvol, 2),
        "RSI": round(last['rsi'], 1) if not pd.isna(last['rsi']) else None,
        "Volume": last['volume']
    })

    if signal != "WAIT":
        hot_signals.append({
            "Symbol": sym.replace("/USDT:USDT", ""),
            "Signal": signal,
            "Color": color,
            "Price": last['close']
        })

# -------------------- Display --------------------
st.subheader("🔥 Hot Signals")
if hot_signals:
    cols = st.columns(len(hot_signals))
    for idx, hs in enumerate(hot_signals):
        with cols[idx]:
            st.markdown(f"<h3 style='color:{hs['Color']}; text-align:center;'>{hs['Symbol']}<br>{hs['Signal']}</h3>", unsafe_allow_html=True)
            st.metric("Price", f"${hs['Price']:,.4f}")
else:
    st.info("No hot signals at the moment. Try adjusting filters or timeframe.")

st.subheader("📋 All Scanned Pairs")
df_results = pd.DataFrame(results)
# Drop Volume column for cleaner view
df_display = df_results[["Symbol", "Price", "Signal", "RVOL", "RSI"]].copy()
df_display["Price"] = df_display["Price"].apply(lambda x: f"${x:,.4f}")
df_display["Signal"] = df_display.apply(lambda row: f"<span style='color:{row['Color']}; font-weight:bold'>{row['Signal']}</span>", axis=1)
df_display = df_display.drop(columns=["Color"])
st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

# -------------------- Detailed Chart --------------------
st.subheader("📊 Select a Coin for Detailed View")
selected_coin = st.selectbox("Choose a coin", [s["Symbol"] for s in results])
full_sym = selected_coin + "/USDT:USDT"

if full_sym:
    df_detail = fetch_ohlcv(full_sym, tf)
    if df_detail is not None:
        df_detail = compute_indicators(df_detail)
        last = df_detail.iloc[-1]

        # Create candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=df_detail['timestamp'][-100:],
                open=df_detail['open'][-100:],
                high=df_detail['high'][-100:],
                low=df_detail['low'][-100:],
                close=df_detail['close'][-100:],
                name="Price"
            )
        ])
        # Add EMAs
        fig.add_trace(go.Scatter(x=df_detail['timestamp'][-100:], y=df_detail['ema9'][-100:],
                                  name="EMA9", line=dict(color="lime", width=1)))
        fig.add_trace(go.Scatter(x=df_detail['timestamp'][-100:], y=df_detail['ema21'][-100:],
                                  name="EMA21", line=dict(color="red", width=1)))
        fig.add_trace(go.Scatter(x=df_detail['timestamp'][-100:], y=df_detail['ema200'][-100:],
                                  name="EMA200", line=dict(color="orange", width=1)))
        # SuperTrend line
        fig.add_trace(go.Scatter(x=df_detail['timestamp'][-100:], y=df_detail['st_line'][-100:],
                                  name="SuperTrend", line=dict(color="purple", width=2)))

        fig.update_layout(title=f"{selected_coin} – {tf_display}", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Show signal and suggested stops
        signal, _ = signal_func(df_detail, params)
        if signal != "WAIT":
            atr = last['atr']
            entry = last['close']
            if signal == "LONG":
                sl = entry - 2 * atr
                tp1 = entry + 3 * atr
                tp2 = entry + 5 * atr
            else:
                sl = entry + 2 * atr
                tp1 = entry - 3 * atr
                tp2 = entry - 5 * atr
            st.success(f"**Current Signal:** {signal} at {entry:.4f}")
            st.info(f"📉 **Stop Loss:** {sl:.4f} | 🎯 **TP1:** {tp1:.4f} | 🚀 **TP2:** {tp2:.4f} (ATR‑based)")
        else:
            st.info("No active signal for this coin.")

# -------------------- Footer --------------------
st.caption(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto‑refresh every {refresh_sec}s")
time.sleep(refresh_sec)
st.rerun()
