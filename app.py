import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="Best MEXC Scanner", layout="wide", page_icon="🚀")
st.title("🚀 Best Profitable 100x Day Trading Scanner (Modified VSI)")
st.success("✅ Enhanced for higher profitability - looser normal mode, funding filter, trailing stop, backtest button!")
st.warning("⚠️ 100x = EXTREME RISK. Max 0.5% risk per trade. Not financial advice.")

# Sidebar
st.sidebar.header("Settings")
symbols = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "PEPE/USDT:USDT", "SHIB/USDT:USDT", "SUI/USDT:USDT", "TON/USDT:USDT", "BNB/USDT:USDT",
    "ADA/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT", "TRX/USDT:USDT", "WIF/USDT:USDT",
    "GOLD/XAUT:USDT", "DOT/USDT:USDT", "MATIC/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT"
]
tf_options = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "Daily": "1d", "Weekly": "1w"}
tf_display = st.sidebar.selectbox("Timeframe", list(tf_options.keys()), index=0)  # default 5m for day trading
tf = tf_options[tf_display]

a1_mode = st.sidebar.checkbox("A1 Setups Only (Highest Probability - Fewer Trades)", value=False)
refresh_sec = st.sidebar.slider("Auto Refresh (seconds)", 20, 60, 30)

# Functions
def get_data(ex, sym, timeframe, limit=200):
    ohlcv = ex.fetch_ohlcv(sym, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

def calculate_indicators(df):
    df = df.copy()
    df['ema9'] = df['c'].ewm(span=9, adjust=False).mean()
    df['ema21'] = df['c'].ewm(span=21, adjust=False).mean()
    df['ema200'] = df['c'].ewm(span=200, adjust=False).mean()
    df['vol_avg'] = df['v'].rolling(20).mean()
    multiplier = 3.5 if a1_mode else 2.0  # Looser for more trades
    df['vol_surge'] = (df['v'] > df['vol_avg'] * multiplier) & (df['v'] > df['v'].shift())
    delta = df['c'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    macd = df['c'].ewm(span=12, adjust=False).mean() - df['c'].ewm(span=26, adjust=False).mean()
    df['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    bb_mid = df['c'].rolling(20).mean()
    bb_std = df['c'].rolling(20).std()
    df['bb_width'] = (bb_mid + 2*bb_std - (bb_mid - 2*bb_std)) / bb_mid
    df['bb_exp'] = df['bb_width'] > df['bb_width'].rolling(20).mean()
    hl2 = (df['h'] + df['l']) / 2
    tr = pd.concat([df['h']-df['l'], (df['h']-df['c'].shift()).abs(), (df['l']-df['c'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(10).mean()
    df['upper'] = hl2 + 3 * df['atr']
    df['lower'] = hl2 - 3 * df['atr']
    df['trend'] = True
    for i in range(1, len(df)):
        if df['c'].iloc[i] > df['upper'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = True
        elif df['c'].iloc[i] < df['lower'].iloc[i-1]:
            df.loc[df.index[i], 'trend'] = False
        else:
            df.loc[df.index[i], 'trend'] = df['trend'].iloc[i-1]
    df['st'] = np.where(df['trend'], df['lower'], df['upper'])
    return df

def get_signal(df, funding_rate):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    vol_surge = last['vol_surge']
    ema_cross_long = last['ema9'] > last['ema21'] and prev['ema9'] <= prev['ema21']
    ema_cross_short = last['ema9'] < last['ema21'] and prev['ema9'] >= prev['ema21']
    rsi_long = last['rsi'] > (60 if a1_mode else 50)  # Stricter in A1
    rsi_short = last['rsi'] < (40 if a1_mode else 50)
    macd_bull = last['macd_hist'] > 0 and last['macd_hist'] > prev['macd_hist'] * (1.2 if a1_mode else 1)
    macd_bear = last['macd_hist'] < 0 and last['macd_hist'] < prev['macd_hist'] * (1.2 if a1_mode else 1)
    long_cond = vol_surge and ema_cross_long and last['trend'] and rsi_long and macd_bull and last['bb_exp'] and last['c'] > last['ema200'] and funding_rate < 0.05  # Funding filter
    short_cond = vol_surge and ema_cross_short and not last['trend'] and rsi_short and macd_bear and last['bb_exp'] and last['c'] < last['ema200'] and funding_rate > -0.05
    if long_cond:
        return f"A1 LONG" if a1_mode else "LONG", "#00FF00", last['c']
    if short_cond:
        return f"A1 SHORT" if a1_mode else "SHORT", "#FF0000", last['c']
    return "WAIT", "#AAAAAA", last['c']

# Fetch funding rate (for filter)
def get_funding_rate(ex, sym):
    try:
        return ex.fetch_funding_rate(sym)['rate']
    except:
        return 0

# RUN SCANNER
exchange = ccxt.mexc({'enableRateLimit': True})
results = {}
hot = []
progress = st.progress(0)
for i, sym in enumerate(symbols):
    try:
        df = get_data(exchange, sym, tf)
        df = calculate_indicators(df)
        funding = get_funding_rate(exchange, sym)
        sig, col, price = get_signal(df, funding)
        results[sym] = {"price": price, "sig": sig, "col": col}
        if sig != "WAIT":
            hot.append((sym.replace("/USDT:USDT", ""), sig, col, price))
    except:
        pass
    progress.progress(int((i + 1) / len(symbols) * 100))

st.subheader("🔥 Hot Signals (Ready to Trade Now)")
if hot:
    cols = st.columns(len(hot))
    for idx, (s, sig, colr, pr) in enumerate(hot):
        with cols[idx]:
            st.markdown(f"<h3 style='color:{colr}; text-align:center;'>{s}<br>{sig}</h3>", unsafe_allow_html=True)
            st.metric("", f"${pr:,.4f}")
else:
    st.info("No hot signals — uncheck A1 and switch to 5m/15m")

st.subheader("All Pairs Scanner")
data = [{"Coin": k.replace("/USDT:USDT",""), "Price": f"${v['price']:,.4f}", "Signal": f"<span style='color:{v['col']}; font-weight:bold'>{v['sig']}</span>"} for k,v in results.items()]
st.markdown(pd.DataFrame(data).to_html(escape=False, index=False), unsafe_allow_html=True)

st.subheader("📊 Select Coin for Full Chart + SL/TP")
selected = st.selectbox("Select Coin", [k.replace("/USDT:USDT","") for k in symbols])
full_sym = selected + "/USDT:USDT"
if full_sym in results:
    df = get_data(exchange, full_sym, tf)
    df = calculate_indicators(df)
    funding = get_funding_rate(exchange, full_sym)
    sig, col, price = get_signal(df, funding)
    fig = go.Figure(data=[go.Candlestick(x=df['ts'][-100:], open=df['o'][-100:], high=df['h'][-100:], low=df['l'][-100:], close=df['c'][-100:])])
    fig.add_trace(go.Scatter(x=df['ts'][-100:], y=df['ema9'][-100:], name="EMA9", line=dict(color="lime")))
    fig.add_trace(go.Scatter(x=df['ts'][-100:], y=df['ema21'][-100:], name="EMA21", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=df['ts'][-100:], y=df['st'][-100:], name="SuperTrend", line=dict(color="purple", width=3)))
    fig.update_layout(title=f"{selected} {tf_display} — {sig}", height=500)
    st.plotly_chart(fig, width='stretch')
    if sig != "WAIT":
        sl = price * (1 - 0.004) if "LONG" in sig else price * (1 + 0.004)
        tp1 = price * (1 + 0.01) if "LONG" in sig else price * (1 - 0.01)
        tp2 = price * (1 + 0.02) if "LONG" in sig else price * (1 - 0.02)
        atr = df['atr'].iloc[-1]
        st.success(f"**ENTRY NOW** | **SL**: {sl:,.4f} | **TP1 (50%)**: {tp1:,.4f} | **TP2**: {tp2:,.4f} (Trail remaining with ATR {atr:,.4f})")

# Simple Backtest Button (for BTC, last 1000 bars)
if st.button("Run Backtest on BTC (Normal Mode)"):
    back_df = get_data(exchange, "BTC/USDT:USDT", tf, limit=1000)
    back_df = calculate_indicators(back_df, a1_mode=False)
    trades = []
    for i in range(200, len(back_df)):
        sub_df = back_df.iloc[0:i]
        signal = get_signal(sub_df, 0)  # Assume neutral funding
        if signal[0] != "WAIT":
            entry = back_df['c'].iloc[i]
            direction = 1 if "LONG" in signal[0] else -1
            sl = entry * (1 - direction * 0.004)
            tp1 = entry * (1 + direction * 0.01)
            tp2 = entry * (1 + direction * 0.02)
            outcome = 0
            tp1_hit = False
            for j in range(i+1, len(back_df)):
                high = back_df['h'].iloc[j]
                low = back_df['l'].iloc[j]
                if direction > 0:
                    if low <= sl:
                        outcome -= 1
                        break
                    if high >= tp1 and not tp1_hit:
                        outcome += 0.5 * 1
                        tp1_hit = True
                        sl = entry + back_df['atr'].iloc[j]  # Trail
                    if high >= tp2:
                        outcome += 0.5 * 2
                        break
                else:
                    if high >= sl:
                        outcome -= 1
                        break
                    if low <= tp1 and not tp1_hit:
                        outcome += 0.5 * 1
                        tp1_hit = True
                        sl = entry - back_df['atr'].iloc[j]
                    if low <= tp2:
                        outcome += 0.5 * 2
                        break
            if outcome == 0:
                outcome = direction * (back_df['c'].iloc[-1] - entry) / entry * 100 / 0.4  # Normalize to RR
            trades.append(outcome)
    if trades:
        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades) * 100
        net_profit = sum(trades)
        st.write(f"Backtest Results: Trades: {len(trades)}, Win Rate: {win_rate:.2f}%, Net RR: {net_profit:.2f} (Profitable if >0)")
    else:
        st.write("No trades in this period.")

st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')} • Auto-refresh every {refresh_sec}s")
time.sleep(refresh_sec)
st.rerun()
