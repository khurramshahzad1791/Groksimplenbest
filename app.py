import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Pro Crypto Scanner",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# Cached resources
# ------------------------------
@st.cache_resource
def get_exchange(exchange_id='mexc'):
    """Return a CCXT exchange object (cached resource)."""
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    return exchange

# ------------------------------
# Data fetching (underscore to skip hashing)
# ------------------------------
@st.cache_data(ttl=300)
def fetch_top_symbols(_exchange, limit=100, quote='USDT'):
    """Fetch top symbols by 24h volume."""
    try:
        tickers = _exchange.fetch_tickers()
        symbols = []
        for sym, data in tickers.items():
            if sym.endswith(f'/{quote}') and data['quoteVolume'] is not None:
                symbols.append({
                    'symbol': sym,
                    'volume': data['quoteVolume'],
                    'last': data['last'],
                    'change': data['percentage']
                })
        symbols.sort(key=lambda x: x['volume'], reverse=True)
        return symbols[:limit]
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return []

@st.cache_data(ttl=60)
def fetch_ohlcv(_exchange, symbol, timeframe='15m', limit=200):
    """Fetch OHLCV data."""
    try:
        ohlcv = _exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.astype(float)
    except Exception as e:
        return pd.DataFrame()

# ------------------------------
# Indicator calculation
# ------------------------------
def calculate_indicators(df):
    """Add technical indicators using pandas_ta."""
    if df.empty or len(df) < 50:
        return df

    # Moving averages
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['EMA_20'] = ta.ema(df['close'], length=20)

    # RSI
    df['RSI'] = ta.rsi(df['close'], length=14)

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = df.join(macd)

    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None:
        df = df.join(bbands)

    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if stoch is not None:
        df = df.join(stoch)

    # Volume SMA
    df['Volume_SMA'] = ta.sma(df['volume'], length=20)

    # ATR for stop loss / take profit
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    df.dropna(inplace=True)
    return df

# ------------------------------
# Helper to find dynamic column names
# ------------------------------
def find_columns(df, prefix):
    """Return the first column that starts with the given prefix."""
    cols = [col for col in df.columns if col.startswith(prefix)]
    return cols[0] if cols else None

# ------------------------------
# Grade and probability based on strength
# ------------------------------
def get_grade_and_prob(strength):
    if strength >= 6:
        return 'A+', 90
    elif strength == 5:
        return 'A', 80
    elif strength == 4:
        return 'B+', 70
    elif strength == 3:
        return 'B', 60
    elif strength == 2:
        return 'C+', 50
    elif strength == 1:
        return 'C', 40
    else:
        return 'D', 20

# ------------------------------
# Signal generation with trade levels
# ------------------------------
def generate_signal(df):
    """
    Returns a dictionary with signal details:
    - signal: 'BUY', 'SELL', 'NEUTRAL'
    - strength: integer score
    - description: string
    - grade: A+, A, B, C, etc.
    - probability: confidence %
    - entry: current close
    - stop_loss: price level
    - take_profit_1: first target
    - take_profit_2: second target
    """
    if df.empty or len(df) < 10:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    # Dynamically find column names
    macd_col = find_columns(df, 'MACD_')
    macd_signal_col = find_columns(df, 'MACDs_')
    stoch_k_col = find_columns(df, 'STOCHk_')
    stoch_d_col = find_columns(df, 'STOCHd_')
    bb_lower = find_columns(df, 'BBL_')
    bb_upper = find_columns(df, 'BBU_')

    signals = []
    strength = 0

    # Trend
    if latest['close'] > latest['SMA_20']:
        signals.append("Price above SMA20")
        strength += 1
    if latest['SMA_20'] > latest['SMA_50']:
        signals.append("Golden cross (SMA20 > SMA50)")
        strength += 1

    # RSI
    if latest['RSI'] < 30:
        signals.append("Oversold (RSI < 30)")
        strength += 2
    elif latest['RSI'] > 70:
        signals.append("Overbought (RSI > 70)")
        strength -= 1

    # MACD
    if macd_col and macd_signal_col:
        if latest[macd_col] > latest[macd_signal_col] and (prev is None or prev[macd_col] <= prev[macd_signal_col]):
            signals.append("MACD bullish cross")
            strength += 2
        elif latest[macd_col] < latest[macd_signal_col] and (prev is None or prev[macd_col] >= prev[macd_signal_col]):
            signals.append("MACD bearish cross")
            strength -= 2

    # Bollinger Bands
    if bb_lower and bb_upper:
        if latest['close'] < latest[bb_lower]:
            signals.append("Below lower Bollinger")
            strength += 2
        elif latest['close'] > latest[bb_upper]:
            signals.append("Above upper Bollinger")
            strength -= 1

    # Stochastic
    if stoch_k_col and stoch_d_col:
        if latest[stoch_k_col] < 20 and latest[stoch_k_col] > latest[stoch_d_col]:
            signals.append("Stochastic oversold & rising")
            strength += 1
        elif latest[stoch_k_col] > 80 and latest[stoch_k_col] < latest[stoch_d_col]:
            signals.append("Stochastic overbought & falling")
            strength -= 1

    # Volume spike
    if latest['volume'] > 1.5 * latest['Volume_SMA']:
        signals.append("High volume")
        strength += 1

    # Determine signal type
    if strength >= 2:
        signal_type = "BUY"
    elif strength <= -2:
        signal_type = "SELL"
    else:
        signal_type = "NEUTRAL"

    # Grade & probability
    grade, probability = get_grade_and_prob(abs(strength))

    # Calculate entry, stop loss, take profit (based on ATR)
    atr = latest['ATR']
    entry = latest['close']

    if signal_type == "BUY":
        stop_loss = entry - 1.5 * atr
        take_profit_1 = entry + 2.0 * atr
        take_profit_2 = entry + 3.0 * atr
    elif signal_type == "SELL":
        stop_loss = entry + 1.5 * atr
        take_profit_1 = entry - 2.0 * atr
        take_profit_2 = entry - 3.0 * atr
    else:
        stop_loss = take_profit_1 = take_profit_2 = np.nan

    return {
        'signal': signal_type,
        'strength': strength,
        'description': ", ".join(signals) if signals else "No clear signal",
        'grade': grade,
        'probability': probability,
        'entry': entry,
        'stop_loss': stop_loss,
        'take_profit_1': take_profit_1,
        'take_profit_2': take_profit_2,
        'risk_reward_ratio': round((take_profit_1 - entry) / (entry - stop_loss), 2) if signal_type == "BUY" and not pd.isna(stop_loss) else (
            round((entry - take_profit_1) / (stop_loss - entry), 2) if signal_type == "SELL" and not pd.isna(stop_loss) else np.nan)
    }

# ------------------------------
# Scan function
# ------------------------------
def scan_symbols(_exchange, symbol_list, min_volume=1_000_000, timeframe='15m'):
    results = []
    progress_bar = st.progress(0)
    status = st.empty()

    for i, item in enumerate(symbol_list):
        symbol = item['symbol']
        status.text(f"Scanning {symbol} ({i+1}/{len(symbol_list)})...")
        progress_bar.progress((i+1)/len(symbol_list))

        df = fetch_ohlcv(_exchange, symbol, timeframe=timeframe, limit=200)
        if df.empty:
            continue

        df = calculate_indicators(df)
        if df.empty:
            continue

        signal_dict = generate_signal(df)
        if signal_dict is None:
            continue

        results.append({
            'Symbol': symbol,
            'Price': round(signal_dict['entry'], 6),
            'Volume (24h)': f"${item['volume']:,.0f}",
            'Signal': signal_dict['signal'],
            'Grade': signal_dict['grade'],
            'Probability': f"{signal_dict['probability']}%",
            'Entry': round(signal_dict['entry'], 6),
            'Stop Loss': round(signal_dict['stop_loss'], 6) if not pd.isna(signal_dict['stop_loss']) else '-',
            'TP1': round(signal_dict['take_profit_1'], 6) if not pd.isna(signal_dict['take_profit_1']) else '-',
            'TP2': round(signal_dict['take_profit_2'], 6) if not pd.isna(signal_dict['take_profit_2']) else '-',
            'R/R': signal_dict['risk_reward_ratio'] if not pd.isna(signal_dict['risk_reward_ratio']) else '-',
            'Strength': signal_dict['strength'],
            'Description': signal_dict['description']
        })

    progress_bar.empty()
    status.empty()
    return pd.DataFrame(results)

# ------------------------------
# Plot function
# ------------------------------
def plot_symbol(_exchange, symbol, timeframe='15m'):
    df = fetch_ohlcv(_exchange, symbol, timeframe=timeframe, limit=200)
    if df.empty:
        return None
    df = calculate_indicators(df)
    if df.empty:
        return None

    # Find dynamic column names
    macd_col = find_columns(df, 'MACD_')
    macd_signal_col = find_columns(df, 'MACDs_')
    macd_hist_col = find_columns(df, 'MACDh_')

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.15, 0.15])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                  open=df['open'],
                                  high=df['high'],
                                  low=df['low'],
                                  close=df['close'],
                                  name='Price'), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1), name='SMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='blue', width=1), name='SMA50'), row=1, col=1)

    # Volume
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='black'), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # MACD
    if macd_col and macd_signal_col and macd_hist_col:
        fig.add_trace(go.Scatter(x=df.index, y=df[macd_col], line=dict(color='blue'), name='MACD'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[macd_signal_col], line=dict(color='red'), name='Signal'), row=4, col=1)
        colors_macd = ['green' if val >= 0 else 'red' for val in df[macd_hist_col]]
        fig.add_trace(go.Bar(x=df.index, y=df[macd_hist_col], name='Histogram', marker_color=colors_macd), row=4, col=1)

    fig.update_layout(title=f'{symbol} - {timeframe} chart', xaxis_rangeslider_visible=False, height=800)
    return fig

# ------------------------------
# Main app
# ------------------------------
st.title("📈 Pro Crypto Day Trading Scanner")
st.markdown("Scan top cryptocurrencies for high-probability trade setups with entry, stop loss, and take profit levels.")

with st.sidebar:
    st.header("Settings")
    exchange_id = st.selectbox("Exchange", ['mexc', 'binance', 'kucoin', 'bybit'], index=0)
    quote = st.selectbox("Quote currency", ['USDT', 'BTC', 'ETH'], index=0)
    num_symbols = st.slider("Number of symbols to scan", 10, 200, 50, step=5)
    timeframe = st.selectbox("Timeframe", ['1m', '5m', '15m', '30m', '1h', '4h', '1d'], index=2)
    min_volume = st.number_input("Minimum 24h volume (USDT)", value=1_000_000, step=100_000, format="%d")
    st.markdown("---")
    st.header("Signal Filters")
    show_buy = st.checkbox("Show only BUY signals", value=True)
    show_sell = st.checkbox("Show only SELL signals", value=False)
    min_grade = st.selectbox("Minimum grade", ['D', 'C', 'C+', 'B', 'B+', 'A', 'A+'], index=2)  # default C
    run_scan = st.button("🔍 Run Scan", type="primary")

exchange = get_exchange(exchange_id)

with st.spinner("Fetching top symbols..."):
    symbol_list = fetch_top_symbols(exchange, limit=num_symbols, quote=quote)

if not symbol_list:
    st.error("Could not fetch symbols. Using fallback list.")
    fallback = [f"{s}/{quote}" for s in ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'MATIC', 'DOT', 'LTC']]
    symbol_list = [{'symbol': s, 'volume': 0, 'last': 0, 'change': 0} for s in fallback]

st.success(f"Loaded {len(symbol_list)} symbols.")

if run_scan:
    results_df = scan_symbols(exchange, symbol_list, min_volume=min_volume, timeframe=timeframe)

    if results_df.empty:
        st.warning("No signals generated.")
    else:
        # Apply filters
        filtered = results_df.copy()
        if show_buy and not show_sell:
            filtered = filtered[filtered['Signal'] == 'BUY']
        elif show_sell and not show_buy:
            filtered = filtered[filtered['Signal'] == 'SELL']
        elif show_buy and show_sell:
            filtered = filtered[filtered['Signal'].isin(['BUY', 'SELL'])]

        # Grade filter
        grade_order = {'D':0, 'C':1, 'C+':2, 'B':3, 'B+':4, 'A':5, 'A+':6}
        min_grade_val = grade_order[min_grade]
        filtered = filtered[filtered['Grade'].map(grade_order) >= min_grade_val]

        st.subheader(f"Scan Results ({len(filtered)} signals)")
        # Color rows by signal
        def color_signal(val):
            color = 'green' if val == 'BUY' else 'red' if val == 'SELL' else 'gray'
            return f'background-color: {color}; color: white'

        styled = filtered.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled, width='stretch', height=600)  # updated

        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download results as CSV", csv, "scan_results.csv", "text/csv")

st.markdown("---")
st.subheader("📉 Detailed Chart")
chart_symbol = st.selectbox("Select a symbol to view chart", [s['symbol'] for s in symbol_list], index=0)
if chart_symbol:
    fig = plot_symbol(exchange, chart_symbol, timeframe=timeframe)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not load chart data.")

st.markdown("---")
st.caption("Data provided by CCXT. Indicators: pandas_ta. Trade at your own risk.")
