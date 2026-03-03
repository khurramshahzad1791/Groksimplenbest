import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from ta import add_all_ta_features
from ta.utils import dropna

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="Crypto Stock Scanner",
    page_icon="📈",
    layout="wide"
)

# ------------------------------
# Helper functions with caching
# ------------------------------
@st.cache_resource
def get_exchange():
    """Create and return a CCXT exchange object (cached as resource)."""
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    return exchange

# Note: All cached functions that receive the exchange object use a leading underscore
# to tell Streamlit NOT to include that argument in the cache key.
@st.cache_data(ttl=300)  # cache for 5 minutes
def get_top_symbols(_exchange, limit=50):
    """
    Fetch top symbols by 24h volume from the exchange.
    The leading underscore in _exchange prevents hashing errors.
    """
    try:
        tickers = _exchange.fetch_tickers()
        # Filter out pairs that are not USDT or have no volume
        symbols = []
        for symbol, data in tickers.items():
            if symbol.endswith('/USDT') and data['quoteVolume'] is not None:
                symbols.append({
                    'symbol': symbol,
                    'volume': data['quoteVolume'],
                    'last': data['last'],
                    'change': data['percentage']
                })
        # Sort by volume descending and take top N
        symbols.sort(key=lambda x: x['volume'], reverse=True)
        return symbols[:limit]
    except Exception as e:
        st.error(f"Error fetching top symbols: {e}")
        return []

@st.cache_data(ttl=60)  # cache for 1 minute
def get_historical_data(_exchange, symbol, timeframe='1h', limit=100):
    """
    Fetch OHLCV data for a given symbol.
    """
    try:
        ohlcv = _exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.warning(f"Could not fetch data for {symbol}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def calculate_indicators(_exchange, symbol, df):
    """Add technical indicators to the dataframe."""
    if df.empty:
        return df
    # Clean data
    df = dropna(df)
    # Add all TA features
    df = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )
    return df

def scan_stocks(_exchange, symbol_list, min_volume=1_000_000):
    """
    Scan through symbols and compute signals.
    Returns a DataFrame with scan results.
    """
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, item in enumerate(symbol_list):
        symbol = item['symbol']
        status_text.text(f"Scanning {symbol}... ({i+1}/{len(symbol_list)})")
        progress_bar.progress((i+1)/len(symbol_list))

        # Fetch historical data
        df = get_historical_data(_exchange, symbol, timeframe='1h', limit=100)
        if df.empty:
            continue

        # Calculate indicators
        df = calculate_indicators(_exchange, symbol, df)
        if df.empty or len(df) < 50:
            continue

        # Latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Simple signal logic (customize as needed)
        signals = []
        if latest['close'] > latest['SMA_20'] and prev['close'] <= prev['SMA_20']:
            signals.append("SMA20 cross above")
        if latest['rsi'] < 30:
            signals.append("Oversold (RSI)")
        if latest['rsi'] > 70:
            signals.append("Overbought (RSI)")
        if latest['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 1.5:
            signals.append("High volume")

        results.append({
            'Symbol': symbol,
            'Price': latest['close'],
            'Volume (24h)': item['volume'],
            'Change %': item['change'],
            'RSI': round(latest['rsi'], 2),
            'SMA_20': round(latest['SMA_20'], 4),
            'SMA_50': round(latest['SMA_50'], 4),
            'Signal': ', '.join(signals) if signals else 'None'
        })

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(results)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Crypto Stock Scanner")
st.markdown("Scan top cryptocurrencies for technical signals.")

# Sidebar controls
st.sidebar.header("Settings")
num_symbols = st.sidebar.slider("Number of symbols to scan", 10, 200, 50)
min_volume = st.sidebar.number_input("Minimum 24h volume (USDT)", value=1_000_000, step=100_000)
timeframe = st.sidebar.selectbox("Timeframe", ['1h', '4h', '1d'], index=0)

# Get exchange (cached resource)
exchange = get_exchange()

# Fetch top symbols
with st.spinner("Fetching top symbols by volume..."):
    symbol_list = get_top_symbols(exchange, limit=num_symbols)

if not symbol_list:
    st.error("No symbols retrieved. Using fallback list.")
    # Fallback common symbols
    fallback = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    symbol_list = [{'symbol': s, 'volume': 0, 'last': 0, 'change': 0} for s in fallback]

st.success(f"Loaded {len(symbol_list)} symbols.")

# Start scan button
if st.button("🔍 Run Scan", type="primary"):
    results_df = scan_stocks(exchange, symbol_list, min_volume=min_volume)

    if not results_df.empty:
        st.subheader("Scan Results")
        # Filter out symbols with no signal or low volume
        filtered = results_df[results_df['Signal'] != 'None']
        if not filtered.empty:
            st.dataframe(filtered, use_container_width=True)
        else:
            st.info("No stocks with signals found.")
    else:
        st.warning("No data available.")

# Optional: display raw top symbols
with st.expander("View top symbols by volume"):
    if symbol_list:
        df_display = pd.DataFrame(symbol_list)
        st.dataframe(df_display, use_container_width=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Data provided by MEXC exchange via CCXT. Indicators calculated with TA library.")
