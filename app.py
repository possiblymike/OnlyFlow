import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Yahoo Finance GEX Table", layout="wide")
st.title("Net Gamma Exposure (GEX) Data Table")

# --- Auto-Refresh Logic (5 times per minute) ---
count = st_autorefresh(interval=12000, limit=None, key="yf_refresh")

# --- Sidebar UI Controls ---
st.sidebar.header("Screener Controls")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="SPY", max_chars=5).upper()

# --- Fetch Available Expirations ---
@st.cache_data(ttl=60)
def get_expirations(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    return ticker.options

expirations = get_expirations(ticker_input)

if not expirations:
    st.error(f"No options data found for {ticker_input}. Please check the ticker symbol.")
    st.stop()

selected_expiry = st.sidebar.selectbox("Expiration Date", expirations)

# --- Black-Scholes Gamma Calculation ---
def calculate_gamma(S, K, T, r, sigma):
    if T <= 0.0: T = 1e-5 
    if sigma <= 0.0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# --- Yahoo Finance API Engine ---
@st.cache_data(ttl=10) # 10-second cache to sync with the 12-second refresh
def get_yf_gex(ticker_symbol, expiry_date):
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        spot_price = ticker.fast_info['lastPrice']
    except:
        spot_price = ticker.history(period="1d")['Close'].iloc[-1]
        
    opt_chain = ticker.option_chain(expiry_date)
    calls = opt_chain.calls
    puts = opt_chain.puts
    
    # Calculate Time to Expiration (T)
    exp_date = datetime.strptime(expiry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days_to_exp = (exp_date - now).days
    T = max(days_to_exp / 365.0, 1e-5) 
    
    risk_free_rate = 0.05 
    contract_multiplier = 100
    
    # Process Greeks & GEX
    calls['gamma'] = calls.apply(lambda row: calculate_gamma(spot_price, row['strike'], T, risk_free_rate, row['impliedVolatility']), axis=1)
    calls['Call GEX'] = calls['gamma'] * calls['openInterest'] * contract_multiplier * spot_price
    
    puts['gamma'] = puts.apply(lambda row: calculate_gamma(spot_price, row['strike'], T, risk_free_rate, row['impliedVolatility']), axis=1)
    puts['Put GEX'] = puts['gamma'] * puts['openInterest'] * contract_multiplier * spot_price * -1
    
    # Merge and Calculate Net GEX
    df = pd.merge(calls[['strike', 'Call GEX']], puts[['strike', 'Put GEX']], on='strike', how='outer').fillna(0)
    df.rename(columns={'strike': 'Strike'}, inplace=True)
    df['Net GEX'] = df['Call GEX'] + df['Put GEX']
    
    # Filter bounds (+/- 10% of spot)
    df = df[(df['Strike'] >= spot_price * 0.90) & (df['Strike'] <= spot_price * 1.10)]
    
    # Sort by Strike descending
    df = df.sort_values(by="Strike", ascending=False).reset_index(drop=True)
    
    return df, spot_price

# --- Styling Function for the Table ---
def style_dataframe(df, spot_price):
    if df.empty: return df.style
    
    closest_idx = (df['Strike'] - spot_price).abs().idxmin()
    
    def apply_row_styles(row):
        if row.name == closest_idx:
            return ['background-color: #374151; font-weight: bold'] * len(row)
        return [''] * len(row)
        
    def color_gex(val):
        if val > 0:
            return 'color: #10B981;'
        elif val < 0:
            return 'color: #EF4444;'
        return 'color: #9CA3AF;'

    styled_df = (df.style
        .apply(apply_row_styles, axis=1)
        .map(color_gex, subset=['Call GEX', 'Put GEX', 'Net GEX'])
        .format({
            'Strike': '${:.2f}', 
            'Call GEX': '${:,.0f}', 
            'Put GEX': '${:,.0f}', 
            'Net GEX': '${:,.0f}'
        })
    )
    return styled_df

# --- App Execution ---
with st.spinner(f"Pulling options chain for {ticker_input}..."):
    df, current_spot = get_yf_gex(ticker_input, selected_expiry)

if df is not None and not df.empty:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**{ticker_input} Options** • Expiry: `{selected_expiry}`")
    with col2:
        st.markdown(f"Spot Price: **`${current_spot:.2f}`**")
        st.caption(f"Auto-refreshing 5x/min (Update #{count})")

    # Apply styles and render the table
    styled_table = style_dataframe(df, current_spot)
    
    st.dataframe(
        styled_table,
        use_container_width=True,
        height=600,
        hide_index=True
    )
else:
    st.warning("No valid GEX data found for this selection.")
