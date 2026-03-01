import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="GEX Heatseeker UI", layout="wide")
st.title("Interactive Net Gamma Exposure (GEX)")

# --- Sidebar UI Controls ---
st.sidebar.header("Screener Controls")
ticker_input = st.sidebar.text_input("Ticker Symbol", value="SPY", max_chars=5).upper()

# --- Black-Scholes Gamma Calculation ---
def calculate_gamma(S, K, T, r, sigma):
    if T <= 0.0: T = 1e-5 
    if sigma <= 0.0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

# --- Fetch Available Expirations ---
@st.cache_data(ttl=60) # Cache for 1 minute to keep UI snappy
def get_expirations(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    return ticker.options

expirations = get_expirations(ticker_input)

if not expirations:
    st.error(f"No options data found for {ticker_input}. Please check the ticker symbol.")
    st.stop()

selected_expiry = st.sidebar.selectbox("Expiration Date", expirations)

# Manual refresh button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()

# --- Data Fetching Engine ---
@st.cache_data(ttl=60) # Short cache to allow near real-time updates when refreshed
def get_gex_data(ticker_symbol, expiry_date):
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
    calls['call_gex'] = calls['gamma'] * calls['openInterest'] * contract_multiplier * spot_price
    
    puts['gamma'] = puts.apply(lambda row: calculate_gamma(spot_price, row['strike'], T, risk_free_rate, row['impliedVolatility']), axis=1)
    puts['put_gex'] = puts['gamma'] * puts['openInterest'] * contract_multiplier * spot_price * -1
    
    # Merge and Calculate Net GEX
    df = pd.merge(calls[['strike', 'call_gex']], puts[['strike', 'put_gex']], on='strike', how='outer').fillna(0)
    df['net_gex'] = df['call_gex'] + df['put_gex']
    
    # Filter bounds (+/- 10% of spot for a wider view)
    df = df[(df['strike'] >= spot_price * 0.90) & (df['strike'] <= spot_price * 1.10)]
    
    return df, spot_price

# --- App Execution ---
with st.spinner(f"Pulling options chain for {ticker_input}..."):
    df, current_spot = get_gex_data(ticker_input, selected_expiry)

st.markdown(f"**{ticker_input} Options** • Expiry: `{selected_expiry}` • Spot: `{current_spot:.2f}`")

# --- Plotly Visualization ---
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df['strike'], y=df['call_gex'], name='Call GEX', marker_color='#10B981'
))

fig.add_trace(go.Bar(
    x=df['strike'], y=df['put_gex'], name='Put GEX', marker_color='#EF4444'
))

fig.update_layout(
    barmode='relative',
    plot_bgcolor='#111827',
    paper_bgcolor='#111827',
    font=dict(color='white'),
    xaxis=dict(title="Strike Price", showgrid=False),
    yaxis=dict(title="GEX ($)", showgrid=True, gridcolor='#374151'),
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20)
)

fig.add_vline(x=current_spot, line_dash="dash", line_color="#FBBF24", 
              annotation_text="Spot Price", annotation_position="top right")

st.plotly_chart(fig, use_container_width=True)
