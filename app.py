import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Configure Streamlit page
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS styling for a modern dashboard look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #007bff;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("📈 Demand Forecasting & Analytics Hub")
st.markdown("Welcome to the **Interactive Demand Forecasting Dashboard**. This platform leverages your pre-trained AutoARIMA model to project future demand.")

st.divider()

# Load Data and Model
@st.cache_data
def load_data():
    df = pd.read_csv("/content/demand_data.csv")
    df['Month'] = pd.to_datetime(df['Month'], format='%b-%Y')
    df.rename(columns={'Month': 'Month-Year'}, inplace=True)
    df.set_index('Month-Year', inplace=True)
    return df

@st.cache_resource
def load_model():
    return joblib.load("/content/arima_model.joblib")

try:
    df = load_data()
    model = load_model()
    
    # Layout: Sidebar Controls
    st.sidebar.header("⚙️ Configuration")
    forecast_steps = st.sidebar.slider(
        "Select Months to Forecast",
        min_value=1,
        max_value=24,
        value=12,
        step=1
    )
    
    # Generate Forecasts
    forecast_series = model.predict(n_periods=forecast_steps)
    
    # Create a nice forecast DataFrame
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
    forecast_df = pd.DataFrame({'Forecasted Demand (000L)': forecast_series.values}, index=forecast_dates)
    forecast_df.index.name = 'Month-Year'
    
    # Main Dashboard Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'><h4>Latest Actual Demand</h4><h2>{}K L</h2></div>".format(int(df['Demand_000L'].iloc[-1])), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h4>Max Projected Demand</h4><h2>{}K L</h2></div>".format(int(forecast_df['Forecasted Demand (000L)'].max())), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card' style='border-left-color: #28a745;'><h4>ARIMA Model Status</h4><h2>Active (MAPE: 2.97%)</h2></div>", unsafe_allow_html=True)
        
    st.write("")
    
    # Plotly interactive chart
    st.subheader("🔮 Demand Forecast Visualization")
    
    fig = go.Figure()
    
    # Historical Trace
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Demand_000L'],
        mode='lines+markers',
        name='Historical Demand',
        line=dict(color='#007bff', width=3)
    ))
    
    # Forecasted Trace
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Forecasted Demand (000L)'],
        mode='lines+markers',
        name='Forecasted Demand',
        line=dict(color='#ff7f0e', width=3, dash='dash')
    ))
    
    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Demand (000L)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Breakdown
    st.subheader("📊 Detailed Data view")
    tab1, tab2 = st.tabs(["🔮 Forecast Table", "📈 Historical Data"])
    
    with tab1:
        st.dataframe(forecast_df.style.format("{:.2f}"), use_container_width=True)
    with tab2:
        st.dataframe(df, use_container_width=True)
        
except Exception as e:
    st.error(f"Error loading dashboard components: {e}")
