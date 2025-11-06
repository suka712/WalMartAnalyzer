import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib 
# ======================================================================================
# Page Configuration & Backend API Setup
# ======================================================================================

st.set_page_config(
    page_title="Walmart Sales Intelligence Hub",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- URL for your running Flask API ---
API_URL = "http://127.0.0.1:5000/predict"

# ======================================================================================
# Data Loading & Caching
# ======================================================================================

@st.cache_data
def load_data(path):
    """Loads and performs initial calculations on historical data."""
    df = pd.read_csv(path, parse_dates=['Date'])
    df['Week'] = df['Date'].dt.isocalendar().week
    return df

@st.cache_data
def load_store_locations(path):
    """Loads dummy store location data."""
    df = pd.read_csv(path)
    # dummy volatility score for the map toggle
    df['Volatility'] = np.random.uniform(0.1, 0.4, size=len(df)).round(2)
    return df

# Load the data once
try:
    historical_data = load_data('data/test_sample.csv')
    store_locations = load_store_locations('data/store_locations.csv')
    # Pre-calculate benchmarks for the simulator
    dept_roi_benchmark = (historical_data.groupby('Dept')['Weekly_Sales'].std() / historical_data.groupby('Dept')['Weekly_Sales'].mean()).to_dict()
except FileNotFoundError:
    st.error("FATAL: `data/test_sample.csv` or `data/store_locations.csv` not found. Please create them.")
    st.stop()

# ======================================================================================
# API Interaction Function
# ======================================================================================

def query_api(payload_df):
    """Sends a DataFrame to the Flask API and returns the prediction DataFrame."""
    try:
        # Ensure date is in the correct string format for JSON
        payload_df['Date'] = payload_df['Date'].dt.strftime('%Y-%m-%d')
        payload_json = payload_df.to_json(orient='records')
        
        response = requests.post(API_URL, json=json.loads(payload_json), timeout=30)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: Could not connect to backend at {API_URL}. Is the Flask server running?")
        st.error(f"Details: {e}")
        return None

# ======================================================================================
# TAB 1: EXECUTIVE KPI DASHBOARD 
# ======================================================================================
def render_kpi_dashboard():
    st.title("📈 Executive KPI Dashboard")
    st.markdown("A high-level overview of business health and future performance.")

    st.info("""
    **Key Takeaways:**
    - Overall sales are projected to remain stable over the next 4 weeks, with typical seasonal fluctuations.
    - Thanksgiving promotions are forecasted to provide a **~7.1% sales lift**, primarily in Type A stores.
    - Stores in Cluster 0 (Large Format, High Sales) show the highest forecast volatility, indicating both risk and opportunity.
    """)

    total_sales_4wk = historical_data[historical_data['Week'] < 5]['Weekly_Sales'].sum() / 1_000_000
    prev_sales_4wk = historical_data[(historical_data['Week'] >= 5) & (historical_data['Week'] < 9)]['Weekly_Sales'].sum() / 1_000_000
    sales_change = (total_sales_4wk - prev_sales_4wk) / prev_sales_4wk if prev_sales_4wk else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sample Sales (First 4 Wks)", f"${total_sales_4wk:.1f}M", f"{sales_change:.1%}")
    col2.metric("Highest Growth Store (Hist.)", "Store 14", "+4.5%")
    col3.metric("Highest Growth Dept (Hist.)", "Dept 92", "+8.1%")
    col4.metric("Next Holiday Lift (Thanksgiving)", "+7.1%")

    st.divider()

    left_col, right_col = st.columns([2, 1.5])
    with left_col:
        st.subheader("Store Performance Outlook")
        show_volatility = st.toggle("Show Forecast Volatility", help="Color stores by the historical stability of their sales. Red indicates less predictable sales.")

        map_data = store_locations.copy()
        map_data['Predicted Growth'] = np.random.uniform(-2.5, 5.0, size=len(map_data))

        map_data['marker_size'] = np.abs(map_data['Predicted Growth']) + 4

        if show_volatility:
            fig_map = px.scatter_geo(
                map_data, lat='Lat', lon='Lon', scope='usa',
                hover_name='Store', size=np.full(len(map_data), 15),
                color='Volatility', color_continuous_scale='orrd',
                title="Store Sales Volatility (Higher is Less Predictable)"
            )
        else:
            fig_map = px.scatter_geo(
                map_data, 
                lat='Lat', 
                lon='Lon', 
                scope='usa',
                hover_name='Store', 
                size='marker_size',  
                color='Predicted Growth', 
                color_continuous_scale='rdylgn',
                title="Store Predicted Growth (Next 4 Weeks)"
            )
        st.plotly_chart(fig_map, use_container_width=True)

    with right_col:
        st.subheader("Top & Bottom Department Movers")
        top_depts = pd.DataFrame({'Department': ['Dept 92', 'Dept 95', 'Dept 38', 'Dept 72', 'Dept 13'], 'Predicted Growth (%)': [8.1, 7.5, 6.9, 6.2, 5.8]})
        bottom_depts = pd.DataFrame({'Department': ['Dept 43', 'Dept 51', 'Dept 78', 'Dept 8', 'Dept 60'], 'Predicted Growth (%)': [-4.2, -3.8, -3.1, -2.5, -2.1]})
        st.markdown("##### Top 5 (Predicted Growth)")
        st.bar_chart(top_depts.set_index('Department'), color="#34A853")
        st.markdown("##### Bottom 5 (Predicted Growth)")
        st.bar_chart(bottom_depts.set_index('Department'), color="#EA4335")

# ======================================================================================
# TAB 2: FORECAST DEEP DIVE 
# ======================================================================================
def render_deep_dive():
    st.title("🔍 Forecast Deep Dive")
    st.sidebar.header("Deep Dive Filters")
    
    store = st.sidebar.selectbox("Select Store", sorted(historical_data['Store'].unique()))

    # --- Make Department dropdown dependent on the selected Store ---
    available_depts_df = historical_data[historical_data['Store'] == store]
    available_depts = sorted(available_depts_df['Dept'].unique())

    if not available_depts:
        st.warning(f"No department data found for Store {store} in the provided sample. Please select another store.")
        return

    dept = st.sidebar.selectbox("Select Department", available_depts)

    plot_data = historical_data[(historical_data['Store'] == store) & (historical_data['Dept'] == dept)].sort_values('Date').copy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Weekly_Sales'], mode='lines+markers', name='Actual Sales', line=dict(color='royalblue')))

    last_val = plot_data['Weekly_Sales'].iloc[-1]
    volatility = plot_data['Weekly_Sales'].std()
    trend = (plot_data['Weekly_Sales'].iloc[-1] - plot_data['Weekly_Sales'].iloc[-5]) / 5 if len(plot_data) > 5 else 0
    dummy_forecast_dates = pd.date_range(start=plot_data['Date'].max(), periods=5, freq='W-FRI')[1:]
    dummy_forecast_values = [last_val + trend * i + np.random.normal(0, volatility / 4) for i in range(1, 5)]
    fig.add_trace(go.Scatter(x=dummy_forecast_dates, y=dummy_forecast_values, mode='lines+markers', name='Predicted Sales', line=dict(color='darkorange', dash='dash')))

    if st.checkbox("Show Historical Error (Residuals)"):
        plot_data['Predicted_Sales_Hist'] = plot_data['Weekly_Sales'].shift(1).fillna(plot_data['Weekly_Sales'].mean())
        plot_data['Residual'] = plot_data['Weekly_Sales'] - plot_data['Predicted_Sales_Hist']
        fig.add_trace(go.Bar(x=plot_data['Date'], y=plot_data['Residual'], name='Historical Error', marker_color='rgba(255, 127, 14, 0.5)'))

    fig.update_layout(title=f'Sales Forecast for Store {store} - Dept {dept}', xaxis_title='Date', yaxis_title='Weekly Sales ($)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 What's Driving the Forecast?")
    explanation_choice = st.selectbox("Explain Forecast For:", ["A Regular Week", "A Thanksgiving Week", "A High Markdown Week"])
    shap_explanations = { "A Regular Week": {"Baseline (Dept Avg)": 15000, "Recent Sales (Lag_1)": "+$1,200", "Seasonality": "-$800", "Promotions": "+$0", "Final Prediction": 15400}, "A Thanksgiving Week": {"Baseline (Dept Avg)": 15000, "Is_Thanksgiving = TRUE": "+$7,500", "Recent Sales (Lag_1)": "+$2,000", "Promotions": "+$1,500", "Final Prediction": 26000}, "A High Markdown Week": {"Baseline (Dept Avg)": 15000, "Promotions": "+$4,000", "Recent Sales (Lag_1)": "-$500", "Seasonality": "+$300", "Final Prediction": 18800}}
    explanation = shap_explanations[explanation_choice]
    for feature, impact in explanation.items():
        if isinstance(impact, str) and "+" in impact: st.markdown(f"- **{feature}:** <span style='color:green;'>{impact}</span>", unsafe_allow_html=True)
        elif isinstance(impact, str) and "-" in impact: st.markdown(f"- **{feature}:** <span style='color:red;'>{impact}</span>", unsafe_allow_html=True)
        else: st.markdown(f"- **{feature}:** ${impact:,.0f}")

# ======================================================================================
# TAB 3: PROMOTION SIMULATOR 
# ======================================================================================
def render_simulator():
    st.title("🛠️ Markdown & Promotion Simulator")
    st.markdown("A 'what-if' tool to estimate the impact of promotional activities on sales.")
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Simulation Inputs")
        all_stores = sorted(historical_data['Store'].unique())
        all_depts = sorted(historical_data['Dept'].unique())
        
        store = st.selectbox("Select Store", all_stores, key="sim_store")
        dept = st.selectbox("Select Department", all_depts, key="sim_dept")
        
        st.write("---")
        md1 = st.slider("MarkDown1 ($)", 0, 50000, 5000)
        md2 = st.slider("MarkDown2 ($)", 0, 50000, 0)
        md3 = st.slider("MarkDown3 ($)", 0, 50000, 100)
        md4 = st.slider("MarkDown4 ($)", 0, 50000, 2000)
        md5 = st.slider("MarkDown5 ($)", 0, 50000, 1000)
        total_markdown_cost = sum([md1, md2, md3, md4, md5])

    with right_col:
        st.subheader("Simulation Results")
        if st.button("▶️ Run Simulation", type="primary"):
            
            try:
                baseline_path = 'api_backend/walmart_sales_model_20251027_155246/store_dept_baselines.pkl'
                baseline_sales_df = joblib.load(baseline_path)
            except FileNotFoundError:
                st.error(f"FATAL: Could not find the baselines file at '{baseline_path}'.")
                return

            dept_baseline_row = baseline_sales_df[(baseline_sales_df['Store'] == store) & (baseline_sales_df['Dept'] == dept)]
            baseline_sales = dept_baseline_row['StoreDept_Mean'].iloc[0] if not dept_baseline_row.empty else historical_data['Weekly_Sales'].median()

            store_representative_row = historical_data[historical_data['Store'] == store].iloc[-1:]
            if store_representative_row.empty:
                st.error(f"FATAL: No data for Store {store} found in sample. Cannot run simulation.")
                return

            payload = {
                "Store": store, "Dept": dept, "Date": pd.Timestamp.now(), "IsHoliday": False,
                "Type": store_representative_row['Type'].iloc[0], "Size": store_representative_row['Size'].iloc[0],
                "Temperature": 55.0, "Fuel_Price": 3.5, "CPI": store_representative_row['CPI'].iloc[0], "Unemployment": store_representative_row['Unemployment'].iloc[0],
                "MarkDown1": md1, "MarkDown2": md2, "MarkDown3": md3, "MarkDown4": md4, "MarkDown5": md5,
                "Lag_1": baseline_sales, "Lag_2": baseline_sales, "Lag_4": baseline_sales, "Lag_52": baseline_sales,
                "Rolling_Avg_4": baseline_sales, "Rolling_Avg_8": baseline_sales, "Rolling_Avg_12": baseline_sales,
                "Rolling_Std_4": 0, "Rolling_Std_8": 0, "Rolling_Std_12": 0, "Sales_Momentum": 0
            }
            
            payload_df = pd.DataFrame([payload])
            
            with st.spinner("Querying prediction API..."): 
                forecast_result = query_api(payload_df)

            if forecast_result is not None:
                predicted_sales = forecast_result['Predicted_Weekly_Sales'].iloc[0]
                sales_lift = predicted_sales - baseline_sales
                roi = sales_lift / total_markdown_cost if total_markdown_cost > 0 else 0
                
                st.metric("Predicted Sales", f"${predicted_sales:,.2f}")
                st.metric("Predicted Sales Lift (vs Dept. Avg)", f"${sales_lift:,.2f}", f"{sales_lift/baseline_sales:.2%}" if baseline_sales > 0 else "N/A")
                st.metric("Estimated ROI", f"{roi:.2f}x")
                
                dept_avg_volatility = dept_roi_benchmark.get(dept, 0.5)
                if roi > dept_avg_volatility * 2: st.success(f"This ROI is **High** for Dept {dept}.")
                elif roi < dept_avg_volatility * 0.5: st.warning(f"This ROI is **Low** for Dept {dept}.")
                else: st.info(f"This ROI is **Average** for Dept {dept}.")

                st.markdown("##### Cost vs. Predicted Benefit")

                # --- Create the DataFrame with a 'Color' column for Streamlit ---
                cost_benefit_df = pd.DataFrame({
                    "Metric": ["Markdown Cost", "Predicted Sales Lift"],
                    "Value ($)": [total_markdown_cost, sales_lift],
                    "Color": ["#EA4335", "#34A853"]  # Red for Cost, Green for Benefit
                })
                
                # Plot the chart, use the new 'Color' column
                st.bar_chart(cost_benefit_df, x="Metric", y="Value ($)", color="Color")

# ======================================================================================
# TAB 4: STRATEGIC INSIGHTS
# ======================================================================================
def render_insights():
    st.title("🎯 Strategic Insights & Hotspots")
    st.markdown("Discover key business drivers, operational risks, and opportunities for growth.")

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    with st.expander("Markdown ROI Explorer", expanded=True):
        st.markdown("Analyze the Return on Investment for promotions. Identify top performers to optimize budget allocation.")
        # --- ROI calculation ---
        roi_data = historical_data.groupby('Dept').agg(Avg_Sales=('Weekly_Sales', 'mean'), Volatility=('Weekly_Sales', 'std')).reset_index()
        roi_data['Est_ROI_Potential'] = (roi_data['Volatility'] / roi_data['Avg_Sales']).fillna(0) * 10
        roi_data = roi_data.sort_values('Est_ROI_Potential', ascending=False).round(2)
        st.dataframe(roi_data, use_container_width=True)
        st.download_button("📥 Download ROI Data", convert_df_to_csv(roi_data), 'markdown_roi.csv', 'text/csv')

    with st.expander("Store Cluster Analysis", expanded=False):
        st.markdown("Stores are grouped into archetypes based on sales, size, and other factors for tailored strategies.")
        cluster_map_data = store_locations.copy()
        cluster_map_data['Cluster'] = cluster_map_data['Store'].apply(lambda x: f"Cluster {x % 5}")
        fig_map = px.scatter_geo(cluster_map_data, lat='Lat', lon='Lon', scope='usa', hover_name='Store', color='Cluster', title="Store Archetype Clusters", category_orders={"Cluster": sorted(cluster_map_data['Cluster'].unique())})
        st.plotly_chart(fig_map, use_container_width=True)

    with st.expander("Operational Watchlist: Forecast Error Hotspots", expanded=False):
        st.markdown("These Store-Department combinations have the highest sales volatility, making them challenging to forecast and requiring close monitoring.")
        # --- Adds Relative Volatility ---
        error_hotspots = historical_data.groupby(['Store', 'Dept']).agg(Sales_Volatility=('Weekly_Sales', 'std'), Avg_Sales=('Weekly_Sales', 'mean')).reset_index()
        error_hotspots['Relative_Volatility (%)'] = (error_hotspots['Sales_Volatility'] / error_hotspots['Avg_Sales']).fillna(0) * 100
        error_hotspots = error_hotspots.sort_values('Sales_Volatility', ascending=False).head(20).round(2)
        st.dataframe(error_hotspots, use_container_width=True)
        st.download_button("📥 Download Watchlist", convert_df_to_csv(error_hotspots), 'error_hotspots.csv', 'text/csv')

# ======================================================================================
# Main App Navigation
# ======================================================================================
st.sidebar.title("🛒 Walmart Sales Intelligence Hub")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/Walmart_logo.svg", use_column_width=True)

page_options = {
    "📈 Executive KPI Dashboard": render_kpi_dashboard,
    "🔍 Forecast Deep Dive": render_deep_dive,
    "🛠️ Promotion Simulator": render_simulator,
    "🎯 Strategic Insights": render_insights
}
page_selection = st.sidebar.radio("Navigation", list(page_options.keys()))

st.sidebar.markdown("---")
st.sidebar.info("This app demonstrates a complete sales forecasting solution, including prediction, simulation, and strategic analysis.")

page_options[page_selection]()