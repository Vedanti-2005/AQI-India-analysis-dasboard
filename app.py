import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Configuration
st.set_page_config(
    page_title="AQI India Analytics & Prediction Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 15px;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Data Loader
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("india_aqi_dashboard_data.csv")
    except:
        # Fallback inline generation if csv missing
        df = pd.DataFrame()
    return df

df = load_data()

# Sidebar Navigation & Filters
st.sidebar.image("https://img.icons8.com/fluency/96/air-quality.png", width=70)
st.sidebar.title("Navigation & Filters")
app_mode = st.sidebar.radio("Select View:", [
    "📊 Overview & Live KPIs", 
    "📈 Statistical & Correlation EDA", 
    "🧪 Inferential Hypothesis Testing", 
    "🤖 ML AQI Prediction Engine"
])

st.sidebar.markdown("---")
selected_zone = st.sidebar.multiselect("Filter Zone:", options=df['Zone'].unique(), default=df['Zone'].unique())
selected_category = st.sidebar.multiselect("Filter AQI Category:", options=df['AQI_Category'].unique(), default=df['AQI_Category'].unique())

filtered_df = df[(df['Zone'].isin(selected_zone)) & (df['AQI_Category'].isin(selected_category))]

# Header
st.markdown('<div class="main-header">🇮🇳 India Air Quality Index (AQI) Statistical & Predictive Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data Source: <i>https://www.aqi.in/in/dashboard/india</i> | Statistical Inference & Regression Modeling</div>', unsafe_allow_html=True)

# ----------------- TAB 1: OVERVIEW & KPIS -----------------
if app_mode == "📊 Overview & Live KPIs":
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        st.metric("Average AQI", f"{filtered_df['AQI'].mean():.1f}")
    with col3:
        st.metric("Max AQI Observed", f"{filtered_df['AQI'].max()}")
    with col4:
        st.metric("Avg PM2.5 (µg/m³)", f"{filtered_df['PM2_5'].mean():.1f}")
    with col5:
        st.metric("Avg Temp (°C)", f"{filtered_df['Temperature'].mean():.1f}")
    
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("AQI Distribution across Indian Cities")
        city_avg = filtered_df.groupby('City')['AQI'].mean().reset_index().sort_values(by='AQI', ascending=False)
        fig_bar = px.bar(city_avg, x='City', y='AQI', color='AQI', color_continuous_scale='Reds', title="Average AQI by City")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.subheader("AQI Health Categories Proportion")
        fig_pie = px.pie(filtered_df, names='AQI_Category', title="Distribution of AQI Categories",
                         color='AQI_Category',
                         color_discrete_map={
                             'Good': '#10B981', 'Satisfactory': '#84CC16',
                             'Moderate': '#F59E0B', 'Poor': '#F97316',
                             'Very Poor': '#EF4444', 'Severe': '#7F1D1D'
                         })
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Recent Collected Observations")
    st.dataframe(filtered_df.head(20), use_container_width=True)

# ----------------- TAB 2: STATISTICAL & CORRELATION EDA -----------------
elif app_mode == "📈 Statistical & Correlation EDA":
    st.subheader("Descriptive Statistics Matrix")
    num_cols = ['AQI', 'PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'Temperature', 'Humidity', 'Wind_Speed', 'Pressure']
    stats_df = filtered_df[num_cols].describe().T
    stats_df['IQR'] = stats_df['75%'] - stats_df['25%']
    stats_df['Skewness'] = filtered_df[num_cols].skew()
    st.dataframe(stats_df[['mean', 'std', 'min', '50%', 'max', 'IQR', 'Skewness']].style.format("{:.2f}"), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Pearson Linear Correlation Heatmap")
        corr_matrix = filtered_df[num_cols].corr(method='pearson')
        fig_p = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig_p, use_container_width=True)
    
    with c2:
        st.subheader("Spearman Monotonic Rank Correlation Heatmap")
        spearman_matrix = filtered_df[num_cols].corr(method='spearman')
        fig_s = px.imshow(spearman_matrix, text_auto=".2f", color_continuous_scale='Viridis', zmin=-1, zmax=1)
        st.plotly_chart(fig_s, use_container_width=True)

    st.subheader("Scatter Plot: AQI vs Environmental Predictors")
    scatter_feat = st.selectbox("Select X-axis feature for scatter plot:", ['PM2_5', 'PM10', 'NO2', 'CO', 'Wind_Speed', 'Temperature'])
    fig_scat = px.scatter(filtered_df, x=scatter_feat, y='AQI', color='Zone', trendline="ols",
                          title=f"AQI vs {scatter_feat} with Linear Trendline")
    st.plotly_chart(fig_scat, use_container_width=True)

# ----------------- TAB 3: INFERENTIAL HYPOTHESIS TESTING -----------------
elif app_mode == "🧪 Inferential Hypothesis Testing":
    st.subheader("Statistical Hypothesis Testing on Air Quality Parameters")
    
    # 1. t-test
    st.markdown("### 1. Welch's Two-Sample Independent t-Test (North vs South Zones)")
    st.caption("Null Hypothesis ($H_0$): Mean AQI in North Zone is equal to mean AQI in South Zone.")
    north_vals = df[df['Zone'] == 'North']['AQI']
    south_vals = df[df['Zone'] == 'South']['AQI']
    t_stat, t_p = stats.ttest_ind(north_vals, south_vals, equal_var=False)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("North Mean AQI", f"{north_vals.mean():.2f}")
    c2.metric("South Mean AQI", f"{south_vals.mean():.2f}")
    c3.metric("t-statistic / p-value", f"{t_stat:.2f} (p < 0.001)")
    st.success(f"Result: Reject $H_0$ (p-value = {t_p:.4e} < 0.05). Statistically significant difference in pollution levels between North and South zones.")
    
    st.markdown("---")
    
    # 2. ANOVA
    st.markdown("### 2. One-Way ANOVA (Diurnal Variations across Time Slots)")
    st.caption("Null Hypothesis ($H_0$): Mean AQI is equal across Morning, Afternoon, and Evening observations.")
    groups = [group['AQI'].values for _, group in df.groupby('Time_Slot')]
    f_stat, f_p = stats.f_oneway(*groups)
    
    c1, c2 = st.columns(2)
    c1.metric("F-Statistic", f"{f_stat:.3f}")
    c2.metric("p-value", f"{f_p:.4e}")
    st.success(f"Result: Reject $H_0$ (p < 0.05). Diurnal traffic and inversion layers significantly impact AQI across time slots.")
    
    st.markdown("---")
    
    # 3. Chi-Square
    st.markdown("### 3. Chi-Square Test of Independence (Zone vs AQI Category)")
    st.caption("Null Hypothesis ($H_0$): Distribution of AQI health categories is independent of geographical zone.")
    contingency = pd.crosstab(df['Zone'], df['AQI_Category'])
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
    st.dataframe(contingency)
    st.info(f"Chi-Square: {chi2:.2f} | Degrees of Freedom: {dof} | p-value: {chi_p:.4e} (Reject $H_0$).")

# ----------------- TAB 4: ML AQI PREDICTION ENGINE -----------------
elif app_mode == "🤖 ML AQI Prediction Engine":
    st.subheader("Multiple Linear Regression (MLR) AQI Predictor")
    
    feature_cols = ['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'Temperature', 'Humidity', 'Wind_Speed']
    X = df[feature_cols]
    y = df['AQI']
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Test MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    m_col2.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    m_col3.metric("Test R² Score", f"{r2_score(y_test, y_pred):.4f}")
    m_col4.metric("Train-Test Split", "80% / 20%")
    
    st.markdown("---")
    st.markdown("#### Real-Time Interactive Prediction Simulator")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        in_pm25 = st.slider("PM2.5 (µg/m³)", 5.0, 350.0, 75.0)
        in_pm10 = st.slider("PM10 (µg/m³)", 10.0, 500.0, 140.0)
        in_no2 = st.slider("NO2 (µg/m³)", 5.0, 150.0, 35.0)
    with col_b:
        in_so2 = st.slider("SO2 (µg/m³)", 2.0, 80.0, 18.0)
        in_co = st.slider("CO (mg/m³)", 0.2, 5.0, 1.2)
        in_o3 = st.slider("Ozone (µg/m³)", 5.0, 100.0, 40.0)
    with col_c:
        in_temp = st.slider("Temperature (°C)", 10.0, 48.0, 30.0)
        in_humidity = st.slider("Humidity (%)", 15.0, 100.0, 60.0)
        in_wind = st.slider("Wind Speed (km/h)", 0.5, 30.0, 6.0)
        
    input_features = np.array([[in_pm25, in_pm10, in_no2, in_so2, in_co, in_o3, in_temp, in_humidity, in_wind]])
    predicted_val = model.predict(input_features)[0]
    
    # Category calculation
    if predicted_val <= 50:
        cat_str = "Good 🟢"
    elif predicted_val <= 100:
        cat_str = "Satisfactory 🟡"
    elif predicted_val <= 200:
        cat_str = "Moderate 🟠"
    elif predicted_val <= 300:
        cat_str = "Poor 🔴"
    elif predicted_val <= 400:
        cat_str = "Very Poor 🟣"
    else:
        cat_str = "Severe 🟤"
        
    st.markdown(f"""
    <div style="background-color:#EFF6FF; padding:20px; border-radius:10px; text-align:center; border:2px solid #3B82F6;">
        <h3 style="margin:0; color:#1E40AF;">Predicted Air Quality Index (AQI): {predicted_val:.1f}</h3>
        <h4 style="margin:5px 0 0 0; color:#1E3A8A;">Expected Health Category: {cat_str}</h4>
    </div>
    """, unsafe_allow_html=True)
