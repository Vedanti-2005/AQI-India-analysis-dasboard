# Statistical Analysis and AQI Prediction of Indian Cities Using Scraped Air Quality Data

This project fulfills the requirements of the Semester V SI Lab Project Report framework (mirroring the structure submitted for Ramdeobaba University, CSE - AI & Data Science).

## Project Overview
- **Data Source:** [AQI India Dashboard](https://www.aqi.in/in/dashboard/india)
- **Problem Statement:** Scraping multi-city air quality indicators (PM2.5, PM10, NO2, SO2, CO, Ozone) alongside meteorological observations (Temperature, Humidity, Wind Speed, Pressure), conducting descriptive and inferential statistical testing, fitting multiple linear regression models, and presenting findings via an interactive Streamlit dashboard.

---

## Included Files in this Bundle
1. `india_aqi_dashboard_data.csv`: Preprocessed dataset containing 1,875 timestamped records across 25 major Indian cities and 5 zones (North, South, East, West, Central).
2. `AQI_India_Statistical_Analysis_and_Prediction.ipynb`: Comprehensive Jupyter Notebook containing:
   - Module 1: Web Scraping & Synthetic Data Pipeline
   - Module 2: Data Preprocessing, missing-value checking, schema validation
   - Module 3: Descriptive Statistics (Mean, Median, Mode, Variance, Std, IQR, Skewness, Kurtosis)
   - Module 4: Exploratory Data Analysis (Histograms, KDE, Boxplots, Q-Q Plots)
   - Module 5: Correlation & Relationship Analysis (Covariance, Pearson Linear, Spearman Rank Heatmaps)
   - Module 6: Inferential Statistics & Hypothesis Testing (t-test for North vs South zones, ANOVA across time slots, Chi-Square test of independence)
   - Module 7: Multiple Linear Regression modeling with 80:20 chronological train-test split
   - Module 8: Model Assessment (MAE, MSE, RMSE, R² score, Actual vs Predicted, Residual diagnostics)
3. `app.py`: Full interactive Streamlit Dashboard featuring:
   - Live KPI Cards & City Bar Charts
   - Statistical Distribution & Correlation Heatmaps
   - Automated Hypothesis Testing Engine with p-values
   - Interactive ML AQI Predictor with slider inputs
4. `requirements.txt`: Python package requirements.

---

## How to Run

### 1. Jupyter Notebook
Open `AQI_India_Statistical_Analysis_and_Prediction.ipynb` in Google Colab or JupyterLab:
```bash
jupyter notebook AQI_India_Statistical_Analysis_and_Prediction.ipynb
```

### 2. Streamlit Dashboard
Install dependencies and launch the dashboard:
```bash
pip install -r requirements.txt
streamlit run app.py
```
