# FINAL-DATA-SCIENCE-CAPSTONE-PROJECT
# POWERPULSE KENYA: FORECASTING MONTHLY ELECTRICITY CONSUMPTION USING TIME SERIES ANALYSIS AND MACHINE LEARNING


## 1. Background
Electricity is essential for Kenya's households, businesses and industrial activities. Understanding historical electricity consumption patterns can help stakeholders identify trends, anticipate future consumption and make better-informed planning decisions.

PowerPulse Kenya is a data science project that will investigate historical monthly electricity sales in Kenya and develop forecasting models to predict future consumption. The project will use historical data published by the Kenya National Bureau of Statistics (KNBS), with particular attention to Kenya Power and Lighting Company (KPLC) electricity sales.

The project will combine exploratory data analysis, statistical testing, time series forecasting and machine learning to determine which forecasting approach produces the most accurate and useful predictions.

## 2. Problem Statement

Electricity consumption changes over time in response to seasonal patterns, economic activity and other factors. Without reliable forecasts, stakeholders may face difficulties anticipating changes in electricity usage and supporting effective planning.

Although historical electricity statistics are published by KNBS, these records must be compiled, verified and analysed before they can be used reliably for forecasting. Furthermore, different forecasting techniques may perform differently on Kenya's historical electricity sales data.

This project therefore seeks to determine whether historical monthly KPLC electricity sales data can support sufficiently accurate forecasts and to identify which forecasting methods perform best.

## 3. Main Objective

To develop and evaluate forecasting models for predicting Kenya's monthly KPLC electricity sales using historical data, statistical analysis and machine learning techniques.

## 4. Specific Objectives

1. To collect, clean and verify historical monthly electricity sales data for Kenya from official sources.
2. To explore long-term trends, seasonal patterns, unusual observations and changes in electricity sales over time.
3. To establish a baseline forecasting model for comparison with more advanced approaches.
4. To develop and compare selected statistical and machine learning forecasting models.
5. To evaluate model performance using appropriate forecasting metrics and identify the most suitable approach.
6. To develop an interactive PowerPulse dashboard for communicating historical patterns, forecasts and practical insights.

## 5. Research Questions

1. What trends and seasonal patterns are present in Kenya's historical monthly KPLC electricity sales?
2. Can historical monthly electricity sales data be used to produce sufficiently accurate forecasts?
3. Which selected statistical and machine learning models provide the best forecasting performance?
4. How do the models compare in terms of forecast errors and their ability to generalise to unseen months?
5. What insights from the historical analysis and forecasts could support electricity planning and decision-making?

## 6. Data Sources

The primary data source will be the **Kenya National Bureau of Statistics (KNBS)**, particularly its monthly Leading Economic Indicators reports and relevant statistical publications.

* KNBS Leading Economic Indicators archive: https://www.knbs.or.ke/reports_category/leading-economic-indicators/
* KNBS Statistical Abstracts: https://www.knbs.or.ke/statistical-abstracts/

The target variable will be monthly KPLC electricity sales, measured in gigawatt-hours (GWh), subject to verification of definitions and units across historical reporting periods.

The intended study period is January 2010 to March 2026, depending on the availability and verification of comparable historical records. The current working dataset covers January 2024 to March 2026 and is undergoing historical expansion and verification.

## 7. Methodology

The project will follow these stages:

**Stage 1: Data collection and verification**

Obtain historical records, reconcile reporting revisions, check missing months and duplicates, verify units, document sources and flag anomalies.

**Stage 2: Exploratory Data Analysis (EDA)**

Use descriptive statistics and visualisations to investigate long-term trends, monthly seasonality, year-on-year changes, outliers and potential structural breaks.

**Stage 3: Statistical analysis**

Assess stationarity and seasonal patterns using suitable statistical tests. Investigate unusual observations and possible changes in the data-generating process.

**Stage 4: Baseline forecasting**

Establish a simple naïve or seasonal-naïve forecast to provide a benchmark against which more advanced models will be evaluated.

**Stage 5: Model development and comparison**

Subject to the characteristics of the verified dataset, candidate methods will include:

* Seasonal ARIMA (SARIMA)
* Exponential smoothing (ETS)
* Linear regression with time and seasonal features
* Random Forest regression using appropriately constructed lagged and calendar features

Additional models may be considered if justified by the data. Model selection will depend on data sufficiency, assumptions and validation results.

**Stage 6: Model evaluation**

Use chronological train-validation-test splits or rolling-origin time series validation to prevent future information from leaking into model training. Compare models using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and, where appropriate, Mean Absolute Percentage Error (MAPE).

**Stage 7: Dashboard and reporting**

Develop a PowerPulse dashboard showing historical electricity sales, trends, seasonal patterns, forecast comparisons and model performance. Present findings, limitations and recommendations in a final report and presentation.

## 8. Tools and Technologies

| Tool                        | Purpose                                               |
| --------------------------- | ----------------------------------------------------- |
| Python                      | Data analysis and forecasting                         |
| Pandas and NumPy            | Data cleaning, manipulation and numerical computation |
| Matplotlib and Seaborn      | Exploratory visualisation                             |
| Statsmodels                 | Statistical tests, SARIMA and exponential smoothing   |
| Scikit-learn                | Regression models and performance evaluation          |
| Jupyter Notebook or VS Code | Development and documentation                         |
| Excel or CSV                | Initial data inspection and storage                   |
| Tableau or Power BI         | Interactive dashboard development                     |
| GitHub                      | Version control and project documentation             |

## 9. Expected Outcomes

The project is expected to produce:

1. A verified historical dataset of monthly KPLC electricity sales.
2. An exploratory analysis of trends, seasonality and unusual observations.
3. A comparison of baseline, statistical and machine learning forecasting methods.
4. An evidence-based assessment of the accuracy and limitations of the selected forecasting models.
5. An interactive PowerPulse dashboard and a final presentation of the findings.

The project will not assume that any particular model is superior in advance. Conclusions will be based on observed performance on appropriately held-out historical data.


## 10. Scope and Limitations

Potential limitations include historical reporting revisions, changes in data definitions, structural breaks and unusual events such as the COVID-19 period. These issues will be documented and considered during analysis and evaluation.

**Proposed project title:** PowerPulse Kenya: Forecasting Monthly Electricity Consumption Using Time Series Analysis and Machine Learning.

**Current status:** Historical data verification and expansion are in progress. Model training will begin only after the dataset is sufficiently complete and verified.
