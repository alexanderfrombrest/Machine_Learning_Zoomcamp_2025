# Project Description: Automated Real Estate Valuation Model

## 1. Problem Definition and Business Goal

The objective of this project is to build an **Automated Valuation Model (AVM)** for the real estate market in Poland, specifically focusing on the Warsaw area.

The core business problem is the **inefficiency of identifying undervalued assets**. 
In 2025 housing market, private buyers often rely on human intuition and cannot eficiently assess the price of property of interest. Futhermore, the amount of offers buyers need to analyze daily is enormous and with often dynamic prices, it is extremelly important to have automated service or model that will point buyers attention on some specific, undervalued offers.

The solution aims to provide a reliable, objective, and near real-time "Fair Value" estimate for any given property listing, allowing users to **identify undervalued assets.** 

## 2. Machine Learning Formulation

This project is framed as a **Supervised Regression** problem:

* **Input (Features - X):** Structural, locational, and convinience-based property characteristics (e.g., `area`, `buildYear`, `floor_numeric`, `distance_from_center`, and engineered boolean features like `has_elevator`, `has_terrace`).
* **Output (Target - y):** The logarithm of the property's asking price (`price_log`).
* **Model:** An optimized **Extreme Gradient Boosting (XGBoost)** Regressor, which is highly effective at capturing the non-linear feature interactions common in real estate data.

## 3. Data Processing and Evaluation

The model was developed using a dataset of **20,000+ flat listings** by cleaning, outlier removal, and feature engineering, including:

* **Target Transformation:** Using $\ln(1 + \text{price})$ to stabilize the price distribution and train the model to minimize **percentage error** rather than absolute error.
* **High-Cardinality Encoding:** Employing **Target Encoding** for the high-cardinality `location_district` feature to efficiently capture the value hierarchy of different city zones.