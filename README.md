# 🌍 Tourist Demand Prediction Based on Destination Weather

Can weather conditions predict how many tourists a country receives? This project investigates the relationship between destination climate and international tourist arrivals using regression models and SHAP-based feature analysis.

## Overview

**Question:** Do temperature, precipitation, and humidity explain international tourist volumes at the country level?

**Approach:**
1. Merge two independent datasets on country — weather snapshots and historical tourist arrivals
2. Explore distributions and feature relationships through EDA
3. Train and compare multiple regression models across three rounds of increasing complexity
4. Interpret results with SHAP feature importance

**Short answer:** Not really — and that's an interesting finding. Country identity dominates over raw climate features, suggesting destination-specific factors (culture, infrastructure, visa policy) matter far more than weather alone.

## Data Sources

| Dataset | Source | Features |
|---------|--------|----------|
| Global Weather Repository | Kaggle | Temperature (°C), precipitation (mm), humidity per country |
| International Tourist Arrivals | Our World in Data | Annual tourist arrivals by country and year |

## Methods

**Preprocessing**
- Aggregated weather readings to country-year means
- Log-transformed tourist arrivals to handle heavy right skew (France, Spain, US dominate raw counts)
- One-hot encoded country dummies for Round 3 models

**Models Compared**

| Round | Features | Models |
|-------|----------|--------|
| Round 1 | Weather only (temp, precip, humidity) | Linear Regression, Random Forest |
| Round 2 | Weather + log transform | Linear Regression, Random Forest |
| Round 3 | Weather + country dummies | Linear Regression, Random Forest |

**Evaluation:** RMSE and R² on held-out test set

## Key Findings

- **Weather features alone are weak predictors** of tourist volumes (low R² in Rounds 1–2)
- **Adding country identity (Round 3) dramatically improved model performance** — confirming that destination-specific characteristics dominate
- SHAP analysis showed country dummies consistently ranked as the top features, with temperature and precipitation contributing marginally
- Tourist arrivals are heavily right-skewed — log transformation was essential for stable model training

## Limitations & Honest Reflection

- The Global Weather Repository covers only a single year, so every year of tourism data for a country gets the same weather values — year-to-year variation in arrivals can't be explained by weather
- Aggregating all weather readings to a single country mean discards regional and seasonal variation
- The merged dataset after inner join is relatively small, limiting model capacity

These limitations are acknowledged in the notebook and motivate clear directions for future work.

## Tech Stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `shap`

## Setup

```bash
git clone https://github.com/2005yoheiUBC/tourist-demand-prediction
cd tourist-demand-prediction
pip install -r requirements.txt
jupyter notebook notebooks/travel_analysis.ipynb
```

## Project Structure

```
tourist-demand-prediction/
├── data/
│   ├── GlobalWeatherRepository.csv
│   └── international-tourist-trips.csv
├── notebooks/
│   └── travel_analysis.ipynb
├── requirements.txt
└── README.md
```

## Future Work

- Incorporate economic indicators (GDP per capita, flight connectivity)
- Use multi-year weather data to capture year-over-year climate trends
- Add seasonality features (monthly breakdown instead of annual means)
- Experiment with gradient boosting models (XGBoost, LightGBM)

