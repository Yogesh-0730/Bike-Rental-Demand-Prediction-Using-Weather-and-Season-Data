# Bike Rental Demand Prediction Using Weather and Season Data

## CLA-3 — Machine Learning Case Study (Regression)

### 📋 Problem Statement

Predicting hourly bike rental demand based on weather conditions, seasonal factors, and temporal features using regression models. Accurate demand prediction helps bike-sharing operators optimize fleet distribution, reduce maintenance costs, and improve customer satisfaction.

### 📊 Dataset

- **Source**: [UCI Machine Learning Repository / Kaggle Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)
- **File**: `hour.csv`
- **Records**: 17,379 hourly observations (January 2011 – December 2012)
- **Target Variable**: `cnt` (Total bike rentals per hour)

#### Features

| Feature      | Description                                          |
|-------------|------------------------------------------------------|
| `season`     | Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)       |
| `yr`         | Year (0=2011, 1=2012)                                |
| `mnth`       | Month (1–12)                                         |
| `hr`         | Hour (0–23)                                          |
| `holiday`    | Whether the day is a holiday                         |
| `weekday`    | Day of the week (0–6)                                |
| `workingday` | Whether the day is a working day                     |
| `weathersit` | Weather condition (1=Clear, 2=Mist, 3=Light Rain, 4=Heavy Rain) |
| `temp`       | Normalized temperature in Celsius                    |
| `atemp`      | Normalized feeling temperature                       |
| `hum`        | Normalized humidity                                  |
| `windspeed`  | Normalized wind speed                                |

### 🛠️ Models Used

1. **Linear Regression** — Baseline model
2. **Decision Tree Regressor** — Non-linear tree-based model
3. **Random Forest Regressor** — Ensemble of decision trees
4. **Gradient Boosting Regressor** — Sequential boosting ensemble

### 📈 Results

| Model               | MAE    | RMSE   | R² Score |
|---------------------|--------|--------|----------|
| Linear Regression   | ~101   | ~142   | ~0.39    |
| Decision Tree       | ~41    | ~65    | ~0.87    |
| Random Forest       | ~33    | ~55    | ~0.91    |
| Gradient Boosting   | ~36    | ~56    | ~0.91    |

> *Note: Exact values may vary slightly. Run the code to get precise results.*

**Best Model**: Random Forest / Gradient Boosting (R² ≈ 0.91)

### 🔑 Key Findings

- **Hour of day** is the strongest predictor — demand peaks at 8 AM and 5–6 PM on working days
- **Temperature** has a strong positive correlation with bike rentals
- **Humidity** and **bad weather** significantly reduce demand
- **Seasonal effect**: Fall has the highest average demand; Spring has the lowest
- Tree-based ensemble methods dramatically outperform Linear Regression

### 🗂️ Project Structure

```
Bike-Rental-Demand-Prediction/
├── bike_rental_prediction.py    # Complete ML pipeline code
├── outputs/                     # Generated plots and visualizations
│   ├── 01_target_distribution.png
│   ├── 02_hourly_pattern.png
│   ├── 03_season_weather.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_temp_hum_scatter.png
│   ├── 06_workingday_pattern.png
│   ├── 07_model_comparison.png
│   ├── 08_actual_vs_predicted.png
│   ├── 09_residuals.png
│   └── 10_feature_importance.png
├── README.md
└── requirements.txt
```

### ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python bike_rental_prediction.py
```

### 📦 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn

### 📚 References

1. Fanaee-T, Hadi, and Gama, Joao. "Event labeling combining ensemble detectors and background knowledge." *Progress in Artificial Intelligence*, vol. 2, no. 2–3, pp. 113–127, 2014.
2. UCI Machine Learning Repository — Bike Sharing Dataset: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
3. Scikit-learn Documentation: https://scikit-learn.org/stable/

### 📝 License

This project is for academic purposes (CLA-3 Case Study Submission).
