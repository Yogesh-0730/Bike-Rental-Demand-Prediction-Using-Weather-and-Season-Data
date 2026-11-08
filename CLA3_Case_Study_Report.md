# CLA-3 — Case Study Report

**Student Template**

| | |
|---|---|
| **Case study title** | Bike Rental Demand Prediction Using Weather and Season Data |
| **Base paper** | Fanaee-T, H. and Gama, J., "Event labeling combining ensemble detectors and background knowledge," *Progress in Artificial Intelligence*, vol. 2, no. 2–3, pp. 113–127, 2014. |
| **Date of submission** | 07-11-2026 |

| Roll No. | Name |
|---|---|
| [Your Roll No.] | Yogesh |

> [!NOTE]
> **Instructions**: Length 3–5 pages. This report documents a **Regression** problem solved using four supervised learning models on the Bike Sharing dataset.

---

## 1. Introduction and Problem Statement

Urban bike-sharing systems have become a critical component of sustainable transportation infrastructure in cities worldwide. With over 500 bike-sharing programs globally, accurately predicting rental demand is essential for operational efficiency — it determines how bikes should be distributed across stations, when maintenance should be scheduled, and how many bikes are needed during peak hours.

**The Problem**: Bike rental demand fluctuates significantly based on weather conditions, time of day, season, and whether it is a working day or holiday. Under-prediction leads to stock-outs (empty stations frustrating riders), while over-prediction wastes resources on unused fleet repositioning. Both hurt profitability and user experience.

**The Specific Question This Case Study Answers**: *Can we build a regression model that accurately predicts the total number of bike rentals per hour (the `cnt` variable) using weather, seasonal, and temporal features — and which regression algorithm performs best on this real-world dataset?*

We use the UCI/Kaggle Bike Sharing Dataset (17,379 hourly records from Washington D.C., 2011–2012) and compare four regression models: Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.

---

## 2. Base Paper Summary

The base paper by **Fanaee-T and Gama (2014)** introduced the Bike Sharing Dataset and proposed using it as a benchmark for event detection and regression analysis. Their key idea was to combine ensemble detectors with background knowledge (calendar, weather, time) to label events that cause anomalous demand patterns.

**Key Contributions**:
- They demonstrated that bike rental counts are strongly influenced by environmental and temporal factors — temperature, humidity, wind, hour of day, and season.
- They showed that ensemble methods can detect unusual demand spikes (events) by modelling the normal demand pattern first.
- The dataset became one of the most popular regression benchmarks on UCI and Kaggle, cited in over 500 studies.

**Main Results**: The paper established that weather and temporal features alone can explain a large portion of demand variance, making the dataset ideal for supervised regression experimentation.

**Citation**: Fanaee-T, H. and Gama, J., "Event labeling combining ensemble detectors and background knowledge," *Progress in Artificial Intelligence*, vol. 2, no. 2–3, pp. 113–127, 2014.

---

## 3. Method / Model Explanation

We applied four regression algorithms, progressing from simple to complex:

### 3.1 Linear Regression (Baseline)

Linear Regression fits a straight-line relationship between features and the target:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$$

where $\hat{y}$ is the predicted rental count, $\beta_0$ is the intercept, $\beta_i$ are learned coefficients, and $x_i$ are the input features (temperature, humidity, hour, etc.). The model minimises the sum of squared errors (SSE).

### 3.2 Decision Tree Regressor

A Decision Tree recursively splits the data on feature thresholds to minimise variance within each leaf node. For a node $t$:

$$\text{MSE}(t) = \frac{1}{|t|} \sum_{i \in t} (y_i - \bar{y}_t)^2$$

At each split, the feature and threshold that produce the largest reduction in MSE are chosen. We set `max_depth=10` to prevent overfitting.

### 3.3 Random Forest Regressor

Random Forest is an ensemble of $B$ decision trees trained on bootstrapped samples with random feature subsets. The final prediction averages all trees:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

We used $B = 100$ trees with `max_depth=15`. The bagging and feature randomness reduce overfitting compared to a single tree.

### 3.4 Gradient Boosting Regressor

Gradient Boosting builds trees sequentially, each correcting the previous model's residual errors:

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

where $\eta = 0.1$ is the learning rate and $h_m$ is the new tree fitted to the negative gradient (residuals). We used 200 boosting stages with `max_depth=5`.

### Worked Example

Consider a data point: **hr=17** (5 PM), **workingday=1**, **temp=0.58**, **weathersit=1** (clear). Intuitively, this is an evening commute hour on a clear working day — we'd expect high demand. The Gradient Boosting model predicted **422 rentals**, very close to the actual value of **435** (3% error).

---

## 4. Implementation

### Tools and Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.15 | Programming language |
| pandas | 3.1.0 | Data loading and manipulation |
| NumPy | 2.5.4 | Numerical computation |
| scikit-learn | 1.9.1 | ML models, metrics, preprocessing |
| Matplotlib | 3.11.2 | Plotting and visualization |
| Seaborn | 0.13.2 | Statistical plots |

### Dataset Details

- **Source**: UCI Machine Learning Repository / Kaggle — Bike Sharing Dataset
- **File**: `hour.csv` — 17,379 hourly records from January 2011 to December 2012
- **Target**: `cnt` (total rentals per hour, range 1–977)
- **Features used**: 12 features (season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed)

### Steps Followed

1. **Data Loading**: Loaded CSV with pandas; confirmed 17,379 records, 17 columns, zero missing values.
2. **EDA**: Generated distribution plots, correlation heatmap, seasonal/hourly patterns.
3. **Feature Engineering**: Dropped `instant` (row index), `dteday` (date string), `casual` and `registered` (they sum to `cnt` — data leakage).
4. **Train/Test Split**: 80/20 split (13,903 train, 3,476 test) with `random_state=42` for reproducibility.
5. **Scaling**: Applied `StandardScaler` for Linear Regression; tree-based models used unscaled features.
6. **Model Training**: Trained all four models and evaluated on the test set.
7. **Visualization**: Generated 10 plots covering EDA, model comparison, residual analysis, and feature importance.

### Problems Faced and Solutions

| Problem | Solution |
|---------|----------|
| `casual` + `registered` = `cnt` (leakage) | Dropped both columns before modelling |
| Linear Regression assumes linearity but demand is nonlinear (peaks at 8AM, 5PM) | Compared with tree-based models that capture nonlinearity |
| Large output range (1–977) | Used RMSE alongside MAE and R² for robust evaluation |

**GitHub Link**: *[To be updated after upload]*

---

## 5. Results and Discussion

### 5.1 Model Comparison

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 104.80 | 139.21 | 0.3880 |
| Decision Tree | 35.71 | 58.73 | 0.8911 |
| Random Forest | 25.25 | 42.58 | 0.9427 |
| **Gradient Boosting** | **26.61** | **42.30** | **0.9435** |

### Model Comparison Visualization

![Model Comparison](C:/Users/Yogesh/.gemini/antigravity/brain/03f74536-f09c-4c12-8d52-8e2ff8f95ef7/07_model_comparison.png)

### 5.2 EDA Insights

#### Hourly Rental Pattern
![Hourly Pattern](C:/Users/Yogesh/.gemini/antigravity/brain/03f74536-f09c-4c12-8d52-8e2ff8f95ef7/02_hourly_pattern.png)

#### Season & Weather Impact
![Season Weather](C:/Users/Yogesh/.gemini/antigravity/brain/03f74536-f09c-4c12-8d52-8e2ff8f95ef7/03_season_weather.png)

#### Feature Correlation Heatmap
![Correlation](C:/Users/Yogesh/.gemini/antigravity/brain/03f74536-f09c-4c12-8d52-8e2ff8f95ef7/04_correlation_heatmap.png)

#### Working Day vs Non-Working Day
![Working Day Pattern](C:/Users/Yogesh/.gemini/antigravity/brain/03f74536-f09c-4c12-8d52-8e2ff8f95ef7/06_workingday_pattern.png)

### 5.3 Best Model Analysis (Gradient Boosting, R² = 0.9435)

#### Actual vs Predicted
![Actual vs Predicted](C:/Users/Yogesh/.gemini/antigravity/brain/03f74536-f09c-4c12-8d52-8e2ff8f95ef7/08_actual_vs_predicted.png)

#### Residual Analysis
![Residuals](C:/Users/Yogesh/.gemini/antigravity/brain/03f74536-f09c-4c12-8d52-8e2ff8f95ef7/09_residuals.png)

#### Feature Importance
![Feature Importance](C:/Users/Yogesh/.gemini/antigravity/brain/03f74536-f09c-4c12-8d52-8e2ff8f95ef7/10_feature_importance.png)

### 5.4 Correct vs Incorrect Prediction Example

**✓ Correct Prediction** (0.00% error):
- **Input**: Season=Fall, Month=July, Hour=9 AM, Working Day, Clear Weather, Temp=0.74
- **Actual**: 203 rentals → **Predicted: 203.0** rentals
- **Explanation**: A typical weekday morning with good weather — the model learned this common pattern accurately.

**✗ Incorrect Prediction** (3015.87% error):
- **Input**: Season=Summer, Month=May, Hour=3 AM, Working Day, Light Rain, Temp=0.42, High Wind
- **Actual**: 1 rental → **Predicted: −29.2** rentals
- **Explanation**: At 3 AM with rain and high wind, demand drops to nearly zero. The model overshot into negative territory. This extreme edge case (very low actual demand combined with adverse weather at an off-peak hour) is hard to predict precisely because even small absolute errors produce huge percentage errors when the actual value is 1.

### 5.5 Limitations

- **Temporal leakage**: Year (`yr`) feature encodes time trend — model may not generalize to future years.
- **Negative predictions**: Gradient Boosting can predict negative values for extreme low-demand hours; a post-processing clamp (max of 0 and prediction) would fix this.
- **City-specific**: Trained on Washington D.C. data only — transfer to other cities would need retraining.
- **No external events**: Model doesn't account for special events (concerts, protests) that affect demand.

---

## 6. Conclusion

This case study demonstrated that **hourly bike rental demand can be predicted with high accuracy (R² = 0.9435) using weather, seasonal, and temporal features with a Gradient Boosting Regressor**. The model explains over 94% of the variance in hourly demand, with a mean absolute error of only 26.61 rentals (compared to an average of 189 rentals/hour). The study confirmed that **hour of day** is the single most important predictor, followed by **temperature** and **year** (indicating a growing trend in bike-sharing adoption). Ensemble tree-based methods (Random Forest, Gradient Boosting) vastly outperformed Linear Regression, demonstrating that the relationship between weather/temporal features and bike demand is fundamentally nonlinear. These models can be deployed in real-time fleet management systems to optimise bike distribution across stations.

---

## 7. References

[1] Fanaee-T, H. and Gama, J., "Event labeling combining ensemble detectors and background knowledge," *Progress in Artificial Intelligence*, vol. 2, no. 2–3, pp. 113–127, 2014.

[2] UCI Machine Learning Repository, "Bike Sharing Dataset," [Online]. Available: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset. [Accessed: Nov. 2026].

[3] Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

## 8. Declaration

I confirm that this case study is my own work and that all results reported were obtained by me.

**Signature**: _______________

**Roll No.**: [Your Roll No.]

**Name**: Yogesh
