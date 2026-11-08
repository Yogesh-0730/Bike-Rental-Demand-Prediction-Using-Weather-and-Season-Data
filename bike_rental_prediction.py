"""
=============================================================================
Bike Rental Demand Prediction Using Weather and Season Data
=============================================================================
Case Study: CLA-3 — Regression Problem
Dataset: Bike Sharing Dataset (hour.csv) from UCI / Kaggle
Target Variable: cnt (Total bike rentals per hour)
Models Used: Linear Regression, Decision Tree, Random Forest, Gradient Boosting
=============================================================================
"""

# ── 1. Import Libraries ──────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 2. Configuration ─────────────────────────────────────────────────────────
# Try local directory first, then parent directory (works for .py scripts)
# For Jupyter Notebook / Google Colab, set DATASET_PATH and OUTPUT_DIR manually
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _script_dir = os.getcwd()  # Fallback for Jupyter/Colab

DATASET_PATH = os.path.join(_script_dir, 'hour.csv')
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(_script_dir, '..', 'hour.csv')
OUTPUT_DIR = os.path.join(_script_dir, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  BIKE RENTAL DEMAND PREDICTION USING WEATHER AND SEASON DATA")
print("  Problem Type: Regression")
print("=" * 70)

# ── 3. Load and Explore Dataset ──────────────────────────────────────────────
print("\n[1] LOADING DATASET...")
df = pd.read_csv(DATASET_PATH)

print(f"\n  Dataset Shape: {df.shape}")
print(f"  Total Records: {df.shape[0]}")
print(f"  Total Features: {df.shape[1]}")

print("\n  First 5 Rows:")
print(df.head().to_string(index=False))

print("\n  Column Data Types:")
print(df.dtypes.to_string())

print("\n  Statistical Summary:")
print(df.describe().round(2).to_string())

print("\n  Missing Values:")
print(df.isnull().sum().to_string())
print(f"\n  Total Missing Values: {df.isnull().sum().sum()}")

# ── 4. Exploratory Data Analysis (EDA) ───────────────────────────────────────
print("\n[2] EXPLORATORY DATA ANALYSIS...")

# 4.1 Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['cnt'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Bike Rentals (cnt)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Total Rentals per Hour')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['cnt'].mean(), color='red', linestyle='--', label=f"Mean = {df['cnt'].mean():.0f}")
axes[0].legend()

axes[1].boxplot(df['cnt'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[1].set_title('Box Plot of Bike Rentals', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Total Rentals per Hour')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_target_distribution.png")

# 4.2 Hourly rental pattern
fig, ax = plt.subplots(figsize=(12, 5))
hourly = df.groupby('hr')['cnt'].mean()
ax.bar(hourly.index, hourly.values, color='steelblue', edgecolor='black', alpha=0.8)
ax.set_title('Average Bike Rentals by Hour of Day', fontsize=13, fontweight='bold')
ax.set_xlabel('Hour (0–23)')
ax.set_ylabel('Average Rentals')
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_hourly_pattern.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_hourly_pattern.png")

# 4.3 Seasonal rental pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
season_labels = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
weather_labels = {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain'}

seasonal = df.groupby('season')['cnt'].mean()
colors_season = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
axes[0].bar([season_labels[s] for s in seasonal.index], seasonal.values, color=colors_season, edgecolor='black')
axes[0].set_title('Average Rentals by Season', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Average Rentals')

weather = df.groupby('weathersit')['cnt'].mean()
colors_weather = ['#f1c40f', '#95a5a6', '#3498db', '#2c3e50']
axes[1].bar([weather_labels.get(w, f'W{w}') for w in weather.index],
            weather.values, color=colors_weather[:len(weather)], edgecolor='black')
axes[1].set_title('Average Rentals by Weather Condition', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Average Rentals')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_season_weather.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_season_weather.png")

# 4.4 Correlation heatmap (excluding casual & registered to avoid leakage)
fig, ax = plt.subplots(figsize=(12, 9))
numeric_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, ax=ax, square=True,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap of Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 04_correlation_heatmap.png")

# 4.5 Temperature vs Rentals scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df['temp'], df['cnt'], alpha=0.1, c='steelblue', s=10)
axes[0].set_title('Temperature vs Bike Rentals', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Normalized Temperature')
axes[0].set_ylabel('Total Rentals')

axes[1].scatter(df['hum'], df['cnt'], alpha=0.1, c='coral', s=10)
axes[1].set_title('Humidity vs Bike Rentals', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Normalized Humidity')
axes[1].set_ylabel('Total Rentals')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_temp_hum_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_temp_hum_scatter.png")

# 4.6 Working day vs non-working day hourly pattern
fig, ax = plt.subplots(figsize=(12, 5))
for wd, label, color in [(1, 'Working Day', 'steelblue'), (0, 'Non-Working Day', 'coral')]:
    subset = df[df['workingday'] == wd].groupby('hr')['cnt'].mean()
    ax.plot(subset.index, subset.values, marker='o', label=label, color=color, linewidth=2)
ax.set_title('Hourly Rental Pattern: Working Day vs Non-Working Day', fontsize=13, fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Rentals')
ax.legend()
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_workingday_pattern.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 06_workingday_pattern.png")

# ── 5. Feature Engineering & Data Preprocessing ─────────────────────────────
print("\n[3] DATA PREPROCESSING & FEATURE ENGINEERING...")

# Drop non-predictive columns: instant (index), dteday (date string),
# casual and registered (leakage — they sum to cnt)
drop_cols = ['instant', 'dteday', 'casual', 'registered']
df_model = df.drop(columns=drop_cols)
print(f"  Dropped columns: {drop_cols}")

# Add cyclical encoding for hour (23:00 and 00:00 are adjacent)
df_model['hr_sin'] = np.sin(2 * np.pi * df_model['hr'] / 24)
df_model['hr_cos'] = np.cos(2 * np.pi * df_model['hr'] / 24)
print("  ✓ Added cyclical hour features: hr_sin, hr_cos")

# Feature set
feature_cols = [c for c in df_model.columns if c != 'cnt']
X = df_model[feature_cols]
y = df_model['cnt']

print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
print(f"  Target: cnt")
print(f"  X shape: {X.shape}, y shape: {y.shape}")

# Chronological Train-Test Split (80-20)
# Since this is time-series data, we split by time order to avoid data leakage
split_idx = int(len(X) * 0.80)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]
print(f"\n  Train set: {X_train.shape[0]} samples (first 80% chronologically)")
print(f"  Test set:  {X_test.shape[0]} samples (last 20% chronologically)")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Features standardized using StandardScaler")

# ── 6. Model Training & Evaluation ──────────────────────────────────────────
print("\n[4] MODEL TRAINING & EVALUATION...")
print("-" * 70)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Use scaled features for Linear Regression, unscaled for tree-based
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Clamp predictions to non-negative (rental count cannot be negative)
    y_pred = np.maximum(y_pred, 0)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'y_pred': y_pred
    }
    
    print(f"    MAE  = {mae:.2f}")
    print(f"    RMSE = {rmse:.2f}")
    print(f"    R²   = {r2:.4f}")

# ── 7. Results Comparison ────────────────────────────────────────────────────
print("\n[5] RESULTS COMPARISON")
print("=" * 70)
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<25} {r['MAE']:>8.2f} {r['RMSE']:>8.2f} {r['R²']:>8.4f}")
print("=" * 70)

best_model_name = max(results, key=lambda k: results[k]['R²'])
print(f"\n  ★ Best Model: {best_model_name} (R² = {results[best_model_name]['R²']:.4f})")

# 7.1 Model comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
model_names = list(results.keys())
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

# MAE comparison
mae_vals = [results[m]['MAE'] for m in model_names]
axes[0].barh(model_names, mae_vals, color=colors, edgecolor='black')
axes[0].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('MAE (lower is better)')
for i, v in enumerate(mae_vals):
    axes[0].text(v + 1, i, f'{v:.2f}', va='center', fontsize=10)

# RMSE comparison
rmse_vals = [results[m]['RMSE'] for m in model_names]
axes[1].barh(model_names, rmse_vals, color=colors, edgecolor='black')
axes[1].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('RMSE (lower is better)')
for i, v in enumerate(rmse_vals):
    axes[1].text(v + 1, i, f'{v:.2f}', va='center', fontsize=10)

# R² comparison
r2_vals = [results[m]['R²'] for m in model_names]
axes[2].barh(model_names, r2_vals, color=colors, edgecolor='black')
axes[2].set_title('R² Score', fontsize=12, fontweight='bold')
axes[2].set_xlabel('R² (higher is better)')
for i, v in enumerate(r2_vals):
    axes[2].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✓ Saved: 07_model_comparison.png")

# 7.2 Actual vs Predicted scatter plot for best model
best_pred = results[best_model_name]['y_pred']
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, best_pred, alpha=0.3, s=15, c='steelblue', edgecolors='none')
max_val = max(y_test.max(), best_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Actual vs Predicted — {best_model_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Actual Rentals (cnt)')
ax.set_ylabel('Predicted Rentals')
ax.legend(fontsize=11)
ax.set_xlim(0, max_val + 20)
ax.set_ylim(0, max_val + 20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 08_actual_vs_predicted.png")

# 7.3 Residual plot for best model
residuals = y_test.values - best_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(best_pred, residuals, alpha=0.2, s=10, c='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title(f'Residual Plot — {best_model_name}', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Rentals')
axes[0].set_ylabel('Residuals (Actual − Predicted)')
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Residual Value')
axes[1].set_ylabel('Frequency')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_residuals.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 09_residuals.png")

# 7.4 Feature importance (for best tree-based model)
if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    best_model_obj = models[best_model_name]
    importances = best_model_obj.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    feat_imp.plot(kind='barh', color='steelblue', edgecolor='black', ax=ax)
    ax.set_title(f'Feature Importance — {best_model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '10_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 10_feature_importance.png")

# ── 8. Sample Predictions ───────────────────────────────────────────────────
print("\n[6] SAMPLE PREDICTIONS (Best Model: {})".format(best_model_name))
print("-" * 70)

# Show 10 sample predictions
sample_idx = np.random.RandomState(42).choice(len(y_test), 10, replace=False)
y_test_arr = y_test.values
print(f"{'#':<4} {'Actual':>8} {'Predicted':>10} {'Error':>8} {'Error%':>8}")
print("-" * 42)
correct_count = 0
for i, idx in enumerate(sample_idx):
    actual = y_test_arr[idx]
    predicted = best_pred[idx]
    error = actual - predicted
    pct = abs(error) / max(actual, 1) * 100
    status = "✓" if pct < 20 else "✗"
    if pct < 20:
        correct_count += 1
    print(f"{i+1:<4} {actual:>8.0f} {predicted:>10.1f} {error:>8.1f} {pct:>7.1f}% {status}")

print(f"\n  Predictions within 20% error: {correct_count}/10")

# ── 9. Prediction Error Analysis ─────────────────────────────────────────────
print("\n[7] PREDICTION ERROR ANALYSIS: LOWEST vs HIGHEST ERROR")
print("-" * 70)

# Find the prediction with the lowest and highest absolute error
abs_errors = np.abs(y_test.values - best_pred)
pct_errors = abs_errors / np.maximum(y_test.values, 1) * 100
best_idx = np.argmin(abs_errors)
worst_idx = np.argmax(abs_errors)

print(f"\n  ↓ Lowest Prediction Error:")
print(f"    Actual: {y_test.values[best_idx]:.0f}, Predicted: {best_pred[best_idx]:.1f}")
print(f"    Absolute Error: {abs_errors[best_idx]:.2f} rentals")
print(f"    Features: {X_test.iloc[best_idx].to_dict()}")

print(f"\n  ↑ Highest Prediction Error:")
print(f"    Actual: {y_test.values[worst_idx]:.0f}, Predicted: {best_pred[worst_idx]:.1f}")
print(f"    Absolute Error: {abs_errors[worst_idx]:.2f} rentals")
print(f"    Features: {X_test.iloc[worst_idx].to_dict()}")

# ── 10. Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"""
  Dataset:        Bike Sharing Dataset (hour.csv)
  Records:        {df.shape[0]} hourly records (2011–2012)
  Features Used:  {len(feature_cols)} ({', '.join(feature_cols)})
  Target:         cnt (total bike rentals per hour)
  Train/Test:     80/20 chronological split
  
  Models Trained:
    1. Linear Regression  → R² = {results['Linear Regression']['R²']:.4f}
    2. Decision Tree      → R² = {results['Decision Tree']['R²']:.4f}
    3. Random Forest      → R² = {results['Random Forest']['R²']:.4f}
    4. Gradient Boosting  → R² = {results['Gradient Boosting']['R²']:.4f}
  
  Best Model: {best_model_name}
    - MAE  = {results[best_model_name]['MAE']:.2f}
    - RMSE = {results[best_model_name]['RMSE']:.2f}
    - R²   = {results[best_model_name]['R²']:.4f}
  
  Key Findings:
    - Hour of day is the strongest predictor of bike rental demand
    - Temperature has a strong positive correlation with rentals
    - Humidity and bad weather reduce demand significantly
    - Working days show commute peaks (8AM, 5-6PM), weekends show midday peaks
    - Tree-based ensemble models significantly outperform Linear Regression
""")
print("  All plots saved to:", OUTPUT_DIR)
print("=" * 70)
print("  ✓ Case study execution complete!")
