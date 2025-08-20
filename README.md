# Time Series Forecasting of U.S. Energy Consumption using XGBoost

A comprehensive machine learning project that predicts hourly energy consumption in the United States using XGBoost regression. This model leverages historical power consumption data from PJM Interconnection LLC to provide accurate forecasting for energy planning and optimization.

## 📊 Project Overview

This project implements a robust time series forecasting system to predict electricity consumption patterns using XGBoost. The model incorporates temporal features, lag variables, and cross-validation techniques to achieve high accuracy in energy demand forecasting.

## 🎯 Objectives

- Predict hourly energy consumption (MW) using historical data
- Implement proper time series cross-validation for model evaluation
- Create meaningful temporal and lag features for improved accuracy
- Provide reliable forecasts for energy planning and grid management
- Build a production-ready model with save/load functionality

## 📁 Dataset

**Source**: PJM Interconnection LLC (Regional Transmission Organization)  
**File**: `PJME_hourly.csv`  
**Frequency**: Hourly measurements  
**Target Variable**: `PJME_MW` (Power consumption in Megawatts)  
**Time Period**: Historical hourly energy consumption data

### Data Characteristics
- **Temporal Resolution**: Hourly energy consumption data
- **Geographic Coverage**: PJM Interconnection service area
- **Data Quality**: Preprocessed with outlier removal (< 19,000 MW)

## 🔧 Methodology

### 1. Data Preprocessing & Quality Control

#### Outlier Detection and Removal
- **Histogram Analysis**: Identified consumption patterns and anomalies
- **Threshold Application**: Removed unrealistic low consumption values (< 19,000 MW)
- **Data Validation**: Ensured data quality for reliable forecasting

#### Time Series Preparation
- **Index Conversion**: Proper datetime indexing for time series operations
- **Data Sorting**: Chronological ordering for time series cross-validation
- **Missing Value Handling**: Addressed gaps in hourly recordings

### 2. Feature Engineering

#### Temporal Features
- **Hour of Day**: Captures daily consumption patterns
- **Day of Week**: Weekly seasonality patterns  
- **Month & Quarter**: Seasonal variations
- **Day of Year**: Annual cyclical patterns
- **Week of Year**: ISO calendar week numbers

#### Lag Features (Historical Context)
- **1-Year Lag**: Same day/hour from previous year (364 days)
- **2-Year Lag**: Two-year historical comparison (728 days)
- **3-Year Lag**: Three-year trend analysis (1092 days)

*Note: 364-day lag ensures exact weekly alignment (364 ÷ 7 = 52 weeks)*

### 3. Time Series Cross-Validation

#### Validation Strategy
- **Method**: TimeSeriesSplit with 5 folds
- **Test Size**: 1 year (24 × 365 × 1 hours)
- **Gap**: 24 hours between train/validation sets
- **Approach**: Forward-chaining validation maintaining temporal order

#### Benefits
- **No Data Leakage**: Future data never used for past predictions
- **Realistic Evaluation**: Mimics real-world forecasting scenarios
- **Robust Performance**: Multiple validation folds ensure stability

### 4. Model Architecture

#### XGBoost Configuration
```python
XGBRegressor(
    base_score=0.5,
    booster="gbtree",
    n_estimators=1000,
    early_stopping_rounds=50,
    objective="reg:linear",
    max_depth=3,
    learning_rate=0.01
)
```

#### Key Parameters
- **Tree-based Boosting**: Gradient boosting with decision trees
- **Early Stopping**: Prevents overfitting (50 rounds patience)
- **Conservative Learning**: Low learning rate (0.01) for stability
- **Moderate Depth**: max_depth=3 prevents overfitting

## 📈 Results & Performance

### Cross-Validation Results
- **Evaluation Metric**: Root Mean Square Error (RMSE)
- **Validation Method**: 5-fold time series cross-validation
- **Performance**: Consistent accuracy across all folds
- **Generalization**: Strong performance on out-of-sample data

### Model Capabilities
- **Short-term Accuracy**: Excellent hourly predictions
- **Seasonal Patterns**: Captures daily, weekly, and yearly cycles
- **Trend Following**: Adapts to long-term consumption changes
- **Anomaly Handling**: Robust to unusual consumption patterns

## 🛠️ Technologies & Libraries

- **Python 3.x** - Primary programming language
- **pandas** - Data manipulation and time series handling
- **NumPy** - Numerical computations
- **matplotlib** - Data visualization and plotting
- **seaborn** - Statistical visualization and styling
- **XGBoost** - Gradient boosting framework
- **scikit-learn** - Machine learning utilities and metrics

## 📋 Installation & Requirements

### Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
```

### Required Libraries
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
xgboost>=1.5.0
scikit-learn>=1.0.0
```

## 🚀 Usage

### 1. Data Preparation
```python
# Load and preprocess data
df = pd.read_csv("PJME_hourly.csv")
df = df.set_index("Datetime")
df.index = pd.to_datetime(df.index)

# Remove outliers
df = df.query("PJME_MW > 19000").copy()
```

### 2. Feature Engineering
```python
# Create temporal features
df = create_features(df)

# Add lag features
df = add_lags(df)
```

### 3. Model Training
```python
# Train with cross-validation
features = ["dayofyear", "hour", "dayofweek", "quarter", "month", "year", "lag1", "lag2", "lag3"]
reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3)
reg.fit(X_train, y_train)
```

### 4. Future Predictions
```python
# Create future date range
future = pd.date_range("2018-08-03", "2019-08-01", freq="1h")

# Generate predictions
future_predictions = reg.predict(future_features)
```

## 📊 Key Features

### Advanced Time Series Techniques
- **Proper Cross-Validation**: TimeSeriesSplit prevents data leakage
- **Feature Engineering**: Comprehensive temporal and lag features
- **Outlier Management**: Statistical outlier detection and removal
- **Model Persistence**: Save/load functionality for production deployment

### Forecasting Capabilities
- **Multiple Horizons**: Short-term (hours) to long-term (years)
- **Seasonal Awareness**: Captures daily, weekly, and annual patterns
- **Trend Adaptation**: Responds to changing consumption patterns
- **Production Ready**: Robust model with error handling

## 🔍 Model Insights

### Feature Importance
The model leverages multiple feature types:
- **Temporal Features**: Capture cyclical patterns
- **Lag Features**: Provide historical context
- **Seasonal Components**: Account for weather and behavioral patterns

### Validation Strategy Benefits
- **Temporal Integrity**: Maintains chronological order
- **Realistic Testing**: Simulates actual forecasting conditions
- **Performance Stability**: Multiple folds ensure robust evaluation

## 🚀 Future Enhancements

### Model Improvements
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Feature Selection**: Automated feature importance analysis
- **Ensemble Methods**: Combine multiple algorithms
- **Deep Learning**: LSTM/GRU for complex temporal patterns

### Additional Features
- **Weather Integration**: Temperature and seasonal data
- **Economic Indicators**: GDP, industrial activity metrics
- **Holiday Effects**: Special event consumption patterns
- **Real-time Updates**: Streaming data integration

### Production Features
- **API Development**: REST API for real-time predictions
- **Monitoring Dashboard**: Performance tracking and alerts
- **Automated Retraining**: Scheduled model updates
- **Scalability**: Distributed computing for large datasets

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Time Series Plots**: Historical consumption trends
- **Cross-Validation Splits**: Training/validation set visualization
- **Prediction Plots**: Future forecasting results
- **Distribution Analysis**: Outlier detection histograms

## 💡 Key Insights

### Energy Consumption Patterns
- **Daily Cycles**: Peak consumption during business hours
- **Weekly Patterns**: Lower weekend consumption
- **Seasonal Trends**: Higher consumption in summer/winter
- **Annual Growth**: Long-term demand increases

### Model Performance
- **High Accuracy**: Consistent RMSE across validation folds
- **Pattern Recognition**: Excellent seasonal pattern capture
- **Robustness**: Stable performance across different time periods
- **Scalability**: Efficient training on large datasets

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature engineering techniques
- Alternative model architectures
- Performance optimization
- Visualization enhancements
- Documentation improvements

---

**Note**: This project provides valuable insights for energy grid management, capacity planning, and demand forecasting in the electrical power industry.
