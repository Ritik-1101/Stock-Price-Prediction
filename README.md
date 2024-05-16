# Stock Price Prediction

This Jupyter Notebook contains code to predict stock prices using various machine learning regression models. The notebook includes data preprocessing, feature engineering, model training, and evaluation. It also visualizes the predictions and extends them to future dates.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Notebook Structure](#notebook-structure)
4. [Models Used](#models-used)
5. [Contributing](#contributing)
6. [License](#license)

## Installation

To run this notebook, you need to have Python and Jupyter Notebook installed. You can install the necessary packages using the following command:

```bash
pip install pandas numpy matplotlib plotly scikit-learn
```
## Usage

- Load Data: Place your stock data CSV file in the same directory as the notebook.
- Run the Notebook: Execute each cell in the notebook sequentially.
- Visualize Predictions: The notebook will generate plots showing the predictions made by each model.

## Notebook Structure

### Imports: Import necessary libraries.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
```
### Load and Preprocess Data: Load the data and perform initial preprocessing.

```python

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Close/Last'] = data['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
    data = data.dropna().reset_index(drop=True)
    return data
```
### Feature Engineering: Add technical indicators as features.

```python

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data['SMA_10'] = data['Close/Last'].rolling(window=10).mean()
    data['SMA_30'] = data['Close/Last'].rolling(window=30).mean()
    data['RSI'] = compute_rsi(data['Close/Last'])
    return data
```
### Model Training and Evaluation: Train and evaluate different regression models.

```python

def train_and_evaluate_models(data: pd.DataFrame):
    features = ['SMA_10', 'SMA_30', 'RSI']
    X = data[features]
    Y = data['Close/Last']
    dates = data['Date']
    tscv = TimeSeriesSplit(n_splits=5)

    models = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor()]
    model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']

    for model, name in zip(models, model_names):
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            dates_test = dates.iloc[test_idx]

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, Y_train)
            y_pred = model.predict(X_test_scaled)

            scores.append({
                'R^2 Score': r2_score(Y_test, y_pred),
                'MSE': mean_squared_error(Y_test, y_pred),
                'MAE': mean_absolute_error(Y_test, y_pred)
            })
            plot_predictions(Y_test, y_pred, f'{name} Predictions', dates_test)
```
### Future Predictions: Extend predictions to future dates.

```python

    def extend_predictions(data: pd.DataFrame, model, scaler, future_steps: int):
        features = ['SMA_10', 'SMA_30', 'RSI']
        last_data = data.iloc[-future_steps:][features]
        last_data_scaled = scaler.transform(last_data)

        future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=future_steps + 1, closed='right')
        future_predictions = model.predict(last_data_scaled)
        
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close/Last'], label='Historical Data')
        plt.plot(future_dates, future_predictions, label='Future Predictions', linestyle='--')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Stock Price Predictions')
        plt.show()
```
### Models Used
```
    Linear Regression
    Random Forest Regressor
    Gradient Boosting Regressor
```