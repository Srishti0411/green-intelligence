import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def prepare_features(df: pd.DataFrame):
    y = df["energy_kwh"]

    X = df.drop(columns=["energy_kwh", "co2_kg"])

    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded, y

def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,      
        random_state=42 
    )    
    model = LinearRegression()
    
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred) -> dict:
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

def get_coefficients(model: LinearRegression, X: pd.DataFrame) -> pd.DataFrame:

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    return coef_df

def run_regression_pipeline(csv_path: str):

    df = load_data(csv_path)
    X, y = prepare_features(df)
    model, X_test, y_test, y_pred = train_model(X, y)
    metrics = evaluate_model(y_test, y_pred)
    coefficients = get_coefficients(model, X)

    return {
        "model": model,
        "metrics": metrics,
        "coefficients": coefficients,
        "y_test": y_test,
        "y_pred": y_pred
    }