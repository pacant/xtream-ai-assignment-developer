import datetime
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
import os
import optuna


# load csv file into a dataframe
def load_data(data_file):
    data = pd.read_csv(data_file)
    return data

# drop rows with missing values, zero values in x, y, z and price


def preprocessing(data):
    data_new = data.dropna()
    data_new = data_new[(data_new.x * data_new.y *
                         data_new.z != 0) & (data_new.price > 0)]
    return data_new

# drop columns depth, table, y, z and one-hot encode cut, color and clarity


def preprocessing_linear_regressor(data):
    data_lr = preprocessing(data)
    data_lr = data_lr.drop(columns=['depth', 'table', 'y', 'z'])
    data_lr = pd.get_dummies(
        data_lr, columns=['cut', 'color', 'clarity'], drop_first=True)
    return data_lr

# convert cut, color and clarity to ordered categorical


def preprocessing_xgb(data):
    data_xgb = preprocessing(data)
    data_xgb['cut'] = pd.Categorical(data_xgb['cut'], categories=[
                                     'Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
    data_xgb['color'] = pd.Categorical(data_xgb['color'], categories=[
                                       'D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
    data_xgb['clarity'] = pd.Categorical(data_xgb['clarity'], categories=[
                                         'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
    return data_xgb

# train a linear regressor model


def train_linear_regressor(X_train, y_train):
    y_train_log = np.log(y_train)
    model = LinearRegression()
    model.fit(X_train, y_train_log)
    return model

# train a xgboost  model


def train_xgb(X_train, y_train):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize', study_name='Diamonds XGBoost')
    study.optimize(lambda trial: objective(trial,
                   X_train, y_train), n_trials=100)

    model = xgb.XGBRegressor(
        **study.best_params, enable_categorical=True, random_state=42)
    model.fit(X_train, y_train)
    return model

# print the scores


def score_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'R2 Score: {round(r2, 4)}')
    print(f'MAE: {round(mae, 2)}$')
    return mae, r2
# evaluate the linear regression model


def evaluate_model_lr(model, X_test, y_test):
    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    mae, r2 = score_model(y_test, y_pred)
    return mae, r2

# evaluate the xgb model


def evaluate_model_xgb(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae, r2 = score_model(y_test, y_pred)
    return mae, r2

# save the model to a file in the models directory


def save_model(model, model_type,  mae, r2, directory='models'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_directory = os.path.join(directory, model_type)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    file_directory = os.path.join(model_directory, f"{model_type}_{timestamp}")
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    model_name = f"{model_type}_{timestamp}.pkl"
    model_path = os.path.join(file_directory, model_name)

    joblib.dump(model, model_path)

    log_file = os.path.join(
        file_directory, f"{model_type}_{timestamp}_metrics.log")
    with open(log_file, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"MAE: {round(mae,2)}\n")
        f.write(f"R2: {round(r2,4)}\n\n")


# Function for Optuna to optimize hyperparameters
def objective(trial: optuna.trial.Trial, X_train_xgb, y_train_xgb) -> float:
    # Define hyperparameters to tune
    param = {
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
        'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'random_state': 42,
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'enable_categorical': True
    }

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        X_train_xgb, y_train_xgb, test_size=0.2, random_state=42)

    # Train the model
    model = xgb.XGBRegressor(**param)
    model.fit(x_train, y_train)

    # Make predictions
    preds = model.predict(x_val)

    # Calculate MAE
    mae = mean_absolute_error(y_val, preds)

    return mae
