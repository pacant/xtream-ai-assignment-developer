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
import glob


# load csv file into a dataframe
def load_data(data_file):
    data = pd.read_csv(data_file)
    return data


# drop columns depth, table, y, z and one-hot encode cut, color and clarity


def preprocessing_linear_regression(data, predict=False):
    data_lr = data.drop(columns=['depth', 'table', 'y', 'z'])
    data_lr = data_lr[(data_lr.x != 0)]
    if not predict:
        data_lr = data_lr[data_lr.price > 0]

    # drop rows with missing values
    if data_lr.isnull().any().any():
        data_lr = data_lr.dropna()

    data_lr = pd.get_dummies(
        data_lr, columns=['cut', 'color', 'clarity'], drop_first=True)

    return data_lr

# convert cut, color and clarity to ordered categorical (if predict=True the preprocessing is for predictions)


def preprocessing_xgb(data, predict=False):
    # dropping rows with missing values
    if data.isnull().any().any():
        data_xgb = data.dropna()
    else:
        data_xgb = data.copy()

    data_xgb = data_xgb[(data_xgb.x * data_xgb.y * data_xgb.z != 0)]
    if not predict:
        data_xgb = data_xgb[data_xgb.price > 0]

    data_xgb['cut'] = pd.Categorical(data_xgb['cut'], categories=[
                                     'Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
    data_xgb['color'] = pd.Categorical(data_xgb['color'], categories=[
                                       'D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
    data_xgb['clarity'] = pd.Categorical(data_xgb['clarity'], categories=[
                                         'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
    return data_xgb

# train a linear regressor model


def train_linear_regression(X_train, y_train):
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
    return mae, r2

# evaluate the linear regression model (log_transf=True if the model was trained with the log of the target variable)


def evaluate_model(model, X_test, y_test, log_transf=False):
    y_pred = model.predict(X_test)
    if log_transf:
        y_pred = np.exp(y_pred)
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

    model_timestamp = f"{model_type}_{timestamp}"
    file_directory = os.path.join(model_directory, model_timestamp)
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    model_path = os.path.join(file_directory, "model.pkl")

    joblib.dump(model, model_path)

    log_file = os.path.join(
        file_directory, "metrics.log")
    with open(log_file, 'a') as f:
        f.write(f"Model: {model_timestamp}\n")
        f.write(f"MAE: {round(mae,2)}\n")
        f.write(f"R2: {round(r2,4)}\n\n")
    print(f"Model and metrics saved in {file_directory}")


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

# load the latest model from the models directory


def get_latest_model(model_type, directory='models'):
    model_directory = os.path.join(directory, model_type)
    subdirectories = glob.glob(os.path.join(
        model_directory, f'{model_type}_*'))

    if not subdirectories:
        raise FileNotFoundError(
            f"No model found in {model_directory}")

    subdirectories.sort(key=os.path.getmtime, reverse=True)
    latest_subdir = subdirectories[0]

    latest_model_path = os.path.join(latest_subdir, 'model.pkl')

    if not os.path.exists(latest_model_path):
        raise FileNotFoundError(
            f"No file found in {latest_subdir}")

    return joblib.load(latest_model_path)


def get_similar_diamonds(diamonds, cut, color, clarity, carat, n=6):
    diamonds_sorted = diamonds.copy()

    diamonds_sorted['carat_diff'] = np.abs(diamonds_sorted['carat'] - carat)

    # sort by carat difference
    diamonds_sorted = diamonds_sorted.sort_values('carat_diff')

    # get the top n similar diamonds with the same cut, color and clarity
    similar_diamonds = diamonds_sorted[
        (diamonds_sorted['cut'] == cut) &
        (diamonds_sorted['color'] == color) &
        (diamonds_sorted['clarity'] == clarity)
    ]

    similar_diamonds = similar_diamonds.drop(columns='carat_diff')

    similar_diamonds = similar_diamonds.head(n)

    return similar_diamonds
