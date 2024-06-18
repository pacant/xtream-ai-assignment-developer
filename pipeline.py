import argparse
from utils import load_data, preprocessing_linear_regression, train_linear_regression, evaluate_model, save_model, preprocessing_xgb, train_xgb
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='data/diamonds.csv',
                    help='Path to csv file(default: data/diamonds.csv)')

args = parser.parse_args()

data = load_data(args.csv)

## -- preprocessing train and evaluation of linear regression model -- ##
data_lr = preprocessing_linear_regression(data)

X_train, X_test, y_train, y_test = train_test_split(
    data_lr.drop(columns='price'), data_lr['price'], test_size=0.2, random_state=42)

model_lr = train_linear_regression(X_train, y_train)

mae, r2 = evaluate_model(model_lr, X_test, y_test, log_transf=True)

save_model(model_lr, "linear_regression", mae, r2)

##  -- preprocessing train and evaluation of xgboost model -- ##
data_xgb = preprocessing_xgb(data)

X_train, X_test, y_train, y_test = train_test_split(
    data_xgb.drop(columns='price'), data_xgb['price'], test_size=0.2, random_state=42)

model_xgb = train_xgb(X_train, y_train)

mae, r2 = evaluate_model(model_xgb, X_test, y_test)

save_model(model_xgb, "xgb", mae, r2)
