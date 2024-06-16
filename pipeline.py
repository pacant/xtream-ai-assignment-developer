import argparse
import utils
from sklearn.model_selection import train_test_split

# models accepted by the pipeline, add more models here if needed.
valid_models = ['linear_regressor', 'xgb']

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default='data/diamonds.csv',
                    help='Path to csv file(default: data/diamonds.csv)')
parser.add_argument('--model', type=str, default='linear_regressor',
                    choices=valid_models, help=f'Model to use, choose from {valid_models} (default: linear_regressor)')
args = parser.parse_args()

data = utils.load_data(args.csv)

if args.model == 'linear_regressor':
    data_lr = utils.preprocessing_linear_regressor(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data_lr.drop(columns='price'), data_lr['price'], test_size=0.2, random_state=42)

    # train the model
    model_lr = utils.train_linear_regressor(X_train, y_train)

    # evaluate the model
    mae, r2 = utils.evaluate_model_lr(model_lr, X_test, y_test)

    utils.save_model(model_lr, "linear_regressor", mae, r2)

elif args.model == 'xgb':
    data_xgb = utils.preprocessing_xgb(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data_xgb.drop(columns='price'), data_xgb['price'], test_size=0.2, random_state=42)

    # train the model
    model_xgb = utils.train_xgb(X_train, y_train)

    # evaluate the model
    mae, r2 = utils.evaluate_model_xgb(model_xgb, X_test, y_test)

    utils.save_model(model_xgb, "xgb", mae, r2)

# elif args.model ==
