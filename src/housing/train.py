import argparse
import os
import pickle
import shutil
from logging import Logger

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from housing.logger import configure_logger

model_names = ["lin_model", "tree_model", "forest_model", "grid_search_model"]

def get_path():
    path_parent = os.getcwd()
    while os.path.basename(os.getcwd()) != "assignment-4":
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)
    return os.getcwd() + "/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputpath",
        help="path to the input dataset ",
        type=str,
        default="data/processed/",
    )
    parser.add_argument(
        "--outputpath", help="path to store the output ", type=str, default="artifacts"
    )
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default=get_path()+"logs/logs.log")
    return parser.parse_args()


def train(housing_prepared, housing_labels):
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    return lin_reg, tree_reg, forest_reg, grid_search


def load_data(in_path):
    prepared = pd.read_csv(in_path + "/train_X.csv")
    lables = pd.read_csv(in_path + "/train_y.csv")
    lables = lables.values.ravel()
    return prepared, lables


def rem_artifacts(out_path):
    if os.path.exists(out_path + "/models" ):
        shutil.rmtree(out_path + "/models")


def model(lin_reg, tree_reg, forest_reg, grid_search, out_path):
    out_path=out_path+'/models'
    os.makedirs(out_path)
    pickle.dump(lin_reg, open(out_path+'/lin_model.pkl', 'wb'))
    pickle.dump(tree_reg, open(out_path+'/tree_model.pkl', 'wb'))
    pickle.dump(forest_reg, open(out_path+'/forest_model.pkl', 'wb'))
    pickle.dump(grid_search, open(out_path+'/grid_search_model.pkl', 'wb'))


if __name__ == "__main__":

    args = parse_args()
    logger = configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )
    path_parent = get_path()
    in_path = path_parent + args.inputpath
    out_path = path_parent + args.outputpath
    rem_artifacts(out_path)
    prepared, labels = load_data(in_path)
    logger.debug("Loaded training data")
    lin_reg, tree_reg, forest_reg, grid_search = train(prepared, labels)
    logger.debug("Training completed")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    model(lin_reg, tree_reg, forest_reg, grid_search, out_path)
    logger.debug(f"Models stored at {out_path}.")
