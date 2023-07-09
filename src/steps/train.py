import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import dvc.api


# Define the custom scoring function for MAPE
def mape_scorer(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    return mean_absolute_percentage_error(y_test, y_pred)

def retrain_model(X_train, y_train, output_dir):
    """Function that retrains a model on new data
    
    Args:
        X_train (pandas.DataFrame): The training features.
        y_train (pandas.Series or numpy.ndarray): The training labels.
        output_dir (str): The directory to save the retrained model.
    """

    estimators = [
        ('rf', RandomForestRegressor()),
        ('en', ElasticNet(alpha=0.001)),
        ('xgb', XGBRegressor(eta=0.05, subsample=1.0, max_depth=5, min_child_weight=3)),
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=XGBRegressor(eta=0.05, max_depth=5)
    )

    stacking_regressor.fit(X_train, y_train)
    pickle.dump(stacking_regressor, open(os.path.join(output_dir, 'stacking.pkl'), 'wb'))
    print('Model was retrained')


def hyper_parameter_tuning(X_train, y_train, output_dir):
    """Function that searches for the best parameters for model"""

    # This model training too long, will be trained later

    # TODO: all this init steps can be moved to arguments
    # Define the base models
    base_models = [
        ('rf', RandomForestRegressor()),
        ('en', ElasticNet()),
        ('svr', SVR()),
        ('xgb', XGBRegressor(eta=0.05, subsample=1.0, max_depth=5, min_child_weight=3))
    ]

    # Define the final estimator model
    final_estimator = XGBRegressor()

    # Create the stacking regressor
    stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=final_estimator)

    # Define the parameter grid for grid search
    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 5, 10],
        'en__alpha': [0.001, 0.01, 0.1],
        'svr__C': [0.1, 1, 10],
        'final_estimator__eta': [0.01, 0.05],
        'final_estimator__max_depth': [3, 5]
    }

    # Perform grid search with MAPE scoring
    grid_search = GridSearchCV(stacking_regressor, param_grid, cv=5, scoring=mape_scorer, verbose=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best stacking regressor model
    best_stacking_regressor = grid_search.best_estimator_
    pickle.dump(best_stacking_regressor, open(os.path.join(output_dir, 'stacking.pkl'), 'wb'))
    print('Model was retrained')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython3 train.py input_dir output_dir \n')
        sys.exit(1)

    np.random.seed(1234)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    params = dvc.api.params_show()

    use_grid_search = params['train']['use_grid_search']

    X_train = pd.read_csv(os.path.join(input_dir, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
    
    if use_grid_search:
        hyper_parameter_tuning(X_train, y_train, output_dir)
    else:
        retrain_model(X_train, y_train, output_dir)
