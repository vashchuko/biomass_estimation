import sys
import os
import yaml
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 split.py features_file output_dir \n'
    )
    sys.exit(1)

features_file = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(os.path.join(output_dir), exist_ok=True)

data = pd.read_csv(features_file)

params = yaml.safe_load(open('params.yaml'))['split']
results_column = params['results_column']
group_column = params['group_column']

X, y = data[['shot_number', 'xr1',	'xr2', 'xr3',	'xr4',	'xr5',	'xr6',	'xr7',	'xr8',	'xr9', 
            'xg1', 'xg2', 'xg3',	'xg4',	'xg5',	'xg6',	'xg7',	'xg8',	'xg9', 'xb1',	'xb2', 'xb3',
            'xb4',	'xb5',	'xb6',	'xb7',	'xb8',	'xb9']], data[[results_column]]

gs = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
train_ix, test_ix = next(gs.split(X, y, groups=X[group_column]))

X = X.drop(columns=[group_column])
X_train = X.iloc[train_ix]
y_train = y.iloc[train_ix][results_column]

X_test = X.iloc[test_ix]
y_test = y.iloc[test_ix][results_column]

X_train.to_csv(os.path.join(output_dir, 'x_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'x_test.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

