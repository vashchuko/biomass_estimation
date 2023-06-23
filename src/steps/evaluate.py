import sys
import os 
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import json


if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 evaluate.py input_data input_model output_dir \n'
    )
    sys.exit(1)

input_dir = sys.argv[1]
input_model = sys.argv[2]
output_dir = sys.argv[3]

os.makedirs(os.path.join(output_dir), exist_ok=True)

model = pickle.load(open(input_model, 'rb'))


X_test = pd.read_csv(os.path.join(input_dir, 'x_test.csv'))
y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)


metrics = {}
metrics['mape'] = mape

print("Best Stacking Regressor MAPE:", mape)

with open(os.path.join(output_dir, "metrics.json"), "w") as mf:
    json.dump(metrics, mf)
