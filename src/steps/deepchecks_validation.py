import pandas as pd
import os
import sys
from deepchecks.tabular.checks import TrainTestSamplesMix, LabelDrift, FeatureDrift
from deepchecks.tabular import Dataset


if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 deepchecks_validation.py features_file output_dir \n'
    )
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(os.path.join(output_dir), exist_ok=True)

X_train = pd.read_csv(os.path.join(input_dir, 'x_train.csv'))
y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
X_test = pd.read_csv(os.path.join(input_dir, 'x_test.csv'))
y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))

ds_train = Dataset(pd.concat([X_train, y_train]), label='agbd', cat_features=[])
ds_test =  Dataset(pd.concat([X_test, y_test]), label='agbd', cat_features=[])

check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal()
train_vs_test = check.run(ds_train, ds_test)

assert (
    train_vs_test.passed_conditions()
), "Data from Test dataset also present in Train dataset"

train_vs_test.save_as_html(f"{output_dir}/train_test_samples_mix.html")

check = LabelDrift().add_condition_drift_score_less_than(0.4)
label_drift = check.run(
    train_dataset=ds_train, test_dataset=ds_test
)
label_drift.save_as_html(f"{output_dir}/label_drift.html")
drift_score = label_drift.reduce_output()
assert label_drift.passed_conditions(), (
    f"Drift score above threshold: {drift_score['Label Drift Score']} vs "
    f"{0.4}"
)

check = FeatureDrift().add_condition_drift_score_less_than(0.4)
label_drift = check.run(
    train_dataset=ds_train, test_dataset=ds_test
)
label_drift.save_as_html(f"{output_dir}/feature_drift.html")
drift_score = label_drift.reduce_output()
assert label_drift.passed_conditions(), (
    f"Drift score above threshold: {drift_score['Label Drift Score']} vs "
    f"{0.4}"
)