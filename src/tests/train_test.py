import os
import pickle
import unittest
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from src.steps.train import mape_scorer, retrain_model


class TrainTest(unittest.TestCase):
    def setUp(self):
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'feature3': [3, 6, 9, 12, 15]
        })
        self.y_train = pd.Series([10, 20, 30, 40, 50])

    def test_mape_scorer(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.2, 3.3, 4.4, 5.5]

        class DummyModel():
            def predict(self, x):
                return y_pred
            
        clf = DummyModel()
        
        expected_mape = mean_absolute_percentage_error(y_true, y_pred)

        calculated_mape = mape_scorer(clf, [], y_true)

        self.assertAlmostEqual(calculated_mape, expected_mape, places=6)

    def test_retrain_model(self):
        output_dir = "src/tests/test_outputs"

        # Perform retraining
        retrain_model(self.X_train, self.y_train, output_dir)

        # Check if the model file was created
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'stacking.pkl')))

        # Load the model and check if it's an instance of StackingRegressor
        loaded_model = pickle.load(open(os.path.join(output_dir, 'stacking.pkl'), 'rb'))
        self.assertIsInstance(loaded_model, StackingRegressor)

        # Check if the model was retrained correctly
        y_pred = loaded_model.predict(self.X_train)
        self.assertEqual(len(y_pred), len(self.y_train))

if __name__ == '__main__':
    unittest.main()
