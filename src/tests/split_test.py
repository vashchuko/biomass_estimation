import unittest
import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.steps.split import split_data


class SplitDataTest(unittest.TestCase):
    def setUp(self):
        # Name of test file
        self.features_file = 'test_features.csv'
        # Create a temporary directory for test output
        self.output_dir = "src/tests/test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def test_split_data(self):
        # Prepare some random test data
        data = pd.DataFrame({
            'shot_number': [1, 2, 3, 4, 5],
            'xr1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'xr2': [0.2, 0.4, 0.6, 0.8, 1.0],
            'xr3': [0.3, 0.6, 0.9, 1.2, 1.5],
            'xr4': [0.4, 0.8, 1.2, 1.6, 2.0],
            'xg1': [0.5, 1.0, 1.5, 2.0, 2.5],
            'xg2': [0.6, 1.2, 1.8, 2.4, 3.0],
            'xg3': [0.7, 1.4, 2.1, 2.8, 3.5],
            'xb1': [0.8, 1.6, 2.4, 3.2, 4.0],
            'xb2': [0.9, 1.8, 2.7, 3.6, 4.5],
            'xr5': [0.5, 1.0, 1.5, 2.0, 2.5],
            'xr6': [0.6, 1.2, 1.8, 2.4, 3.0],
            'xr7': [0.7, 1.4, 2.1, 2.8, 3.5],
            'xr8': [0.8, 1.6, 2.4, 3.2, 4.0],
            'xr9': [0.9, 1.8, 2.7, 3.6, 4.5],
            'xg4': [0.5, 1.0, 1.5, 2.0, 2.5],
            'xg5': [0.6, 1.2, 1.8, 2.4, 3.0],
            'xg6': [0.7, 1.4, 2.1, 2.8, 3.5],
            'xg7': [0.8, 1.6, 2.4, 3.2, 4.0],
            'xg8': [0.9, 1.8, 2.7, 3.6, 4.5],
            'xg9': [0.5, 1.0, 1.5, 2.0, 2.5],
            'xb3': [0.6, 1.2, 1.8, 2.4, 3.0],
            'xb4': [0.7, 1.4, 2.1, 2.8, 3.5],
            'xb5': [0.8, 1.6, 2.4, 3.2, 4.0],
            'xb6': [0.9, 1.8, 2.7, 3.6, 4.5],
            'xb7': [0.5, 1.0, 1.5, 2.0, 2.5],
            'xb8': [0.6, 1.2, 1.8, 2.4, 3.0],
            'xb9': [0.7, 1.4, 2.1, 2.8, 3.5],
            'agbd': 12345
        })

        data.to_csv(os.path.join(self.output_dir, self.features_file), index=False)

        # Run the split_data function
        split_data(os.path.join(self.output_dir, self.features_file), self.output_dir)

        # Check if the output files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "x_train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "y_train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "x_test.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "y_test.csv")))

        # Check the distribution of files
        x_train = pd.read_csv(os.path.join(self.output_dir, "x_train.csv"))
        self.assertEqual(len(x_train), 3)  # Expect 70% of the data for training

        y_train = pd.read_csv(os.path.join(self.output_dir, "y_train.csv"))
        self.assertEqual(len(y_train), 3)  # Expect 70% of the data for training

        x_test = pd.read_csv(os.path.join(self.output_dir, "x_test.csv"))
        self.assertEqual(len(x_test), 2)  # Expect 30% of the data for testing

        y_test = pd.read_csv(os.path.join(self.output_dir, "y_test.csv"))
        self.assertEqual(len(y_test), 2)  # Expect 30% of the data for testing


if __name__ == '__main__':
    unittest.main()
