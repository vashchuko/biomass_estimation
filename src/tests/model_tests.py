import unittest
from ..model import EstimateModel


class EstimateModelTest(unittest.TestCase):
    def test_EstimateModel_should_predict_values_when_happy_path(self):
        # Arrange
        estimateModel = EstimateModel('.\\.\\data\\model\\stacking.pkl')

        # Act
        prediction = estimateModel.predict('.\\.\\data\\nature_reserves\\nature_reserves.geojson')

        # Arrange
        self.assertEqual(prediction['estimated_abgd'], 219343920)
        print(prediction)


if __name__=='__main__':
	unittest.main()
