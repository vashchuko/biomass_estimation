import unittest
from ..model import EstimateModel


class EstimateModelTest(unittest.TestCase):
    def test_EstimateModel_should_predict_values_when_happy_path(self):
        # Arrange
        estimateModel = EstimateModel('././data/model/stacking.pkl')

        # Act
        prediction = estimateModel.predict('././data/nature_reserves/sub_regions/nature_reserves_sub1.geojson',
                                           use_hub=False)

        # Assert
        self.assertIn('estimated_abgd', prediction.keys(), 'Missing estimated_abgd key in result dictionary')
        self.assertIn('estimated_carbon', prediction.keys(), 'Missing estimated_carbon key in result dictionary')
        self.assertIn('estimated_co2e', prediction.keys(), 'Missing estimated_co2e key in result dictionary')

    def test_EstimateModel_should_raise_error_when_dump_file_not_found(self):
        # Arrange
        # Act
        # Assert
        
        with self.assertRaises(FileNotFoundError):
            _ = EstimateModel('')

    def test_EstimateModel_should_return_correct_estimations_for_specific_input(self):
        # Arrange
        estimateModel = EstimateModel('././data/model/stacking.pkl')

        # Act
        prediction = estimateModel.predict('././data/nature_reserves/sub_regions/nature_reserves_sub3.geojson',
                                           use_hub=False)
        
        print(prediction)

        # Assert
        self.assertAlmostEqual(prediction['estimated_abgd'], 20030974.0, delta=1, msg='Incorrect estimated ABGD')
        self.assertAlmostEqual(prediction['estimated_carbon'], 10015487.0, delta=1, msg='Incorrect estimated carbon')
        self.assertAlmostEqual(prediction['estimated_co2e'], 36756837.29, delta=1, msg='Incorrect estimated CO2e')


        

        

if __name__=='__main__':
	unittest.main()
