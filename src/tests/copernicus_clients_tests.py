import unittest
import json
import os
from ..copernicus_client import CopernicusClient
from sentinelsat.sentinel import SentinelAPI


class CopernicusClientTest(unittest.TestCase):

    def test_class_should_raise_exception_when_credentials_files_doesnt_exist(self):
        # Arrange
        # Act
        # Assert
        with self.assertRaises(FileNotFoundError):
            _ = CopernicusClient().with_credentials('notexistingfile.json')

    
    def test_class_should_read_creds_when_credentials_files_exist(self):
        # Arrange

        test_creds = {
            "username": "test",
            "password": "test"
        }
        file_name = 'creds_test.json'

        f = open(file_name, "a")
        f.write(json.dumps(test_creds))
        f.close()

        # Act
        client = CopernicusClient().with_credentials(file_name)
        os.remove(file_name)

        # Assert
        self.assertIsInstance(client._api, SentinelAPI)


    def test_class_should_get_images_when_geo_json_provided(self):
        # Arrange
        # Act
        products = CopernicusClient().with_credentials()\
            .download_images(aoi_path='data\\nature_reserves\\sub_regions\\map.geojson')
        
        # Assert
        self.assertIsInstance(products, list)
        self.assertGreater(len(products), 0)



if __name__=='__main__':
	unittest.main()