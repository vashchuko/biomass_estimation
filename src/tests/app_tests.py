import unittest
from flask_testing import TestCase
from werkzeug.datastructures import FileStorage
from app import app, allowed_file

UPLOAD_FOLDER = 'data/nature_reserves'

class AppTestCase(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        return app

    def test_allowed_file(self):
        # Test allowed file extensions
        self.assertTrue(allowed_file('file.geojson'))
        self.assertTrue(allowed_file('file.GEOJSON'))
        self.assertFalse(allowed_file('file.txt'))
        self.assertFalse(allowed_file('file.jpg'))
        self.assertFalse(allowed_file('file.doc'))

    def test_upload_file(self):
        
        file = FileStorage(stream=open('data/nature_reserves/sub_regions/nature_reserves_sub5_simple.geojson', 'rb'), filename='nature_reserves_sub5_simple.geojson')

        response = self.client.post('/', data={'file': file})
        self.assert_template_used('results.html')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Estimated ABGD:', response.data)
        self.assertIn(b'Estimated CO2e:', response.data)

if __name__ == '__main__':
    unittest.main()
