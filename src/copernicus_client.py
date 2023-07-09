import json
import os.path
from sentinelsat import make_path_filter
from sentinelsat.sentinel import SentinelAPI, geojson_to_wkt, read_geojson


class CopernicusClient():
    KEYS_PATH:str = 'copernicus_key.json'
    BAND_NAME:str = 'TCI'
    BASE_DIRECTORY:str = 'temp'

    def with_credentials(self, keys_path: str=None) -> object:
        file_location = self.KEYS_PATH if not keys_path else keys_path
        if not os.path.isfile(file_location):
            raise FileNotFoundError('File with copernicus creds not found!')

        file = open(self.KEYS_PATH if not keys_path else keys_path)
        copernicus_account = json.load(file)
        file.close()
        
        self._api = SentinelAPI(copernicus_account['username'], 
                          copernicus_account['password'], 
                          'https://scihub.copernicus.eu/dhus')
        return self

    def download_images(self, aoi_path: str, 
                   start_date: str = '20220601', 
                   end_date: str = '20221001',
                   max_cloudness: int = 30,
                   limit: int = 1):
        footprint = geojson_to_wkt(read_geojson(aoi_path))
        path_filter = make_path_filter(f"*_{self.BAND_NAME}.jp2")

        try: 
            products = self._api.query(footprint,
                        date = (start_date, end_date),
                        platformname='Sentinel-2',
                        cloudcoverpercentage=(0, max_cloudness),
                        order_by="cloudcoverpercentage",
                        limit=limit)
            
            products = self._api.download_all(products, directory_path=self.BASE_DIRECTORY, nodefilter=path_filter)
            
            pictures_path = []
            for item in products.downloaded.values():
                for node in item["nodes"].values():
                    if 'path' in node: 
                        pictures_path.append(os.path.join(self.BASE_DIRECTORY, node['product_root_dir'], node['node_path']))

            return pictures_path
        except Exception as e:
            print(e)

        return None
