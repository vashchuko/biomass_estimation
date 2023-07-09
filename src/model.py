import numpy as np
import geopandas as gpd
import pandas as pd
import pickle
import os
import re
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point
from sklearn import preprocessing
from rasterio.mask import mask
from src.copernicus_client import CopernicusClient


class EstimateModel():

    def __init__(self, dump_path: str) -> None:
        """
        Initialize the estimator model.

        Args:
            dump_path (str, required): Path to pretrained estimator model.
        
        Raises:
            FileNotFoundError: If the dump file doesn't exist.
        """
        self.IMAGES_PATH = 'data/nature_reserves'
        if os.path.isfile(dump_path):
            self.__estimator = pickle.load(open(dump_path, 'rb'))
        else:
            raise FileNotFoundError('Model dump file doesn\'t exist')
        
    def predict(self, geojson_file, use_hub:bool = True) -> dict:
        images_path = None
        if use_hub:
            images_path = CopernicusClient().with_credentials().download_images(geojson_file)

        aoi_geojson = gpd.read_file(geojson_file)
        pictures_dict = self.__get_pictures(pictures_path=images_path)

        output_images_results = []
        for picture_key in pictures_dict.keys():
            raster_image = pictures_dict[picture_key]
            aoi_geojson = aoi_geojson.to_crs(raster_image.crs)

            for _, polygon_row in aoi_geojson.iterrows():
                image_polygons = [aoi_geojson.iloc[_]['geometry']]
                try:
                    image, image_transform = mask(raster_image, image_polygons, crop=True)
                    image_points = self.__get_points_grid_over_image_polygon(image, image_transform, image_polygons, raster_image.crs)
                    output_images_results.append((image, image_transform, image_points))
                except Exception as e:
                    print(e)
                    pass
                    # For now we are ignoring this behaviour
                    
        output_features_df = self.__extract_features(output_images_results)
        y_pred_estimation = self.__estimator.predict(output_features_df)

        result = {}
        result['estimated_abgd'] = y_pred_estimation.sum()
        result['estimated_carbon'] = result['estimated_abgd'] * 0.5
        result['estimated_co2e'] = result['estimated_carbon'] * 3.67
        return result
            
    
    def __get_feature_names(self) -> np.array:
        feature_names_tags = ["xr", "xg", "xb"]
        feature_names_indices = range(1, 10, 1)
        feature_names = np.array([[f"{tag}{index}" for index in feature_names_indices] for tag in feature_names_tags]).flatten()
        return feature_names
    

    def __get_pictures(self, pictures_path:list = None) -> dict:
        # Collect subdirectories in pictures folder. Avoid any additional folders
        if pictures_path is None:
            pictures_path = []
            subfolders = [f.path for f in os.scandir(self.IMAGES_PATH) if f.is_dir()]
            
            for dirname in list(subfolders):
                pictures_path.extend([os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) & ('TCI' in f) & ('2022' in f)])

        pictures_dict = {}
        for picture_path in pictures_path:
            pictures_dict[re.search(r'\d{8}', picture_path).group()] = \
                rasterio.open(picture_path, "r", driver="JP2OpenJPEG")
        return pictures_dict
    
    
    def __get_epsg_code(self, longitude, latitude) -> str:
        """
        Generates EPSG code from lon, lat
        :param longitude: float
        :param latitude: float
        :return: int, EPSG code
        """

        def _zone_number(lat, lon):
            if 56 <= lat < 64 and 3 <= lon < 12:
                return 32
            if 72 <= lat <= 84 and lon >= 0:
                if lon < 9:
                    return 31
                elif lon < 21:
                    return 33
                elif lon < 33:
                    return 35
                elif lon < 42:
                    return 37

            return int((lon + 180) / 6) + 1

        zone = _zone_number(latitude, longitude)

        if latitude > 0:
            return 32600 + zone
        else:
            return 32700 + zone
        

    def __get_points_grid_over_image_polygon(self, image, image_transform, image_polygons, epsg):

        # Get image boundaries
        x_min_pixel, x_max_pixel = 0, image.shape[1]
        y_min_pixel, y_max_pixel = 0, image.shape[2]

        # Define the grid spacing (in pixels)
        grid_spacing = 3

        # Fill up pixels range
        x_pixel_coords = np.arange(x_min_pixel, x_max_pixel, grid_spacing)
        y_pixel_coords = np.arange(y_min_pixel, y_max_pixel, grid_spacing)

        # Generate a gird of pixels 
        x_pixel_coords, y_pixel_coords = np.meshgrid(x_pixel_coords, y_pixel_coords)
        x_pixel_coords, y_pixel_coords = x_pixel_coords.flatten(), y_pixel_coords.flatten()

        # Transform a gird of pixels into real coordinates
        x_real_coords, y_real_coords = xy(image_transform, x_pixel_coords, y_pixel_coords)

        # Convert a grid of real coordinates into shapely.geometry.Points
        real_points = np.vectorize(Point)(x_real_coords, y_real_coords)
        real_points = gpd.GeoSeries(real_points)
        real_points.crs = epsg

        print("Done generating points for an image!")
        print("Starting a filtering process of generated point for polygons...")

        # Filter out a grid of shapely.geometry.Points, so that it contain only those points that fit the polygon
        real_points_in_polygon_list = []
        
        for i, image_polygon in enumerate(image_polygons):
            print(f"Filtering for a polygon {i+1} out of {len(image_polygons)}...")

            real_points_polygon_fit = image_polygon.contains(real_points)
            real_points_polygon_fit_indexes = np.where(real_points_polygon_fit == True)[0]
            real_points_in_polygon = real_points[real_points_polygon_fit_indexes]
            real_points_in_polygon_list.append(real_points_in_polygon)

            print(f"Done filtering for a polygon {i+1} out of {len(image_polygons)}!")

        real_points_in_polygons = pd.concat(real_points_in_polygon_list)

        return real_points_in_polygons
    

    def __read_window(self, image, window):
        window_values = (image[:, window.row_off:window.row_off + window.height, window.col_off:window.col_off + window.width])
        
        if window_values.size != 27:
            return np.append(window_values.flatten(), np.zeros(27-window_values.size))
        return window_values.flatten()


    def __map_points_to_images(self, points, image, image_transform):
        # Extract coordinates of points
        coords = [(point.x, point.y) for point in points]
        x_coords, y_coords = zip(*coords)

        # Transform real coordinates to image pixel locations
        pixel_rows, pixel_cols = rasterio.transform.rowcol(image_transform, x_coords, y_coords)
        pixels_coords = zip(pixel_rows, pixel_cols)

        # Wrap pixel locations with 3x3 grid
        point_windows = [rasterio.windows.Window(pixel_cols - 1, pixel_rows - 1, 3, 3) for pixel_rows, pixel_cols in pixels_coords if (pixel_rows > 0) & (pixel_cols > 0)]

        point_grids = [self.__read_window(image, point_window) for point_window in point_windows]

        return point_grids
    

    def __extract_features(self, images_results):
        final_features_df = None

        for image, image_transform, image_points in images_results:
            # Get features from image
            features = self.__map_points_to_images(image_points, image, image_transform)
            if len(features[0]) == 0:
                continue

            feature_names = self.__get_feature_names()

            # Normalize features
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled_features = min_max_scaler.fit_transform(features)
            scaled_features = np.array(scaled_features).reshape((scaled_features.size // 27, 27))
            
            # Save normalized features
            features_df = pd.DataFrame(scaled_features, columns=feature_names)

            if final_features_df is None:
                final_features_df = features_df
            else:
                final_features_df = pd.concat([final_features_df, features_df], ignore_index=True)

        return final_features_df