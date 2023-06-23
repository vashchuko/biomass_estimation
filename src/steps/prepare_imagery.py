import sys
import os
import yaml
import rasterio
import rasterio.mask
import re
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
from datetime import datetime
from sklearn import preprocessing

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare_imagery.py subset_file input-dir output-dir\n'
    )
    sys.exit(1)

subset_file = sys.argv[1]
imagery_dir = sys.argv[2]
output_dir = sys.argv[3]

os.makedirs(os.path.join(output_dir), exist_ok=True)

params = yaml.safe_load(open('params.yaml'))['prepare_imagery']
features_channel_key = params['features_channel_key']
cloud_channel_key = params['cloud_channel_key']

pictures_path = []
sub_folders = [f.path for f in os.scandir(imagery_dir) if f.is_dir()]
for dir_name in list(sub_folders):
    pictures_path.extend([os.path.join(dir_name, f) for f in os.listdir(dir_name)
                          if os.path.isfile(os.path.join(dir_name, f))])

pictures_dict = {}
for picture_path in list(filter(lambda path: features_channel_key in path, pictures_path)):
    pictures_dict[re.search(r'\d{8}', picture_path).group()] = rasterio.open(picture_path, "r", driver="JP2OpenJPEG")

clouds_dict = {}
for picture_path in list(filter(lambda path: cloud_channel_key in path, pictures_path)):
    clouds_dict[re.search(r'\d{8}', picture_path).group()] = rasterio.open(picture_path, "r", driver="JP2OpenJPEG")

gedi_subset = pd.read_csv(subset_file)
start_date = datetime(2022, 6, 1)
end_date = datetime(2022, 9, 1)

gedi_subset['date'] = pd.to_datetime(gedi_subset['date'])
gedi_subset = gedi_subset[(gedi_subset['date'] >= start_date) & (gedi_subset['date'] <= end_date)]
gedi_subset = gedi_subset[gedi_subset['agbd'] != -9999]

gedi_points = [shapely.geometry.Point(lon, lat) for lon,lat in zip(gedi_subset["lon_lowestmode"], gedi_subset["lat_lowestmode"])]
gedi_df = gpd.GeoDataFrame(gedi_subset, crs="EPSG:4326", geometry=gedi_points)

def get_feature_names(sentinel2_image):
  feature_names_tags = ["x"] if sentinel2_image.count == 1 else ["xr", "xg", "xb"]
  feature_names_indices = range(1, 10, 1)
  feature_names = np.array([[f"{tag}{index}" for index in feature_names_indices] for tag in feature_names_tags]).flatten()
  return feature_names

def map_gedi_to_sentinel2(gedi_points, sentinel2_image, sentinel2_clouds):
  # Transform GEDI points to Sentinel2 coordinate system
  GEDI_EPSG_CODE = "EPSG:4326"
  SENTINEL2_EPSG_CODE = "EPSG:32634"
  gedi_points = rasterio.warp.transform_geom(src_crs=GEDI_EPSG_CODE, dst_crs=SENTINEL2_EPSG_CODE, geom=gedi_points)

  # Extract coordinates of GEDI points
  gedi_coords = [gedi_point["coordinates"] for gedi_point in gedi_points]
  gedi_x_coords, gedi_y_coords = zip(*gedi_coords)

  # Transform GEDI coordinates to Sentinel2 image pixel locations
  pixel_rows, pixel_cols = rasterio.transform.rowcol(sentinel2_image.transform, gedi_x_coords, gedi_y_coords)
  pixels_coords = zip(pixel_rows, pixel_cols)

  # Wrap GEDI pixel locations with 3x3 grid
  gedi_point_windows = [rasterio.windows.Window(pixel_cols - 1, pixel_rows - 1, 3, 3) for pixel_rows, pixel_cols in pixels_coords]
  gedi_point_grids = [sentinel2_image.read(window=gedi_point_window).flatten() for gedi_point_window in gedi_point_windows]

  #Transfrom GEDI coordinates to Sentinel2 cloud prediction pixel locations
  cloud_pixel_rows, cloud_pixel_cols = rasterio.transform.rowcol(sentinel2_clouds.transform, gedi_x_coords, gedi_y_coords)
  cloud_pixels_coords = zip(cloud_pixel_rows, cloud_pixel_cols)

  # Wrap GEDI pixel locations with 2x2 grid for cloud predictions
  gedi_point_windows_clouds = [rasterio.windows.Window(cloud_pixel_cols - 1, cloud_pixel_rows - 1, 2, 2) for cloud_pixel_rows, cloud_pixel_cols in cloud_pixels_coords]
  gedi_point_grid_clouds = [sentinel2_clouds.read(window=gedi_point_window).flatten() for gedi_point_window in gedi_point_windows_clouds]

  return gedi_point_grids, gedi_point_grid_clouds

def extract_features(gedi_df, sentinel2_images, sentinel2_clouds):
  final_features_df = None

  for date, sentinel2_image in sentinel2_images.items():
    # Get features from Sentinel2 and GEDI
    gedi_points = gedi_df.geometry.to_numpy()
    (features, features_clouds) = map_gedi_to_sentinel2(gedi_points, sentinel2_image, sentinel2_clouds[date])
    feature_names = get_feature_names(sentinel2_image)

    # Normalize features
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    
    # Save normalized features
    features_df = gedi_df.copy()
    features_df["sentinel2_date"] = datetime.strptime(date, "%Y%m%d").date()
    features_df[feature_names] = features
    features_df["cloud_pred"] = [feature_clouds.mean() for feature_clouds in features_clouds]

    if final_features_df is None:
      final_features_df = features_df
    else:
      final_features_df = pd.concat([final_features_df, features_df], ignore_index=True)

  return final_features_df

features_for_all_images_df = extract_features(gedi_df, pictures_dict, clouds_dict)
features_for_all_images_df = features_for_all_images_df[features_for_all_images_df['cloud_pred'] <= 20.0]
features_for_all_images_df = features_for_all_images_df[(features_for_all_images_df['agbd'] >= 200) & (features_for_all_images_df['agbd'] <= 600)]
features_for_all_images_df.to_csv(os.path.join(output_dir, 'subset_raster.csv'), index=False)