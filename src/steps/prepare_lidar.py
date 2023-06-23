import os.path
import sys
import geopandas as gpd
import pandas as pd
from shapely.ops import orient
from tqdm import tqdm
from glob import glob
from os import path
import h5py
import numpy as np


if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare_lidar.py aoi_shape input-dir output-dir\n'
    )
    sys.exit(1)

aoi_shape = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

os.makedirs(os.path.join(output_dir), exist_ok=True)

grsm_poly = gpd.read_file(aoi_shape)
grsm_poly.geometry = grsm_poly.geometry.apply(orient, args=(1,))

# converting to WGS84 coordinate system
grsm_epsg4326 = grsm_poly.to_crs(epsg=4326)

for infile in tqdm(glob(path.join(input_dir, 'GEDI04_A*.h5'))):
    name, ext = path.splitext(path.basename(infile))
    subfilename = "{name}_sub{ext}".format(name=name, ext=ext)
    outfile = path.join(output_dir, path.basename(subfilename))
    hf_in = h5py.File(infile, 'r')
    hf_out = h5py.File(outfile, 'w')

    # copy ANCILLARY and METADATA groups
    var1 = ["/ANCILLARY", "/METADATA"]
    for v in var1:
        hf_in.copy(hf_in[v], hf_out)

    # loop through BEAMXXXX groups
    for v in list(hf_in.keys()):
        if v.startswith('BEAM'):
            beam = hf_in[v]
            # find the shots that overlays the area of interest (GRSM)
            lat = beam['lat_lowestmode'][:]
            lon = beam['lon_lowestmode'][:]
            i = np.arange(0, len(lat), 1)  # index
            geo_arr = list(zip(lat, lon, i))
            l4adf = pd.DataFrame(geo_arr, columns=["lat_lowestmode", "lon_lowestmode", "i"])
            l4agdf = gpd.GeoDataFrame(l4adf, geometry=gpd.points_from_xy(l4adf.lon_lowestmode, l4adf.lat_lowestmode))
            l4agdf.crs = "EPSG:4326"
            l4agdf_gsrm = l4agdf[l4agdf['geometry'].within(grsm_epsg4326.geometry[0])]
            indices = l4agdf_gsrm.i

            # copy BEAMS to the output file
            for key, value in beam.items():
                if isinstance(value, h5py.Group):
                    for key2, value2 in value.items():
                        group_path = value2.parent.name
                        group_id = hf_out.require_group(group_path)
                        dataset_path = group_path + '/' + key2
                        hf_out.create_dataset(dataset_path, data=value2[:][indices])
                        for attr in value2.attrs.keys():
                            hf_out[dataset_path].attrs[attr] = value2.attrs[attr]
                else:
                    group_path = value.parent.name
                    group_id = hf_out.require_group(group_path)
                    dataset_path = group_path + '/' + key
                    hf_out.create_dataset(dataset_path, data=value[:][indices])
                    for attr in value.attrs.keys():
                        hf_out[dataset_path].attrs[attr] = value.attrs[attr]

    hf_in.close()
    hf_out.close()

subset_df = pd.DataFrame()
date_str_list = []

for subfile in tqdm(glob(path.join(output_dir, 'GEDI04_A*_sub.h5'))):
    datetime_str = subfile.split('_')[2]

    # convert the date and time string to a datetime object
    datetime_obj = pd.to_datetime(datetime_str, format='%Y%j%H%M%S')

    # format the datetime object as a string in 'YYYY-MM-DD' format
    date_str = datetime_obj.strftime('%Y-%m-%d')

    hf_in = h5py.File(subfile, 'r')
    for v in list(hf_in.keys()):
        if v.startswith('BEAM'):
            col_names = []
            col_val = []
            beam = hf_in[v]
            # copy BEAMS
            for key, value in beam.items():
                # looping through subgroups
                if isinstance(value, h5py.Group):
                    for key2, value2 in value.items():
                        if (key2 != "shot_number"):
                            # xvar variables have 2D
                            if (key2.startswith('xvar')):
                                for r in range(4):
                                    col_names.append(key2 + '_' + str(r + 1))
                                    col_val.append(value2[:, r].tolist())
                            else:
                                col_names.append(key2)
                                col_val.append(value2[:].tolist())

                # looping through base group
                else:
                    # xvar variables have 2D
                    if (key.startswith('xvar')):
                        for r in range(4):
                            col_names.append(key + '_' + str(r + 1))
                            col_val.append(value[:, r].tolist())
                    else:
                        col_names.append(key)
                        col_val.append(value[:].tolist())

            # create a pandas dataframe
            beam_df = pd.DataFrame(map(list, zip(*col_val)), columns=col_names)
            beam_df['date'] = date_str
            # Inserting BEAM names
            beam_df.insert(0, 'BEAM', np.repeat(str(v), len(beam_df.index)).tolist())
            # Appending to the subset_df dataframe
            subset_df = pd.concat([subset_df, beam_df])

    hf_in.close()

subset_df = subset_df.set_index('shot_number')
subset_df = subset_df[subset_df['agbd'] != -9999.0]
subset_df.to_csv(os.path.join(output_dir, 'subset.csv'))
