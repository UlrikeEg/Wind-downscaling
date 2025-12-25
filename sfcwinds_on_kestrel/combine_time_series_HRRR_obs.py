
"""Combine HRRR station time series with observation time series and plot example.

MPI Execution Note (HOW TO RUN):
salloc -n 16 --time=10:00:00 --account=sfcwinds --mem=50G --qos=high
module load anaconda3
conda activate thrive

mpirun -n 16 python3 /home/uegerer/sfcwinds/sfcwinds_on_kestrel/combine_time_series_HRRR_obs.py  > output.log 2>&1


This script:
 1. Loads station mapping information (stations_found, x/y indices) from the pickle produced by the HRRR station extraction script.
 2. Scans a folder for per-station merged HRRR time series files (HRRR_<station>.nc).
 3. Builds a combined pandas DataFrame of HRRR wind speed/direction (10 m) for all stations.
 4. Loads observation parquet files for a chosen station and aligns timestamps (drops timezone, sorts).
 5. Generates a comparison plot (wind speed and wind direction) for the chosen station.
"""


import os
import time
import glob
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

from Functions_sfcwinds import *
from mpi4py import MPI

# Enable interactive backend when running in IPython, ignore in plain Python scripts (this is for data inspection with interactive plots)
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass


def filter_valid_netcdf(file_list):
    valid_files = []
    for f in file_list:
        try:
            with xr.open_dataset(f, engine='netcdf4') as ds:
                pass
            valid_files.append(f)
        except Exception as e:
            print(f"[WARNING] Skipping unreadable file: {f} ({e})", flush=True)
    return valid_files

# Set paths
save_folder = "/kfs2/projects/sfcwinds/merged_station_data/"
save_folder = "/kfs2/projects/sfcwinds/merged_station_data_updated/"
hrrr_folder = "/kfs2/projects/sfcwinds/HRRR_station_data/"
ERA5_folder = "/kfs2/projects/sfcwinds/ERA5_station_data/"
base_dir = "/kfs2/projects/sfcwinds/observations/"
plots_dir = os.path.join(save_folder, "plots")
os.makedirs(plots_dir, exist_ok=True)




# === MPI setup ===
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()


# Get metadata for all files
if rank == 0:
    # Metadata table
    meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")

    # Load the elevation dataset
    ncfile = '/kfs2/projects/sfcwinds/environmental_data/CONUS_elevation_1km.nc'
    ds = xr.open_dataset(ncfile)
    ds = ds.fillna(0)   # NaN values are where no elevation is available (e.g. ocean) - set to zero
    ds['elevation_std'] = ds['elevation_std'].where(ds['elevation_std'] <= 100)  
    mean_elev = ds['elevation_mean'].values
    std_elev = ds['elevation_std'].values

    # Add terrain to metadata
    mean_std_list = []
    closest_elev_list = []

    # Precompute once
    from scipy.spatial import cKDTree
    lon_flat = ds['lon'].values.ravel()
    lat_flat = ds['lat'].values.ravel()
    grid_coords = np.column_stack([lon_flat, lat_flat])
    tree = cKDTree(grid_coords)
    grid_shape = ds['lon'].values.shape  # needed to unravel indices

    def find_closest_grid_point_fast(lon, lat, tree, grid_shape):
        # Query nearest grid point
        dist, idx_flat = tree.query([lon, lat], k=1)
        return np.unravel_index(idx_flat, grid_shape)

    # Vectorized for many stations
    def find_closest_many(lons, lats, tree, grid_shape):
        pts = np.column_stack([lons, lats])
        dists, idxs_flat = tree.query(pts, k=1)
        return np.array([np.unravel_index(i, grid_shape) for i in idxs_flat])


    for i, row in meta.iterrows():
        #print(f"station {i} of {len(meta)}: {row.station_id}")
        lat_stat = row.lat
        lon_stat = row.lon

        # tree built once above
        idx = find_closest_grid_point_fast(lon_stat, lat_stat, tree, grid_shape)

        ds_loc = ds.isel(x = idx[1], y = idx[0])
        distance_m = haversine(ds_loc.lat.values, ds_loc.lon.values, lat_stat, lon_stat)
        # Checking that the terrain coordinates are within 2kms of the station coordinates
        if distance_m <= 2000.:
            pass
            # print(f"Distance is {distance_m}. Data accepted.")
        elif distance_m > 2000.:
            pass
            #print("Data was measured more than 2km away.")

        closest_elev = mean_elev[idx]
        mean_std_vicinity = get_vicinity_mean_std(std_elev, idx, radius=20)

        if mean_std_vicinity > 1000:
            mean_std_vicinity = 0

        closest_elev_list.append(closest_elev)
        mean_std_list.append(mean_std_vicinity)

    meta['closest_elev'] = closest_elev_list
    meta['std_elev_within_20km'] = mean_std_list

    
else:
    meta = None





# Find all stations in the ERA5 folder (rank 0)
if rank == 0:
    stations = set()
    pattern = re.compile(r"ERA5_(.+)_(\d{4})\.nc$")  # captures station + year  # re.compile(r"ERA5_(.+)_2023.nc$")   # 
    for filename in os.listdir(ERA5_folder):
        match = pattern.match(filename)
        if match:
            station = match.group(1)
            stations.add(station)
    stations = sorted(list(stations))
else:
    stations = None

# stations = stations[2:3]
#stations = ["US-ASH"]  # "ALLN2""LVYN2"    NSWM  NSTV   NSSV   NSPA   NSNA   NSMV  NROG NREE  NPWL  NPVA
#stations = stations[1500:1501]

# Broadcast station list and metadatato all ranks
meta = comm.bcast(meta, root=0)
stations = comm.bcast(stations, root=0)
num_stations = len(stations)

# Partition stations among ranks
stations_per_rank = num_stations // comm_size
remainder = num_stations % comm_size
if rank < remainder:
    start = rank * (stations_per_rank + 1)
    end = start + stations_per_rank + 1
else:
    start = rank * stations_per_rank + remainder
    end = start + stations_per_rank

my_stations = stations[start:end]

comm.barrier()


# Performance timer for the main loop
loop_start_time = time.perf_counter()


for i, station in enumerate(my_stations):

    # Skip if merged file already exists
    merged_path = os.path.join(save_folder, f"merged_HRRR_ERA5_obs_{station}.nc")
    if os.path.exists(merged_path):
        print(f"[RANK {rank}] Skipping station {station} because merged file already exists: {merged_path}", flush=True)
        continue 

    print(f"[RANK {rank}] Processing station: {station}, {i+start}/{num_stations}", flush=True)

    ### Read data

    # Read HRRR data
    hrrr_files = glob.glob(os.path.join(hrrr_folder, f"HRRR_{station}_*.nc"))
    if hrrr_files == []:
        print(f"[RANK {rank}] [WARNING] No HRRR files found for station {station}. Skipping.", flush=True)
        continue
    per_station = xr.open_mfdataset(hrrr_files, engine='netcdf4', 
                                    concat_dim ="valid_time",combine='nested', chunks={})
    per_station = per_station.sortby('valid_time')

    if per_station[["u10_h", "v10_h"]].to_array().isnull().all().compute().item():
        print(f"[RANK {rank}] [WARNING] Skipping station {station} because all HRRR data is NaN.", flush = True)
        continue

    # Read ERA5 data
    ERA5_folder_files = glob.glob(os.path.join(ERA5_folder, f"ERA5_{station}_*.nc"))
    ERA5_folder_files = filter_valid_netcdf(ERA5_folder_files)
    if ERA5_folder_files == []:
        print(f"[RANK {rank}] [WARNING] No ERA5 files found for station {station}. Skipping.", flush=True)
        continue
    per_station_era5 = xr.open_mfdataset(ERA5_folder_files, engine='netcdf4', 
                                    concat_dim ="valid_time",combine='nested', chunks={})
    per_station_era5 = per_station_era5.sortby('valid_time')

    if per_station_era5[["u10", "v10"]].to_array().isnull().all().compute().item():
        print(f"[RANK {rank}] [WARNING] Skipping station {station} because all ERA5 data is NaN.", flush = True)
        # Some ERA5 stations are all NaN. This is probably because the closest gridpoint is over water and not included in ERA5-Land.
        continue

    # Read observation data
    all_dfs = []

    pattern = os.path.join(base_dir, "*", station, "*.parquet")
    found = sorted(glob.glob(pattern))

    if not found:
        print(f"[RANK {rank}] [WARNING] No observation files found for station {station}. Skipping.", flush=True)
        continue
    for f in found:
        try:
            df = pd.read_parquet(f)
            station_id = os.path.basename(os.path.dirname(f))
            df["station_id"] = station_id
            all_dfs.append(df)
        except Exception as e:
            print(f"[RANK {rank}] Failed to read {f}: {e}", flush=True)

    obs = pd.concat(all_dfs, ignore_index=True)

    if obs.drop(columns=['station_id', "timestamp"], errors='ignore').isna().all().all():
        print(f"[RANK {rank}] [WARNING] Skipping station {station} because all obs data is NaN.", flush = True)
        continue

    ### Modify data

    # Obs data QC
    obs.index = obs.timestamp
    obs = obs.sort_index()

    # plt.figure()
    # plt.plot(obs['windspeed'])

    obs['windspeed'] = obs['windspeed'].where(obs['windspeed'] < 40, np.nan)
    obs['windspeed'] = obs['windspeed'].where(obs['windspeed'] > 0, np.nan)

    filter_window = "3h"
    obs['windspeed'] = obs['windspeed'].where( np.abs(obs['windspeed'] - obs['windspeed'].rolling(filter_window, center=True, min_periods=1).median() ) 
                         <= (3* obs['windspeed'].rolling(filter_window, center=True, min_periods=1).std() ) , np.nan)  
         
    # Ensure 'winddirection' column exists
    if 'winddirection' not in obs.columns:
        obs['winddirection'] = np.nan                                                 
    
    obs['winddirection'] = obs['winddirection'].where((obs['winddirection'] <= 360) & (obs['winddirection'] >= 0), np.nan)

    # plt.plot(obs['windspeed'])
    # plt.show()


    # Calculate u and v for obs
    wind_speed = obs['windspeed'].values
    wind_direction = obs['winddirection'].values
    obs['u'] , obs['v'] = uv_from_wspd_wdir(wind_speed, wind_direction)

    # Station metadata
    row = meta[meta['station_id'] == station].iloc[0]
    lon = row['lon']
    lat = row['lat']
    obs_height = row['height']


    # # Lat and lon in HRRR change between HRRR v2 and v3 (23 Aug 2016) - the difference in only about 53m and seems to be constant across stations
    # dist = haversine(per_station.latitude.min().values, per_station.longitude.min().values, per_station.latitude.max().values, per_station.longitude.max().values)

    # # Fsr varies with seasons for some stations (CA_MERCED_23_WSW) and not for others ()
    # # per_station["fsr_smooth"] = per_station["fsr"].where(per_station["fsr"] !=0.011, np.nan)   # 0.011 seems to be default value for missing
    # # per_station["fsr_smooth"] = per_station["fsr_smooth"].rolling(valid_time=500, center=True, min_periods=1).median()

    # Define an average surface roughness per station (use HRRRv4 - 2 Dec 2020, and exclude the default value of 0.011)
    per_station_v4 = per_station.sel(valid_time=slice(pd.to_datetime("2020-12-02"), None))
    fsr_avg = per_station_v4["fsr"].where(per_station_v4["fsr"] !=0.011, np.nan).dropna(dim="valid_time").mean().values
    if np.isnan(fsr_avg):
        fsr_avg = 0.011  # default if no valid data


    # plt.figure()
    # plt.title(f"{station}, dist in lan/lot = {dist:.1f} m")
    # plt.plot(per_station.valid_time.values, per_station.fsr.values, ".", label ="FSR")
    # #plt.plot(per_station.valid_time.values, per_station.fsr_smooth.values, ".", label ="FSR Smooth")
    # plt.hlines(fsr_avg, label="FSR Avg", xmin=per_station.valid_time.values.min(), xmax=per_station.valid_time.values.max(), colors='black')
    # plt.plot(per_station.valid_time.values, (per_station.latitude.values-per_station.latitude.mean().values)*100,
             
    #           ".", label ="(Latitude - mean)*100")
    # plt.plot(per_station.valid_time.values, (per_station.longitude.values-per_station.longitude.mean().values)*100, 
    #          ".", label ="(Longitude - mean)*100")
    # plt.legend()
    # plt.show()



    # Calculate 3m and 10m wind for obs
    obs['windspeed_3m'] = standardize_wspd_height(wind_speed, obs_height, target_height=3.0, z0=fsr_avg)
    obs['u_3m'], obs['v_3m'] = uv_from_wspd_wdir(obs['windspeed_3m'].values, wind_direction)

    obs['windspeed_10m'] = standardize_wspd_height(wind_speed, obs_height, target_height=10.0, z0=fsr_avg)
    obs['u_10m'], obs['v_10m'] = uv_from_wspd_wdir(obs['windspeed_10m'].values, wind_direction)

    # plt.figure()
    # plt.plot(obs['windspeed_3m'], label='Obs 3m')
    # plt.plot(obs['windspeed_10m'], label='Obs 10m')
    # plt.plot(obs['windspeed'], "--", label=f'Obs original {obs_height}m')
    # plt.legend()
    # plt.show()


    # Drop duplicates
    if per_station.indexes['valid_time'].has_duplicates:
        _, unique_idx = np.unique(per_station['valid_time'].values, return_index=True)
        per_station = per_station.isel(valid_time=np.sort(unique_idx))

    if per_station_era5.indexes['valid_time'].has_duplicates:
        _, unique_idx = np.unique(per_station_era5['valid_time'].values, return_index=True)
        per_station_era5 = per_station_era5.isel(valid_time=np.sort(unique_idx))

    # Add prefixes to variable names
    rename_dict = {v: f'hrrr_{v}' for v in per_station.data_vars}
    per_station = per_station.rename(rename_dict)

    rename_dict_era5 = {v: f'era5_{v}' for v in per_station_era5.data_vars}
    per_station_era5 = per_station_era5.rename(rename_dict_era5)

    rename_dict = {}
    if 'latitude' in per_station_era5.coords:
        rename_dict['latitude'] = 'era5_latitude'
    if 'longitude' in per_station_era5.coords:
        rename_dict['longitude'] = 'era5_longitude'
    if rename_dict:
        per_station_era5 = per_station_era5.rename(rename_dict)

    # Make hrrr_longitude and hrrr_latitude variables scalar coordinates (not depending on valid_time)
    for coord_var in ['hrrr_longitude', 'hrrr_latitude']:
        if coord_var in per_station:
            # Extract as scalar if possible
            value = per_station[coord_var].values
            # If it's an array, take the last value (assume constant)
            if hasattr(value, '__len__') and not isinstance(value, str):
                value = value.flat[-1]
            per_station = per_station.drop_vars(coord_var)
            per_station = per_station.assign_coords({coord_var: value})


    obs_df = obs.copy()
    obs_df = obs_df.rename(columns={'timestamp': 'valid_time'})
    obs_vars = [c for c in obs_df.columns if c not in ['valid_time', 'station_id']]
    obs_df = obs_df.rename(columns={v: f'obs_{v}' for v in obs_vars})
    obs_df = obs_df.sort_values('valid_time').drop_duplicates('valid_time')

    ### Merge HRRR, ERA5 and obs
    obs_xr = xr.Dataset()
    obs_xr = obs_xr.assign_coords(valid_time=(['valid_time'], pd.to_datetime(obs_df['valid_time'].values)))
    obs_xr = obs_xr.assign_coords(station=per_station['station'].values.item())
    for v in obs_vars:
        obs_xr[f'obs_{v}'] = (['valid_time'], obs_df[f'obs_{v}'].values)

    merged = xr.merge([per_station, per_station_era5, obs_xr], join="outer")

    # Add obs_longitude and obs_latitude as coordinates and height and elev from meta
    assign_coords_dict = {}
    assign_coords_dict['obs_longitude'] = float(lon)
    assign_coords_dict['obs_latitude'] = float(lat)
    row = meta[meta['station_id'] == station].iloc[0]
    if 'height' in row:
        assign_coords_dict['obs_height'] = float(row['height'])
    if 'elev' in row:
        assign_coords_dict['elevation'] = float(row['elev'])
    if 'std_elev_within_20km' in row:
        assign_coords_dict['std_elev_within_20km'] = float(row['std_elev_within_20km'])
    assign_coords_dict['fsr_avg'] = fsr_avg
    if assign_coords_dict:
        merged = merged.assign_coords(**assign_coords_dict)

    if 'station' in merged and 'valid_time' in merged['station'].dims:
        station_val = merged['station'].values[0]
        merged = merged.drop_vars('station')
        merged = merged.assign_coords(station=station_val)

    # Save to NetCDF
    merged.to_netcdf(merged_path)
    print(f"[RANK {rank}] Saved merged NetCDF: {merged_path}", flush=True)


    # Plotting

    #per_station = per_station.fillna(0)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axs[0].plot(merged.valid_time.values, merged.era5_wspd10.values, ".", ms = 1, color="red", label="ERA5")
    axs[0].plot(merged.valid_time.values, merged.hrrr_wspd10_h.values, ".", ms = 1, color="blue", label="HRRR")
    axs[0].plot(merged.valid_time.values, merged.obs_windspeed.values, ".", ms = 1, color="green", label=f"Obs {merged.obs_height.values}m {station}")
    axs[0].set_ylabel("Wind Speed (m/s)")
    axs[0].legend(loc = "upper right", markerscale=5)
    axs[0].grid(True)
    axs[1].plot(merged.valid_time.values, merged.era5_wdir10.values, ".", ms = 1, color="red", label="ERA5")
    axs[1].plot(merged.valid_time.values, merged.hrrr_wdir10_h.values, ".", ms = 1, color="blue", label="HRRR")
    axs[1].plot(merged.valid_time.values, merged.obs_winddirection.values, ".", ms = 1, color="green", label=f"Obs {merged.obs_height.values}m {station}")
    axs[1].set_ylabel("Wind Direction (deg)")
    axs[1].set_xlabel("Time")
    axs[1].grid(True)
    plt.tight_layout()
    # plt.show(block=False)
    plt.savefig(os.path.join(plots_dir, f"{station}.png"))
    plt.close()


# Print elapsed time
comm.barrier()
loop_end_time = time.perf_counter()
print(f"[TIMER] Loop over stations took {(loop_end_time - loop_start_time)/60:.2f} minutes.", flush=True)
    