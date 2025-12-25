"""
MPI Execution Note (HOW TO RUN):
This script is designed for parallel execution with mpi4py.

Examples:

Local / simple:
module load mpi
module load anaconda3
conda activate thrive
mpirun -n 8 python3 extract_data_UE.py
mpirun --use-hwthread-cpus -n 16 python3 extract_data_UE.py

Or on a SLURM cluster:
salloc -n 16 --time=24:00:00 --account=sfcwinds --mem-per-cpu=5G --qos=high
module load anaconda3
conda activate thrive
srun -n 16 python3 /home/uegerer/sfcwinds/sfcwinds_on_kestrel/extract_data_UE.py

Runtime behavior:
  - Rank 0 builds/loads the station mapping pickle (station_mapping_<area>.pkl) then broadcasts indices to all ranks.
  - Each rank processes a disjoint subset of ERA5 files based on naive partitioning.
  - Daily per-station files are written under <save_folder>/daily by all ranks.
  - Final merging per station on all ranks after an MPI barrier.

"""

import os
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
from Functions_sfcwinds import *
from mpi4py import MPI
import warnings
import time
warnings.simplefilter("ignore")
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys


# Enable interactive backend when running in IPython, ignore in plain Python scripts (this is for data inspection with interactive plots)
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass


t0 = time.perf_counter()


comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
comm_size = comm.Get_size() 

print(f" MPI_COMM {comm} RANK {rank} SIZE {comm_size}\n")


# === MPI-safe setup: Only rank 0 does setup, then broadcasts ===
if rank == 0:

    print("[STATUS] Rank 0: Starting setup (directory creation, file finding, metadata reading, mapping)...", flush=True)

    # Where to save the files
    save_folder = "/kfs2/projects/sfcwinds/ERA5_station_data"
    if not os.path.exists(save_folder):
        print(f"[STATUS] Creating save folder: {save_folder}", flush=True)
        os.makedirs(save_folder)
    if not os.path.exists(save_folder+"/daily"):
        print(f"[STATUS] Creating daily subfolder: {save_folder}/daily", flush=True)
        os.makedirs(save_folder+"/daily")

    # ERA5 data
    ERA5_folder = "/kfs2/projects/sfcwinds/ERA5"

    # Find all ERA5 files 
    file_pattern = os.path.join(ERA5_folder, f"ERA5_land_*.nc")
    file_list = sorted(glob.glob(file_pattern))[::]   # !!! for testing only every 100th file
    print(f"[STATUS] Found {len(file_list)} ERA5 files.", flush=True)


    # Directory for the daily files
    daily_dir = os.path.join(save_folder, "daily")

    # Delete corrupted processed files so that they can be processed again (hopefully correctly) 
    corrupted_file_check = False
    if corrupted_file_check == True:
        
        print("[STATUS] Deleting corrupted processed files...", flush=True)
        file_part_pattern = os.path.join(daily_dir, f"ERA5_all_stations_*.nc")
        file_part_list = sorted(glob.glob(file_part_pattern))[::]
        good_files = []
        for i, f in enumerate(file_part_list):
            if i % 50 == 0:
                print(f"[STATUS] Checking file {i+1} of {len(file_part_list)}: {f}", flush=True)
            try:
                with xr.open_dataset(f, engine='netcdf4') as ds:
                    pass  # File is readable
                good_files.append(f)
            except Exception as e:
                try:
                    os.remove(f)
                    print(f"[INFO] Deleted corrupted file: {f}", flush=True)
                except Exception as del_e:
                    print(f"[ERROR] Could not delete {f}: {del_e}", flush=True)
        if not good_files:
            raise RuntimeError("[ERROR] No valid NetCDF files found!", flush=True)
        del ds
        print(f"[STATUS]  {len(file_part_list) - len(good_files)} corrupted ERA files out of {len(file_part_list)} deleted.", flush=True)

    # Check which daily ERA5 files are already processed
    processed_files = set()
    for fname in os.listdir(daily_dir):
        if fname.startswith("ERA5_all_stations_") and fname.endswith(".nc"):
            # Extract the date string from the filename
            processed_files.add(fname[-13:-3])
    print(f"[STATUS] Found {len(processed_files)} already processed daily files in {daily_dir}.", flush=True)

    # Filter file_list to only unprocessed files
    file_list = [f for f in file_list if f[-13:-3] not in processed_files]
    print(f"[STATUS] {len(file_list)} ERA5 files remain to be processed.", flush=True)

    # Station mapping cache   (defined in HRRR extraction script)
    mapping_pickle = os.path.join("/kfs2/projects/sfcwinds/HRRR_station_data", f"station_mapping_US_SW.pkl")
    if os.path.exists(mapping_pickle):
        print(f"[STATUS] Loading cached station mapping from {mapping_pickle}...", flush=True)
        with open(mapping_pickle, 'rb') as f:
            mapping = pickle.load(f)
        stations_found = mapping.get('stations_found', [])
        print(f"[STATUS] Loaded cached station mapping (n={len(stations_found)})", flush=True)
    else:
        print("[STATUS] Compute station mapping in the HRRR script!", flush=True)
        sys.exit(1)

    # Build lat/lon arrays for stations (ERA5 has lat/lon as dimensions)
    meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")
    station_lons = []
    station_lats = []
    for station in stations_found:
        row = meta[meta['station_id'] == station].iloc[0]
        station_lons.append(row['lon'])
        station_lats.append(row['lat'])

else:
    save_folder = None
    ERA5_folder = None
    file_pattern = None
    file_list = None
    meta = None
    mapping_pickle = None
    stations_found = None
    station_lons = None
    station_lats = None

# Broadcast all setup variables to all ranks
save_folder = comm.bcast(save_folder, root=0)
ERA5_folder = comm.bcast(ERA5_folder, root=0)
file_pattern = comm.bcast(file_pattern, root=0)
file_list = comm.bcast(file_list, root=0)
stations_found = comm.bcast(stations_found, root=0)
station_lons = comm.bcast(station_lons, root=0)
station_lats = comm.bcast(station_lats, root=0)


# Ensure that all threads have stations_found, x_closest, and y_closest prior to the file sweep 
comm.barrier()


# Parameters for naive partitioning
# Length of file list
len_file_list=len(file_list)
# Number of files per rank without the residual
deltafile=len_file_list // comm_size
# Residual number of files when len_file_list/comm_size is not an integer
deltafile_res=len_file_list % comm_size
# Lower bound of the files to be swept over for this rank
lbound=deltafile*rank
# Upper bound of the files to be swept over for this rank
if (rank==comm_size-1):
    ubound=deltafile*(rank+1)+deltafile_res-1
else:
    ubound=(rank+1)*deltafile-1



# Sweep over all files, one file at a time
for file_ind in range (lbound, ubound+1):  # []: #

    # Calculate the local file index for this rank
    local_idx = file_ind - lbound + 1
    local_total = ubound - lbound + 1
    print(f"[STATUS] RANK {rank}: Processing ERA5 file {file_ind+1} of {len_file_list} (file {local_idx} of {local_total} for this rank)", flush=True)
    ifile = file_list[file_ind]
    try:
        # Opening ifile, which is only one file
        era = xr.open_dataset(ifile, engine='netcdf4', chunks={'valid_time':24,'isobaricInhPa':3,'y':10,'x':10})
    except Exception as e:
        print(f"[ERROR] RANK {rank}: Failed to open {ifile}: {e}", flush=True)
        continue

    # Reduce the ERA5 file (this is all we donwloaded anyways)
    needed = [ 'u10','v10','t2m','d2m','sp']
    era = era[ [v for v in needed if v in era] ]

    # Drop unused spatial dimensions
    drop_coords = ["number", "expver"]
    era = era.drop_vars(drop_coords)

    # Build DataArrays for station lat/lon
    station_lon_da = xr.DataArray(station_lons, dims="station")
    station_lat_da = xr.DataArray(station_lats, dims="station")

    # Extract all stations from ERA5 using nearest lat/lon
    era_pts = era.sel(longitude=(station_lon_da), 
                      latitude=(station_lat_da),
                      method="nearest")
    era_pts = era_pts.assign_coords(station=("station", stations_found))

    # Close daily ERA5 file after all stations extracted
    era.close()
    del era

    # Wind processing vectorized over station
    era_pts["wspd10"], era_pts["wdir10"] = wspd_wdir_from_uv(era_pts["u10"], era_pts["v10"])

    # Daily file string
    sub_str = ifile[-13:-3]

    # Cast float64 to float32
    for v in era_pts.data_vars:
        if era_pts[v].dtype == "float64":
            era_pts[v] = era_pts[v].astype("float32")

    # save daily file with all stations
    file_path = os.path.join(save_folder, "daily", f"ERA5_all_stations_{sub_str}.nc")
    era_pts.to_netcdf(file_path)

# Ensure that all threads are here prior to the merging of all datewise written files
comm.barrier()



#%% Read all-station ERA5 files


print(f"[STATUS] RANK {rank}: Loading and concatenating daily files...", flush=True)
file_part_pattern = os.path.join(save_folder, "daily", f"ERA5_all_stations_*.nc")
file_part_list = sorted(glob.glob(file_part_pattern))[::]   # !!! for testing only first 10 files

# # Exclude bad files from merging
# good_files = []
# for i, f in enumerate(file_part_list):
#     if i % 50 == 0:
#         print(f"[STATUS] Checking file {i+1} of {len(file_part_list)}: {f}", flush=True)
#     try:
#         with xr.open_dataset(f, engine='netcdf4') as ds:
#             pass  # File is readable
#         good_files.append(f)
#     except Exception as e:
#         print(f"[WARNING] Skipping and deleting corrupted file: {f} ({e})", flush=True)
#         try:
#             os.remove(f)
#             print(f"[INFO] Deleted corrupted file: {f}", flush=True)
#         except Exception as del_e:
#             print(f"[ERROR] Could not delete {f}: {del_e}", flush=True)
# if not good_files:
#     raise RuntimeError("[ERROR] No valid NetCDF files found!", flush=True)
# del ds

good_files = file_part_list
concat_data = xr.open_mfdataset(good_files, engine='netcdf4', 
                                concat_dim ="valid_time",combine='nested', chunks={})
concat_data = concat_data.sortby("valid_time").drop_duplicates("valid_time")
concat_data = concat_data.chunk({'valid_time': 4*24 * 30, 'station': 1})
print(f"[STATUS] ERA5 files loaded and transformed.", flush=True)

# Broadcast the number of stations to all ranks
if rank == 0:
    num_stations = concat_data.sizes['station']
else:
    num_stations = None
num_stations = comm.bcast(num_stations, root=0)

# Partition stations among ranks
stations_per_rank = num_stations // comm_size
remainder = num_stations % comm_size
if rank < remainder:
    start = rank * (stations_per_rank + 1)
    end = start + stations_per_rank + 1
else:
    start = rank * stations_per_rank + remainder
    end = start + stations_per_rank

# Each rank loads the data if needed (xarray lazy loading, so this is fine)
if rank == 0:
    print(f"[STATUS] Saving station data in parallel across ranks...", flush=True)

for local_idx, i in enumerate(range(start, end)[::-1], 1):
    print(f"[STATUS] RANK {rank}: Processing station {i+1} of {num_stations} (station {local_idx} of {end-start})", flush=True)
    per_station = concat_data.isel(station=i)
    # Ensure valid_time is datetime64[ns]
    if not np.issubdtype(per_station['valid_time'].dtype, np.datetime64):
        per_station = per_station.assign_coords(valid_time=per_station['valid_time'].astype('datetime64[ns]'))
    # Loop over years
    years = np.arange(2000, 2026)  # [2000, 2022] # 
    for year in years:
        total_output_path = os.path.join(save_folder, f"ERA5_{per_station.station.values}_{year}.nc")
        if os.path.exists(total_output_path):
            print(f"[STATUS] RANK {rank}: File already exists, skipping: {total_output_path}", flush=True)
            continue
        # Select data for this year
        year_mask = (per_station['valid_time'].dt.year == year)
        if year_mask.sum().item() == 0:
            continue  # No data for this year
        per_station_year = per_station.sel(valid_time=year_mask)
        encoding = {v: {'zlib': True, 'complevel': 1, 'shuffle': True} for v in per_station_year.data_vars}
        encoding = {v: {} for v in per_station_year.data_vars}
        enc = encoding.copy()
        enc['valid_time'] = {
            'units': 'seconds since 1970-01-01 00:00:00',
            'calendar': 'standard',
            'dtype': 'float64'
        }
        per_station_year.to_netcdf(total_output_path, encoding=enc)
        per_station_year.close()
        print(f"[STATUS] RANK {rank}: Saved {total_output_path}", flush=True)

if rank == 0:
    print("[STATUS] Done writing all station files.", flush=True)
    print("[STATUS] Elapsed time after writing all complete time series (min):", (time.perf_counter() - t0)/60, flush=True)






