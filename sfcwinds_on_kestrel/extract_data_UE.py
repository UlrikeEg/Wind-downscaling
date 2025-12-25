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
salloc -n 16 --time=5:00:00 --account=sfcwinds --mem-per-cpu=2G --qos=high
module load anaconda3
conda activate thrive
srun -n 16 python3 /home/uegerer/sfcwinds/sfcwinds_on_kestrel/extract_data_UE.py

Runtime behavior:
  - Rank 0 builds/loads the station mapping pickle (station_mapping_<area>.pkl) then broadcasts indices to all ranks.
  - Each rank processes a disjoint subset of HRRR files based on naive partitioning.
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
    save_folder = "/kfs2/projects/sfcwinds/HRRR_station_data"
    if not os.path.exists(save_folder):
        print(f"[STATUS] Creating save folder: {save_folder}", flush=True)
        os.makedirs(save_folder)
    if not os.path.exists(save_folder+"/daily"):
        print(f"[STATUS] Creating daily subfolder: {save_folder}/daily", flush=True)
        os.makedirs(save_folder+"/daily")

    # HRRR data
    HRRR_folder = "/kfs2/projects/sfcwinds/HRRR"
    area = "US_SW" # keep this (we only have HRRR data for the US southwest downloaded for the entire 2014-2024 period)

    # Find all HRRR files for this area
    file_pattern = os.path.join(HRRR_folder, f"hrrr_{area}_*.nc")
    file_list = sorted(glob.glob(file_pattern))[::]   # !!! for testing only every 100th file
    print(f"[STATUS] Found {len(file_list)} HRRR files.", flush=True)
    if file_list:
        print("[STATUS] Opening base HRRR file for mapping...", flush=True)
        hrrr_base = xr.open_dataset(file_list[-1], chunks="auto", decode_timedelta=True, engine='netcdf4')
    else:
        hrrr_base = None

    # Directory for the daily files
    daily_dir = os.path.join(save_folder, "daily")

    # Delete corrupted processed files so that they can be processed again (hopefully correctly) 
    corrupted_file_check = False
    if corrupted_file_check == True:
        
        print("[STATUS] Deleting corrupted processed files...", flush=True)
        file_part_pattern = os.path.join(daily_dir, f"HRRR_all_stations_*.nc")
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
        print(f"[STATUS]  {len(file_part_list) - len(good_files)} corrupted HRRR files out of {len(file_part_list)} deleted.", flush=True)

    # Check which daily HRRR files are already processed
    processed_files = set()
    for fname in os.listdir(daily_dir):
        if fname.startswith("HRRR_all_stations_") and fname.endswith(".nc"):
            # Extract the date string from the filename
            processed_files.add(fname[-13:-3])
    print(f"[STATUS] Found {len(processed_files)} already processed daily files in {daily_dir}.", flush=True)

    # Filter file_list to only unprocessed files
    file_list = [f for f in file_list if f[-13:-3] not in processed_files]
    print(f"[STATUS] {len(file_list)} HRRR files remain to be processed.", flush=True)

    # Wind variables specifications in HRRR
    wind_specs = [
        {"u": "u",      "v": "v",      "suffix": ""},
        {"u": "u10",    "v": "v10",    "suffix": "10"},
        {"u": "u10_h",  "v": "v10_h",  "suffix": "10_h"},
        {"u": "u80",    "v": "v80",    "suffix": "80"},
    ]

    # Station metadata
    meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")
    base_dir = "/kfs2/projects/sfcwinds/observations/"
    stations = ["NMC60", "NMC63", 'STN01']   # Test for one station, a state, or similar

    # Station mapping cache  
    mapping_pickle = os.path.join(save_folder, f"station_mapping_{area}.pkl")
    if os.path.exists(mapping_pickle):
        print(f"[STATUS] Loading cached station mapping from {mapping_pickle}...", flush=True)
        with open(mapping_pickle, 'rb') as f:
            mapping = pickle.load(f)
        stations_found = mapping.get('stations_found', [])
        x_closest = mapping.get('x_closest', [])
        y_closest = mapping.get('y_closest', [])
        print(f"[STATUS] Loaded cached station mapping (n={len(stations_found)})", flush=True)
    else:
        print("[STATUS] Computing station mapping (this may take a while)...", flush=True)
        stations_found = []
        x_closest = []
        y_closest = []
        for i, station in enumerate(stations):
            if i % 10 == 0:
                print(f"[STATUS] Mapping stations: {i}/{len(stations)}", flush=True)

            row = meta[meta['station_id'] == station].iloc[0]

            lon = row['lon']
            lat = row['lat']

            # Find closest HRRR location for this station (maybe also the 4 sourrounding stations)
            closest, closest_lat, closest_lon, idx = find_closest_HRRR_loc(hrrr_base, [lat, lon]) 

            # Distance between HRRR loc and station
            distance_m = haversine(closest_lat, closest_lon, lat, lon)

            # Exception handling for stations that cannot be matched within 2 km to a measurement
            if(distance_m>2500.):
                print(f"Actual location for station {station} is {lon:.5f} {lat:.5f}") 
                print(f"Closest found HRRR loc for  {station} is {closest_lon:.5f} {closest_lat:.5f}") 
                print(f"Distance: {distance_m/1000:.2f} km") 
                print("The data is more than 2.5 km away. Rejected.")
                continue
            else:
                stations_found.append(station)
                y_closest.append(idx[0])
                x_closest.append(idx[1])

        # commented to not overwrite the file
        # with open(mapping_pickle, 'wb') as f:
        #     pickle.dump({
        #         'stations_found': stations_found,
        #         'x_closest': x_closest,
        #         'y_closest': y_closest
        #     }, f)

    # Diagnostic plot of HRRR domain and stations using first time slice
    make_map_plot = False
    if make_map_plot  == True and 'u10' in hrrr_base.data_vars and len(stations_found) > 0:
        plots_dir = os.path.join(save_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        gust2d = hrrr_base['u10'].isel(valid_time=0)
        lat_base = hrrr_base.isel(valid_time=0)['latitude']
        lon_base = hrrr_base.isel(valid_time=0)['longitude']
        st_lons = []
        st_lats = []
        for st in stations_found:
            row = meta[meta['station_id'] == st].iloc[0]
            st_lons.append(row['lon'])
            st_lats.append(row['lat'])

        fig = plt.figure(figsize=(9,6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        pcm = ax.pcolormesh(lon_base, lat_base, gust2d.squeeze(), cmap='viridis', shading='auto', transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':')
        gl.bottom_labels = True; gl.left_labels = True; gl.right_labels = False; gl.top_labels = False
        cbar = fig.colorbar(pcm, ax=ax, shrink=0.8); cbar.set_label('U10 component (m/s)')
        ax.scatter(st_lons, st_lats, s=15, c='black', edgecolors='white', linewidth=0.5, label='Stations', transform=ccrs.PlateCarree())
        ax.set_title('HRRR U10 (t0) with Station Locations')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
        fig.tight_layout()
        plt.show()
        plot_path = os.path.join(plots_dir, 'hrrr_u10_station_map.png')
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
else:
    save_folder = None
    HRRR_folder = None
    area = None
    file_pattern = None
    file_list = None
    hrrr_base = None
    wind_specs = None
    meta = None
    base_dir = None
    stations = None
    mapping_pickle = None
    stations_found = None
    x_closest = None
    y_closest = None

# Broadcast all setup variables to all ranks
save_folder = comm.bcast(save_folder, root=0)
HRRR_folder = comm.bcast(HRRR_folder, root=0)
area = comm.bcast(area, root=0)
x_closest = comm.bcast(x_closest,root=0)
y_closest = comm.bcast(y_closest,root=0)
file_pattern = comm.bcast(file_pattern, root=0)
file_list = comm.bcast(file_list, root=0)
wind_specs = comm.bcast(wind_specs, root=0)
base_dir = comm.bcast(base_dir, root=0)
stations = comm.bcast(stations, root=0)
stations_found = comm.bcast(stations_found, root=0)
x_closest = comm.bcast(x_closest, root=0)
y_closest = comm.bcast(y_closest, root=0)
meta = comm.bcast(meta, root=0)
# hrrr_base is not broadcast (xarray objects are not MPI-serializable); only used for mapping on rank 0



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
for file_ind in []: # range (lbound, ubound+1):  #

    # Calculate the local file index for this rank
    local_idx = file_ind - lbound + 1
    local_total = ubound - lbound + 1
    print(f"[STATUS] RANK {rank}: Processing HRRR file {file_ind+1} of {len_file_list} (file {local_idx} of {local_total} for this rank)", flush=True)
    ifile = file_list[file_ind]
    try:
        # Opening ifile, which is only one file
        hrrr = xr.open_dataset(ifile, engine='netcdf4', chunks={'valid_time':96,'isobaricInhPa':3,'y':10,'x':10})
    except Exception as e:
        print(f"[ERROR] RANK {rank}: Failed to open {ifile}: {e}", flush=True)
        continue

    # Reduce the HRRR file
    needed = ['latitude', 'longitude', 'u','v','u10','v10','u10_h','v10_h','u80','v80','t2m','sp','blh','fsr','tp','snowc','gust','veg','gppbfas','d2m','tcc','cape']
    hrrr = hrrr[ [v for v in needed if v in hrrr] ]

    # bring the HRRR latitude between -180 and 180
    hrrr["longitude"] = ((hrrr["longitude"] + 180) % 360) - 180

    # Build indexers with a new 'station' dimension
    x_idx = xr.DataArray(x_closest, dims="station")
    y_idx = xr.DataArray(y_closest, dims="station")

    # Extract all stations simultaneously
    hrrr_pts = hrrr.isel(x=x_idx, y=y_idx).assign_coords(station=("station", stations_found))

    # Close daily HRRR file after all stations extracted
    hrrr.close()
    del hrrr

    # Wind processing vectorized over station
    present = set(hrrr_pts.data_vars)
    for spec in wind_specs:
        u_name, v_name, suffix = spec["u"], spec["v"], spec["suffix"]
        if u_name in present and v_name in present:
            u_rot, v_rot = rotate_to_true_north(hrrr_pts[u_name], hrrr_pts[v_name], hrrr_pts["longitude"])
            wspd_da, wdir_da = wspd_wdir_from_uv(u_rot, v_rot)
            hrrr_pts[u_name] = u_rot
            hrrr_pts[v_name] = v_rot
            wspd_name = f"wspd{suffix}" if suffix else "wspd"
            wdir_name = f"wdir{suffix}" if suffix else "wdir"
            hrrr_pts[wspd_name] = wspd_da
            hrrr_pts[wdir_name] = wdir_da

    # Daily file string
    sub_str = ifile[-13:-3]

    # Cast float64 to float32 excluding longitude/latitude
    for v in hrrr_pts.data_vars:
        if v in ("longitude", "latitude"):
            continue
        if hrrr_pts[v].dtype == "float64":
            hrrr_pts[v] = hrrr_pts[v].astype("float32")

    # save daily file with all stations
    file_path = os.path.join(save_folder, "daily", f"HRRR_all_stations_{sub_str}.nc")
    hrrr_pts.to_netcdf(file_path)

# Ensure that all threads are here prior to the merging of all datewise written files
comm.barrier()



#%% Read all-station HRRR files


print(f"[STATUS] RANK {rank}: Loading and concatenating daily files...", flush=True)
file_part_pattern = os.path.join(save_folder, "daily", f"HRRR_all_stations_*.nc")
file_part_list = sorted(glob.glob(file_part_pattern))[::]

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

# Partition stations among ranks
num_stations = len(stations_found)
stations_per_rank = num_stations // comm_size
remainder = num_stations % comm_size
if rank < remainder:
    start = rank * (stations_per_rank + 1)
    end = start + stations_per_rank + 1
else:
    start = rank * stations_per_rank + remainder
    end = start + stations_per_rank

if rank == 0:
    print(f"[STATUS] Saving station data in parallel across ranks...", flush=True)

# Loop over years first, then over stations
years = np.arange(2014, 2025)[::-1]
for year in years:
    
    year_files = [f for f in good_files if f"_{year}-" in f]

    if not year_files:
        continue
    print(f"[STATUS] RANK {rank}: Opening {len(year_files)} daily files for year {year}...", flush=True)
    concat_data_year = xr.open_mfdataset(year_files, engine='netcdf4',
                                        concat_dim="valid_time", combine='nested', chunks={})
    concat_data_year = concat_data_year.sortby("valid_time").drop_duplicates("valid_time")
    concat_data_year = concat_data_year.chunk({'valid_time': 4*24*30, 'station': 1})
    
    for local_idx, i in enumerate(range(start, end)[::-1], 1):
        print(f"[STATUS] RANK {rank}: Processing station {i+1} of {num_stations} (station {local_idx} of {end-start})", flush=True)
        per_station = concat_data_year.isel(station=i)
        total_output_path = os.path.join(save_folder, f"HRRR_{per_station.station.values}_{year}.nc")
        if os.path.exists(total_output_path):
            print(f"[STATUS] RANK {rank}: File already exists, skipping: {total_output_path}", flush=True)
            continue
        # Ensure valid_time is datetime64[ns]
        if not np.issubdtype(per_station['valid_time'].dtype, np.datetime64):
            per_station = per_station.assign_coords(valid_time=per_station['valid_time'].astype('datetime64[ns]'))
        # Only save if there is data for this year
        if per_station.sizes.get('valid_time', 0) == 0:
            continue
        enc = {'valid_time': {
                'units': 'seconds since 1970-01-01 00:00:00',
                'calendar': 'standard',
                'dtype': 'float64'
            }}
        per_station.to_netcdf(total_output_path, encoding=enc)
        per_station.close()
        print(f"[STATUS] RANK {rank}: Saved {total_output_path}", flush=True)
    concat_data_year.close()

if rank == 0:
    print("[STATUS] Done writing all station files.", flush=True)
    print("[STATUS] Elapsed time after writing all complete time series (min):", (time.perf_counter() - t0)/60, flush=True)






