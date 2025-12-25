import os
import pandas as pd
import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
from Functions_sfcwinds import *
from mpi4py import MPI
import warnings
warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
comm_size = comm.Get_size() 

print(f" MPI_COMM {comm} RANK {rank} SIZE {comm_size}\n")

# Enable interactive backend when running in IPython, ignore in plain Python scripts (this is for data inspection with interactive plots)
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass


# Where to save the files
#save_folder = "/kfs2/projects/sfcwinds/HRRR_station_data"
save_folder = "/scratch/itopcuog/sfcwinds/HRRR_station_data"
if rank==0:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)



# %% HRRR data
#HRRR_folder = "/kfs2/projects/sfcwinds/HRRR"
HRRR_folder = "/home/itopcuog/sfcwinds_dir/reduced_data/HRRR"
#HRRR_folder = "/scratch/itopcuog/medium_data/HRRR_small"
area = "US_SW" # keep this (we only have HRRR data for the US southwest downloaded for the entire 2014-2024 period)
date_String = "*"   # keep * or select a date for testing

# Find all HRRR files for this date and area
file_pattern = os.path.join(HRRR_folder, f"hrrr_{area}_{date_String}*.nc")
file_list = sorted(glob.glob(file_pattern))

# One base HRRR file for getting lon and lat
if file_list:
    hrrr_base = xr.open_dataset(file_list[-1], chunks="auto", decode_timedelta=True, engine='netcdf4')


#%% Filter stations we want so that we can loop over them

# Read metadata of observations (this includes the lon/lat of each station)
meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")

# filter which data to read
base_dir = "/kfs2/projects/sfcwinds/observations/"
#stations = ["NMC60"]   #  meta[meta.state == "NM"].station_id.values  # Test for one station, a state, or similar
stations = ["NMC60", "NMC63"]   #  meta[meta.state == "NM"].station_id.values  # Test for one station, a state, or similar
                       #  eventually use all stations in the US southwest. Maybe filter by state: CA, AZ, NM, UT, NV, CO, TX, WY, OK, KS, NE, SD, OR, ID



# Array to hold all the found stations
stations_found = []
# Array to hold all the station locations
x_closest=[]
y_closest=[]

# Sweeping over all stations to determine the
# closest coordinates for each one
if rank==0 : 
    for station in stations:

        row = meta[meta['station_id'] == station].iloc[0]
        st_ind=stations.index(station);

        lon = row['lon']
        lat = row['lat']
        if lon < 0:
            lon = lon + 360 # convert to 0-360 (HRRR convention)
        print(f"Station: {station}, Longitude: {lon}, Latitude: {lat}")

    # Find closest HRRR location for this station (maybe also the 4 sourrounding stations)
        closest, closest_lat, closest_lon, idx = find_closest_HRRR_loc(hrrr_base, [lat, lon]) 
        print(f"Coordinates are: {closest_lat}, {closest_lon}")

        distance_m = haversine(closest_lat, closest_lon, lat, lon)

        print(f"Actual location for station {station} is {lon} {lat}") 
        print(f"Closest found location for station {station} is {closest_lon} {closest_lat}") 

    # Exception handling for stations that cannot be matched
    # within 2 km to a measurement
        if(distance_m>2000.):
            print("The data is more than 2kms away. Rejected.")
            continue
        else:
            stations_found.append(station)
            y_closest.append(idx[0])
            x_closest.append(idx[1])
            print(f"station {stations_found[-1]}, xc {x_closest[-1]}, yc {y_closest[-1]}")
        

    # loop over all HRRR files and extract data for this location
    # something like 

    print(f"Stations found: {stations_found}\n")
    print(f"x and y closest found: {x_closest} {y_closest}\n")
    print(f"File list: {file_list[:]}\n")

# Broadcast stations_found, x_closest, and y_closest
# to all ranks
stations_found = comm.bcast(stations_found,root=0)
x_closest = comm.bcast(x_closest,root=0)
y_closest = comm.bcast(y_closest,root=0)

# Ensure that all threads have stations_found,
# x_closest, and y_closest priot to the file sweep 
comm.barrier()
print(f"RANK {rank} stations {stations_found} x{x_closest} y {y_closest}\n")

# Parameters for naive partitioning
# Length of file list
len_file_list=len(file_list)
# Number of files per rank without the residual
deltafile=len_file_list // comm_size
# Residual number of files when len_file_list/comm_size
# is not an integer
deltafile_res=len_file_list % comm_size
# Lower bound of the files to be swept over for this rank
lbound=deltafile*rank
# Upper bound of the files to be swept over for this rank
if (rank==comm_size-1):
    #ubound=(rank+1)*deltafile+deltafile_res
    ubound=deltafile*(rank+1)+deltafile_res-1
else:
    ubound=(rank+1)*deltafile-1


# Sweep over all files, one file at a time
#for ifile in file_list:
for file_ind in range (lbound,ubound+1):
    ifile=file_list[file_ind]
    # Opening ifile, which is only one file
    hrrr = xr.open_dataset(ifile, chunks="auto", engine='netcdf4', decode_timedelta= True)

# Sweep over all stations within each file loop
# This way, only one file is open whiledata for several
# stations are being extracted
    for station in stations_found:
        print(f"Station name is {station}")
        st_ind=stations_found.index(station)
        print(f"Station index is {st_ind}")
        hrrr_loc = hrrr.isel(x = x_closest[st_ind], y = y_closest[st_ind]) 

    # correct the HRRR wind direction (model coordinate system into Noth-East)
        # u10
        if (('u10' in hrrr_loc.data_vars) and ('v10' in hrrr_loc.data_vars)):
            hrrr_loc["u10"], hrrr_loc["v10"] = rotate_to_true_north(hrrr_loc["u10"], hrrr_loc["v10"], hrrr_loc["longitude"])
            # calculate wind speed and direction for 10 m
            hrrr_loc["wspd10"], hrrr_loc["wdir10"] = wspd_wdir_from_uv(hrrr_loc["u10"], hrrr_loc["v10"])

        # u10_h
        print(f'{('u10_h' in hrrr_loc.data_vars)}')
        if (('u10_h' in hrrr_loc.data_vars) and ('v10_h' in hrrr_loc.data_vars)):
            hrrr_loc["u10_h"], hrrr_loc["v10_h"] = rotate_to_true_north(hrrr_loc["u10_h"], hrrr_loc["v10_h"], hrrr_loc["longitude"])
        # calculate wind speed and direction for 10_h
            hrrr_loc["wspd10_h"], hrrr_loc["wdir10_h"] = wspd_wdir_from_uv(hrrr_loc["u10_h"], hrrr_loc["v10_h"])

        # u80
        if (('u80' in hrrr_loc.data_vars) and ('v80' in hrrr_loc.data_vars)):
            hrrr_loc["u80"], hrrr_loc["v80"] = rotate_to_true_north(hrrr_loc["u80"], hrrr_loc["v80"], hrrr_loc["longitude"])
        # calculate wind speed and direction for 80 m
            hrrr_loc["wspd80"], hrrr_loc["wdir80"] = wspd_wdir_from_uv(hrrr_loc["u80"], hrrr_loc["v80"])

    # Make some plots comparing wind speed and direction for one / a few stations (compare HRRR to observations)
    # see below for code example
    # This is yet to be done properly

    # save HRRR data for the current station
        # Extracting the date suffix from each hrrr file name
        sub_str=ifile[-13:-3]
        print(f"Subtracted string {sub_str}")
        # Appending the extracted string to the station name
        # so that each station can have individual files for each date
        # that are to be merged later
        file_path = os.path.join(save_folder, f"HRRR_{station}_{sub_str}.nc")
        print(f"Generated file path {file_path}")
        hrrr_loc.to_netcdf(file_path,mode='a')  # there are some encoding options if file size or write speed is an issue, see the download scripts.

    hrrr.close()

# Ensure that all threads are here prior to the
# merging of all datewise written files
comm.barrier()

# Merge the individual files for each station on rank 0
if rank==0:
    for station in stations_found:
        date_String = "*"   # wildcard for dates
        file_part_pattern = os.path.join(save_folder, f"HRRR_{station}_{date_String}*.nc")
        file_part_list = sorted(glob.glob(file_part_pattern))
    # Load the datewise separated files for one station only
    # using open_mfdataset
    # This allows for concatenating over valid_time
        concat_data = xr.open_mfdataset(file_part_list[:], concat_dim ="valid_time",combine='nested', chunks="auto", parallel = True, engine='netcdf4', decode_timedelta= True)
        total_output_path = os.path.join(save_folder, f"HRRR_{station}.nc")
        concat_data.to_netcdf(total_output_path,mode='w')  
    

# # #Read observations according to station filter -= read the actual data
# print (f"Processing station {station}")
#all_dfs = []
 #years    = ["2024"]  # for testing shorter time series
#years    = ["2025"]  # for testing shorter time series
#for year in years:
#    pattern = os.path.join(base_dir, "*", station, f"{year}.parquet")
#    found = glob.glob(pattern)

#    for f in found:
#        try:
#            df = pd.read_parquet(f)
#        # Extract station_id from the path (e.g., .../CoAgMet/HOT01/2023.parquet)
#            station_id = os.path.basename(os.path.dirname(f))  # 'HOT01'
#            df["station_id"] = station_id
#            all_dfs.append(df)
#        except Exception as e:
#            print(f"Failed to read {f}: {e}")

#    print(f"{len(all_dfs)} files found.")
#    obs = pd.concat(all_dfs, ignore_index=True)
#    obs["timestamp"] = obs["timestamp"].dt.tz_convert(None)

# # Plor obs vs. HRRR for this station
#    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# # Wind speed
#    axs[0].plot(obs.timestamp.values, obs.windspeed.values, ".", color="green", label="Obs "+station)
#    axs[0].plot(hrrr_loc.valid_time.values, hrrr_loc.wspd10.values, ".", color="blue", label="HRRR")
#    axs[0].set_ylabel("Wind Speed (m/s)")
#    axs[0].legend()
#    axs[0].grid(True)

# # Wind direction
#    axs[1].plot(obs.timestamp.values, obs.winddirection.values, ".", color="green", label="Obs "+station)
#    axs[1].plot(hrrr_loc.valid_time.values, hrrr_loc.wdir10.values, ".", color="blue", label="HRRR")
#    axs[1].set_ylabel("Wind Direction (deg)")
#    axs[1].set_xlabel("Time")
#    axs[1].grid(True)
#
#    plt.tight_layout()
#    plt.savefig(f"{year}.png")
    #plt.show()






