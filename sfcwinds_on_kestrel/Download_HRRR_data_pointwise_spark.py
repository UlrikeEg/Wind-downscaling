from herbie import Herbie, FastHerbie
import herbie
#from toolbox import EasyMap, pc
#from paint.standard2 import cm_tmp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import numpy as np
#from toolbox.cartopy_tools import common_features, pc
import pandas as pd
import numpy as np
import glob
import matplotlib.cm as cm
import xarray as xr
from datetime import datetime, timedelta
import time
import os
import warnings
import pickle
warnings.filterwarnings("ignore", message="Will not remove GRIB file because it previously existed.")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*decode_timedelta.*")


# PySpark (new)
try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None
import subprocess
import re
import sys




"""

HRRR model documentation:
https://rapidrefresh.noaa.gov/hrrr/

Documentation for the HRRR data download wrapper:
https://herbie.readthedocs.io/en/stable/

Variables:
https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfnatf00.grib2.shtml


"""


#%% Define

# Folder to save datasets
save_folder = "/kfs2/projects/sfcwinds/HRRR_pointwise_new"

    
G3P3 = (-106.510100,  34.962400, "G3P3")


# date range with one-day frequency
date_range = pd.date_range(datetime(2025, 3, 21), 
                           datetime.now(), 
                           freq="h").tolist()[::-1]   # list starts from end

# # Get time ranges from obs file
# obs = pd.read_pickle("Tower_obs_2020.pkl")
# timestamps = obs.index.round("h").unique()
# all_ranges = [pd.date_range(end=ts+pd.to_timedelta("2h"), periods=21, freq="h") for ts in timestamps.dropna()] # Create a full hourly range for each timestamp: from t-20h to t
# date_range = pd.DatetimeIndex(sorted(set(ts for rng in all_ranges for ts in rng)))[::-1]# Flatten the list of ranges and remove duplicates


# Forecast hour (0=analysis, 2 is recommended)
fxx = list(range(0, 18)) #  (i.e., 0, 18)




# Chunk dictionaries
chunk_dict_hourly = {"valid_time": 1, "isobaricInhPa": 1, "y": 200, "x": 200}
chunk_dict_daily = {"valid_time": 12, "isobaricInhPa": 1, "y": 200, "x": 200}




# Define compression settings
comp = {"zlib": True, "complevel": 4}  


#%% Functions


def safe_xarray(herbie_obj, search_str):
    
    """ accessing the Herbie object sometimes gives a permission error, adding a wait time helps."""
    
    for _ in range(5):  # Retry up to 5 times
        try:
            return herbie_obj.xarray(search_str, remove_grib=True)
        except PermissionError:
            time.sleep(1)  # Wait and retry
    raise Exception(f"Failed to load {search_str} after multiple attempts.")
    
    
def process_hourly_data(timestamp, loc, fxx):
    """Retrieve, process, and return hourly HRRR dataset"""
    
    max_retries = 10  # Set a limit to avoid infinite loops
    attempt = 0
    
    while attempt < max_retries:
        
        try:

            H_subh = FastHerbie([timestamp], model='hrrr', product='subh', fxx=fxx, verbose=False)  
            
            #H_subh.inventory().variable.values   
            #H_subh.inventory()[["variable", "search_this"]].values 

            
            df = H_subh.inventory()[["variable", "search_this"]]
            filtered_df = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd):10\b", case=False, na=False) 
                             & ~df["search_this"].str.contains("ave", case=False, na=False)
                             ]
            wind_search_string_10 = "|".join(filtered_df["search_this"].drop_duplicates().values)
     
            filtered_df = df[df["search_this"].str.contains(r"\b(?:ugrd|vgrd):80\b", case=False, na=False) 
                             & ~df["search_this"].str.contains("ave", case=False, na=False)
                             ]
            wind_search_string_80 = "|".join(filtered_df["search_this"].drop_duplicates().values)

            # Subhourly data
            ds_subh = xr.merge([
                safe_xarray(H_subh, wind_search_string_10),
                safe_xarray(H_subh, wind_search_string_80).rename({"u": "u80", "v": "v80"}),
                safe_xarray(H_subh, ':TMP:2 m|DPT'),
                safe_xarray(H_subh, 'GUST')
            ], compat='minimal')
            
            
            # Select location
            points = pd.DataFrame(
                {
                    "latitude": [loc[1]],
                    "longitude": [loc[0]],
                    "stid": [loc[2]],
                }
            )
            
            hrrr = ds_subh.herbie.pick_points(points).isel(point=0)
            
            # # use valid_time instead of forecast step
            # hrrr = hrrr.swap_dims({"step": "valid_time"})
            
    
            # Cleanup variables and attributes
            hrrr = hrrr.drop_vars(["boundaryLayerCloudLayer", "heightAboveGroundLayer", "metpy_crs", 
                                   "surface", "gribfile_projection", "point", "point_latitude", 
                                   "point_longitude", "point_stid", "point_grid_distance"], errors="ignore")
            
            # Add the valid_time coordinate
            hrrr = hrrr.assign_coords(valid_time=hrrr.time + hrrr.step)
            
            # Add wind speed and direction
            hrrr['wspd10'] = (hrrr.u10**2 + hrrr.v10**2)**0.5
            hrrr['wdir10'] = np.degrees(np.arctan2(hrrr.u10, hrrr.v10))
            hrrr['wspd80'] = (hrrr.u80**2 + hrrr.v80**2)**0.5
            hrrr['wdir80'] = np.degrees(np.arctan2(hrrr.u80, hrrr.v80))
    
            return hrrr
        except Exception as e:
    
            print(f"Error: {e}")
            
            attempt += 1
            print(f"Attempt {attempt} failed. Retrying...")
            time.sleep(1)
    raise Exception(f"Failed to process data for {timestamp} after {max_retries} attempts.")

def _process_and_save(ts, loc, fxx_list, save_folder, comp_dict):
    """Process and save HRRR data for a given timestamp and location."""
    
    # Rebuild file path
    hourly_file_path = os.path.join(save_folder, f"hrrr_{loc[2]}_{ts.date()}_{ts.hour:02d}h.nc")
    if os.path.exists(hourly_file_path):
        return
    hrrr = process_hourly_data(ts, loc=loc, fxx=fxx_list)
    if hrrr is None:
        return
    encoding = {v: {**comp_dict, "chunksizes": (1,)} for v in hrrr.data_vars}
    hrrr.to_netcdf(hourly_file_path)

def _java_version_ok(min_major=17):
    """Return True if system java major version >= min_major."""
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT).decode()
        m = re.search(r'version\s+"(\d+)(?:\.(\d+)\.(\d+))?', out)
        if not m:
            return False
        major = int(m.group(1))
        return major >= min_major
    except Exception:
        return False

# New lightweight diagnostics
def _spark_env_diag():
    info = {}
    info["JAVA_HOME"] = os.environ.get("JAVA_HOME", "UNSET")
    info["SPARK_HOME"] = os.environ.get("SPARK_HOME", "UNSET")
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT).decode()
        info["java_version_output"] = " | ".join(out.strip().splitlines())
    except Exception as e:
        info["java_version_output"] = f"java -version failed: {e}"
    return info

#%%

if __name__ == "__main__":
    # Build list of timestamps needing processing
    pending = []
    for date in date_range:
        hourly_file_path = os.path.join(save_folder, f"hrrr_{G3P3[2]}_{date.date()}_{date.hour:02d}h.nc")
        if not os.path.exists(hourly_file_path):
            pending.append(date)
    if SparkSession is not None and not _java_version_ok(17):
        print("Java version too old for Spark; falling back to sequential. Need Java 17+.")
        SparkSession = None
    if not pending:
        print("No new timestamps to process.")
    elif SparkSession is None:
        print("PySpark not available; running sequential.")
        for date in pending:
            print(date)
            hrrr = process_hourly_data(date, loc=G3P3, fxx=fxx)
            if hrrr is None:
                continue
            hourly_file_path = os.path.join(save_folder, f"hrrr_{G3P3[2]}_{date.date()}_{date.hour:02d}h.nc")
            encoding = {var: {**comp, "chunksizes": (1,)} for var in hrrr.data_vars}
            hrrr.to_netcdf(hourly_file_path)
            print("Saved hourly file.")
    else:
        # Force consistent Python and master
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
        try:
            spark = (SparkSession.builder
                     .master("local[*]")  # explicit
                     .appName("HRRR_pointwise_download")
                     .config("spark.sql.execution.arrow.pyspark.enabled", "false")
                     .getOrCreate())
        except Exception as e:
            print(f"SparkSession creation failed: {e}")
            print("Diagnostics:")
            for k, v in _spark_env_diag().items():
                print(f"  {k}: {v}")
            print("Falling back to sequential execution.")
            for date in pending:
                print(date)
                hrrr = process_hourly_data(date, loc=G3P3, fxx=fxx)
                if hrrr is None:
                    continue
                hourly_file_path = os.path.join(save_folder, f"hrrr_{G3P3[2]}_{date.date()}_{date.hour:02d}h.nc")
                encoding = {var: {**comp, "chunksizes": (1,)} for var in hrrr.data_vars}
                hrrr.to_netcdf(hourly_file_path)
                print("Saved hourly file.")
        else:
            sc = spark.sparkContext
            loc_bc = sc.broadcast(G3P3)
            fxx_bc = sc.broadcast(fxx)
            save_bc = sc.broadcast(save_folder)
            comp_bc = sc.broadcast(comp)
            rdd = sc.parallelize(pending, numSlices=min(len(pending), sc.defaultParallelism))
            def _worker(ts):
                try:
                    _process_and_save(ts, loc_bc.value, fxx_bc.value, save_bc.value, comp_bc.value)
                except Exception as e:
                    print(f"Worker error {ts}: {e}")
            rdd.foreach(_worker)
            spark.stop()
            print("Parallel processing complete.")
        
#%% Plots
    
    
    
    # # Make a map plot
    # to_plot = ds_sfc.isel(valid_time=0).isel()
    
    # fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    # plt.pcolormesh(to_plot.longitude, to_plot.latitude, to_plot.gust, cmap="viridis", shading="auto", transform=ccrs.PlateCarree())
    # plt.colorbar(label="Wind Speed (m/s)")
    
    # # Add map features
    # ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.STATES, linestyle='-', linewidth=0.5, edgecolor="gray") 
    # ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.3)
    # ax.add_feature(cfeature.LAKES, edgecolor='black', alpha=0.3)
    # ax.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.3)
    # terrain = cfeature.NaturalEarthFeature(
    #     category='physical',
    #     name='shaded_relief',
    #     scale='110m'  # High resolution
    # )
    
    
    
    
    
    
    # # Timeseries of wind
    # fig = plt.figure(figsize=(11,4))
     
    # ax2 = plt.subplot(2, 1, 1)
    # plt.ylabel('Wind speed (m s$^{-1}$)')
    # ax2.plot( hrrr.valid_time, hrrr.speed,'.', label = "HRRR, 10m", ms=3)
    # ax2.plot( hrrr.valid_time, hrrr.gust,'.', label = "", ms=3, color="C2", alpha = 0.5)
    # plt.legend(loc=1,  markerscale=2)
    # plt.grid()
    # ax2.set_zorder(100)
     
    # ax1 = plt.subplot(2, 1, 2, sharex=ax2)  # 
    # ax1.set_ylabel('Wind dir ($^\circ$)')
    # hrrr['wdir'] = np.degrees(np.arctan2(hrrr.u10, hrrr.v10)) +180
    # ax1.plot( hrrr.valid_time, hrrr.wdir,'.', label = "", ms=3)
    # ax1.set_ylim(0, 360)    
    # plt.grid()    
     
    # #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    # fig.autofmt_xdate()
    # ax1.set_xlabel("Time (UTC)")
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.1)

