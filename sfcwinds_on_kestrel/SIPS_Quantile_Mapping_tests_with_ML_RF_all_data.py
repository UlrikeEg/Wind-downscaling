import os
import pandas as pd
import xarray as xr
import glob
#matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
from datetime import datetime
import numpy as np
from scipy.stats import norm, gamma, erlang, expon
import matplotlib.dates as mdates
from sklearn.metrics import root_mean_squared_error
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp  # KS test for distribution similarity
from sklearn.model_selection import cross_val_score  
from sklearn.model_selection import KFold 
from sklearn.base import clone 
import cartopy.crs as ccrs
import cartopy.feature as cfeature



"""
To do:
- add more features (terrain std dev)
- tune hyperparameters (grid search)
- test other models

"""

from Functions_sfcwinds import *

def quantile_mapping(model_series, model_cdf, obs_cdf, quantiles):
    # Find the quantile of each model value in the model CDF
    Quantiles = np.interp(model_series, model_cdf, quantiles)
    # Map these quantiles to observed values using the obs CDF
    corrected_series = np.interp(Quantiles, quantiles, obs_cdf)
    return corrected_series


# Enable interactive backend when running in IPython, ignore in plain Python scripts
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass



#%% Read data
# Read metadata to filter what we want
meta = pd.read_csv("/kfs2/projects/sfcwinds/observations/metadata_CONUS.csv")
states_in_US_SW = ["CA", "AZ", "NM", "UT", "NV", "CO", "TX", "WY", "OK", "KS", "NE", "SD", "OR", "ID"]
meta['state'] = meta['state'].astype(str).str.strip()
meta = meta[meta.state.isin(states_in_US_SW)]
meta = meta.reset_index(drop=True) 


# Directory with observations and model data merged files
base_dir = "/kfs2/projects/sfcwinds/test_case/Obs hrrr comparison results May 2024/" # test case data
base_dir = "/kfs2/projects/sfcwinds/merged_station_data_updated/"  # all data in US Southwest as nc files


# Load the elevation dataset
ncfile = '/kfs2/projects/sfcwinds/environmental_data/CONUS_elevation_1km.nc'
ds = xr.open_dataset(ncfile)
ds = ds.fillna(0)   # NaN values are where no elevation is available (e.g. ocean) - set to zero
ds['elevation_std'] = ds['elevation_std'].where(ds['elevation_std'] <= 100)  
mean_elev = ds['elevation_mean'].values
std_elev = ds['elevation_std'].values

# # Terrain complexity map
# std_elev = ds['elevation_std'].values
# plt.figure(figsize=(9,6))
# ax = plt.axes(projection=ccrs.PlateCarree())
# mesh = ax.pcolormesh(ds['lon'], ds['lat'], ds['elevation_std'],
#                         cmap='terrain', shading='auto',
#                         transform=ccrs.PlateCarree())
# ax.add_feature(cfeature.BORDERS, linewidth=1)
# ax.add_feature(cfeature.STATES, linewidth=0.5)
# ax.add_feature(cfeature.COASTLINE, linewidth=1)
# gl = ax.gridlines(draw_labels=True, alpha=0.4, linestyle=':')
# gl.bottom_labels = True
# gl.left_labels = True
# gl.right_labels = False
# gl.top_labels = False
# cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
# cbar.set_label("Terrain complexity (elevation std dev, m)")
# plt.tight_layout()
# plt.show()


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
meta['mean_std_vicinity'] = mean_std_list

# plt.figure()
# plt.plot(meta.elev, meta.closest_elev, ".")
# plt.plot(meta.elev, meta.mean_std_vicinity, ".")
# plt.show()

# Cut terrain to NM or US SW region
#NM borders
max_lat = 37.2
min_lat = 31.3
max_lon = -103.0   
min_lon = -109.2
# US SW borders
max_lat = 47
min_lat = 25
max_lon = -93  
min_lon = -125

ds = ds.where( (ds.lon > min_lon) & (ds.lon < max_lon), drop=True)  # Cut to NM borders
ds = ds.where((ds.lat > min_lat) & (ds.lat < max_lat), drop=True)
ds = ds.fillna(0)   # NaN values are where no elevation is available (e.g. ocean) - set to zero
mean_elev = ds['elevation_mean'].values
std_elev = ds['elevation_std'].values
lat = ds['lat'].values
lon = ds['lon'].values
ds.close()

print("Data reading and preprocessing complete.")


#%% Test for one station
test_one_station = False

if test_one_station == True:

    # Pick one station for testing
    stid = "SLMN5"  # "SVAN5"   


    # Read files from base_dir
    merged = pd.read_csv(os.path.join(base_dir, f"{stid}_3m_hrrr_10m_merged.csv"))
    # quantile_10m_data = pd.read_csv(os.path.join(base_dir, f"{stid}_3m_hrrr_10m_quantile_data.csv"))
    # quantile_10m_stats = pd.read_csv(os.path.join(base_dir, f"{stid}_3m_hrrr_10m_quantile_stats.csv"))




    ### Plot original data


    # Wspd time series
    fig, ax = plt.subplots()
    ax.plot(merged.valid_time, merged.obs_wspd, ".", label=f"{stid} {meta.loc[meta['station_id'] == stid].height.values[0]}m (obs orig)")
    ax.plot(merged.valid_time[~np.isnan(merged.obs_wspd3)],merged.obs_wspd3[~np.isnan(merged.obs_wspd3)],label= f"{stid}"' 3m')
    ax.plot(merged.valid_time[~np.isnan(merged.obs_wspd3)],merged.hrrr_wspd10[~np.isnan(merged.obs_wspd3)],label='HRRR 10m')
    ax.legend(loc='upper right',fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Wind speed (m/s)')
    ax.legend(loc='upper right',fontsize=10)
    fig.autofmt_xdate()
    #fig.savefig(f"{stid} 3m vs hrrr 10m wind speed test case NM May 2024.png")
    #plt.tight_layout()
    #plt.savefig("obs num time resolution NM.png")
    #plt.xlim([datetime.datetime(2024,8,26,0,0), datetime.datetime(2024,9,10,0,0)])
    #plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # Wdir time series
    fig, ax = plt.subplots()
    ax.plot(merged.valid_time[~np.isnan(merged.obs_wdir)],merged.obs_wdir[~np.isnan(merged.obs_wdir)],label= f"{stid}"' 3m')
    ax.plot(merged.valid_time[~np.isnan(merged.obs_wdir)],merged.hrrr_wdir10_corr[~np.isnan(merged.obs_wdir)],label='HRRR 10m')
    ax.legend(loc='upper right',fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Wind direction (deg)')
    ax.legend(loc='upper right',fontsize=10)
    fig.autofmt_xdate()
    # fig.savefig(f"{stid} 3m  hrrr 10m wind direction test case NM May 2024.png")
        

    # Wsdp histogram
    plt.figure()
    hist1 = plt.hist(merged.obs_wspd3, bins=np.arange(0, 16, 0.1), edgecolor='none',
                    density=False,alpha=0.5,label= f"{stid}"' 3m')
    hist2 = plt.hist(merged.hrrr_wspd10, bins=np.arange(0, 16, 0.1), edgecolor='none',
                    density=False,alpha=0.5,label='hrrr 10m')
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind speed histogram test case NM May 2024.png")
    plt.show()

    # Wdir histogram
    plt.figure()
    hist1 = plt.hist(merged.obs_wdir, bins=np.arange(0, 370, 5), edgecolor='none',
                    density=False,alpha=0.5,label= f"{stid}"' 3m')
    hist2 = plt.hist(merged.hrrr_wdir10_corr, bins=np.arange(0, 370, 5), edgecolor='none',
                    density=False,alpha=0.5,label='hrrr 10m')
    plt.xlabel("Wind direction (deg)")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind direction histogram test case NM May 2024.png")
    plt.show()



    ### Wspd quantile mapping

    # Define number of quantiles
    q_ = 50 

    # Wind speed time series
    obs_windspeed_3m = merged.obs_wspd3[~np.isnan(merged.obs_wspd3)]
    hrrr_windspeed_10m = merged.hrrr_wspd10[~np.isnan(merged.obs_wspd3)]


    ### Quantiles based on normal distribution (parametric)

    # quantiles = [round(x*0.05,2) for x in range(1,q_)]      # for 5% increments up to 95%

    # dist_obs_ws = norm(loc=obs_obs_wspd3.mean(), scale=obs_obs_wspd3.std())
    # dist_hrrr_ws = norm(loc=hrrr_windspeed_10m.mean(), scale=hrrr_windspeed_10m.std())

    # q_dist_obs_ws = [dist_obs_ws.ppf(x*0.05) for x in range(1,q_)]
    # q_dist_hrrr_ws = [dist_hrrr_ws.ppf(x*0.05) for x in range(1,q_)]
    # q_dist_diff_ws = np.array(q_dist_obs_ws)-np.array(q_dist_hrrr_ws)

    # #Quantiles Obs and HRRR
    # plt.figure()
    # plt.plot(quantiles,q_dist_obs_ws,alpha=0.5,label= f"{stid}")
    # plt.plot(quantiles,q_dist_hrrr_ws,alpha=0.5,label='hrrr')
    # plt.xlabel("Quantiles")
    # plt.ylabel("Values")
    # plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    # plt.legend(loc='upper left',fontsize=10)
    # plt.savefig(f"{stid} 3m vs hrrr 10m wind speed quantiles test case NM May 2024.png")
    # plt.show()



    ### Empirical quantiles (non-parametric CDF)
    quantiles = np.arange(0,1+1/q_,1/q_)
    quantiles_obs_ws_emp = np.quantile(obs_windspeed_3m, quantiles)
    quantiles_hrrr_ws_emp = np.quantile(hrrr_windspeed_10m, quantiles)
    quantiles_diff_ws_emp = quantiles_obs_ws_emp - quantiles_hrrr_ws_emp

    # CDFs Obs and HRRR
    plt.figure()
    plt.plot(quantiles,quantiles_obs_ws_emp,alpha=0.5,label= f"{stid}")
    plt.plot(quantiles,quantiles_hrrr_ws_emp,alpha=0.5,label='hrrr')
    plt.xlabel("Quantiles")
    plt.ylabel("Values")
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.legend(loc='upper left',fontsize=10)
    plt.savefig(f"{stid} 3m vs hrrr 10m wind speed quantiles test case NM May 2024.png")
    plt.show()



    # Reconstruct the time series based on quantile mapping

    corrected_hrrr_windspeed = quantile_mapping(
        hrrr_windspeed_10m,
        quantiles_hrrr_ws_emp,
        quantiles_obs_ws_emp,
        quantiles
    )

    # Time series plot of original and corrected model data
    fig, ax = plt.subplots()
    ax.plot(merged.valid_time[~np.isnan(merged.obs_wspd3)], obs_windspeed_3m, label=f"{stid} 3m (obs)")
    ax.plot(merged.valid_time[~np.isnan(merged.obs_wspd3)], hrrr_windspeed_10m, label="HRRR 10m (model)")
    ax.plot(merged.valid_time[~np.isnan(merged.obs_wspd3)], corrected_hrrr_windspeed, label="Corrected 10m (quantile-mapped)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wind speed (m/s)")
    ax.legend(loc="upper right", fontsize=10)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    plt.title("Observed, Model, and Corrected Wind Speed Time Series")
    plt.show()

    # Histogram of wind speed values for obs, model, and corrected
    plt.figure()
    plt.hist(obs_windspeed_3m, bins=np.arange(0, 16, 0.2), alpha=0.5, label=f"{stid} 3m (obs)", edgecolor='black')
    plt.hist(hrrr_windspeed_10m, bins=np.arange(0, 16, 0.2), alpha=0.5, label="HRRR 10m (model)", edgecolor='black')
    plt.hist(corrected_hrrr_windspeed, bins=np.arange(0, 16, 0.2), alpha=0.5, label="Corrected 10m (quantile-mapped)", edgecolor='black')
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right", fontsize=10)
    plt.title("Histogram of Wind Speed: Obs, Model, Corrected")
    plt.show()


    # Correlation scatter plot: obs vs. original model and obs vs. corrected model

    # Calculate metrics for original model
    mask = ~np.isnan(obs_windspeed_3m) & ~np.isnan(hrrr_windspeed_10m)
    r_model = np.corrcoef(obs_windspeed_3m[mask], hrrr_windspeed_10m[mask])[0,1]
    rmse_model = root_mean_squared_error(obs_windspeed_3m[mask], hrrr_windspeed_10m[mask])
    bias_model = np.mean(hrrr_windspeed_10m[mask] - obs_windspeed_3m[mask])

    # Calculate metrics for corrected model
    mask_corr = ~np.isnan(obs_windspeed_3m) & ~np.isnan(corrected_hrrr_windspeed)
    r_corr = np.corrcoef(obs_windspeed_3m[mask_corr], corrected_hrrr_windspeed[mask_corr])[0,1]
    rmse_corr = root_mean_squared_error(obs_windspeed_3m[mask_corr], corrected_hrrr_windspeed[mask_corr])
    bias_corr = np.mean(corrected_hrrr_windspeed[mask_corr] - obs_windspeed_3m[mask_corr])

    # Scatter plots: obs vs. original model and obs vs. corrected model
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(obs_windspeed_3m, hrrr_windspeed_10m, alpha=0.5, label=f"r = {r_model:.2f}\nRMSE = {rmse_model:.2f}\nBias = {bias_model:.2f}")
    plt.xlabel(f"{stid} 3m (obs)")
    plt.ylabel("HRRR 10m (model)")
    plt.title("Obs vs. Original Model")
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)

    plt.subplot(1,2,2)
    plt.scatter(obs_windspeed_3m, corrected_hrrr_windspeed, alpha=0.5, label=f"r = {r_corr:.2f}\nRMSE = {rmse_corr:.2f}\nBias = {bias_corr:.2f}")
    plt.xlabel(f"{stid} 3m (obs)")
    plt.ylabel("Corrected 10m (quantile-mapped)")
    plt.title("Obs vs. Corrected Model")
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()








#%% Test: Quantile Mapping with Machine Learning to New Stations



def build_quantile_mapping_pairs(model, obs, quantiles, stid, plot = False):
    # Create mapping pairs for one station.
    model_q = np.quantile(model, quantiles)
    obs_q = np.quantile(obs, quantiles)

    if plot == True:
        # CDFs Obs and HRRR
        plt.figure()
        plt.plot(quantiles,obs_q,alpha=0.5,label= f"{stid}")
        plt.plot(quantiles,model_q,alpha=0.5,label='hrrr')
        plt.xlabel("Quantiles")
        plt.ylabel("Values")
        plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        plt.legend(loc='upper left',fontsize=10)
        plt.title(f"{stid} {obs_train.name} vs {model_train.name} quantiles.png")
        plt.show()

    return model_q, obs_q

# -------------------------------
# Step 1: Build training dataset
# -------------------------------

# Define quantiles
q_ = 50 
quantiles = np.arange(0,1+1/q_,1/q_)


# Find all station IDs (stids) available in base_dir
#pattern = os.path.join(base_dir, "*_3m_hrrr_10m_merged.csv")   # Test case data files
pattern = os.path.join(base_dir, "merged_HRRR_ERA5_obs_*.nc")   # US SW all data files
merged_files = glob.glob(pattern)
#all_stids = [os.path.basename(f).replace("_3m_hrrr_10m_merged.csv", "") for f in merged_files]
all_stids = [os.path.basename(f).replace("merged_HRRR_ERA5_obs_", "").replace(".nc", "") for f in merged_files]



# Pre-filter stations with insufficient usable data
valid_stids = []
invalid_stids = []
for stid in all_stids:
    fpath = os.path.join(base_dir, f"merged_HRRR_ERA5_obs_{stid}.nc")
    try:    
        # df_tmp = pd.read_csv(fpath, usecols=["obs_wspd3", "hrrr_wspd10"]).dropna()
        df_tmp = xr.open_dataset(fpath, engine='netcdf4')[["hrrr_wspd10_h", "obs_windspeed", "era5_wspd10"]]
    except Exception:
        invalid_stids.append(stid)
        continue
    n_obs = len(df_tmp)
    # Require minimum counts
    if n_obs >= 10:
        valid_stids.append(stid)
    else:
        invalid_stids.append(stid)

print(f"Total stations found: {len(all_stids)}")
print(f"Using {len(valid_stids)} stations with sufficient data; excluded {len(invalid_stids)}.")

if len(valid_stids) == 0:
    raise RuntimeError("No stations with sufficient data available for training/testing.")

# Random 80/20 split into training and test station IDs (from valid stations only)
rng = np.random.default_rng(42)  # reproducible
shuffled = rng.permutation(valid_stids)
n_train = max(1, int(len(shuffled) * 0.8))
training_stids = list(shuffled[:n_train])
test_stids = list(shuffled[n_train:])

print(f"Training stations: {len(training_stids)}  Test stations: {len(test_stids)}")

empty_map = True
if empty_map == True:

    train_meta = meta.loc[meta.station_id.isin(training_stids), ["station_id", "lat", "lon"]].dropna()
    test_meta = meta.loc[meta.station_id.isin(test_stids), ["station_id", "lat", "lon", "elev"]].dropna()

    plt.figure(figsize=(9,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    mesh = ax.pcolormesh(lon, lat, mean_elev, cmap='terrain', shading='auto', transform=ccrs.PlateCarree(),vmin = 0,vmax=4000)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':')
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.top_labels = False

    # Training stations (black)
    ax.scatter(train_meta.lon, train_meta.lat, c='k', s=30, marker='^', label='Training stations', alpha=0.8)
    # Test stations (colored by r_corr)
    sc = ax.scatter(test_meta.lon, test_meta.lat, c=test_meta.elev, cmap='terrain', vmin=0, vmax=4000,
                    s=40, edgecolor='black', linewidth=1,label = "Test stations")
    ax.set_title("Station Map: Training and Test")
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()


# Build features (metadata) for each station
def build_features(stid, merged, meta):
    row = meta.loc[meta['station_id'] == stid]
    if row.empty:
        return {
            "lat": np.nan,
            "lon": np.nan,
            "elev": np.nan,
            # "state": None,   # model does not take strings
            # "source_network": None,
            "model_mean_ws": float(np.nan),
            "model_std_ws": float(np.nan),
            "model_mean_ws80": float(np.nan),
            "model_std_ws80": float(np.nan),
            "model_mean_temp": float(np.nan),
            "model_std_temp": float(np.nan),
            "model_mean_blh": float(np.nan),
            "model_std_blh": float(np.nan),
            "veg": float(np.nan),
            "z0": float(np.nan),
            "terrain_complex": float(np.nan),
        }
    r = row.iloc[0]
    return {
        "lat": float(r.get("lat", np.nan)),
        "lon": float(r.get("lon", np.nan)),
        "elev": float(r.get("elev", np.nan)),
        # "state": r.get("state", None),
        # "source_network": r.get("source_network", None),
        # simple climatology features
        "model_mean_ws": float(np.nanmean(merged.hrrr_wspd10)),
        "model_std_ws": float(np.nanstd(merged.hrrr_wspd10)),
        "model_mean_ws80": float(np.nanmean(merged.hrrr_wspd80)),
        "model_std_ws80": float(np.nanstd(merged.hrrr_wspd80)),
        "model_mean_temp": float(np.nanmean(merged.hrrr_t2m)),
        "model_std_temp": float(np.nanstd(merged.hrrr_t2m)),
        "model_mean_blh": float(np.nanmean(merged.hrrr_blh)),
        "model_std_blh": float(np.nanstd(merged.hrrr_blh)),
        "veg": float(np.nanmean(merged.hrrr_veg)),
        "z0": float(np.nanmean(merged.hrrr_fsr)),
        "terrain_complex": float(r.get("mean_std_vicinity", np.nan)),
    }

all_records = []
for i, station_id in enumerate(training_stids):

    print(f"Processing station {i+1}/{len(training_stids)} ({station_id}) for training data...")

    # Read merged data
    merged = pd.read_csv(os.path.join(base_dir, f"{station_id}_3m_hrrr_10m_merged.csv"))

    merged["hrrr_wspd80"] = np.sqrt(merged.hrrr_u80**2 + merged.hrrr_v80**2)

    # Data QC
    merged['obs_wspd3'] = merged.obs_wspd3.where((merged.obs_wspd3 >= 0) & (merged.obs_wspd3 <= 60), np.nan)

    # Extract training data (non-missing obs)
    model_train = merged.hrrr_wspd10[~np.isnan(merged.obs_wspd3)]
    obs_train = merged.obs_wspd3[~np.isnan(merged.obs_wspd3)]

    # Build features
    features = build_features(station_id, merged, meta)

    # Build quantile mapping pairs
    if i == 5:  # plots only for one test station
        plot = True
    else:
        plot = False
    model_q, obs_q = build_quantile_mapping_pairs(model_train, obs_train, quantiles, station_id, plot=plot)


    for p, mq, oq in zip(quantiles, model_q, obs_q):
        rec = {
            "station_id": station_id,
            "quantile": p,
            "model_q": mq,
            "obs_q": oq,
            **features  # unpack station metadata
        }
        all_records.append(rec)

df = pd.DataFrame(all_records)

# -------------------------------
# Step 2: Train ML model
# -------------------------------
X = df.drop(columns=["station_id", "obs_q"])
y = df["obs_q"]

ml_model = RandomForestRegressor(n_estimators=200, random_state=0)
ml_model.fit(X, y)

# --- Model evaluation & feature importance ---
try:
    # In-sample performance
    y_pred_train = ml_model.predict(X)
    train_rmse = np.sqrt(np.mean((y_pred_train - y.values)**2))
    train_bias = np.mean(y_pred_train - y.values)

    # Cross-validation (5-fold) RMSE
    cv_scores = cross_val_score(ml_model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    cv_rmse_mean = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()

    # Cross-validation bias
    K = 10
    kf = KFold(n_splits=K, shuffle=True, random_state=0)
    cv_biases = []
    for train_idx, test_idx in kf.split(X):
        m_fold = clone(ml_model)
        m_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_fold = m_fold.predict(X.iloc[test_idx])
        cv_biases.append(np.mean(y_pred_fold - y.iloc[test_idx].values))
    cv_bias_mean = np.mean(cv_biases)
    cv_bias_std = np.std(cv_biases)

    print(f"Training RMSE: {train_rmse:.3f}  Bias: {train_bias:.3f}")
    print(f"{K}-fold CV RMSE: {cv_rmse_mean:.3f} (+/- {cv_rmse_std:.3f})")
    print(f"{K}-fold CV Bias: {cv_bias_mean:.3f} (+/- {cv_bias_std:.3f})")


    # Feature importance (exclude 'quantile' and 'model_q')
    importances = ml_model.feature_importances_
    keep_cols = [c for c in X.columns if c not in ("quantile", "model_q")]

    keep_idx = [i for i, c in enumerate(X.columns) if c in keep_cols]
    importances_keep = importances[keep_idx]
    order = np.argsort(importances_keep)

    plt.figure(figsize=(5, 0.35*len(keep_cols)+1))
    plt.barh(np.array(keep_cols)[order], importances_keep[order], color="teal")
    plt.xlabel("Gini Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

    # Optional permutation importance on kept cols (exclude quantile, model_q correctly)
    from sklearn.inspection import permutation_importance
    # Compute over full X (estimator expects same columns), then filter
    perm = permutation_importance(ml_model, X, y, n_repeats=8, random_state=0, n_jobs=-1)
    perm_mean_keep = perm.importances_mean[keep_idx]
    perm_std_keep = perm.importances_std[keep_idx]
    order_p = np.argsort(perm_mean_keep)

    plt.figure(figsize=(5, 0.35*len(keep_cols)+1))
    plt.barh(np.array(keep_cols)[order_p],
             perm_mean_keep[order_p],
             xerr=perm_std_keep[order_p],
             color="darkorange", alpha=0.85)
    plt.xlabel("Permutation Importance (Δ score)")
    plt.title("Permutation Importance")
    plt.tight_layout()
    plt.show()

    """
    Gini vs. permutation:

    Definition
    Gini (mean decrease impurity): Sums each split’s impurity reduction (variance for regression, Gini/entropy for classification) attributed to a feature across all trees, normalized to sum to 1.
    Permutation importance: Measures drop in a chosen performance metric after randomly shuffling a single feature’s values (breaking its relation to target).
    What it captures
    Gini: How often and how effectively a feature was used inside the forest to create pure (low-variance) nodes.
    Permutation: Actual predictive dependence of the model on that feature given all others.
    Biases
    Gini: Biased toward features with many unique values / continuous variables; splits among correlated features dilute importance arbitrarily.
    Permutation: Less biased toward cardinality; still affected when features are strongly correlated (shuffling one may not hurt because others substitute).
    Scale
    """




except Exception as e:
    print(f"Model evaluation / feature importance failed: {e}")



# -------------------------------
# Step 3: Apply to test stations
# -------------------------------


test_results = {}

# Columns used in training (exclude station_id, obs_q)
feature_cols = X.columns.tolist()

for i, stid in enumerate(test_stids):
    print(f"Applying quantile mapping to test station: {stid}")
    merged_test = pd.read_csv(os.path.join(base_dir, f"{stid}_3m_hrrr_10m_merged.csv"))
    merged_test["hrrr_wspd80"] = np.sqrt(merged_test.hrrr_u80**2 + merged_test.hrrr_v80**2)

    # build feature dict for this test station
    feats = build_features(stid, merged_test, meta)

    # Model series (use all available model values)
    full_model_series = merged_test.hrrr_wspd10.values
    valid_model_mask = ~np.isnan(full_model_series)
    model_series = full_model_series[valid_model_mask]

    # Observation series for test station
    full_obs_series = merged_test.obs_wspd3.values
    obs_mask = ~np.isnan(full_obs_series) & valid_model_mask
    obs_series = full_obs_series[obs_mask]

    # Time vectors
    times_full = pd.to_datetime(merged_test.valid_time)
    times_model = times_full[valid_model_mask]
    times_obs = times_full[obs_mask]

    # Station-specific model quantiles
    model_q_test = np.quantile(model_series, quantiles)

    # Build prediction rows (one per quantile)
    pred_rows = []
    for p, mq in zip(quantiles, model_q_test):
        row = {"quantile": p, "model_q": mq}
        for col in feature_cols:
            if col in ("quantile", "model_q"):
                continue
            row[col] = feats.get(col, np.nan)
        pred_rows.append(row)

    X_pred = pd.DataFrame(pred_rows)[feature_cols]  # align columns with training
    predicted_obs_q = ml_model.predict(X_pred)

    # Map full model time series to “corrected” series
    corrected_series = np.interp(model_series, model_q_test, predicted_obs_q)
    # Add observed empirical CDF at same quantiles
    obs_q_test = np.quantile(obs_series, quantiles)

    # Metrics (overlapping obs/model times)
    model_series_overlap = full_model_series[obs_mask]
    corrected_series_overlap = np.interp(model_series_overlap, model_q_test, predicted_obs_q)

    def safe_corr(a, b):
        return np.corrcoef(a, b)[0, 1] if len(a) > 1 else np.nan

    r_model = safe_corr(obs_series, model_series_overlap)
    r_corr = safe_corr(obs_series, corrected_series_overlap)
    rmse_model = root_mean_squared_error(obs_series, model_series_overlap)
    rmse_corr = root_mean_squared_error(obs_series, corrected_series_overlap)
    bias_model = np.mean(model_series_overlap - obs_series)
    bias_corr = np.mean(corrected_series_overlap - obs_series)
    ks_model = ks_2samp(obs_series, model_series_overlap).statistic
    ks_corr = ks_2samp(obs_series, corrected_series_overlap).statistic

    print(f"[{stid}] r(model)={r_model:.3f} r(corr)={r_corr:.3f} RMSE(model)={rmse_model:.3f} RMSE(corr)={rmse_corr:.3f} "
          f"Bias(model)={bias_model:.3f} Bias(corr)={bias_corr:.3f} KS(model)={ks_model:.3f} KS(corr)={ks_corr:.3f}")

    test_results[stid] = {
        "quantiles": quantiles,
        "model_q": model_q_test,
        "predicted_obs_q": predicted_obs_q,
        "original_model_series": model_series,
        "corrected_series": corrected_series,
        "metrics": {
            "r_model": r_model,
            "r_corr": r_corr,
            "rmse_model": rmse_model,
            "rmse_corr": rmse_corr,
            "bias_model": bias_model,
            "bias_corr": bias_corr,
            "ks_model": ks_model,
            "ks_corr": ks_corr
        }
    }

    if i == i:  # plots only for one test station

        # Diagnostic CDF plot
        plt.figure(figsize=(5,4))
        plt.plot(quantiles, model_q_test, label="HRRR CDF", alpha=0.6)
        plt.plot(quantiles, predicted_obs_q, label="Pred Obs CDF", alpha=0.6)
        plt.plot(quantiles, obs_q_test, label="Obs CDF (validation)" \
        "", alpha=0.8, linewidth=1.5)
        plt.title(f"{stid} CDFs (Model / Predicted / Observed)")
        plt.xlabel("Quantile")
        plt.ylabel("Wind speed (m/s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # PDF / histogram including observations
        plt.figure(figsize=(6,4))
        bins = np.arange(0, 16, 0.2)
        plt.hist(obs_series, bins=bins, alpha=0.45, label="Obs", density=True, edgecolor='black')
        plt.hist(model_series_overlap, bins=bins, alpha=0.45, label="HRRR", density=True, edgecolor='black')
        plt.hist(corrected_series_overlap, bins=bins, alpha=0.45, label="HRRR corrected", density=True, edgecolor='black')
        plt.title(f"{stid} PDF: Obs vs HRRR vs HRRR corrected")
        plt.xlabel("Wind speed (m/s)")
        plt.ylabel("Density")
        txt = (f"Metrics compared to original observations:\n"
            f"r(HRRR)={r_model:.2f} r(corr HRRR)={r_corr:.2f}\n"
            f"RMSE(HRRR)={rmse_model:.2f} RMSE(corr HRRR)={rmse_corr:.2f}\n"
            f"Bias(HRRR)={bias_model:.2f} Bias(corr HRRR)={bias_corr:.2f}")
           # f"KS(HRRR)={ks_model:.2f} KS(corr HRRR)={ks_corr:.2f}")
        plt.annotate(txt, xy=(0.98, 0.97), xycoords='axes fraction',
                    ha='right', va='top', 
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Time series plot
        plt.figure(figsize=(8,4))
        plt.plot(times_obs, obs_series, label="Obs (validation)", linewidth=1.2)
        plt.plot(times_model, model_series, label="HRRR", alpha=0.6)
        plt.plot(times_model, corrected_series, label="HRRR corrected", alpha=0.8)
        plt.title(f"{stid} Time Series")
        plt.xlabel("Time")
        plt.ylabel("Wind speed (m/s)")
        txt2 = (f"r(HRRR)={r_model:.2f} r(corr HRRR)={r_corr:.2f} "
                f"RMSE(HRRR)={rmse_model:.2f} RMSE(corr HRRR)={rmse_corr:.2f}")
        plt.annotate(txt2, xy=(0.01, 0.97), xycoords='axes fraction',
                    ha='left', va='top',
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    
#%% Summary maps of results


# Map of station locations colored by correlation (r_corr) for test stations
try:
    train_meta = meta.loc[meta.station_id.isin(training_stids), ["station_id", "lat", "lon"]].dropna()
    test_meta = meta.loc[meta.station_id.isin(test_stids), ["station_id", "lat", "lon"]].dropna()

    # Collect r_corr values for test stations (skip those without results)
    r_values = []
    test_lons = []
    test_lats = []
    for _, row in test_meta.iterrows():
        stid = row.station_id
        if stid in test_results and not np.isnan(test_results[stid]["metrics"]["r_corr"]):
            test_lons.append(row.lon)
            test_lats.append(row.lat)
            r_values.append(test_results[stid]["metrics"]["r_corr"])

    if len(r_values) == 0:
        print("No test station correlations available for map plot.")
    else:
        r_values = np.array(r_values)
        vmin, vmax = 0, 1.0  

        plt.figure(figsize=(9,6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon, lat, mean_elev, cmap='terrain', shading='auto', transform=ccrs.PlateCarree(),vmin = 0,vmax=4000)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':')
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False

        # Training stations (black)
        ax.scatter(train_meta.lon, train_meta.lat, c='k', s=30, marker='^', label='Training stations', alpha=0.8)
        # Test stations (colored by r_corr)
        sc = ax.scatter(test_lons, test_lats, c=r_values, cmap='RdYlBu', vmin=vmin, vmax=vmax,
                        s=40, edgecolor='black', linewidth=0.5, label='Test stations (r_corr)')
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Correlation r (Obs vs Corrected HRRR)', fontsize=10)

        ax.set_title("Station Map: Training (black) and Test (color-coded by r_corr)")
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3, linestyle=':')
        plt.tight_layout()
        plt.show()
except Exception as e:
    print(f"Map plotting failed: {e}")


# Bias map (corrected bias: corrected - obs)
try:
    train_meta = meta.loc[meta.station_id.isin(training_stids), ["station_id", "lat", "lon"]].dropna()
    test_meta = meta.loc[meta.station_id.isin(test_stids), ["station_id", "lat", "lon"]].dropna()

    bias_vals = []
    bias_lons = []
    bias_lats = []
    for _, row in test_meta.iterrows():
        sid = row.station_id
        if sid in test_results:
            b = test_results[sid]["metrics"].get("bias_corr", np.nan)
            if not np.isnan(b):
                bias_lons.append(row.lon)
                bias_lats.append(row.lat)
                bias_vals.append(b)

    if len(bias_vals) == 0:
        print("No bias values available for bias map plot.")
    else:
        bias_vals = np.array(bias_vals)
        max_abs = np.max(np.abs(bias_vals))
        vmin, vmax = -max_abs, max_abs  # symmetric for diverging colormap

        plt.figure(figsize=(9,6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon, lat, mean_elev, cmap='terrain', shading='auto', transform=ccrs.PlateCarree(),vmin = 0,vmax=4000)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':')
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False

        ax.scatter(train_meta.lon, train_meta.lat, c='k', s=30, marker='^', label='Training stations', alpha=0.8)
        sc = ax.scatter(bias_lons, bias_lats, c=bias_vals, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                        s=40, edgecolor='black', linewidth=0.5, label='Test stations (bias)')
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Bias (Corrected HRRR - Obs) [m/s]', fontsize=10)

        ax.set_title("Station Map: Training (black) and Test (color-coded by corrected bias)")
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.show()
except Exception as e:
    print(f"Bias map plotting failed: {e}")

# Uncorrected HRRR correlation map (r_model)
try:
    train_meta = meta.loc[meta.station_id.isin(training_stids), ["station_id", "lat", "lon"]].dropna()
    test_meta = meta.loc[meta.station_id.isin(test_stids), ["station_id", "lat", "lon"]].dropna()

    r_model_vals, r_model_lons, r_model_lats = [], [], []
    for _, row in test_meta.iterrows():
        sid = row.station_id
        if sid in test_results:
            rv = test_results[sid]["metrics"].get("r_model", np.nan)
            if not np.isnan(rv):
                r_model_lons.append(row.lon)
                r_model_lats.append(row.lat)
                r_model_vals.append(rv)

    if len(r_model_vals) == 0:
        print("No original HRRR correlations available for map plot.")
    else:
        r_model_vals = np.array(r_model_vals)
        vmin, vmax = 0, 1.0

        plt.figure(figsize=(9,6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon, lat, mean_elev, cmap='terrain', shading='auto', transform=ccrs.PlateCarree(),vmin = 0,vmax=4000)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':')
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False

        ax.scatter(train_meta.lon, train_meta.lat, c='k', s=30, marker='^', label='Training stations', alpha=0.8)
        sc = ax.scatter(r_model_lons, r_model_lats, c=r_model_vals, cmap='RdYlBu', vmin=vmin, vmax=vmax,
                        s=40, edgecolor='black', linewidth=0.5, label='Test stations (r_model)')
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Correlation r (Obs vs Raw HRRR)', fontsize=10)
        ax.set_title("Station Map: Test (color-coded by raw HRRR correlation)")
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.show()
except Exception as e:
    print(f"Raw correlation map plotting failed: {e}")



# Uncorrected HRRR bias map (bias_model)
try:
    train_meta = meta.loc[meta.station_id.isin(training_stids), ["station_id", "lat", "lon"]].dropna()
    test_meta = meta.loc[meta.station_id.isin(test_stids), ["station_id", "lat", "lon"]].dropna()

    bias_model_vals, bias_model_lons, bias_model_lats = [], [], []
    for _, row in test_meta.iterrows():
        sid = row.station_id
        if sid in test_results:
            bm = test_results[sid]["metrics"].get("bias_model", np.nan)
            if not np.isnan(bm):
                bias_model_lons.append(row.lon)
                bias_model_lats.append(row.lat)
                bias_model_vals.append(bm)

    if len(bias_model_vals) == 0:
        print("No raw HRRR bias values available for map plot.")
    else:
        bias_model_vals = np.array(bias_model_vals)
        max_abs = np.max(np.abs(bias_model_vals))
        vmin, vmax = -max_abs, max_abs

        plt.figure(figsize=(9,6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon, lat, mean_elev, cmap='terrain', shading='auto', transform=ccrs.PlateCarree(),vmin = 0,vmax=4000)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':')
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False

        ax.scatter(train_meta.lon, train_meta.lat, c='k', s=30, marker='^', label='Training stations', alpha=0.8)
        sc = ax.scatter(bias_model_lons, bias_model_lats, c=bias_model_vals, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                        s=40, edgecolor='black', linewidth=0.5, label='Test stations (bias_model)')
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Bias (Raw HRRR - Obs) [m/s]', fontsize=10)
        ax.set_title("Station Map: Test (color-coded by raw HRRR bias)")
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.show()
except Exception as e:
    print(f"Raw bias map plotting failed: {e}")

# Bias improvement map: (bias_model - bias_corr)
try:
    train_meta = meta.loc[meta.station_id.isin(training_stids), ["station_id", "lat", "lon"]].dropna()
    test_meta = meta.loc[meta.station_id.isin(test_stids), ["station_id", "lat", "lon"]].dropna()

    diff_vals = []
    diff_lons = []
    diff_lats = []
    improved = 0
    total = 0
    for _, row in test_meta.iterrows():
        sid = row.station_id
        if sid in test_results:
            bm = test_results[sid]["metrics"].get("bias_model", np.nan)
            bc = test_results[sid]["metrics"].get("bias_corr", np.nan)
            if np.isfinite(bm) and np.isfinite(bc):
                total += 1
                if abs(bc) < abs(bm):
                    improved += 1
                diff_vals.append(abs(bm) - abs(bc))
                diff_lons.append(row.lon)
                diff_lats.append(row.lat)

    if len(diff_vals) == 0:
        print("No stations with both raw and corrected bias for difference map.")
    else:
        diff_vals = np.array(diff_vals)
        max_abs = np.max(np.abs(diff_vals))
        vmin, vmax = -max_abs, max_abs
        frac = improved / total if total > 0 else np.nan

        plt.figure(figsize=(9,6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        mesh = ax.pcolormesh(lon, lat, mean_elev, cmap='terrain', shading='auto', transform=ccrs.PlateCarree(),vmin = 0,vmax=4000)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        gl = ax.gridlines(draw_labels=True, alpha=0.5, linestyle=':')
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.top_labels = False

        ax.scatter(train_meta.lon, train_meta.lat, c='k', s=30, marker='^', label='Training', alpha=0.8)
        sc = ax.scatter(diff_lons, diff_lats, c=diff_vals, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                        s=45, edgecolor='black', linewidth=0.5, label='Test (|bias_raw| - |bias_corr|)')
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Bias Difference (|Raw| - |Corrected|) [m/s]', fontsize=10)

        ax.set_title("Bias Difference Map (Positive = correction reduced bias)")
        ax.legend(loc='lower right', fontsize=9)

        txt = f"Improved stations: {improved}/{total} ({frac*100:.1f}%)"
        ax.annotate(txt, xy=(0.99,0.99), xycoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7), fontsize=9)
        plt.tight_layout()
        plt.show()
except Exception as e:
    print(f"Bias difference map plotting failed: {e}")


# %% Bar plotstry:
station_ids = list(test_results.keys())
if len(station_ids) == 0:
    print("No test results available for bar plots.")
else:
    # Optionally limit number of stations for readability
    max_stations = 60  # adjust as needed
    station_ids_plot = station_ids[:max_stations]

    r_model_vals = [test_results[s]["metrics"]["r_model"] for s in station_ids_plot]
    r_corr_vals = [test_results[s]["metrics"]["r_corr"] for s in station_ids_plot]
    bias_model_vals = [test_results[s]["metrics"]["bias_model"] for s in station_ids_plot]
    bias_corr_vals = [test_results[s]["metrics"]["bias_corr"] for s in station_ids_plot]

    x = np.arange(len(station_ids_plot))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(min(14, 0.25*len(station_ids_plot)+4), 8), sharex=True)
    
    # Correlation bars
    ax = axes[0]
    ax.bar(x - width/2, r_model_vals, width, label="Raw HRRR r", color="#1f77b4")
    ax.bar(x + width/2, r_corr_vals, width, label="Corrected r", color="#ff7f0e")
    ax.set_ylabel("Correlation r")
    ax.set_ylim(0, 1.05)
    ax.set_title("Station-wise Correlation Before / After Correction")
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=9)

    # Bias bars
    ax = axes[1]
    ax.bar(x - width/2, bias_model_vals, width, label="Raw HRRR bias", color="#1f77b4")
    ax.bar(x + width/2, bias_corr_vals, width, label="Corrected bias", color="#ff7f0e")
    ax.set_ylabel("Bias (Model - Obs) [m/s]")
    ax.set_title("Station-wise Bias Before / After Correction")
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=9)

    # X tick labels
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(station_ids_plot, rotation=90, fontsize=7)
    axes[1].set_xlabel("Station ID")

    plt.tight_layout()
    plt.show()


# %%
