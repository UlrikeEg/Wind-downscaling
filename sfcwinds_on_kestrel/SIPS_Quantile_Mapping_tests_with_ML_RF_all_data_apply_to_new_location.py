import os
import pickle
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
from sklearn.inspection import permutation_importance
from scipy.spatial import cKDTree 
from sklearn.base import clone 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib

from Functions_sfcwinds import *

# Enable interactive backend when running in IPython, ignore in plain Python scripts
try:
    if callable(globals().get("get_ipython", None)):
        get_ipython().run_line_magic("matplotlib", "widget")  # type: ignore[name-defined]
except Exception:
    pass

"""

- This script works with the reduced dataset "G3P3test" (takes around 30min to run), but takes forever with the full dataset ("G3P3") - runs out of time after a 10h job
- These are created with /projects/sfcwinds/scripts/sfcwinds_on_kestrel/combine_time_series_HRRR_ERA5_obs_new_location.py

- need to look into what is going on with file sizes and memory usage
- one idea to make it faster would be avoiding creating a dataframe from the dataset ( merged = merged.to_dataframe())  and extract all needed information directly from the xarray dataset
"""


# Unknow station we want the time series for (the name defines which file to read)
station = "G3P3" # this is the whole dataset (10 years HRRR, 25 years ERA5)
# station = "G3P3test" # that's a smaller dataset that works well (only has every 100th HRRR and ERA5 file)
lat, lon =  34.962400, -106.510100
print(f"Station at {lon}E, {lat}N", flush=True)



def build_features_new_loc(merged):
    return {
        "lat": float(np.nanmean(merged.station_latitude.values)),
        "lon": float(np.nanmean(merged.station_longitude.values)),
        "elev": float(np.nanmean(merged.elevation.values)),
        # "state": merged.get("state", None),
        # "source_network": merged.get("source_network", None),
        # simple climatology features
        "model_mean_ws": float(np.nanmean(merged.hrrr_wspd10_h.values)),
        "model_std_ws": float(np.nanstd(merged.hrrr_wspd10_h.values)),
        "model_mean_ws80": float(np.nanmean(merged.hrrr_wspd80.values)),
        "model_std_ws80": float(np.nanstd(merged.hrrr_wspd80.values)),
        "model_mean_temp": float(np.nanmean(merged.hrrr_t2m.values)),
        "model_std_temp": float(np.nanstd(merged.hrrr_t2m.values)),
        "model_mean_blh": float(np.nanmean(merged.hrrr_blh.values)),
        "model_std_blh": float(np.nanstd(merged.hrrr_blh.values)),
        "veg": float(np.nanmean(merged.hrrr_veg.values)),
        "z0": float(np.nanmean(merged.fsr_avg.values)),
        "terrain_complex": float(np.nanmean(merged.std_elev_within_20km.values)),
    }


def read_merged_nc_and_make_df_new_loc(station_id, base_dir = "/kfs2/projects/sfcwinds/merged_station_data_new_locations/"):

    # Read merged data (netcdf) - gives memory error, so use pickle
    # fpath = os.path.join(base_dir, f"merged_HRRR_ERA5_{station_id}.nc")
    # merged = xr.open_dataset(fpath, engine='netcdf4')

    # Read merged data (pickle)
    fpath = os.path.join(base_dir, f"merged_HRRR_ERA5_{station_id}.pkl")
    with open(fpath, "rb") as f:
       merged = pickle.load(f)

    merged = merged.chunk({'valid_time': 100})  

    # Exclude hrrr_wspd (depends on isobaricInhPa, may be used in later iterations)
    merged = merged.drop_vars(["hrrr_wspd", "hrrr_wdir", "hrrr_u", "hrrr_v"])
    merged = merged.drop_dims('isobaricInhPa')

    # Convert to DataFrame
    merged = merged.to_dataframe()

    return merged





# Directory with observations and model data merged files
base_dir = "/kfs2/projects/sfcwinds/merged_station_data_new_locations/"  
# combined HRRR-ERA5 files for new locations
# create with /projects/sfcwinds/scripts/sfcwinds_on_kestrel/combine_time_series_HRRR_ERA5_obs_new_location.py




#%% Quantile Mapping with Machine Learning to New Station


# -------------------------------
# Load pre-trained model
# -------------------------------


# Define quantiles
q_ = 50 
quantiles = np.arange(0,1+1/q_,1/q_)

# Load the ML model (this is currently the RF model created with SIPS_Quantile_Mapping_tests_with_ML_RF_all_data.py)
ml_model = joblib.load("/kfs2/projects/sfcwinds/scripts/sfcwinds_on_kestrel/quantile_mapping_rf_model_all_data.pkl")
print("Pre-trained ML model loaded.", flush=True)




# -------------------------------
# Apply to new location
# -------------------------------


# Columns used in training (adjust this to the created model when modifying it)
feature_cols = ["quantile", "model_q",   # use when using pre-trained model
                # adjust to feature list:
                "lat",
                "lon",
                "elev",
                "model_mean_ws",
                "model_std_ws",
                "model_mean_ws80",
                "model_std_ws80",
                "model_mean_temp",
                "model_std_temp",
                "model_mean_blh",
                "model_std_blh",
                "veg",
                "z0",
                "terrain_complex"
                ]



print(f"Applying quantile mapping to new location {station}", flush=True)

try:

    # Read and process merged data
    merged_test = read_merged_nc_and_make_df_new_loc(station)
    print(f"Read and processed merged data for station {station}.", flush=True)

    # build feature dict for this test station
    feats = build_features_new_loc(merged_test)
    print(f"Built feature dictionary for station {station}.", flush=True)

    # Model series (use all available model values)
    full_model_series = merged_test.hrrr_wspd10_h.values
    valid_model_mask = ~np.isnan(full_model_series)
    model_series = full_model_series[valid_model_mask]

    # Time vectors
    times_full = pd.to_datetime(merged_test.index)
    times_model = times_full[valid_model_mask]

    print(f"Processed time vectors for station {station}.", flush=True)

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

    print(f"Mapped full model time series to corrected series for station {station}.", flush=True)


    # Diagnostic CDF plot
    fig1 = plt.figure(figsize=(5,4))
    plt.plot(quantiles, model_q_test, label="HRRR CDF", alpha=0.6)
    plt.plot(quantiles, predicted_obs_q, label="Corrected HRRR CDF", alpha=0.6)
    plt.title(f"{station} CDFs (Model / Corrected)")
    plt.xlabel("Quantile")
    plt.ylabel("Wind speed (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PDF / histogram including observations
    fig2 = plt.figure(figsize=(6,4))
    bins = np.arange(0, 16, 0.2)
    plt.hist(model_series, bins=bins, alpha=0.45, label="HRRR", density=True, edgecolor='black')
    plt.hist(corrected_series, bins=bins, alpha=0.45, label="HRRR corrected", density=True, edgecolor='black')
    plt.title(f"{station} PDF: Obs vs HRRR vs HRRR corrected")
    plt.xlabel("Wind speed (m/s)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Time series plot
    fig3 = plt.figure(figsize=(8,4))
    plt.plot(times_model, model_series, label="HRRR", alpha=0.6)
    plt.plot(times_model, corrected_series, label="HRRR corrected", alpha=0.8)
    plt.title(f"{station} Time Series")
    plt.xlabel("Time")
    plt.ylabel("Wind speed (m/s)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Processing failed for station {station}: {e}", flush=True)



   
# %% Save all figures

for i, fig_num in enumerate(plt.get_fignums()):
    fig = plt.figure(fig_num)
    fig.savefig(f"/kfs2/projects/sfcwinds/scripts/plots/figure_{station}_{i+1}.png", dpi=200)


print("All done.", flush=True)