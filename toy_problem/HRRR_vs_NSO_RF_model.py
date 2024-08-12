#%% Download HRRR data with Herbie

from herbie import Herbie
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
import s3fs
import numpy as np
import pyarrow
import glob
from windrose import WindroseAxes
import matplotlib.cm as cm
import xarray as xr
from datetime import datetime, timedelta
from numpy import cos,sin
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
from scipy.fftpack import *
import scipy as sp
import scipy.signal
import scipy.signal as signal
from scipy.optimize import curve_fit
import sys
import math
import time
import glob
import netCDF4 as nc
import pickle
import datetime as dt
import matplotlib as mpl
import sklearn




NSO = (-114.9753, 35.7970)
CrescentDunes = (-117.36360821139168, 38.23914286091489)


location = NSO # define location


def read_dataset(dataset, year=None, month=None, day=None):
    path = f's3://oedi-data-lake/NSO/{dataset}'
    if year:
        if '*' not in year:
            path += f'/year={year}'
    if month:
        if '*' not in month:
            path += f'/month={month}'
    if day:
        if '*' not in day:
            path += f'/day={day}'

    # Check existence
    if not s3fs.S3FileSystem(anon=True).exists(path):
        raise Exception('No data available for that selection.')

    # Get dataframe
    df = pd.read_parquet(path, storage_options={"anon": True})
    return df

def read_nsrdb(datafile):
    
    data = pd.read_csv(datafile,
                    index_col = None,
                    header = 0,    
                    skiprows = [0, 1], 
                    engine = 'c',
                    on_bad_lines='warn', 
                    na_values = {'NAN'},
                    dtype = float
                        )
    
    data.index = pd.to_datetime(data.iloc[:,:5])

    return data





    
#%% Read all data

## read NSO data
inflow = read_dataset('inflow_mast_1min', year="2021", month="12", day = "14")
inflow = pd.concat( [inflow, read_dataset('inflow_mast_1min', year="2021", month="12", day = "15")]) 


    
## read HRRR in time range
time_range = pd.date_range(datetime(2021, 12, 15, 0, 0, 0), 
                           datetime(2021, 12, 16, 0, 0, 0), 
                           freq=timedelta(hours=1)).tolist()


read_hrrr = 0

if read_hrrr == 1:

    
    hrrr_all = pd.DataFrame()
    
    for date in time_range:
        
        print (date)
        
        hrrr = pd.DataFrame()
    
        # subhourly data
        H = herbie.fast.FastHerbie(
                [date],
                prioriy = 'google',
                model="hrrr",
                product="subh", # to get 15min steps backwards, set fxx to 1 and product subh, otherwise sfc
                fxx=[0,1]
            )
        # list(H.xarray("PRMSL" ).variables.keys())
        
        # ds = H.xarray("(:UGRD:10 m:|:TMP:2 m above ground:)" )  # gives only 2m temp
        # ds = H.xarray("(GUST|PRES)" )   # works!
    
        ds = H.xarray("[U\|V]GRD:10 m" )
        dsi = ds.herbie.nearest_points(
            points=[NSO],
            names=["NSO"],
        ).isel(point=0).isel(step=[0,2,3,4])   # why are there 2 values per step?
        
        hrrr = pd.concat([hrrr, dsi.to_dataframe().set_index('valid_time')[["u10", "v10"]]], axis=1)
         
        ds = H.xarray("GUST|PRES" )       
        dsi = ds.herbie.nearest_points(
            points=[NSO],
            names=["NSO"],
        ).isel(point=0).isel(step=[0,1,2,3]) # here, the steps are correct
        
        hrrr = pd.concat([hrrr, dsi.to_dataframe().set_index('valid_time')[["gust", "sp"]]], axis=1)
        
        ds = H.xarray(":TMP:2 m" )    
        dsi = ds.herbie.nearest_points(
            points=[NSO],
            names=["NSO"],
        ).isel(point=0).isel(step=[0,1,2,3]) # steps are also correct
        
        hrrr = pd.concat([hrrr, dsi.to_dataframe().set_index('valid_time').t2m], axis=1)
    
    
    
        #Hourly data
        H = herbie.fast.FastHerbie(
                [date],
                prioriy = 'google',
                model="hrrr",
                product="sfc", # to get 15min steps backwards, set fxx to 1 and product subh, otherwise sfc
                fxx=[0,1]
            )
        
        ds = H.xarray("HPBL|SHTFL" )   #       |SFCR
        dsi = ds.herbie.nearest_points(
            points=[NSO],
            names=["NSO"],
        ).isel(point=0).isel(step=[0])
        
        hrrr_hourly = dsi.to_dataframe().set_index('valid_time').drop(
            columns=['surface', 'latitude', 'longitude', 'metpy_crs', 
                     'gribfile_projection', 'y', 'x', 'point', 'point_latitude',
                     'point_longitude'])
        
        hrrr = pd.concat([hrrr, hrrr_hourly], axis=1)
        
        hrrr_all = pd.concat([hrrr_all, hrrr], axis=0)
    
    
    # Add columns and interpolate 1h data
    hrrr = hrrr_all
    hrrr['wind_speed'] = (hrrr.u10**2 + hrrr.v10**2)**0.5
    hrrr['wind_dir'] = np.degrees(np.arctan2(hrrr.u10, hrrr.v10)) +180
    hrrr[["blh",  "ishf"]] = hrrr[["blh",  "ishf"]].interpolate()
    
    
    
    # # Add NSO wspd to hrrr
    hrrr = pd.merge(hrrr, inflow.wspd_3m, left_index = True, right_index = True, how="inner")
    
    # Rename the columns
    hrrr.rename(columns={
        'sp': 'surface_pressure',
        't2m': 'Temp_2m',
        'blh': 'ABL_height',
        'ishf': 'sens_heat_flux',
        'wind_speed': 'wind_speed_10m',
        'wind_dir': 'wind_dir_10m',
        'u10': 'u_10m',
        'v10': 'v_10m'
    }, inplace=True)


    # hrrr.to_pickle("hrrr.pkl")
    
else:

    hrrr = pd.read_pickle("hrrr.pkl")




#%% Plot Timeseries of wind

## read nsrdb
nsrdb_files = sorted(glob.glob('359446_35.80_-114.97_2021.csv')) 
nsrdb = pd.DataFrame()    
for datafile in nsrdb_files:
    nsrdb = pd.concat( [nsrdb, read_nsrdb(datafile)]) 
    
nsrdb = nsrdb["2021-12-14 00:00:00":"2021-12-16 00:00:00"]


fig = plt.figure(figsize=(11,4))
 
ax2 = plt.subplot(2, 1, 1)
plt.ylabel('Wind speed (m/s)', fontsize = 'x-large')
ax2.plot(inflow.wspd_3m,'.', label = "NSO", ms=3, zorder = 10)
ax2.fill_between(inflow.index, inflow.wspd_3m, inflow.wspd_3m_max, label = "", color="C0", alpha = 0.3, edgecolor=None, zorder = 10)
ax2.plot( nsrdb['Wind Speed'],'.', label = "MERRA2", ms=2, zorder = 100)
ax2.plot( hrrr.wind_speed_10m,'.', label = "HRRR", ms=3, zorder = 15)
ax2.fill_between(hrrr.index, hrrr.wind_speed_10m, hrrr.gust, label = "", color="C2", alpha = 0.2, edgecolor=None, zorder = 5)
plt.legend(loc=1,  markerscale=3, fontsize = 'x-large')
plt.grid()
ax2.set_zorder(100)
 
ax1 = plt.subplot(2, 1, 2, sharex=ax2)  # 
ax1.set_ylabel('Wind dir ($^\circ$)', fontsize = 'x-large')
ax1.plot(inflow.wdir_3m,'.', label = "", ms=1)
ax1.plot( nsrdb['Wind Direction'],'.', label = "NSRDB", ms=2)
ax1.plot( hrrr.wind_dir_10m,'.', label = "", ms=3)
ax1.set_ylim(0, 360)    
plt.grid()    
 
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
fig.autofmt_xdate()
ax1.set_xlabel("Time (UTC)", fontsize = 'x-large')
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)



#%% RF model

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer


## Pipeline of data transformations

hrrr_test = hrrr.drop(columns=['time', "wind_dir_10m"])

#  pipeline for pre-processing
std_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),    #, fill_value=0 replaces nans with the specified values (mean, median, defined)
    # ("standardize", StandardScaler()),                               # Data scaling: standard deviation = 1 around zero mean 
])

test = std_pipeline.fit_transform(hrrr_test)


df = pd.DataFrame(
    test,
    columns= hrrr_test.columns.tolist(),
    index=hrrr.index)





## Prepare data

# Input and output of model

response = 'wspd_3m'   
out = df[[response]]

predictors = list(df.columns)
predictors.remove(response)
inp = df[predictors]


# Training and test data
from sklearn.model_selection import train_test_split
out_train, out_test, inp_train, inp_test = train_test_split(out, inp, train_size=0.8)   # random_state makes results reproducable

# print ("{}% of data are training data.".format(len(out_train)/len(out)*100))






regression_plots = 1

if regression_plots == 1:

    ## Example regression plots
    fig = plt.figure(figsize=(10,6))
    plt.suptitle("Regression plots")
    N = inp.shape[1]
    rows = math.floor(np.sqrt(N))
    cols = math.ceil(N/rows)
    for i in range(N):
        ax = plt.subplot(rows, cols, i+1)
        plt.scatter(inp_train.iloc[:,i], out_train  , label = "train")    
        plt.scatter(inp_test.iloc[:,i], out_test  , label = "test")
        plt.xlabel(inp.iloc[:,i].name)
        if i%cols==0:
            plt.ylabel(response)
    plt.legend()        
    plt.tight_layout()
    

    # pd.plotting.scatter_matrix(df)  # makes correlation plot of every parameter combination
    

## Train model

# from sklearn.linear_model import LinearRegression
# model = LinearRegression().fit(inp_train, out_train)

# import sklearn.neighbors
# model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3).fit(inp_train, out_train)

from sklearn.ensemble  import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=200, max_features=5)
model = forest.fit(inp_train, out_train.values.ravel())


# Feature importance
importance = forest.feature_importances_
importance_sorted = pd.Series(
    importance, index= model.feature_names_in_
).sort_values(ascending=True)


## Evaluate model

# Score of model
MS = model.score(inp_test, out_test )

# Score of model on training set
MS_train = model.score(inp_train, out_train )

# Predicted output
out_pred = model.predict(inp_test)


# Mean absolute error
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(out_test, out_pred)


print ("MAE={}, training score={}, test score={}".format(MAE.round(2), MS_train.round(2), MS.round(2)))


# Feature importance
for i in range(len(importance)):
    print('Feature: {}, Score: {}'.format(model.feature_names_in_[i],importance[i]))
 

# Plot model evaluation
fig, [ax, ax1, ax2] = plt.subplots(1, 3, figsize=(11,5), gridspec_kw={'width_ratios': [1, 1, 2]})
plt.suptitle(str(model))

ax.set_title("Model evaluation")
ax.scatter(out_test  , out_pred, label = "MAE={}, score={}".format(MAE.round(2), MS.round(2)), color="grey")
ax.set_xlabel(response + " test data")
ax.set_ylabel(response + " predicted")  
ax.legend()   
ax.grid()

ax1.set_title("Feature importance")
ax1.bar([x for x in range(len(importance_sorted))], importance_sorted, tick_label = inp.columns, color="grey")
ax1.tick_params(axis='x', rotation=45)
ax1.grid()

ax2.set_title("Predicted time series")
ax2.set_ylabel('Wind speed (m s$^{-1}$)')
ax2.plot(inflow.wspd_3m,'.', label = "NSO, 3.5m", ms=1)
ax2.plot( hrrr.wind_speed_10m,'.', label = "HRRR, 10m", ms=3, color="C2")
ax2.plot( inp_test.index, out_pred,'.', label = "Predicted, 3.5m", ms=10, color="black")
#ax2.plot(out_train,'.', label = "Predicted, 3.5m", ms=10, color="red")
plt.legend(loc=1,  markerscale=2)
ax2.grid()
ax2.set_xlabel("Date")

fig.autofmt_xdate()
plt.tight_layout()
# ax1 = plt.subplot(2, 1, 2, sharex=ax2)  # 
# ax1.set_ylabel('Wind dir ($^\circ$)')
# ax1.plot(inflow.wdir_3m,'.', label = "", ms=1)
# hrrr['wdir'] = np.degrees(np.arctan2(hrrr.u10, hrrr.v10)) +180
# ax1.plot( hrrr.valid_time, hrrr.wdir,'.', label = "", ms=3)
# ax1.set_ylim(0, 360)    
# plt.grid()    
 
# #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
# 
# 
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.1)








