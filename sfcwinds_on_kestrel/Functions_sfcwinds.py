import numpy as np
import xarray as xr

Sandia_coords = [34.96226779999999, -106.5096873]

def find_closest_HRRR_loc(hrrr, hrrr_coords_to_use):
    
    """
    hrrr: HRRR xarray
    hrrr_coords_to_use: array -> [lat, lon]

    Use this function on a daily HRRR file with variables:
        latitude (y, x)
        longitude (y, x)
    
    """

    # Observations use longitudes between -180 to 180, transform HRRR to that
    #hrrr['longitude'] = xr.where(hrrr['longitude'] > 180, hrrr['longitude'] - 360, hrrr['longitude'])

    # Find the shortest distance to the given lat/lon point
    dist = (hrrr['longitude'].values - hrrr_coords_to_use[1])**2 + (hrrr['latitude'].values - hrrr_coords_to_use[0])**2
    print(f"Coordinates extracted from HRRR file longitude {hrrr['longitude'].values} latitude {hrrr['latitude'].values} ")
    print(f"Argument coordinates for find_cloest_HRRR_loc {hrrr_coords_to_use}")
    print(f"Distance calculated within function {dist}")    
    # Get the index, lat and lon of the minimum distance
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    closest_lat = hrrr.latitude.values[idx]
    closest_lon = hrrr.longitude.values[idx]

    # HRRR data at that index
    #hrrr_loc =hrrr.isel(y=idx[0], x=idx[1])
    hrrr_loc = hrrr.isel(**{dim: int(i) for dim, i in zip(hrrr.latitude.dims, idx)})

    return hrrr_loc, closest_lat, closest_lon, idx



def find_nearest_obs(obs, target_lat = Sandia_coords[0], target_lon = Sandia_coords[1] ):

    # Compute distance
    obs['distance_to_sandia'] = np.sqrt(
        (obs['lat'] - target_lat)**2 + (obs['lon'] - target_lon)**2
    )
    
    # Find the row with the smallest distance
    closest_station = np.unique(obs[obs.distance_to_sandia == np.nanmin(obs['distance_to_sandia'])].stid)[0]
    
    # Filter data for the closest station
    ts = obs[obs.stid == closest_station]

    return ts


# distance in meters between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth (specified in decimal degrees).
    Returns distance in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def find_closest_grid_point(lon, lat, lon_grid, lat_grid):
    # Compute distance to all grid points in terrain file
    dist = np.sqrt((lon_grid - lon)**2 + (lat_grid - lat)**2)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return idx

def get_vicinity_mean_std(std_array, idx, radius=10):
    # get the standard deviation of terrain height in s specified radius
    y, x = idx
    y1 = max(0, y - radius)
    y2 = min(std_array.shape[0], y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(std_array.shape[1], x + radius + 1)
    vicinity = std_array[y1:y2, x1:x2]
    return np.nanmean(vicinity)



def wspd_wdir_from_uv(u, v):
    """ 
        u   eastward wind component
        v   northward wind component
        wdir   direction where wind is coming from
    """
    
    wspd = (u**2 + v**2)**0.5
    wdir = np.degrees(np.arctan2(u, v)) +180

    return wspd, wdir

def uv_from_wspd_wdir(wspd, wdir):
    """ 
        u   eastward wind component
        v   northward wind component
        wdir   direction where wind is coming from, input in degrees
    """

    wind_dir_rad = np.radians(wdir)
    u = -wspd * np.sin(wind_dir_rad)
    v = -wspd * np.cos(wind_dir_rad)    

    return u,v


def rotate_to_true_north(u10, v10, lon):
    """
    Rotate HRRR 10m wind components to true north orientation.

    Parameters:
    - u10, v10: xarray.DataArrays of shape (time, step)
    - lon: longitude (scalar, degrees east)
    - rotcon_p: rotation constant (default: sin(lat_tan), 38.5° for HRRR)
    - lon_xx_p: reference meridian (default: -97.5°)

    Returns:
    - un10, vn10: xarray.DataArrays of rotated wind components relative to True North
    """
    
    lon = ((lon + 180) % 360) - 180  # make between -180 and 180
                             

    rotcon_p = np.sin(np.radians(38.5))  # for Lambert Conformal (HRRR), 0.6225
    lon_xx_p = -97.5  # reference meridian

    angle2 = rotcon_p * (lon - lon_xx_p) * np.pi / 180
    sinx2 = np.sin(angle2)
    cosx2 = np.cos(angle2)

    # Apply rotation
    un10 = cosx2 * u10 + sinx2 * v10
    vn10 = -sinx2 * u10 + cosx2 * v10

    return un10, vn10
