"""
Spatial Grid Utility Module
"""
import numpy as np

class BayOfBengalGrid:
    """
    Standard spatial grid representation for the Bay of Bengal region:
    Latitude: 5.0° N to 22.0° N
    Longitude: 80.0° E to 100.0° E
    Spatial Resolution: 0.25° × 0.25°
    """
    def __init__(self, lat_min=5.0, lat_max=22.0, lon_min=80.0, lon_max=100.0, res=0.25):
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.res = res

        self.lats = np.arange(lat_min, lat_max + res / 2.0, res)
        self.lons = np.arange(lon_min, lon_max + res / 2.0, res)

        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)

        self.grid_lon, self.grid_lat = np.meshgrid(self.lons, self.lats)

    def shape(self):
        return (self.n_lat, self.n_lon)

    def __repr__(self):
        return (f"BayOfBengalGrid(Lat: [{self.lat_min}, {self.lat_max}], "
                f"Lon: [{self.lon_min}, {self.lon_max}], "
                f"Res: {self.res}°, Grid: {self.n_lat}x{self.n_lon})")
