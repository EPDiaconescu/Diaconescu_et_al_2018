# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 12:51:09 2014

@author: Diaconescu Emilia
"""

# example map on North America CORDEX
#for North America the values are:
#rotpole.grid_north_pole_latitude = 42.50
#rotpole.grid_north_pole_longitude = 83.00
#rotpole.north_pole_grid_longitude = 180.0
# must consider lon_0 = rotpole.grid_north_pole_longitude - 180


from __future__ import print_function
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib  as mpl

nc = Dataset('D:/NowWorking/newProjet/data/images_dom_old/dom_CRCM5_NA.nc')
lats = nc.variables['lat'][:]
lons = nc.variables['lon'][:]
data = nc.variables['tas'][0,:,:].squeeze()
data = np.ma.masked_values(data,-999.)

rotpole_grid_north_pole_latitude = 42.50
rotpole_grid_north_pole_longitude = 83.00
rotpole_north_pole_grid_longitude = 180.0

def normalize180(lon):
    """Normalize lon to range [180, 180)"""
    lower = -180.; upper = 180.
    if lon > upper or lon == lower:
        lon = lower + abs(lon + upper) % (abs(lower) + abs(upper))
    if lon < lower or lon == upper:
        lon = upper - abs(lon - lower) % (abs(lower) + abs(upper))
    return lower if lon == upper else lon

lon_0 = normalize180(rotpole_grid_north_pole_longitude-180.)
o_lon_p = rotpole_north_pole_grid_longitude
o_lat_p = rotpole_grid_north_pole_latitude
print( 'lon_0,o_lon_p,o_lat_p=',lon_0,o_lon_p,o_lat_p)

fig = plt.figure(figsize=(8,8), dpi=300)
plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)

m = Basemap(resolution='c',projection='ortho',lat_0=65.,lon_0=-100.)

x,y = m(lons,lats)
m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawmeridians(np.arange(-180,180,20), linewidth=0.2, fontsize=16)
m.drawparallels(np.arange(0,80,20), linewidth=0.2, fontsize=16)

clevs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
cgbvr = mpl.colors.ListedColormap([[1., 0.4, 0.], [.8, 1., .2], [.6, 1., 0.],  [.2, .8, 0.], [0., .6, 0.], [.6, 1., 1.], [0., 1., 1.], [0., 0., 1.], [0., 0., .6], [0., 0., .4], [.8, .6, 1.], [.8, .4, 1.], [.6, 0., 1.],  [.4, 0., .6], [.8, 0., 0.], [.4, 0., 0.]])

cs = m.contourf(x,y,data,clevs, cmap=cgbvr)
plt.tight_layout()
plt.savefig('D:/NowWorking/newProjet/results/images_alte/dom_CRCM5NA_large2.png', dpi=300)



