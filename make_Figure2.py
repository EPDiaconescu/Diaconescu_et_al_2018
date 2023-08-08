

from netCDF4 import Dataset
from sklearn.metrics import mean_squared_error
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib  as mpl
from matplotlib.colors import Normalize
from scipy import *
from shapely.geometry import Polygon,Point, MultiPoint

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print "\t\ttype:", repr(nc_fid.variables[key].dtype)
            for ncattr in nc_fid.variables[key].ncattrs():
                print '\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr))
        except KeyError:
            print "\t\tWARNING: %s does not contain variable attributes" % key

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print "NetCDF Global Attributes:"
        for nc_attr in nc_attrs:
            print '\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print "NetCDF dimension information:"
        for dim in nc_dims:
            print "\tName:", dim
            print "\t\tsize:", len(nc_fid.dimensions[dim])
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print "NetCDF variable information:"
        for var in nc_vars:
            if var not in nc_dims:
                print '\tName:', var
                print "\t\tdimensions:", nc_fid.variables[var].dimensions
                print "\t\tsize:", nc_fid.variables[var].size
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars



def find_index_of_nearest_xy(y_array, x_array, y_point, x_point):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = np.where(distance==distance.min())
    return idy[0],idx[0]


def do_all(y_array, x_array, points, polygon):
    store = []
    list_index = []
    for i in xrange(points.shape[1]):
        if polygon.contains(Point((points[0,i],points[1,i]))):
            store.append(find_index_of_nearest_xy(y_array,x_array,points[1,i],points[0,i]))
            list_index.append(i)
    return store,list_index



def lon_lat_to_cartesian(lon, lat, R = 1):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x =  R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x,y,z



#####################################################################################################
input= 'D:/NowWorking/newProjet/results/climateMean_stations/'
output='D:/NowWorking/newProjet/results/coord_dom_ARC_b/'
varName='pr'
Numarul='15'
periode='1980to2004'
test1='coord_select_ARC_tabG_'+Numarul+'noYY'


from shapely.geometry import MultiPoint
from shapely.geometry import Point
coords_lon=[-137.6,-120,-110,-98,-76,-60,-60,-60,-145,-142,-137.6]
coords_lat=[57,58,60,62,62,59.5,68,84,84,57,57]
xt, yt, zt = lon_lat_to_cartesian(coords_lon,coords_lat)
coords = zip(xt,yt)

poly = MultiPoint(coords).convex_hull

nc = Dataset('D:/NowWorking/newProjet/topoARC44.nc')
lats = nc.variables['lat'][:]
lons = nc.variables['lon'][:]
data = nc.variables['ELEV'][:,:].squeeze()
#data = np.ma.masked_values(data,-999.)
data[data<0]=-1

nc_st = input+'precipitation_tabG_'+Numarul+'noYY_clim_'+periode+'.csv'
df = pd.read_csv(nc_st, sep=',')
(lines, stations) = df.shape
media1 = df.iloc[0,1:]
ec_id = df.columns[1:]
lonsDF =df.iloc[-2,1:]
latsDF =df.iloc[-1,1:]

media=np.array(media1, dtype=float)
lonsSt0=np.array(lonsDF, dtype=float)
latsSt0=np.array(latsDF, dtype=float)
points_1=np.append([lonsSt0], [latsSt0], axis=0)

lonsSt, latsSt, zst = lon_lat_to_cartesian(lonsSt0,latsSt0)

from shapely.geometry import Point
for i in range(0,lonsSt.shape[0]):
    point = Point(lonsSt[i],latsSt[i])
    if poly.contains(point)==False:
        lonsSt0[i]=nan

new_lons=lonsSt0[np.isfinite(lonsSt0)]
new_lats=latsSt0[np.isfinite(lonsSt0)]
new_ec_id=ec_id[np.isfinite(lonsSt0)]

points_2=np.append([new_lons], [new_lats], axis=0)

lonsDFN=(pd.DataFrame(new_lons)).transpose()
lonsDFN.index=['lons']
lonsDFN.columns=new_ec_id
latsDFN=(pd.DataFrame(new_lats)).transpose()
latsDFN.index=['lons']
latsDFN.columns=new_ec_id

final_ARC=lonsDFN.append(latsDFN)

#final_ARC.to_csv(output+test1+'_'+periode+'.csv', sep=',')

clevs = [-1500, 0, 200, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2800, 3200, 3600, 4000]
cgbvr2 = mpl.colors.ListedColormap([[1., 1., .8], [.8, 1., .2], [.6, 1., 0.],  [0., 1., 0.],  [.2, .8, 0.], [0., .6, 0.], [.6, 1., 1.], [0., 1., 1.], [0., .6, 1.],  [0., 0., 1.], [0., 0., .6], [0., 0., .4], [.8, .6, 1.], [.8, .4, 1.], [.6, 0., 1.],  [.4, 0., .6], [1., 0., 0.], [.8, 0., 0.], [.6, 0., 0.], [.4, 0., 0.]])
cgbvr = mpl.colors.ListedColormap([[0., 1., 1.], [.8, 1., .2], [.6, 1., 0.],  [0., 1., 0.],  [.2, .8, 0.], [0., .6, 0.], [.6, 1., 1.], [0., 1., 1.], [0., .6, 1.],  [0., 0., 1.], [0., 0., .6], [0., 0., .4], [.8, .6, 1.], [.8, .4, 1.], [.6, 0., 1.],  [.4, 0., .6], [1., 0., 0.], [.8, 0., 0.], [.6, 0., 0.], [.4, 0., 0.]])



cmaps = [('Perceptually Uniform Sequential',
                            ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]

val=points_2[0,:].size

points_tas=array([[-130.        , -122.59999847, -133.69999695, -139.05000305,
        -140.19999695, -135.86999512, -139.83000183, -137.36999512,
        -137.22000122, -139.13000488, -132.75      , -128.82000732,
        -135.07000732, -124.72000122, -134.8500061 , -109.16999817,
        -121.23000336, -111.97000122, -115.77999878, -133.47999573,
        -126.80000305, -133.        , -114.4300003 ,  -96.08000183,
         -90.72000122, -115.12999725,  -83.37000275,  -89.80000305,
         -93.43000031, -111.23000336,  -62.33000183,  -63.77999878,
        -105.12999725,  -76.5       ,  -61.38000107,  -66.81999969,
         -68.51999664,  -71.16999817,  -85.93000031,  -81.25      ,
         -68.55000305,  -75.12999725,  -84.62000275,  -77.97000122,
         -94.98000336, -125.26999664, -119.34999847],
       [  58.41999817,   58.83000183,   59.56999969,   61.36999893,
          69.62000275,   63.61999893,   67.56999969,   62.81999969,
          68.94999695,   64.05000305,   60.16999817,   60.11999893,
          60.72000122,   70.16999817,   67.40000153,   62.72000122,
          61.77000046,   60.02000046,   60.83000183,   68.30000305,
          65.27999878,   69.44999695,   62.47000122,   64.30000305,
          63.33000183,   67.81999969,   64.19999695,   68.52999878,
          68.81999969,   65.76999664,   82.5       ,   67.52999878,
          69.09999847,   64.19999695,   66.65000153,   68.47000122,
          70.48000336,   68.65000153,   79.98000336,   68.77999878,
          63.75      ,   68.90000153,   72.98000336,   72.69999695,
          74.72000122,   72.        ,   76.23000336]])


fig1 = plt.figure(figsize=(11,8.5))
plt.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.10,wspace=0.05,hspace=0.05)
width = 6000000; height=4000000; lon_0 = -92; lat_0 = 75
m = Basemap(resolution='l',width=width,height=height,projection='aeqd', lat_0=lat_0,lon_0=lon_0)

m.drawcoastlines()
m.drawcountries(linewidth=1)
m.drawmeridians(np.arange(-180,180,20), linewidth=0.2)
m.drawparallels(np.arange(40,90,10),linewidth=0.2)

#################################


xt, yt = m(points_tas[0,:], points_tas[1,:])

x,y = m(lons,lats)
#im=m.contourf(x,y,data,clevs, cmap='terrain')
m.shadedrelief()

# min_marker_size = 2.8
# for lon, lat, mag in zip(points_2[0,:], points_2[1,:], 1.25**(points_2[1,:]/10)):
#     xt, yt = m(points_tas[0,:], points_tas[1,:])
#     msize = round(mag * min_marker_size)
#     m.plot(xt, yt, 'ro', markersize=msize, lw = 0, markeredgecolor='r' )
m.plot(xt, yt, 'bo', markersize=14, lw = 0, markeredgecolor='k' )


min_marker_size = 6
for lon, lat, mag in zip(points_2[0,:], points_2[1,:], 1.1**(points_2[1,:]/10)):
    x,y = m(lon, lat)
    msize = round(mag + min_marker_size)
    m.plot(x, y, 'ro', markersize=msize, lw = 0, markeredgecolor='w' )

# add colorbar
#cb = m.colorbar

plt.tight_layout()
plt.savefig('D:/NowWorking/newProjet/results/images_alte/Figure2_B.png')


plt.show()

#################################################################################
