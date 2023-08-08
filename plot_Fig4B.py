__author__ = 'emiliapauladiaconescu'

import numpy as np
import pandas as pd
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib  as mpl
import datetime
from netCDF4 import num2date, date2num

def figureP(xarray, yarray, zarray,vmin,increment,ncolors,cmap,box_label,title,outputfig):
    norm = Normalize()
    z = norm(np.array(yarray))*400+70
    fig = plt.figure(figsize=(11,8.5), dpi=1000)
    plt.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.01,wspace=0.05,hspace=0.05)
    width = 6000000; height=3500000; lon_0 = -92; lat_0 = 75
    m = Basemap(resolution='l',width=width,height=height,projection='aeqd', lat_0=lat_0,lon_0=lon_0)
    m.drawcoastlines()
    m.drawcountries(linewidth=1)
    m.drawmeridians(np.arange(-180,180,20), linewidth=0.2)
    m.drawparallels(np.arange(40,90,10),linewidth=0.2)
    m.fillcontinents(color=[0.9,0.9,0.9])
    vmax=vmin+increment*ncolors
    x, y = m(xarray, yarray)
    sc = m.scatter(x,y, s=z, zorder=3, marker='o', lw=.3, antialiased=True, cmap=cmap, c=zarray, vmin=vmin, vmax=vmax)
    c = plt.colorbar(sc, orientation='horizontal',extend='both', shrink=0.9, aspect=24, pad=0.07)
    c.set_label(box_label, size=18)
    c.set_ticks(np.linspace(vmin, vmax, ncolors+1))
#    new=[int(w) for w in np.linspace(vmin+increment, vmax, ncolors)]
    font_size = 22 # Adjust as appropriate.
    c.ax.tick_params(labelsize=font_size)
    plt.title(title, size=28)
#    plt.tight_layout()
    plt.savefig(outputfig)
    fig.clf()


#################################################
output= 'D:/NowWorking/newProjet/results/images2016/'
input_a='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/prAnualMean_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_b='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/pr1mmPc_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_c='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/rx1day_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_d='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/rx5days_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_e='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/PrCum_GTp99_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_f='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/PrCum_GTp95_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'

fileName ='Fig4_climatoB'

#################################################

df_a=pd.read_csv(input_a, sep=',')
obs_a=df_a.iloc[0,1:]
data_obs_a=np.array(obs_a, dtype=float)
lonsDF_a =df_a.iloc[-1,1:]
latsDF_a = df_a.iloc[-2,1:]
points_lon_a=np.array(lonsDF_a, dtype=float)
points_lat_a = np.array(latsDF_a, dtype=float)

df_b=pd.read_csv(input_b, sep=',')
obs_b=df_b.iloc[0,1:]
data_obs_b=np.array(obs_b, dtype=float)
lonsDF_b =df_b.iloc[-1,1:]
latsDF_b = df_b.iloc[-2,1:]
points_lon_b=np.array(lonsDF_b, dtype=float)
points_lat_b = np.array(latsDF_b, dtype=float)

df_c=pd.read_csv(input_c, sep=',')
obs_c=df_c.iloc[0,1:]
data_obs_c=np.array(obs_c, dtype=float)
lonsDF_c =df_c.iloc[-1,1:]
latsDF_c = df_c.iloc[-2,1:]
points_lon_c=np.array(lonsDF_c, dtype=float)
points_lat_c = np.array(latsDF_c, dtype=float)

df_d=pd.read_csv(input_d, sep=',')
obs_d=df_d.iloc[0,1:]
data_obs_d=np.array(obs_d, dtype=float)
lonsDF_d =df_d.iloc[-1,1:]
latsDF_d = df_d.iloc[-2,1:]
points_lon_d=np.array(lonsDF_d, dtype=float)
points_lat_d = np.array(latsDF_d, dtype=float)

df_e=pd.read_csv(input_e, sep=',')
obs_e=df_e.iloc[0,1:]
data_obs_e=np.array(obs_e, dtype=float)
lonsDF_e =df_e.iloc[-1,1:]
latsDF_e = df_e.iloc[-2,1:]
points_lon_e=np.array(lonsDF_e, dtype=float)
points_lat_e = np.array(latsDF_e, dtype=float)

df_f=pd.read_csv(input_f, sep=',')
obs_f=df_f.iloc[0,1:]
data_obs_f=np.array(obs_f, dtype=float)
lonsDF_f =df_f.iloc[-1,1:]
latsDF_f = df_f.iloc[-2,1:]
points_lon_f=np.array(lonsDF_f, dtype=float)
points_lat_f = np.array(latsDF_f, dtype=float)


from matplotlib.colors import Normalize
norm = Normalize()


cgbvr1 = mpl.colors.ListedColormap ([[0.6862745098039216, 0.20784313725490197, 0.2784313725490196],
 [0.8470588235294118, 0.3215686274509804, 0.34509803921568627],
 [0.9372549019607843, 0.5215686274509804, 0.47843137254901963],
 [0.9607843137254902, 0.6941176470588235, 0.5450980392156862],
 [0.9764705882352941, 0.8470588235294118, 0.6588235294117647],
 [0.9490196078431372, 0.9333333333333333, 0.7725490196078432],
 [0.8470588235294118, 0.9254901960784314, 0.9450980392156862],
 [0.6039215686274509, 0.8509803921568627, 0.9333333333333333],
 [0.26666666666666666, 0.7803921568627451, 0.9372549019607843],
 [0.0, 0.6666666666666666, 0.8862745098039215],
 [0.0, 0.4549019607843137, 0.7372549019607844]])

cgbvr2a = mpl.colors.ListedColormap ([[0.0, 0.4549019607843137, 0.7372549019607844],
 [0.0, 0.6666666666666666, 0.8862745098039215],
 [0.26666666666666666, 0.7803921568627451, 0.9372549019607843],
 [0.6039215686274509, 0.8509803921568627, 0.9333333333333333],
 [0.8470588235294118, 0.9254901960784314, 0.9450980392156862],
 [0.9490196078431372, 0.9333333333333333, 0.7725490196078432],
 [0.9764705882352941, 0.8470588235294118, 0.6588235294117647],
 [0.9607843137254902, 0.6941176470588235, 0.5450980392156862],
 [0.9372549019607843, 0.5215686274509804, 0.47843137254901963],
 [0.8470588235294118, 0.3215686274509804, 0.34509803921568627],
 [0.6862745098039216, 0.20784313725490197, 0.2784313725490196]])

cgbvr11_RB = mpl.colors.ListedColormap ([[0.6862745098039216, 0.20784313725490197, 0.2784313725490196],
                                      [0.8470588235294118, 0.3215686274509804, 0.34509803921568627],
                                      [0.9372549019607843, 0.5215686274509804, 0.47843137254901963],
                                      [0.9607843137254902, 0.6941176470588235, 0.5450980392156862],
                                      [0.9764705882352941, 0.8470588235294118, 0.6588235294117647],
                                      [0.9490196078431372, 0.9333333333333333, 0.7725490196078432],
                                      [0.6039215686274509, 0.8509803921568627, 0.9333333333333333],
                                      [0.26666666666666666, 0.7803921568627451, 0.9372549019607843],
                                      [0.0, 0.6666666666666666, 0.8862745098039215],
                                      [0.0, 0.4549019607843137, 0.7372549019607844],
                                      [0.1398846663096372, 0.27690888619890397, 0.61514804293127623]])

cgbvr13_RB = mpl.colors.ListedColormap ([[0.4862745098039216, 0.10784313725490197, 0.1784313725490196],
                                      [0.6862745098039216, 0.20784313725490197, 0.2784313725490196],
                                      [0.8470588235294118, 0.3215686274509804, 0.34509803921568627],
                                      [0.9372549019607843, 0.5215686274509804, 0.47843137254901963],
                                      [0.9607843137254902, 0.6941176470588235, 0.5450980392156862],
                                      [0.9764705882352941, 0.8470588235294118, 0.6588235294117647],
                                      [0.9490196078431372, 0.9333333333333333, 0.7725490196078432],
                                      [0.6039215686274509, 0.8509803921568627, 0.9333333333333333],
                                      [0.26666666666666666, 0.7803921568627451, 0.9372549019607843],
                                      [0.0, 0.6666666666666666, 0.8862745098039215],
                                      [0.0, 0.4549019607843137, 0.7372549019607844],
                                      [0.1398846663096372, 0.27690888619890397, 0.61514804293127623],
                                      [0.0, 0.18235294117647061, 0.45490196078431372]])



cgbvr11 = mpl.colors.ListedColormap ([[0.1398846663096372, 0.27690888619890397, 0.61514804293127623],
                                    [0.0, 0.4549019607843137, 0.7372549019607844],
                                    [0.0, 0.6666666666666666, 0.8862745098039215],
                                    [0.26666666666666666, 0.7803921568627451, 0.9372549019607843],
                                    [0.6039215686274509, 0.8509803921568627, 0.9333333333333333],
                                    [0.9490196078431372, 0.9333333333333333, 0.7725490196078432],
                                    [0.9764705882352941, 0.8470588235294118, 0.6588235294117647],
                                    [0.9607843137254902, 0.6941176470588235, 0.5450980392156862],
                                    [0.9372549019607843, 0.5215686274509804, 0.47843137254901963],
                                    [0.8470588235294118, 0.3215686274509804, 0.34509803921568627],
                                    [0.6862745098039216, 0.20784313725490197, 0.2784313725490196]])

cgbvr13 = mpl.colors.ListedColormap ([[0.0, 0.18235294117647061, 0.45490196078431372],
                                    [0.1398846663096372, 0.27690888619890397, 0.61514804293127623],
                                    [0.0, 0.4549019607843137, 0.7372549019607844],
                                    [0.0, 0.6666666666666666, 0.8862745098039215],
                                    [0.26666666666666666, 0.7803921568627451, 0.9372549019607843],
                                    [0.6039215686274509, 0.8509803921568627, 0.9333333333333333],
                                    [0.9490196078431372, 0.9333333333333333, 0.7725490196078432],
                                    [0.9764705882352941, 0.8470588235294118, 0.6588235294117647],
                                    [0.9607843137254902, 0.6941176470588235, 0.5450980392156862],
                                    [0.9372549019607843, 0.5215686274509804, 0.47843137254901963],
                                    [0.8470588235294118, 0.3215686274509804, 0.34509803921568627],
                                    [0.6862745098039216, 0.20784313725490197, 0.2784313725490196],
                                    [0.4862745098039216, 0.10784313725490197, 0.1784313725490196]])



### Mean pr ####
increment=0.1
vmin=0.4
ncolors=13
zarray = data_obs_a
figureP(points_lon_a, points_lat_a, zarray,vmin,increment,ncolors,cgbvr13_RB,' ','a) Mean Pr (mm/day) ',output+fileName+'_a.png')


### R1mm ####
increment=2
vmin=8
ncolors=11
zarray = data_obs_b
figureP(points_lon_b, points_lat_b, zarray,vmin,increment,ncolors,cgbvr11_RB,' ','b) R1mm (%) ',output+fileName+'_b.png')

### RX1day ####
xarray_c = points_lon_c
yarray_c = points_lat_c
increment=2
vmin=10
ncolors=11
zarray = data_obs_c
figureP(points_lon_c, points_lat_c, zarray,vmin,increment,ncolors,cgbvr11_RB,' ','c) RX1day (mm/day) ',output+fileName+'_c.png')

### RX5day ####
increment=4
vmin=18
ncolors=11
zarray = data_obs_d
figureP(points_lon_d, points_lat_d, zarray,vmin,increment,ncolors,cgbvr11_RB,' ','d) RX5day (mm/day) ',output+fileName+'_d.png')

### R99pTOT ####
increment=3
vmin=4
ncolors=11
zarray = data_obs_e
figureP(points_lon_e, points_lat_e, zarray,vmin,increment,ncolors,cgbvr11_RB,' ','e) R99pTOT (mm) ',output+fileName+'_e.png')

### R99pTOT ####
increment=11
vmin=12
ncolors=11
zarray = data_obs_f
figureP(points_lon_f, points_lat_f, zarray,vmin,increment,ncolors,cgbvr11_RB,' ','f) R95pTOT (mm) ',output+fileName+'_f.png')
