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
    z = norm(np.array(yarray))*200+250
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
    font_size = 20 # Adjust as appropriate.
    c.ax.tick_params(labelsize=font_size)
    plt.title(title, size=28)
#    plt.tight_layout()
    plt.savefig(outputfig)
    fig.clf()


#################################################
output= 'D:/NowWorking/newProjet/results/images2016/'
input_a='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/tasAnualMean_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_b='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/HDD17_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_c='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/GDD5_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_d='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/SU15_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_e='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/TXx_K_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_f='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/TX90_K_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_g='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/FD_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_h='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/TNn_K_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'
input_i='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/TN10_K_StationsAnomalies_ARCcordex_15noYY_1980to2004_.csv'


fileName ='Fig3_C'

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

df_g=pd.read_csv(input_g, sep=',')
obs_g=df_g.iloc[0,1:]
data_obs_g=np.array(obs_g, dtype=float)
lonsDF_g =df_g.iloc[-1,1:]
latsDF_g = df_g.iloc[-2,1:]
points_lon_g=np.array(lonsDF_g, dtype=float)
points_lat_g = np.array(latsDF_g, dtype=float)

df_h=pd.read_csv(input_h, sep=',')
obs_h=df_h.iloc[0,1:]
data_obs_h=np.array(obs_h, dtype=float)
lonsDF_h =df_h.iloc[-1,1:]
latsDF_h = df_h.iloc[-2,1:]
points_lon_h=np.array(lonsDF_h, dtype=float)
points_lat_h = np.array(latsDF_h, dtype=float)

df_i=pd.read_csv(input_i, sep=',')
obs_i=df_i.iloc[0,1:]
data_obs_i=np.array(obs_i, dtype=float)
lonsDF_i =df_i.iloc[-1,1:]
latsDF_i = df_i.iloc[-2,1:]
points_lon_i=np.array(lonsDF_i, dtype=float)
points_lat_i = np.array(latsDF_i, dtype=float)

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


### Mean Tmean ####
increment=2
vmin=-20
ncolors=11
zarray = data_obs_a-273.15
figureP(points_lon_a, points_lat_a, zarray,vmin,increment,ncolors,cgbvr11,' ','a) Mean Tmean ($^\circ$C) ',output+fileName+'_a.png')

### HDD ####
increment=1
vmin=3
ncolors=11
zarray = data_obs_b/1000
figureP(points_lon_b, points_lat_b, zarray,vmin,increment,ncolors,cgbvr11_RB,' ','b) HDD (*1000 $^\circ$C) ',output+fileName+'_b.png')

### GDD ####
increment=1
vmin=0
ncolors=13
zarray = data_obs_c/100
figureP(points_lon_c, points_lat_c, zarray,vmin,increment,ncolors,cgbvr13,' ','c) GDD (*100 $^\circ$C) ',output+fileName+'_c.png')


### SU15 ####
increment=10
vmin=0
ncolors=13
zarray = data_obs_d
figureP(points_lon_d, points_lat_d, zarray,vmin,increment,ncolors,cgbvr13,' ','d) SU15 (days) ',output+fileName+'_d.png')

### TXx ####
increment=2
vmin=12
ncolors=11
zarray = data_obs_e-273.15
figureP(points_lon_e, points_lat_e, zarray,vmin,increment,ncolors,cgbvr11,' ','e) TXx ($^\circ$C) ',output+fileName+'_e.png')

### TX90 ####
increment=2
vmin=4
ncolors=11
zarray = data_obs_f-273.15
figureP(points_lon_f, points_lat_f, zarray,vmin,increment,ncolors,cgbvr11,' ','f) TX90 ($^\circ$C) ',output+fileName+'_f.png')


### FD ####
increment=15
vmin=190
ncolors=11
zarray = data_obs_g
figureP(points_lon_g, points_lat_g, zarray,vmin,increment,ncolors,cgbvr11_RB,' ','g) FD (days) ',output+fileName+'_g.png')


### TNn ####
increment=1
vmin=-50
ncolors=13
zarray = data_obs_h-273.15
figureP(points_lon_h, points_lat_h, zarray,vmin,increment,ncolors,cgbvr13,' ','h) TNn ($^\circ$C) ',output+fileName+'_h.png')

### TN10 ####

increment=2
vmin=-45
ncolors=11
zarray = data_obs_i-273.15
figureP(points_lon_i, points_lat_i, zarray,vmin,increment,ncolors,cgbvr11,' ','i) TN10 ($^\circ$C) ',output+fileName+'_i.png')
