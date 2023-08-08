

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib  as mpl
from matplotlib.colors import Normalize
from netCDF4 import Dataset,  num2date, date2num
from scipy.spatial import cKDTree
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy

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


def find_index_of_nearest_xy(y_array, x_array, y_point, x_point):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = np.where(distance==distance.min())
    return idy[0],idx[0]


def do_all(y_array, x_array, points):
    store = []
    for i in xrange(points.shape[1]):
        store.append(find_index_of_nearest_xy(y_array,x_array,points[1,i],points[0,i]))
    return store

def figureT(xarray, yarray, zarray,vmin,increment,ncolors,cmap,outputfig):
    norm = Normalize()
    z = norm(np.array(yarray))*200+200
    fig = plt.figure(figsize=(11,8.5))
    plt.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.10,wspace=0.05,hspace=0.05)
    width = 6000000; height=3500000; lon_0 = -92; lat_0 = 75
    m = Basemap(resolution='l',width=width,height=height,projection='aeqd', lat_0=lat_0,lon_0=lon_0)
    m.drawcoastlines()
    m.drawcountries(linewidth=1)
    m.drawmeridians(np.arange(-180,180,20), linewidth=0.2)
    m.drawparallels(np.arange(40,90,10),linewidth=0.2)
    vmax=vmin+increment*ncolors
    x, y = m(xarray, yarray)
    sc = m.scatter(x,y, s=z, zorder=3, marker='o', lw=.3, antialiased=True, cmap=cmap, c=zarray, vmin=vmin, vmax=vmax)
    c = plt.colorbar(sc, orientation='horizontal', shrink=0.7, aspect=24, pad=0.07)
    c.set_label(box_label, size=18)
    c.set_ticks(np.linspace(vmin+increment, vmax, ncolors))
    new=[int(w) for w in np.linspace(vmin+increment, vmax, ncolors)]
    font_size = 16 # Adjust as appropriate.
    c.ax.tick_params(labelsize=font_size)
    plt.tight_layout()
    plt.savefig(outputfig)
    fig.clf()


def figureP(xarray, yarray, zarray,vmin,increment,ncolors,cmap,outputfig):
     norm = Normalize()
     z = norm(np.array(yarray))*300+150
     fig = plt.figure(figsize=(11,8.5))
     plt.subplots_adjust(left=0.02,right=0.98,top=0.90,bottom=0.10,wspace=0.05,hspace=0.05)
     width = 6000000; height=3500000; lon_0 = -92; lat_0 = 75
     m = Basemap(resolution='l',width=width,height=height,projection='aeqd', lat_0=lat_0,lon_0=lon_0)
     m.drawcoastlines()
     m.drawcountries(linewidth=1)
     m.drawmeridians(np.arange(-180,180,20), linewidth=0.2)
     m.drawparallels(np.arange(40,90,10),linewidth=0.2)
     vmax=vmin+increment*ncolors
     x, y = m(xarray, yarray)
     sc = m.scatter(x,y, s=z, zorder=3, marker='o', lw=.3, antialiased=True, cmap=cmap, c=zarray, vmin=vmin, vmax=vmax)
     c = plt.colorbar(sc, orientation='horizontal', shrink=0.7, aspect=24, pad=0.07)
     c.set_label(box_label, size=18)
     c.set_ticks(np.linspace(vmin+increment, vmax, ncolors))
     new=[int(w) for w in np.linspace(vmin+increment, vmax, ncolors)]
     font_size = 16 # Adjust as appropriate.
     c.ax.tick_params(labelsize=font_size)
     plt.tight_layout()
     plt.savefig(outputfig)
     fig.clf()




#################################################
domeniu='ARCcordexB'

list_varName=['FD', 'GDD5', 'HDD17', 'pr1mm', 'prAnualMean', 'PrCum_GTp95', 'PrCum_GTp99', 'PrTotPc_GTp95',
              'PrTotPc_GTp99', 'rx1day', 'rx5days', 'SU15', 'tasAnualMean', 'TN10_K', 'TN10p', 'TNn_K',
              'TX90_K', 'TX90p', 'TXx_K']
periode='1980to2004'
ValnrYY='15'
fyy='1980'
lyy='2004'
input= 'D:/NowWorking/newProjet/results/domARC_CORDEX_b/simSelected_climMean_ref/'
output='D:/NowWorking/newProjet/results/domARC_CORDEX_b/metricsSPATIAL_final_ref/'

outputScatter='D:/NowWorking/newProjet/results/domARC_CORDEX_b/scaterplotSPATIAL_final_ref/'
outputViolin='D:/NowWorking/newProjet/results/domARC_CORDEX_b/violinPlotSPATIAL_final_ref/'
outputBoxPlot='D:/NowWorking/newProjet/results/domARC_CORDEX_b/boxPlotSPATIAL_final_ref/'
outputClimIm='D:/NowWorking/newProjet/results/domARC_CORDEX_b/images_ClimMean_final_ref2/'
cmapB = mpl.colors.ListedColormap (['midnightblue','mediumblue', 'Blue','DodgerBlue','DeepSkyBlue',  'PaleTurquoise', 'LIGHTPink','LightCoral','Crimson','FireBrick','darkred' ])

cgbvr1 = mpl.colors.ListedColormap ([[ 0.,  0.4,  1],
                                    [ 0., .8, 1.],
                                    [ 0., 1., 1.],
                                    [ 0., 1., 0.6],
                                    [ 1., 1., 0.],
                                    [ 1., .8, 0.],
                                    [ 1., 0.6, 0.],
                                    [ 1., .4, 0.],
                                    [ 1., 0., 0.],
                                    [ .8, 0., 0.],
                                    [ .6, 0., 0.]])
cgbvr2 = mpl.colors.ListedColormap (['midnightblue','mediumblue', 'Blue','DodgerBlue','DeepSkyBlue',  'PaleTurquoise', 'LIGHTPink','LightCoral','Crimson','FireBrick','darkred' ])
ncolors=11
palette=["gray","green","green","green","green","green","cyan","red","red","darkblue","blue","blue","magenta","magenta","magenta","purple","magenta","purple","gray"]

for varName in list_varName:

    if varName=='prAnualMean':
        box_label='Pr [mm/day]'
        cmap=cgbvr1
    elif varName=='tasAnualMean':
        box_label='Tmean [$^\circ$C]'
        cmap=cgbvr2
    elif varName=='pr1mm':
        box_label='R1mm [days]'
        cmap=cgbvr1
    elif varName=='PrTotPc_GTp99':
        box_label='R99p [%]'
        cmap=cgbvr1
    elif varName=='PrTotPc_GTp95':
        box_label='R95p [%]'
        cmap=cgbvr1
    elif varName=='rx5days':
        box_label='RX5day [mm/day]'
        cmap=cgbvr1
    elif varName=='rx1day':
        box_label='RX1day [mm/day]'
        cmap=cgbvr1
    elif varName=='PrCum_GTp99':
        box_label='R99pTOT [mm]'
        cmap=cgbvr1
    elif varName=='PrCum_GTp95':
        box_label='R95pTOT [mm]'
        cmap=cgbvr1
    elif varName=='FD':
        box_label='FD [days]'
        cmap=cgbvr1
    elif varName=='HDD17':
        box_label='HDD  [degree]'
        cmap=cgbvr1
    elif varName=='GDD5':
        box_label='GDD  [degree]'
        cmap=cgbvr1
    elif varName=='SU15':
        box_label='Arctic Summer days [days]'
        cmap=cgbvr1
    elif varName=='TX90p':
        box_label='TX90p [%]'
        cmap=cgbvr1
    elif varName=='TN10p':
        box_label='TN10p [%]'
        cmap=cgbvr1
    elif varName=='TXx_K':
        box_label='TXx [$^\circ$C]'
        cmap=cgbvr2
    elif varName=='TNn_K':
        box_label='TNn  [$^\circ$C]'
        cmap=cgbvr2
    elif varName=='TN10_K':
        box_label='10th percentile of Tmin  [$^\circ$C]'
        cmap=cgbvr2
    elif varName=='TX90_K':
        box_label='90th percentile of Tmax  [$^\circ$C]'
        cmap=cgbvr2

    #####################################################################

    ###################################################################
    df=pd.read_csv(input+varName+'ClimaticMeanMatrix_ARCcordex_15noYY_1980to2004_sptpSelected.csv', sep=',')
    df=df.dropna(axis=0)
    ec_id=df.iloc[:,0]
    lonsDF =df.iloc[:,-1]
    latsDF =df.iloc[:,-2]
    points_lon=np.array(lonsDF, dtype=float)
    points_lat = np.array(latsDF, dtype=float)
    points_Obs=np.append([points_lon], [points_lat], axis=0)
    xarray =points_lon
    yarray =points_lat

    data=df.iloc[:,1:-2]
    dataset=data.columns
    if cmap==cgbvr2:
        dataObs=np.array(df.iloc[:,-3]-273.15, dtype=float)
    else:
        dataObs=np.array(df.iloc[:,-3], dtype=float)
    k=0
    for setdat in dataset:
        if cmap==cgbvr2:
            zarray = np.array(data[setdat]-273.15, dtype=float)
            vmin=round(np.min(zarray))
            if np.max(zarray)-np.min(zarray)<3:
                increment=0.2
            elif np.max(zarray)-np.min(zarray)<6:
                increment=0.5
            else:
                increment=round((np.max(zarray)-np.min(zarray))/11)
            figureT(xarray, yarray, zarray,vmin,increment,ncolors,cmap,outputClimIm+varName+'_'+setdat+'_'+periode+'_climateMean.png')
        else:
            zarray = np.array(data[setdat], dtype=float)
            vmin=round(np.min(zarray))
            if np.max(zarray)-np.min(zarray)<3:
                increment=0.2
            elif np.max(zarray)-np.min(zarray)<6:
                increment=0.5
            else:
                increment=round((np.max(zarray)-np.min(zarray))/11)
            figureP(xarray, yarray, zarray,vmin,increment,ncolors,cmap,outputClimIm+varName+'_'+setdat+'_'+periode+'_climateMean.png')
        fig=plt.figure(figsize=(10,10))
        ax = plt.gca()
        val_min=min([np.min(dataObs),np.min(zarray)])
        val_max=max([np.max(dataObs),np.max(zarray)])
        plt.plot([round(val_min-(val_max-val_min)/10),round(val_max+(val_max-val_min)/10)],[round(val_min-(val_max-val_min)/10),round(val_max+(val_max-val_min)/10)],'k-',linewidth=0.5)
        ax.scatter(dataObs, zarray, facecolor=palette[k],s=200, alpha=.5, edgecolor='black',linewidth=0.5)
        ax.set_xlim([round(val_min-(val_max-val_min)/10),round(val_max+(val_max-val_min)/10)])
        ax.set_ylim([round(val_min-(val_max-val_min)/10),round(val_max+(val_max-val_min)/10)])
        plt.ylabel(setdat+'\n    ', fontsize=18)
        plt.xlabel('   \n Observations', fontsize=18)
        plt.title(box_label+'\n    ', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(outputScatter+varName+'_'+setdat+'_'+periode+'_climateMeanScatter.png')
        fig.clf()
        k=k+1

    biasData=data.sub(df.iloc[:,-3],axis=0)
    for setdat in dataset:
        zarray = np.array(biasData[setdat], dtype=float)
        vmin=round(np.min(zarray))
        if np.max(zarray)-np.min(zarray)<3:
            increment=0.2
        elif np.max(zarray)-np.min(zarray)<6:
            increment=0.5
        else:
            increment=round((np.max(zarray)-np.min(zarray))/11)
        figureT(xarray, yarray, zarray,vmin,increment,ncolors,cgbvr2,outputClimIm+'Bias_'+varName+'_'+setdat+'_'+periode+'_climateMean.png')


    metrixMean=data.mean()
    metrixVar=data.var()
    metrixStd=data.std(ddof=0) # biesed ; it uses division by N
    metrixCV=(data.std()/metrixMean)*100 #coefficient of variation (CV)is std unbiased divided by mean

    metrixStd_raport=metrixStd.div(metrixStd.iloc[-1], axis=0)
    metrixStd_raport.columns=['std_raport']
    metrixVar_raport=metrixVar.div(metrixVar.iloc[-1], axis=0)
    metrixVar_raport.columns=['var_raport']
    metrixR_coef=pd.DataFrame([scipy.stats.pearsonr(data.iloc[:,-1],data.iloc[:,k]) for k in range(0,len(data.columns))], dtype=float)
    metrixR_coef.index=data.columns
    metrixR_coef.columns=['r_coef','p_val']

    metrixMeanBias=metrixMean.sub(metrixMean.iloc[-1], axis=0)
    metrixMSE=pd.DataFrame([mean_squared_error(data.iloc[:,-1],data.iloc[:,k]) for k in range(0,len(data.columns))], dtype=float)
    metrixMSE.index=data.columns
    metrixMSE.columns=['MSE']
    metrixRMSE=pd.DataFrame([sqrt(mean_squared_error(data.iloc[:,-1],data.iloc[:,k])) for k in range(0,len(data.columns))], dtype=float)
    metrixRMSE.index=data.columns
    metrixRMSE.columns=['RMSE']
    #MSE skill score or Nash-Sutcliffe efficiency in hydrology = 1- MSE/var_biased(obs)

    metrixSS1=1-metrixMSE/(data.var(ddof=0)[-1])
    metrixSS1.columns=['MSE_SS1']

    eval_matrix=pd.concat([pd.DataFrame(metrixMean,columns=['SpTmean']),pd.DataFrame(metrixVar,columns=['Var_b']),pd.DataFrame(metrixStd, columns=['STD_b']),
                           pd.DataFrame(metrixCV, columns=['Coef_variation']), pd.DataFrame(metrixStd_raport, columns=['Std_raport']),
                           pd.DataFrame(metrixVar_raport, columns=['Var_raport']),metrixR_coef,pd.DataFrame(metrixMeanBias, columns=['Mean_Bias']),
                            metrixMSE,metrixRMSE,metrixSS1],axis=1)

    eval_matrix.to_csv(output+varName+'_evalMatrix_'+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_spatial.csv', sep=',')
    np.savetxt(output+varName+'_evalMatrix_'+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_spatial.txt', eval_matrix.values, delimiter=',')

    ###################################### figures ##############################

    import seaborn as sns

    sns.set_style("ticks")
    color = dict(boxes='DarkGreen', whiskers='DarkGreen', medians='Black', caps='Black')

    fig0 = plt.figure(figsize=(15,20))
    ax1=sns.boxplot(data, palette=palette,orient='h')
    font_size = 18
    ax1.tick_params(labelsize=font_size)
    plt.xlabel(box_label, fontsize=20)
#    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outputBoxPlot+varName+'_boxPlt_'+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_spatial.png', dpi=1200)
    fig0.clf()


    fig2 = plt.figure(figsize=(15,20))
    ax=sns.violinplot(data, jitter=True, palette=palette, alpha=0.2, cut=0,  inner='quartile',orient='h')
    ax.tick_params(labelsize=font_size)
    plt.xlabel(box_label, fontsize=20)
#    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outputViolin+varName+'_violin_'+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_spatial.png', dpi=1200)
    fig2.clf()

    print('OK'+varName)



