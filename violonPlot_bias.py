

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

#################################################
domeniu='ARCcordexB'

list_varName=['FD', 'GDD5', 'HDD17', 'pr1mm', 'prAnualMean', 'PrCum_GTp95', 'PrCum_GTp99', 'PrTotPc_GTp95',
               'PrTotPc_GTp99', 'rx1day', 'rx5days', 'SU15', 'tasAnualMean', 'TN10_K', 'TN10p', 'TNn_K',
               'TX90_K', 'TX90p', 'TXx_K']

#list_varName=['HDD17']
periode='1980to2004'
ValnrYY='15'
fyy='1980'
lyy='2004'
input= 'D:/NowWorking/newProjet/results/domARC_CORDEX_b/simSelected_climMean_ref/'

outputViolin='D:/NowWorking/newProjet/results/domARC_CORDEX_b/violinPlotSPATIAL_bias_ref/'

palette=[(0.6, 0.6, 0.6),(0.6,1,0.6),(1,0.6, 1)]

for varName in list_varName:


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

    data0=df.iloc[:,-3]
    data1=df.iloc[:,1]
    dataR=df.iloc[:,2:7]
    dataS=df.iloc[:,7:-3]
    data2=dataR.mean(axis=1)
    data3=dataS.mean(axis=1)
    data=pd.concat([data0, data1], axis=1)
    data['REM']=data2
    data['MEM']=data3


    dataset=data.columns

    biasData=data.sub(df.iloc[:,-3],axis=0)
    biasDataF=biasData.iloc[:,1:]
    ObsMean=data0.mean()
    ObsMax=data0.max()
    ObsMin=data0.min()
    ObsSTD=data0.std()
    if varName=='prAnualMean':
        box_label='Mean Pr bias [mm/day]\n '
        box_unit='mm/day'
    elif varName=='tasAnualMean':
        box_label='Mean Tmean bias [$^\circ$C]\n '
        box_unit='$^\circ$C'
        ObsMean=ObsMean-273.15
    elif varName=='pr1mm':
        box_label='R1mm bias [days]\n '
        box_unit='days'
    elif varName=='PrTotPc_GTp99':
        box_label='R99p bias [%]\n '
        box_unit='%'
    elif varName=='PrTotPc_GTp95':
        box_label='R95p bias [%]\n '
        box_unit='%'
    elif varName=='rx5days':
        box_label='RX5day bias [mm/day]\n '
        box_unit='mm/day'
    elif varName=='rx1day':
        box_label='RX1day bias [mm/day]\n '
        box_unit='mm/day'
    elif varName=='PrCum_GTp99':
        box_label='R99pTOT bias [mm]\n '
        box_unit='mm'
    elif varName=='PrCum_GTp95':
        box_label='R95pTOT bias [mm]\n '
        box_unit='mm'
    elif varName=='FD':
        box_label='FD bias [days]\n '
        box_unit='days'
    elif varName=='HDD17':
        box_label='HDD bias [$^\circ$C]\n '
        box_unit='$^\circ$C'
    elif varName=='GDD5':
        box_label='GDD bias [$^\circ$C]\n '
        box_unit='$^\circ$C'
    elif varName=='SU15':
        box_label='SU15 bias [days]\n '
        box_unit='days'
    elif varName=='TX90p':
        box_label='TX90p bias [%]\n '
        box_unit='%'
    elif varName=='TN10p':
        box_label='TN10p bias [%]\n '
        box_unit='%'
    elif varName=='TXx_K':
        box_label='TXx bias [$^\circ$C]\n '
        box_unit='$^\circ$C'
        ObsMean=ObsMean-273.15
    elif varName=='TNn_K':
        box_label='TNn bias [$^\circ$C]\n '
        box_unit='$^\circ$C'
        ObsMean=ObsMean-273.15
    elif varName=='TN10_K':
        box_label='TN10 bias [$^\circ$C]\n '
        box_unit='$^\circ$C'
        ObsMean=ObsMean-273.15
    elif varName=='TX90_K':
        box_label='TX90 bias [$^\circ$C]\n '
        box_unit='$^\circ$C'
        ObsMean=ObsMean-273.15

    ###################################### figures ##############################

    import seaborn as sns
    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2})

    fig1 = plt.figure(figsize=(8,16))
    ax=sns.violinplot(biasDataF,scale="count", jitter=True, palette=palette, alpha=0.2, cut=0,  inner='quartile',orient='v')
    ax.tick_params(labelsize=28)
    box_title=box_label
    plt.title(box_title, fontsize=28)
    ax.text(0.05, 0.97, 'Mean Obs. = '+str('%.2f'%ObsMean)+' '+box_unit,
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=24)
    plt.savefig(outputViolin+varName+'_EM_violin_'+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_BiasSpatial1.png')
    fig1.clf()

    fig2 = plt.figure(figsize=(8,16))
    ax=sns.violinplot(biasDataF,scale="count", jitter=True, palette=palette, alpha=0.2, cut=0,  inner="stick",orient='v')
    ax.tick_params(labelsize=28)
    box_title=box_label
    plt.title(box_title, fontsize=28)
    ax.text(0.05, 0.97, 'Mean Obs. = '+str('%.2f'%ObsMean)+' '+box_unit,
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='black', fontsize=24)
    plt.savefig(outputViolin+varName+'_EM_violin_'+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_BiasSpatial2.png')
    fig2.clf()

    print('OK'+varName)



