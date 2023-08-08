__author__ = 'emiliapauladiaconescu'


import matplotlib  as mpl
from scipy import vstack, hstack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

domeniu='ARCcordexB'
periode='1980to2004'
ValnrYY='15'
statistique='spatial'


input= 'D:/NowWorking/newProjet/results/domARC_CORDEX_b/metricsSPATIAL_final_ref/'
input2='D:/NowWorking/newProjet/results/domARC_CORDEX_b/metricsSPATIAL_EM_final_ref/'
output='D:/NowWorking/newProjet/results/domARC_CORDEX_b/performaneD_ref_Final/'

ax_var=['tasAnualMean','HDD17', 'GDD5', 'SU15','TXx_K','TX90_K','FD','TNn_K','TN10_K', 'prAnualMean', 'pr1mm','rx1day','rx5days','PrCum_GTp95', 'PrCum_GTp99']
ax_oy=['Mean Tmean','HDD', 'GDD','SU15', 'TXx', 'TX90', 'FD','TNn', 'TN10', 'Mean Pr', 'R1mm','Rx1day','Rx5day','R95pTOT', 'R99pTOT']
ax_oy2=['EM reanalyses', 'EM simulations']

ax_ox=['NRCan', 'GMFD', 'CFSR', 'MERRA','JRA-55','ERAI', 'AWI-HIRHAM5-MPI-ESM-LR', 'CCCma-CanRCM4-CanESM2-022',
       'CCCma-CanRCM4-CanESM2', 'UQAM-CRCM5-MPI-ESM-LR', 'UQAM-CRCM5NA-MPI-ESM-LR ', 'UQAM-CRCM5NA-CanESM2',
       'SMHI-RCA4-CanESM2', 'SMHI-RCA4-NorESM1-M', 'SMHI-RCA4-EC-EARTH', 'SMHI-RCA4SN-EC-EARTH', 'SMHI-RCA4-MPI-ESM-LR',
       'SMHI-RCA4SN-MPI-ESM-LR']

arrVar = np.zeros(19)
arrR = np.zeros(19)
arrRV1 = np.zeros(19)
arrRMSE= np.zeros(19)
arrMSE= np.zeros(19)
arrMSESS1= np.zeros(19)
arrMSESS2= np.zeros(19)

bbVar = np.zeros(4)
bbR = np.zeros(4)
bbRV1 = np.zeros(4)
bbRMSE= np.zeros(4)
bbMSE= np.zeros(4)
bbMSESS1= np.zeros(4)
bbMSESS2= np.zeros(4)

for varName in ax_var:
       matricea=input+varName+'_evalMatrix_'+domeniu+'_'+ValnrYY+'noYY_'+periode+'_'+statistique+'.csv'
       df_mat = pd.read_csv(matricea, sep=',')
       Var_raport=np.array(df_mat['Var_raport'])
       r_coef=np.array(df_mat['r_coef'])
       MSE_RV1=np.array(df_mat['MSE_SS1'])
       RMSE=np.array(df_mat['RMSE'])
       MSE=np.array(df_mat['MSE'])
       MSESS2=1- MSE/np.mean(MSE[1:6])


       arrVar = vstack((arrVar, Var_raport))
       arrR = vstack((arrR, r_coef))
       arrRV1 = vstack((arrRV1, MSE_RV1))
       arrRMSE = vstack((arrRMSE, RMSE))
       arrMSE = vstack((arrMSE, MSE))
       arrMSESS2=vstack((arrMSESS2, MSESS2))

       matriceaB=input2+varName+'_EM_evalMatrix_'+domeniu+'_'+ValnrYY+'noYY_'+periode+'_'+statistique+'.csv'
       df_matB = pd.read_csv(matriceaB, sep=',')
       Var_raport_B=np.array(df_matB['Var_raport'])
       r_coef_B=np.array(df_matB['r_coef'])
       MSE_RV1_B=np.array(df_matB['MSE_SS1'])
       RMSE_B=np.array(df_matB['RMSE'])
       MSE_B=np.array(df_matB['MSE'])
       MSESS2_B=1- MSE_B/np.mean(MSE[1:6])
       MSESS1=1- MSE/MSE_B[2]
       MSESS1_B=1- MSE_B/MSE_B[2]


       bbVar = vstack((bbVar, Var_raport_B))
       bbR = vstack((bbR, r_coef_B))
       bbRV1 = vstack((bbRV1, MSE_RV1_B))
       bbRMSE = vstack((bbRMSE, RMSE_B))
       bbMSE = vstack((bbMSE, MSE_B))
       bbMSESS1=vstack((bbMSESS1, MSESS1_B))
       arrMSESS1=vstack((arrMSESS1, MSESS1))
       bbMSESS2=vstack((bbMSESS2, MSESS2_B))

R_Var=arrVar[1:,:-1]
R_R=arrR[1:,:-1]
R_RV1=arrRV1[1:,:-1]
R_RMSE=arrRMSE[1:,:-1]
R_MSE=arrMSE[1:,:-1]
R_MSESS1=arrMSESS1[1:,:-1]
R_MSESS2=arrMSESS2[1:,:-1]

######################################

Rb_Var=bbVar[1:,2:]
Rb_R=bbR[1:,2:]
Rb_RV1=bbRV1[1:,2:]
Rb_RMSE=bbRMSE[1:,2:]
Rb_MSE=bbMSE[1:,2:]
Rb_MSESS1=bbMSESS1[1:,2:]
Rb_MSESS2=bbMSESS2[1:,2:]
#######################################################

ax_oySS=['All T', 'All Pr']

R_VarT=(R_Var[:9,:]).mean(axis=0)
R_VarPr=(R_Var[9:,:]).mean(axis=0)
R_VarSS=vstack((R_VarT,R_VarPr))

R_RT=(R_R[:9,:]).mean(axis=0)
R_RPr=(R_R[9:,:]).mean(axis=0)
R_RSS=vstack((R_RT,R_RPr))

R_RV1T=(R_RV1[:9,:]).mean(axis=0)
R_RV1Pr=(R_RV1[9:,:]).mean(axis=0)
R_RV1SS=vstack((R_RV1T,R_RV1Pr))

R_MSESS1T=(R_MSESS1[:9,:]).mean(axis=0)
R_MSESS1Pr=(R_MSESS1[9:,:]).mean(axis=0)
R_MSESS1SS=vstack((R_MSESS1T,R_MSESS1Pr))

R_MSESS2T=(R_MSESS2[:9,:]).mean(axis=0)
R_MSESS2Pr=(R_MSESS2[9:,:]).mean(axis=0)
R_MSESS2SS=vstack((R_MSESS2T,R_MSESS2Pr))
#########################################################

cgbvr1 = mpl.colors.ListedColormap (['Navy','Blue','DodgerBlue', 'LightSkyBlue', 'lightcyan', 'Ivory','Bisque',  'lightsalmon', 'tomato','red',  'FireBrick', 'DarkRed','DarkRed'])

cgbvr2 = mpl.colors.ListedColormap ([(0.2783814132520015, 0.0, 0.0),
 (0.51516282650400291, 0.0, 0.0),
 (0.75194423975600444, 0.0, 0.0),
 (0.99902049706244078, 0.0, 0.0),
 (1.0, 0.23578488890837906, 0.0),
 (1.0, 0.47254921004871014, 0.0),
 (1.0, 0.70931353118904128, 0.0),
 (1.0, 1.0, 0.2897051720581133),
 (1.0, 1.0, 0.64485258602905671),'Ivory'])

cgbvr3 = mpl.colors.ListedColormap ([(0.75140332123812503, 0.095886198666823255, 0.47366397696382861),
                                     (0.75140332123812503, 0.095886198666823255, 0.47366397696382861),
                                     (0.85136486852870263, 0.39592464618823109, 0.64467514028736184),
                                     (0.85136486852870263, 0.39592464618823109, 0.64467514028736184),
                                     (0.92318339558208695, 0.64106114296352157, 0.80415226080838376),
                                     (0.92318339558208695, 0.64106114296352157, 0.80415226080838376),
                                     (0.97554786766276635, 0.8202998953707078, 0.90818916348850021),
                                     (0.97554786766276635, 0.8202998953707078, 0.90818916348850021),
                                     (0.97923875556272622, 0.92795079245286827, 0.95447904923382931),
                                     (0.97923875556272622, 0.92795079245286827, 0.95447904923382931),
                                     (0.93856209516525269, 0.96509035194621362, 0.8996539852198433),
                                     (0.93856209516525269, 0.96509035194621362, 0.8996539852198433),
                                     (0.83829297388301172, 0.93310265681322879, 0.71326414627187384),
                                     (0.83829297388301172, 0.93310265681322879, 0.71326414627187384),
                                     (0.65582469456336079, 0.83967705684549665, 0.44590544525314779),
                                     (0.65582469456336079, 0.83967705684549665, 0.44590544525314779),
                                     (0.45959246684523203, 0.70495964849696446, 0.23029605313843377),
                                     (0.45959246684523203, 0.70495964849696446, 0.23029605313843377),
                                     (0.28735103005287699, 0.55486353764347007, 0.12633603124641907),
                                     (0.28735103005287699, 0.40486353764347007, 0.12633603124641907)])


cgbvr4 = mpl.colors.ListedColormap ([(0.3545098 ,  0.05333333,  0.3545098),
                                     (0.42823529,  0.02666667,  0.42823529),
                                     (0.50196078,  0.        ,  0.50196078),
                                     (0.75140332123812503, 0.095886198666823255, 0.47366397696382861),
                                     (0.85136486852870263, 0.39592464618823109, 0.64467514028736184),
                                     (0.92318339558208695, 0.64106114296352157, 0.80415226080838376),
                                     (0.97554786766276635, 0.8202998953707078, 0.90818916348850021),
                                     (0.97923875556272622, 0.92795079245286827, 0.95447904923382931),
                                     (0.83829297388301172, 0.93310265681322879, 0.71326414627187384),
                                     (0.45959246684523203, 0.70495964849696446, 0.23029605313843377)])

cgbvr5 = mpl.colors.ListedColormap ([(0.45882352941176485, 0.72941176470588243, 0.81960784313725488),
                                     (0.18823529411764683, 0.59411764705882342, 0.72941176470588232),
                                     (0.0, 0.31764705882352939, 0.54509803921568623),
                                     (0.0, 0.18235294117647061, 0.45490196078431372),
                                     (0.0, 0.15294117647058825, 0.23137254901960785),
                                     (0.0, 0.22941176470588237, 0.1803921568627451),
                                     (0.28735103005287699, 0.55486353764347007, 0.12633603124641907),
                                     (0.65582469456336079, 0.83967705684549665, 0.44590544525314779),
                                     (0.83829297388301172, 0.93310265681322879, 0.71326414627187384),
                                     (0.90272972513647642, 0.96069204386542828, 0.7874048569623161)])

# ###################################VarPerformance###########################
fig, ax = plt.subplots(figsize=(17.2,4), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 2])
ncolors=13
increment=0.2
vmin=0
vmax=vmin+increment*ncolors
plt.pcolor(Rb_Var.T, cmap=cgbvr1, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,2)+0.5,ax_oy2, size=24)
plt.xticks(np.arange(0,15)+0.5,[' ', ' '], ha='right',size=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig(output+'VarPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_C.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(20,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=13
increment=0.2
vmin=0
vmax=vmin+increment*ncolors
plt.pcolor(R_Var.T, cmap=cgbvr1, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
#aaa=plt.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 ], shrink=.7)
#aaa.ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 ], size=24)
plt.savefig(output+'VarPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(22,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=13
increment=0.2
vmin=0
vmax=vmin+increment*ncolors
plt.pcolor(R_Var.T, cmap=cgbvr1, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
aaa=plt.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 ], shrink=.7)
aaa.ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 ], size=24)
plt.savefig(output+'VarPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A1.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(4,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 2, 0, 24])
ncolors=13
increment=0.2
vmin=0
vmax=vmin+increment*ncolors
plt.pcolor(R_VarSS.T, cmap=cgbvr1, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,[' ', ' '], size=1)
plt.xticks(np.arange(0,2)+0.5,ax_oySS, rotation=30,ha='right',size=24)
cbar_ax = fig.add_axes([0.85, 0.15, 0.15, 0.7])
aaa=plt.colorbar(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 ], cax=cbar_ax)
aaa.ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2 ], size=24)
plt.savefig(output+'VarPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_B.png', bbox_inches='tight')

# # ############################R_Coef####################################
fig, ax = plt.subplots(figsize=(17.2,4), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 2])
ncolors=10
increment=0.1
vmin=0
vmax=vmin+increment*ncolors
plt.pcolor(Rb_R.T, cmap=cgbvr2, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,2)+0.5,ax_oy2, size=24)
plt.xticks(np.arange(0,15)+0.5,[' ', ' '], ha='right',size=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig(output+'RcoefPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_C.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(20,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=10
increment=0.1
vmin=0
vmax=vmin+increment*ncolors
plt.pcolor(R_R.T, cmap=cgbvr2, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
#plt.colorbar(ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], shrink=.7)
plt.savefig(output+'RcoefPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(22,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=10
increment=0.1
vmin=0
vmax=vmin+increment*ncolors
plt.pcolor(R_R.T, cmap=cgbvr2, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
aaa=plt.colorbar(ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], shrink=.7)
aaa.ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], size=24)
plt.savefig(output+'RcoefPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A1.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(4,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 2, 0, 18])
ncolors=10
increment=0.1
vmin=0
vmax=vmin+increment*ncolors
plt.pcolor(R_RSS.T, cmap=cgbvr2, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,[' ', ' '], size=1)
plt.xticks(np.arange(0,2)+0.5,ax_oySS, rotation=30,ha='right',size=24)
cbar_ax = fig.add_axes([0.85, 0.15, 0.15, 0.7])
aaa=plt.colorbar(ticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], cax=cbar_ax)
aaa.ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], size=24)
plt.savefig(output+'RcoefPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_B.png', bbox_inches='tight', pad_inches=0.05)


# ##########################RV1sp################################################
fig, ax = plt.subplots(figsize=(17.2,4), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 2])
ncolors=20
increment=0.1
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(Rb_RV1.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,2)+0.5,ax_oy2, size=24)
plt.xticks(np.arange(0,15)+0.5,[' ', ' '], ha='right',size=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig(output+'RV1spPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_C.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(20,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=20
increment=0.1
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_RV1.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
#plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], shrink=.7)
plt.savefig(output+'RV1spPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(22,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=20
increment=0.1
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_RV1.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
aaa=plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], shrink=.7)
aaa.ax.set_yticklabels([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], size=24)
plt.savefig(output+'RV1spPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A1.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(4,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 2, 0, 18])
ncolors=20
increment=0.1
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_RV1SS.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,[' ', ' '], size=1)
plt.xticks(np.arange(0,2)+0.5,ax_oySS, rotation=30,ha='right',size=24)
cbar_ax = fig.add_axes([0.85, 0.15, 0.15, 0.7])
aaa=plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], cax=cbar_ax)
aaa.ax.set_yticklabels([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0], size=24)
plt.savefig(output+'RV1spPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_B.png', bbox_inches='tight', pad_inches=0.05)


# ############################MSSSreanalyses###############################################
fig, ax = plt.subplots(figsize=(17.2,4), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 2])
ncolors=10
increment=0.2
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(Rb_MSESS2.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,2)+0.5,ax_oy2, size=24)
plt.xticks(np.arange(0,15)+0.5,[' ', ' '], ha='right',size=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig(output+'MSESSreanalPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_C.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(20,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=10
increment=0.2
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_MSESS2.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
#plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], shrink=.7)
plt.savefig(output+'MSESSreanalPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(22,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=10
increment=0.2
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_MSESS2.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
aaa=plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], shrink=.7)
aaa.ax.set_yticklabels([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=24)
plt.savefig(output+'MSESSreanalPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A1.png', bbox_inches='tight', pad_inches=0.05)


fig, ax = plt.subplots(figsize=(4,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 2, 0, 18])
ncolors=10
increment=0.2
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_MSESS2SS.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,[' ', ' '], size=1)
plt.xticks(np.arange(0,2)+0.5,ax_oySS, rotation=30,ha='right',size=24)
cbar_ax = fig.add_axes([0.85, 0.15, 0.15, 0.7])
aaa=plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], cax=cbar_ax)
aaa.ax.set_yticklabels([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=24)
plt.savefig(output+'MSESSreanalPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_B.png', bbox_inches='tight', pad_inches=0.05)

# ############################MSSS_EMreanalyses###############################################
fig, ax = plt.subplots(figsize=(17.2,4), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 2])
ncolors=10
increment=0.2
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(Rb_MSESS1.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,2)+0.5,ax_oy2, size=24)
plt.xticks(np.arange(0,15)+0.5,[' ', ' '], ha='right',size=1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.savefig(output+'MSESS_EMreanalPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_C.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(20,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=10
increment=0.2
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_MSESS1.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
#plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], shrink=.7)
plt.savefig(output+'MSESS_EMreanalPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A.png', bbox_inches='tight', pad_inches=0.05)

fig, ax = plt.subplots(figsize=(22,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 15, 0, 18])
ncolors=10
increment=0.2
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_MSESS1.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,ax_ox, size=24)
plt.xticks(np.arange(0,15)+0.5,ax_oy, rotation=30, ha='right',size=24)
aaa=plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], shrink=.7)
aaa.ax.set_yticklabels([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=24)
plt.savefig(output+'MSESS_EMreanalPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_A1.png', bbox_inches='tight', pad_inches=0.05)


fig, ax = plt.subplots(figsize=(4,20), subplot_kw={'aspect': 1.0})
plt.axis([0, 2, 0, 18])
ncolors=10
increment=0.2
vmin=-1
vmax=vmin+increment*ncolors
plt.pcolor(R_MSESS1SS.T, cmap=cgbvr3, edgecolors='k', vmin=vmin, vmax=vmax, linewidths=2)
plt.yticks(np.arange(0,18)+0.5,[' ', ' '], size=1)
plt.xticks(np.arange(0,2)+0.5,ax_oySS, rotation=30,ha='right',size=24)
cbar_ax = fig.add_axes([0.85, 0.15, 0.15, 0.7])
aaa=plt.colorbar(ticks=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], cax=cbar_ax)
aaa.ax.set_yticklabels([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=24)
plt.savefig(output+'MSESS_EMreanalPerformanceD_ALL_'+domeniu+'_'+periode+'_'+statistique+'_B.png', bbox_inches='tight', pad_inches=0.05)
