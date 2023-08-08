import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;
import statsmodels.api as sm
from numpy import sort, amax, arange, exp, sqrt, abs, floor, searchsorted
from scipy.misc import factorial, comb

sns.set()
sns.set(color_codes=True)
sns.set_style("white")
sns.set_style("ticks")
#sns.set_context("poster")
sns.set_context("talk")

def kuiper_FPP(D,N):
    """Compute the false positive probability for the Kuiper statistic.

    Uses the set of four formulas described in Paltani 2004; they report
    the resulting function never underestimates the false positive probability
    but can be a bit high in the N=40..50 range. (They quote a factor 1.5 at
    the 1e-7 level.

    Parameters
    ----------
    D : float
        The Kuiper test score.
    N : float
        The effective sample size.

    Returns
    -------
    fpp : float
        The probability of a score this large arising from the null hypothesis.

    Reference
    ---------
    Paltani, S., "Searching for periods in X-ray observations using
    Kuiper's test. Application to the ROSAT PSPC archive", Astronomy and
    Astrophysics, v.240, p.789-790, 2004.

    """
    if D<0. or D>2.:
        raise ValueError("Must have 0<=D<=2 by definition of the Kuiper test")

    if D<2./N:
        return 1. - factorial(N)*(D-1./N)**(N-1)
    elif D<3./N:
        k = -(N*D-1.)/2.
        r = sqrt(k**2 - (N*D-2.)/2.)
        a, b = -k+r, -k-r
        return 1. - factorial(N-1)*(b**(N-1.)*(1.-a)-a**(N-1.)*(1.-b))/float(N)**(N-2)*(b-a)
    elif (D>0.5 and N%2==0) or (D>(N-1.)/(2.*N) and N%2==1):
        def T(t):
            y = D+t/float(N)
            return y**(t-3)*(y**3*N-y**2*t*(3.-2./N)/N-t*(t-1)*(t-2)/float(N)**2)
        s = 0.
        # NOTE: the upper limit of this sum is taken from Stephens 1965
        for t in xrange(int(floor(N*(1-D)))+1):
            term = T(t)*comb(N,t)*(1-D-t/float(N))**(N-t-1)
            s += term
        return s
    else:
        z = D*sqrt(N)
        S1 = 0.
        term_eps = 1e-12
        abs_eps = 1e-100
        for m in itertools.count(1):
            T1 = 2.*(4.*m**2*z**2-1.)*exp(-2.*m**2*z**2)
            so = S1
            S1 += T1
            if abs(S1-so)/(abs(S1)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        S2 = 0.
        for m in itertools.count(1):
            T2 = m**2*(4.*m**2*z**2-3.)*exp(-2*m**2*z**2)
            so = S2
            S2 += T2
            if abs(S2-so)/(abs(S2)+abs(so))<term_eps or abs(S1-so)<abs_eps:
                break
        return S1 - 8*D/(3.*sqrt(N))*S2

def kuiper(data, cdf=lambda x: x, args=()):
    """Compute the Kuiper statistic.

    Use the Kuiper statistic version of the Kolmogorov-Smirnov test to
    find the probability that something like data was drawn from the
    distribution whose CDF is given as cdf.

    Parameters
    ----------
    data : array-like
        The data values.
    cdf : callable
        A callable to evaluate the CDF of the distribution being tested
        against. Will be called with a vector of all values at once.
    args : list-like, optional
        Additional arguments to be supplied to cdf.

    Returns
    -------
    D : float
        The raw statistic.
    fpp : float
        The probability of a D this large arising with a sample drawn from
        the distribution whose CDF is cdf.

    Notes
    -----
    The Kuiper statistic resembles the Kolmogorov-Smirnov test in that
    it is nonparametric and invariant under reparameterizations of the data.
    The Kuiper statistic, in addition, is equally sensitive throughout
    the domain, and it is also invariant under cyclic permutations (making
    it particularly appropriate for analyzing circular data).

    Returns (D, fpp), where D is the Kuiper D number and fpp is the
    probability that a value as large as D would occur if data was
    drawn from cdf.

    Warning: The fpp is calculated only approximately, and it can be
    as much as 1.5 times the true value.

    Stephens 1970 claims this is more effective than the KS at detecting
    changes in the variance of a distribution; the KS is (he claims) more
    sensitive at detecting changes in the mean.

    If cdf was obtained from data by fitting, then fpp is not correct and
    it will be necessary to do Monte Carlo simulations to interpret D.
    D should normally be independent of the shape of CDF.

    """

    # FIXME: doesn't work for distributions that are actually discrete (for example Poisson).
    data = sort(data)
    cdfv = cdf(data,*args)
    N = len(data)
    D = amax(cdfv-arange(N)/float(N)) + amax((arange(N)+1)/float(N)-cdfv)

    return D, kuiper_FPP(D,N)

def kuiper_two(data1, data2):
    """Compute the Kuiper statistic to compare two samples.

    Parameters
    ----------
    data1 : array-like
        The first set of data values.
    data2 : array-like
        The second set of data values.

    Returns
    -------
    D : float
        The raw test statistic.
    fpp : float
        The probability of obtaining two samples this different from
        the same distribution.

    Notes
    -----
    Warning: the fpp is quite approximate, especially for small samples.

    """
    data1, data2 = sort(data1), sort(data2)

    if len(data2)<len(data1):
        data1, data2 = data2, data1

    cdfv1 = searchsorted(data2, data1)/float(len(data2)) # this could be more efficient
    cdfv2 = searchsorted(data1, data2)/float(len(data1)) # this could be more efficient
    D = (amax(cdfv1-arange(len(data1))/float(len(data1))) +
            amax(cdfv2-arange(len(data2))/float(len(data2))))

    Ne = len(data1)*len(data2)/float(len(data1)+len(data2))
    return D, kuiper_FPP(D, Ne)


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

#################################################
domeniu='ARCcordex'

list_varName=['tas']
periode='1980to2004'
ValnrYY='15'
fyy='1980'
lyy='2004'

input='D:/NowWorking/newProjet/results/domARC_CORDEX_b/Kuiper2_ref/'
output='D:/NowWorking/newProjet/results/domARC_CORDEX_b/clustering_Anom/'
outputAnom='D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_Anom_ref/'
outputMean= 'D:/NowWorking/newProjet/results/domARC_CORDEX_b/stations_climMean_ref/'

for varName in list_varName:

    if varName=='pr':
        input_obs= 'D:/NowWorking/newProjet/data/tabG_stations/anual/precipitation_niv0.8_tabG_anualMean.csv'
        input_coord='D:/NowWorking/newProjet/results/coord_dom_ARC_b/coord_select_ARC_tabG_'+ValnrYY+'noYY_'+periode+'.csv'
        box_label='pr [mm/day]'
        increment=0.2


    else:
        input_obs= 'D:/NowWorking/newProjet/data/ahccd_stations/anual/'+varName+'_niv0.8_ahccd_anualMean.csv'
        input_coord='D:/NowWorking/newProjet/results/coord_dom_ARC_b/coord_select_ARC_ahccd_'+ValnrYY+'noYY_'+varName+'_'+periode+'.csv'
        box_label=str(varName)+'[$^\circ$K]'
        increment=0.5

    ###################################################################
    df_coord=pd.read_csv(input_coord, sep=',')
#    df_coord=df_coord0.drop('1206197', axis=1)
    ec_id=df_coord.columns[1:]
    lonsDF =df_coord.iloc[-2,1:]
    latsDF =df_coord.iloc[-1,1:]
    points_lon=np.array(lonsDF, dtype=float)
    points_lat = np.array(latsDF, dtype=float)
    points_Obs=np.append([points_lon], [points_lat], axis=0)

    #selection observations
    df = pd.read_csv(input_obs, sep=',')
    (nrYY, stations) = df.shape
    obs=df.iloc[:,1:]
    yy = df.iloc[:,0]
    obs.index=yy

    obs_dom=obs[ec_id]
    data_obs=obs_dom[fyy:lyy]
    print ('OK select')

    #we choose only the first 15 years without NaN for each station
    bbb=data_obs.convert_objects(convert_numeric=True)
    aaa=bbb.describe()
    ccc=(aaa.transpose())['mean']
    yyyy=(aaa.transpose())['count']
    newFrame0=pd.DataFrame(data_obs.transpose(), dtype=float)
    existing_df0 = newFrame0.sub(ccc.values, axis=0)
    anomalies=existing_df0.transpose()
    anomalies.to_csv(outputAnom+varName+'AnualMean_StationsAnomalies_'+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_.csv', sep=',')
    climMean=(pd.concat({'climMean':ccc,'lon':lonsDF,'lat':latsDF}, axis=1)).transpose()
    climMean.to_csv(outputMean+varName+'AnualMean_StationsAnomalies_'+domeniu+'_'+str(ValnrYY)+'noYY_'+periode+'_.csv', sep=',')

    i=0
    station=ec_id[i]
    ttest=(existing_df0.loc[station]).dropna()
    ttest2=ttest.iloc[-yyyy.min():]
    existing_df=pd.DataFrame({station: np.array(ttest2, dtype=float)})
    for i in range(1, len(ec_id)):
        station=ec_id[i]
        ttest=(existing_df0.loc[station]).dropna()
        ttest2=ttest.iloc[-yyyy.min():]
        xxxx=pd.DataFrame({station: np.array(ttest2, dtype=float)})
        existing_df=pd.concat([existing_df,xxxx],axis=1)
    existing_df=existing_df.transpose()


    # plot the data as heatmap
    yticks = existing_df.index
    keptticks = yticks[::int(len(yticks)/20)]
    yticks = ['' for y in yticks]
    yticks[::int(len(yticks)/20)] = keptticks

    xticks = existing_df.columns
    sns.heatmap(existing_df, cmap="seismic",yticklabels=yticks,xticklabels=xticks)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(output+varName+'_anomalyHeatmap.png')
    plt.clf()

###########################################################################
X=np.array(existing_df)
from scipy.cluster.hierarchy import dendrogram, linkage
# generate the linkage matrix
Z = linkage(X, 'ward')

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
print c

# calculate troncated dendrogram
fig00= plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.savefig(output+varName+'_dendrogram.png')
fig00.clf()

#get the cluster id Knowing k
#Elbow Method to find automaticly the number of clusters:
last = Z[:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)

fig0= plt.figure(figsize=(12,8))
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]

k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print "clusters:", k
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.title('k='+str(k))
plt.savefig(output+varName+'_nrClusters.png')
fig0.clf()
#######################COMUTE CLUSTER using Kmeans

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(existing_df)
PCA(copy=True, n_components=2, whiten=False)
existing_2d = pca.transform(existing_df)
existing_df_2d = pd.DataFrame(existing_2d)
existing_df_2d.index = existing_df.index
existing_df_2d.columns = ['PC1','PC2']
existing_df_2d.head()
print(pca.explained_variance_ratio_)
totalVar=(pca.explained_variance_ratio_).sum()
########################
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit(existing_df)
existing_df_2d['cluster'] = pd.Series(clusters.labels_, index=existing_df_2d.index)
existing_df_2d['lats'] = pd.Series(points_lat, index=existing_df_2d.index)
existing_df_2d['lons'] = pd.Series(points_lon, index=existing_df_2d.index)
f=existing_df_2d.cluster.astype(np.float)
f[existing_df_2d['cluster']==0]='r'
f[existing_df_2d['cluster']==1]='b'
f[existing_df_2d['cluster']==2]='g'
f[existing_df_2d['cluster']==3]='c'
f[existing_df_2d['cluster']==4]='m'
f[existing_df_2d['cluster']==5]='y'

existing_df_2d.plot(
        kind='scatter',
        x='lons',y='lats', s=300,
        c=f,
        figsize=(16,8))
plt.savefig(output+varName+'_KMeansPoints.png')


aa1=(existing_df_2d[existing_df_2d['cluster']==0]).index
aa2=(existing_df_2d[existing_df_2d['cluster']==1]).index

# aa5=(existing_df_2d[existing_df_2d['cluster']==4]).index
# aa6=(existing_df_2d[existing_df_2d['cluster']==5]).index
# aa7=(existing_df_2d[existing_df_2d['cluster']==6]).index
# aa8=(existing_df_2d[existing_df_2d['cluster']==7]).index

bb1=(existing_df.transpose()) [aa1]
bb2=(existing_df.transpose()) [aa2]

#compute kuiper between clusters

cc1=np.array(bb1)
cc2=np.array(bb2)
cc0=np.array(existing_df.transpose())

dd0=cc0.flatten()
dd1=cc1.flatten()
dd2=cc2.flatten()

ecdfX = sm.distributions.ECDF(dd1)
ecdfY = sm.distributions.ECDF(dd2)
ecdfZ = sm.distributions.ECDF(dd0)

D1, p1 = kuiper_two(dd1, dd0)
D2, p2 = kuiper_two(dd2, dd0)

kn1=len(aa1)
kn2=len(aa2)
fig1 = plt.figure(figsize=(12,8))
ax = fig1.add_subplot(111)
ax.plot( ecdfX.x, ecdfX.y, color='blue',label='Cluster 1 ('+str(kn1)+' stations); D='+str(D1)[:4])
ax.plot( ecdfY.x, ecdfY.y, color='green', label='Cluster 2 ('+str(kn2)+' stations); D='+str(D2)[:4])
ax.plot( ecdfZ.x, ecdfZ.y, color='red', label='All data')
plt.legend(loc=4, fontsize=14)
plt.savefig(output+varName+'_ecdf.png')
fig1.clf()

increment=0.2
valmin=round(min(dd0))
valmax=round(max(dd0))
binsVal=int((valmax-valmin)/increment)
valmax=valmin+increment*binsVal
dist_space = np.linspace( valmin, valmax,binsVal+1)

histX, binsX = np.histogram(dd1,bins=binsVal,range=(dist_space[0],dist_space[-1]) ,density=True)
widthX = 0.95 * (binsX[1] - binsX[0])
centerX = (binsX[:-1] + binsX[1:]) / 2

histY, binsY = np.histogram(dd2,bins=binsVal,range=(dist_space[0],dist_space[-1]) ,density=True)
widthY = 0.95 * (binsY[1] - binsY[0])
centerY = (binsY[:-1] + binsY[1:]) / 2

histZ, binsZ = np.histogram(dd0,bins=binsVal,range=(dist_space[0],dist_space[-1]) ,density=True)
widthZ = 0.95 * (binsZ[1] - binsZ[0])
centerZ = (binsZ[:-1] + binsZ[1:]) / 2

testM1=np.min(np.vstack((histX,histZ)), axis=0)
areaMin1 = np.sum(testM1)/np.sum(histZ)
Pkval1=areaMin1
testM2=np.min(np.vstack((histY,histZ)), axis=0)
areaMin2 = np.sum(testM2)/np.sum(histZ)
Pkval2=areaMin2

fig2 = plt.figure(figsize=(12,8))
ax = fig2.add_subplot(111)
ax.bar(centerX, histX, align='center', width=2*widthX/3, fc='blue',alpha=0.7,label="Cluster 1; P="+str(Pkval1)[:4])
ax.bar(centerY, histY, align='center', width=widthY/3, fc='green',alpha=0.7, label="Cluster 2; P="+str(Pkval2)[:4])
ax.step(centerZ, histZ, 'r' , where='mid',label='All data')
plt.legend(loc=1, fontsize=14)
plt.savefig(output+varName+'_histo.png')
fig2.clf()
