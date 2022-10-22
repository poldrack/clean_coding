# NB: EXAMPLE OF WORST PRACTICES!
# load survey data, perform factor analysis, and compare to mental health data

from pandas import *
from numpy import *
from scipy.stats import *
maxD=12
hc=['Nervous', 'Hopeless', 'RestlessFidgety', 'Depressed', 'EverythingIsEffort', 'Worthless', ]
h=read_csv('https://raw.githubusercontent.com/poldrack/clean_coding/master/data/health.csv',index_col=0)[hc].dropna().mean(1)
data=read_csv('https://raw.githubusercontent.com/poldrack/clean_coding/master/data/meaningful_variables_clean.csv',index_col=0)
sc=[]
for i in range(data.shape[1]):
    if data.columns[i].split('.')[0][-7:] == '_survey':
        sc=sc+[data.columns[i]]
data=data[sc]
gs=[]
for i in range(data.shape[0]):
    if sum(isnan(data.iloc[i, :])) > 0:
        pass
    else:
        gs=gs+[i]
data=data.iloc[gs,:]
from sklearn.preprocessing import scale
data_sc = scale(data)
from sklearn.decomposition import FactorAnalysis
bicv=zeros(maxD)
for i in range(1,maxD+1):
    fa=FactorAnalysis(i)
    fa.fit(data_sc)
    bicv[i-1]=i*2 - 2*fa.score(data_sc)
npD=argmin(bicv)+1
fa=FactorAnalysis(npD)
f=fa.fit_transform(data_sc)
for i in range(npD):
    print(pearsonr(f[:,i],h[gs]))
    idx=argsort(abs(fa.components_[i, :]))[::-1]
    for j in range(3):
        print(data.columns[idx[j]], fa.components_[i, idx[j]])