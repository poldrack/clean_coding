# NB: EXAMPLE OF WORST PRACTICES!
# load survey data, perform factor analysis, and compare to mental health data

# OVERALL:
# no vertical white space to separate logical sections
# unclear what the rationale is in many sections
# indecipherable variable names
# lack of PEP8 style compliance (e.g. spaces around operators)

# should not use wildcard imports ("import *")
# https://docs.quantifiedcode.com/python-anti-patterns/maintainability/from_module_import_all_used.html?highlight=import
from pandas import *
from numpy import *
from scipy.stats import *


maxD=12  # variable defined well before it is used
hc=['Nervous', 'Hopeless', 'RestlessFidgety', 'Depressed', 'EverythingIsEffort', 'Worthless', ]
# hard coded file paths
h=read_csv('https://raw.githubusercontent.com/poldrack/clean_coding/master/data/health.csv',index_col=0)[hc].dropna().mean(1)
data=read_csv('https://raw.githubusercontent.com/poldrack/clean_coding/master/data/meaningful_variables_clean.csv',index_col=0)
sc=[]
for i in range(data.shape[1]):
    if data.columns[i].split('.')[0][-7:] == '_survey':  
        sc=sc+[data.columns[i]]
data=data[sc]
# this next section duplicates the .dropna() functionality of the pandas DataFrame
gs=[]
for i in range(data.shape[0]):
    if sum(isnan(data.iloc[i, :])) > 0:
        pass
    else:
        gs=gs+[i]
data=data.iloc[gs,:]
# imports should go at the top of the file (per PEP8)
from sklearn.preprocessing import scale
data_sc = scale(data)
from sklearn.decomposition import FactorAnalysis
bicv=zeros(maxD)
for i in range(1,maxD+1):
    fa=FactorAnalysis(i)
    fa.fit(data_sc)
    bicv[i-1]=i*2 - 2*fa.score(data_sc)  # this is not actually BIC, it's AIC!
# using offset from index to move from 1-index to zero-indexed is recipe for errors
npD=argmin(bicv)+1
fa=FactorAnalysis(npD)
f=fa.fit_transform(data_sc)
for i in range(npD):
    print(pearsonr(f[:,i],h[gs]))  # we have not idea what the values are that are being printed!
    idx=argsort(abs(fa.components_[i, :]))[::-1]
    for j in range(3):  # embedded magic number - what does 3 refer to?
        print(data.columns[idx[j]], fa.components_[i, idx[j]])