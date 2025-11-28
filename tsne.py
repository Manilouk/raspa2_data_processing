from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import commentjson as json
warnings.filterwarnings("ignore")



targets = [
#    'Kr-1-273',
#    'Kr-10-273',
#    'Xe-1-273',
#    'Xe-10-273',
    'Eth-4-298',
#    'Eth-20-298',
#    'Eth-40-298',
#    'Pro-1-298',
#    'Pro-5-298',
#    'Pro-10-298',
#    'But-0.24-298',
#    'But-1.2-298',
#    'Hex-0.02-298',
#    'Hex-10-495',
#    'Hex-25-495',
#    'DMB-13-477',
]


##########   Read datasets related to ToBBaCo  ######################
dsc_files=['Vprobes_set%iB.csv'%i for i in range(1,6)]
df_probes = pd.DataFrame()
for d in dsc_files:
    print('ml_csv/%s'%d)
    d0 = pd.read_csv('ml_csv/%s'%d, index_col='id')
    if df_probes.empty:
        df_probes=d0.copy()
    else:
        df_probes = pd.merge(df_probes, d0, on='id')
del(d0)
#print('Lentot=', len(df))
for t in targets:
    df_tob = pd.read_csv('./ml_csv/db_%s.csv'%t, index_col='id')
    df_tob = pd.merge(df_tob, df_probes, on='id')

df_tob['db_mof'] = ['tob']*len(df_tob)
print('Len tob=', len(df_tob))
##########   Read datasets related to CoRE 2019 MOFs ######################
dsc_struc = ['sa_tot_m2g', 'sa_tot_m2cm3', 'pld', 'lcd', 'vf1']
dsc_vprb = ['Vprb_e50s2.5_1', 'Vprb_e50s3.0_1', 'Vprb_e50s3.5_1', 'Vprb_e50s4.0_1']
dsc_VFn = []
df_Core2019 = pd.read_csv('ML_CSV/Core2019_ads_Eth_298_20bar_VFn.csv', index_col='id')
df_Core2019 = df_Core2019[dsc_struc + dsc_vprb + dsc_VFn + ['Eth-4-298']]
df_Core2019['db_mof'] = ['Core2019']*len(df_Core2019)
df_Core2019.rename(columns={'vf1': 'vf', 
        'Vprb_e50s2.5_1': 'Vprb_e50s2.5',
        'Vprb_e50s3.0_1': 'Vprb_e50s3.0',
        'Vprb_e50s3.5_1': 'Vprb_e50s3.5',
        'Vprb_e50s4.0_1': 'Vprb_e50s4.0' }, inplace=True)
print('Len Core=', len(df_Core2019))

df = pd.concat([df_tob, df_Core2019])
print('Lentot=', len(df))
#df.to_csv('aa.csv')
#exit()



dsc_all = list(df_probes.head())

dsc_struc = ['vf', 'sa_tot_m2g', 'sa_tot_m2cm3', 'pld', 'lcd']

dsc_sets = [
#        {'name': 'str', 'dsc': dsc_struc},
        {'name': 'str+Vprb', 'dsc':  dsc_struc + ['Vprb_e50s2.5', 'Vprb_e50s3.0', 'Vprb_e50s3.5', 'Vprb_e50s4.0']},
        ]


cmap = 'magma'
cmap = 'inferno'
cmap = 'cividis'

dirres = './results/'
df_tot = df.copy()
del(df)

for t in targets:
    df = df_tot[df_tot['db_mof']=='tob']
#    df = df_tot[df_tot['db_mof']=='Core2019']
#    df = df_tot.copy()
#    fig, ax1 = plt.subplots(1, 1, figsize=[20, 10],
    names_all = list(df.index)
    print('LLL=', len(df), len(names_all))

    for descr in dsc_sets:
        dsc = descr['dsc']
        dsc_name = descr['name']
      
        X = df[dsc]
        y = np.ravel(df[[t]])
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        z = tsne.fit_transform(X)

#        df = pd.DataFrame()
        df["y"] = y
        df["comp-1"] = z[:,0]
        df["comp-2"] = z[:,1]

        print('oooo=',len(df.loc[names_all][t]))
        
        img = plt.scatter(df["comp-1"],  df["comp-2"], c=df.loc[names_all][t], s=15, cmap=cmap)
        plt.title(descr['name'] + '   ' + t)
#        c=df.loc[names_all][targets[i]], s=2, cmap=cmap)

#        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
##                palette=sns.color_palette("hls", 20),
#                data=df).set(title=descr['name'] + t) 

        plt.show()