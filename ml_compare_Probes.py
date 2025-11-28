import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, mean_absolute_percentage_error
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
import commentjson as json
import pickle
warnings.filterwarnings("ignore")



def st_scores(yref, ytest):
    r2 = r2_score(yref, ytest)
    mae = mean_absolute_error(yref, ytest)
    rmse = np.sqrt(mean_squared_error(yref, ytest))
    wape = mae / np.average(yref)*100
    mape = mean_absolute_percentage_error(yref, ytest)#*100
#    mmape = 0.
#    n = yref.size
#    for i in range(n):
#        mmape +=  abs(    (yref[i] -ytest[i])  / ytest[i])*100
#    mmape /= n
    st = {'r2': r2, 'mae': mae, 'rmse': rmse, 'wape': wape, 'mape': mape}#, 'mmape': mmape}
    return st



targets = [
    'Kr-1-273',
    'Kr-10-273',
    'Xe-1-273',
    'Xe-10-273',
    'Eth-4-298',
    'Eth-20-298',
    'Eth-40-298',
    'Pro-1-298',
    'Pro-5-298',
    'Pro-10-298',
    'But-0.24-298',
    'But-1.2-298',
    'Hex-0.02-298',
    'Hex-10-495',
    'Hex-25-495',
    'DMB-13-477',
]
dsc_files=['Vprobes_set%iB.csv'%i for i in range(1,6)] #preparing files
df_probes = pd.DataFrame()
for d in dsc_files:
    print('ml_csv/%s'%d)
    d0 = pd.read_csv('ml_csv/%s'%d, index_col='id')
    if df_probes.empty: # the first time is empty
        df_probes=d0.copy()
    else: #after the first time, add df_probes
        df_probes = pd.merge(df_probes, d0, on='id') #entry the data
		# if print df_probes shows 
del(d0) # delete the d0 after completing the loop
print(df_probes.head()) # print first rows from the top


dsc_all = list(df_probes.head()) #list the names of descriptors like Vprob_e10s1.0

dsc_prb = ['Vprb1', 'Vprb2', 'Vprb3', 'Vprb4']
dsc_prb = [i for i in dsc_all if 'e100s'  in i]
dsc_prb = []
for i in dsc_all:
    if 'e10s' in i or 'e100s' in i or 'e50s' in i or 'e30s' in i or 'e70s' in i:
        dsc_prb.append(i) #add in list the titles of descriptors
print('DDD=', dsc_prb)
dsc_struc = ['vf', 'sa_tot_m2g', 'sa_tot_m2cm3', 'pld', 'lcd'] #structural descriptors
dsc_hen = ['henry_ch3_298K_mol_kg_pa', 'henry_kr_273K_mol_kg_pa', 'henry_xe_273K_mol_kg_pa']
dsc_prb = ['Vprb_e50s2.5', 'Vprb_e50s3.0', 'Vprb_e50s3.5', 'Vprb_e50s4.0']
dsc_prb = ['Vprb_e50s2.5', 'Vprb_e50s3.0', 'Vprb_e50s3.5', 'Vprb_e50s4.0', 'Vprb_e50s4.5', 'Vprb_e50s5.0']
dsc_prb = ['Vprb_e100s2.5', 'Vprb_e100s3.0', 'Vprb_e100s3.5', 'Vprb_e100s4.0', 'Vprb_e100s4.5', 'Vprb_e100s5.0']

dsc_sets = [
#        {'name': 'str', 'dsc': dsc_struc},
#        {'name': 'Prb', 'dsc': dsc_prb},
#        {'name': 'strPrb', 'dsc': dsc_struc + dsc_prb},
#        {'name': 'strHen', 'dsc': dsc_struc + dsc_hen},
#        {'name': 'strHenPrb', 'dsc': dsc_struc + dsc_hen + dsc_prb},
        ]
dsc_sets = [ #put as a list the descriptors
        {'name': 'str', 'dsc': dsc_struc},
        {'name': 'Vprb', 'dsc':  ['Vprb_e50s2.5', 'Vprb_e50s3.0', 'Vprb_e50s3.5', 'Vprb_e50s4.0']},
        {'name': 'prbset1', 'dsc':  [i for i in dsc_all if 'e10s' in i]}, #prbsets are the new descriptors
        {'name': 'prbset2', 'dsc':  [i for i in dsc_all if 'e30s' in i]},
        {'name': 'prbset3', 'dsc':  [i for i in dsc_all if 'e50s' in i]},
        {'name': 'prbset4', 'dsc':  [i for i in dsc_all if 'e70s' in i]},
        {'name': 'prbset5', 'dsc':  [i for i in dsc_all if 'e100s' in i]},
        {'name': 'prbset1-5', 'dsc': dsc_all},
        {'name': 'str+Vprb', 'dsc':  dsc_struc + ['Vprb_e50s2.5', 'Vprb_e50s3.0', 'Vprb_e50s3.5', 'Vprb_e50s4.0']},
        {'name': 'str+prbset1', 'dsc': dsc_struc + [i for i in dsc_all if 'e10s' in i]},
        {'name': 'str+prbset2', 'dsc': dsc_struc + [i for i in dsc_all if 'e30s' in i]},
        {'name': 'str+prbset3', 'dsc': dsc_struc + [i for i in dsc_all if 'e50s' in i]},
        {'name': 'str+prbset3', 'dsc': dsc_struc + [i for i in dsc_all if 'e50s' in i]},
        {'name': 'str+prbset4', 'dsc': dsc_struc + [i for i in dsc_all if 'e70s' in i]},
        {'name': 'str+prbset5', 'dsc': dsc_struc + [i for i in dsc_all if 'e100s' in i]},
        {'name': 'str+prbset1-5', 'dsc': dsc_struc + dsc_all},
        ]

#dsc_sets = [{'name': 'str+prb_orig', 'dsc': dsc_struc + dsc_prb}]

methods = [
         {'name': 'RF'},
#        {'name': 'XGB'},
#        {'name': 'ERT'},
#        {'name': 'Lasso'},
        ]

#dsc = dsc_struc  + dsc_hen # + dsc_hen
#print(dsc)

irun=0; iter=0
#clf = RandomForestRegressor(random_state=10000+1000*irun+iter)
#clf = ExtraTreesRegressor(random_state=10000+1000*irun+iter)
l_futureimportance = False

dirres = './results/'
dat = {}
dat['dsc_sets'] = dsc_sets #the begin is dictionary of {dsc_sets}: [after in the list the content of dsc_sets] 
for meth in methods:
    m = meth['name']
    dat[m]={}
    if m=='RF':
        irun=0; iter=0
        clf = RandomForestRegressor(random_state=10000+1000*irun+iter)
    if m=='ERT':
        irun=0; iter=0
        clf = ExtraTreesRegressor(random_state=10000+1000*irun+iter)
    if m=='XGB':
        irun=0; iter=0
        clf = XGBRegressor(verbosity=0, random_state=10000+1000*irun+iter)
    if m=='Lasso':
        irun=0; iter=0
        clf = Lasso(alpha=0.5)
     
    for t in targets:
        dat[m][t]={}
        df0 = pd.read_csv('./ml_csv/db_%s.csv'%t, index_col='id')
#        df = df0.copy()
        df = pd.merge(df0, df_probes, on='id')
#        df.to_csv('justforfun.csv')
#        exit()
    
        df_train = df[df['datatype_%s'%t]=='Training']
        df_test = df[df['datatype_%s'%t]=='Testing']
        print('Ntrain= %i   Ntest= %i'%(len(df_train), len(df_test)))
    
    
        for descr in dsc_sets:
            dsc = descr['dsc']
            dsc_name = descr['name']
#            print('method=', meth, 'target: ',t)
#            print('Tha kanw ' + './ml_csv/db_%s.csv'%t)
#            print('descriptors name', dsc_name)
#            print('descriptors ', dsc)
#            print('DSC=', dsc)
            dat[m][t][dsc_name]={'descriptors': dsc}
          
            X_train = df_train[dsc]
            X_test = df_test[dsc]
            y_train = np.ravel(df_train[[t]])
            y_test = np.ravel(df_test[[t]])
            model = clf.fit(X_train, y_train)
            yhat_train = model.predict(X_train)
            sc_train = st_scores(y_train, yhat_train)
    
            yhat_test= model.predict(X_test)
            sc_test = st_scores(y_test, yhat_test)
            dat[m][t][dsc_name]=sc_test
            print('%30s: %20s   %10s   %.3f %.3f  %.3f %.3f  '
               %( t,  m, descr['name'], sc_test['r2'], sc_test['mae'], 
                   sc_test['rmse'], sc_test['wape']))
            plt.title(t)
            plt.scatter(y_test, yhat_test)
            amax = max(y_train)
#            print('---=', m, t, descr)
            fpng = '%s_%s_%s.png'%(m,  t, descr['name'])
            print('fpng=', fpng)
            plt.text(0.2*amax,0.8*amax,'Ntrain=%i\n$R^2=$%.3f\nMAE=%.3f\nRMSE=%.3f\nWAPE=%.1f'%
                     (len(df_train), sc_train['r2'], sc_train['mae'], sc_train['rmse'], sc_train['wape']))
            plt.text(0.8*amax,0.2*amax,'Ntest=%i\n$R^2=$%.3f\nMAE=%.3f\nRMSE=%.3f\nWAPE=%.1f'%
                     (len(df_test), sc_test['r2'], sc_test['mae'], sc_test['rmse'], sc_test['wape']))
            plt.plot([0, 1.1*amax], [0, 1.1*amax])
#            plt.show()
#            plt.savefig('./png/' + fpng)

    pickle.dump(dat, open('%s%s_compareProbes.pickle'%(dirres, m), 'wb'))
     
