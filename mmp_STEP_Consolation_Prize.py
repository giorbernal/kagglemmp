import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.trainFold import getVector

import sys
sys.path.append('../dsbase/src/main')
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModel

if (len(sys.argv) != 2):
    print('Error. Lack of index param ....')
    sys.exit(-1)

index = sys.argv[1]
print('read file ',index)
df = pd.read_csv('datasets/test_reduced.csv.' + str(index))

def loadColumnCategoricalOrder(df, columns_categorical):
    columns_categorical_order_dict = {}
    for x in columns_categorical:
        columns_categorical_order_dict[x] = np.where(df.columns == x)[0][0]
    return columns_categorical_order_dict


def loadColumnCategoricalVectors(fold_id, columns_categorical):
    columns_categorical_vectors_dict = {}
    out_path = 'models/fold' + str(fold_id)
    for c in columns_categorical:
        vec = np.load('models/fold' + str(fold_id) + "/" + c + ".sav.npy")
        columns_categorical_vectors_dict[c] = vec
    return columns_categorical_vectors_dict

def loadModel(fold_id):
   # --------------------------------------
    # Load the i-th model and process
    print('   loading model ...')    
    model = AdaBoostClassificationDSBaseModel('AB2',None,None,None,None,None,None)
    model.load('models/fold' + str(fold_id))
    return model

std=0.0029913775285332835

def calculate(x, cc, cc_o, cc_v, model, df_clean):    
    xp = x.values
    mi=xp[0]
    #xnp = xp[1:]
    acc=0
    try:
        #print('mi: ' + mi)
        xnp = df_clean.loc[mi].values
        #print(' process:' + str(xnp))
        for c in cc:
            index = cc_o[c] + acc
            vec = cc_v[c]
            new = getVector(xnp[index], vec)
            #print('  new: ' + str(new.shape))
            xnp = np.delete(xnp, index)
            #print('  xnp1: ' + str(xnp.shape))
            xnp = np.insert(xnp, index, new)
            #print('  xnp2: ' + str(xnp.shape))
            acc += (new.size - 1)
        pre_result = model.scalerX.transform(xnp.reshape(1,-1))
        result = model.model.predict_proba(pre_result)
        return result[0,1]
    except KeyError:
        return 0.5 + np.random.normal(0,std,1)[0]


df_w = df.drop(['Census_ProcessorClass','Census_InternalBatteryType','DefaultBrowsersIdentifier','Census_IsFlightingInternal','Census_ThresholdOptIn','Census_IsWIMBootEnabled','PuaMode'], axis=1)
columns_categorical = df_w.select_dtypes(include=['object']).columns[1:]

df_w.dropna(inplace=True)

df_w.set_index(['MachineIdentifier'], inplace=True)

df_w.shape

cc_order = loadColumnCategoricalOrder(df_w,columns_categorical)

i = 2
print('-------- Process Fold ',i,' -------------------')
print('loading vectors ...')
cc_values_f = loadColumnCategoricalVectors(i,columns_categorical)
print('loading model ...')
model_f = loadModel(i)
print('applying folding prediction ...')
df['HasDetections'] = df.apply(func=calculate, axis=1, args=(columns_categorical, cc_order, cc_values_f, model_f, df_w))

df[['MachineIdentifier','HasDetections']].to_csv('datasets/submission.csv.' + str(index), index=False)

print('End of process ',index,'!!')
# # End of submission setting!! 
