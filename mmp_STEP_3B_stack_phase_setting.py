import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import sys
sys.path.append('../dsbase/src/main')
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModel
from utils.trainFold import getVector

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
    sys.stdout.write('   loading model ...\n')
    model = AdaBoostClassificationDSBaseModel('AB2',None,None,None,None,None,None)
    model.load('models/fold' + str(fold_id))
    return model

def calculate(x, cc, cc_o, cc_v, model):
    xnp = x.values
    acc=0
    for c in cc:
        index = cc_o[c] + acc
        vec = cc_v[c]
        new = getVector(xnp[index], vec)
        xnp = np.delete(xnp, index)
        xnp = np.insert(xnp, index, new)
        acc += (new.size - 1)
    pre_result = model.scalerX.transform(xnp.reshape(1,-1))
    result = model.model.predict_proba(pre_result)
    return result[0,1]


def main():

    sys.stdout.write('loading dataset ...\n')
    #df = pd.read_csv('datasets/train_stack_shuffled_reduced.csv.1')
    df = pd.read_csv('datasets/train_stack_shuffled_super_reduced.csv')
    sys.stdout.write('dataset loaded!\n')

    sys.stdout.write(str(df.shape) + '\n')

    df_w = df.drop(['Unnamed: 0','MachineIdentifier','HasDetections','fold'], axis=1)
    columns_categorical = df_w.select_dtypes(include=['object']).columns

    cc_order = loadColumnCategoricalOrder(df_w,columns_categorical)

    N = 9
    for i in range(1,N+1):
        sys.stdout.write('-------- Process Fold ' + str(i) +' -------------------\n')
        sys.stdout.write('loading vectors ...\n')
        cc_values_f = loadColumnCategoricalVectors(i,columns_categorical)
        sys.stdout.write('loading model ...\n')
        model_f = loadModel(i)
        sys.stdout.write('applying folding prediction ...\n')
        df['f' + str(i)] = df_w.apply(func=calculate, axis=1, args=(columns_categorical, cc_order, cc_values_f, model_f))
        # save security DatFrame
        df['f' + str(i)].to_csv('datasets/f_stack.csv.' + str(i))

    #df[['HasDetections','fold','f1','f2','f3','f4','f5','f6','f7','f8','f9']].describe()

    sys.stdout.write('exporting result\n')
    df.to_csv('datasets/train_stack_set.csv')
    sys.stdout.write('End of process!\n')

if __name__ == "__main__":
    main()