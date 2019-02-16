
# coding: utf-8

# # Setting of stack phase 

# At this phase we are going to set the stacked-phase dataset 

# In[1]:
print('init')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:


from utils.trainFold import loadColumnsCategorical


# In[3]:


import sys
sys.path.append('/opt/dsbase')
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModel


# ## Loading the original stacked dataset and shuffle it

# In[4]:

print('initiating data load')
df = pd.read_csv('datasets/train_stack.csv')
print('end of data load')


# In[5]:


df_frac = df.sample(frac=0.30)


# In[6]:


df_frac.shape


# ## Defining the Fold X processing 

# In[7]:


def getColumnFoldX(df, fold_id):
    # Convert every element to a one-elenet List
    print('   dataframe to list ...')
    df_w = df.drop(['MachineIdentifier','HasDetections','fold'], axis=1)
    columns_categorical = df_w.select_dtypes(include=['object']).columns

    # Comluns transformation
    df_end = loadColumnsCategorical(fold_id, df_w, columns_categorical)
    
    # --------------------------------------
    # Load the i-th model and process
    print('   loading model ...')    
    model = AdaBoostClassificationDSBaseModel('AB2',None,None,None,None,None,None)
    model.load('models/fold' + str(fold_id))
    
    print('   Calculating: normalization ...')    
    pre_result = model.scalerX.transform(df_end.values)
    print('   Calculating: probabilities ...')    
    result = model.model.predict_proba(pre_result)
    
    # Set the result as a one-column DataFrame
    print('   Creating result dataset ...')        
    columns_name = [str('f' + str(fold_id))]
    df_result = pd.DataFrame(result[:,1])
    df_result.columns = columns_name
    return df_result


# ## Lets obtain the final stacked dataset 

# In[8]:


N = 9 # Number of folds
df_stack_set = df_frac.reset_index(drop=True)
for i in range(9):
    print('processing fold ' + str(i+1) + " ...")
    c = getColumnFoldX(df_frac, i+1)
    c.to_csv('datasets/f_stack.csv.' + str(i+1))
    df_stack_set = df_stack_set.join(c)


# In[9]:


df_stack_set[['HasDetections','fold','f1','f2','f3','f4','f5','f6','f7','f8','f9']].describe()


# In[10]:


df_stack_set.to_csv('datasets/train_stack_set.csv')


# # End of stack train setting!! 
print('end of process!!')
