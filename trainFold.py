import numpy as np
import pandas as pd
import sys
sys.path.append('../dsbase/src/main')

from sklearn.model_selection import train_test_split
from ModelDSBase import ModelDSBaseWrapper
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModelParamsToMap
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModel

def train_test(fold_id):
	print('Initiating training of fold ' + str(fold_id) + ' ...')
	in_path = 'datasets/train_fold' + str(fold_id) + '.csv'
	out_path = 'models/fold' + str(fold_id)
	print('Training of fold ' + str(fold_id) + ' finalized!')

	return (0.6455, np.zeros((2,3)))


def train(fold_id):

	print('Initiating training of fold ' + str(fold_id) + ' ...')

	in_path = 'datasets/train_fold' + str(fold_id) + '.csv'
	out_path = 'models/fold' + str(fold_id)

	# Data Loading 
	df = pd.read_csv(in_path, index_col='MachineIdentifier')


	# Splitting label information
	df_y = df['HasDetections']
	df.drop(labels=['HasDetections'], axis=1, inplace=True)

	# Categorical to Numerical 
	columns_categorical = df.select_dtypes(include=['object']).columns
	df_num=pd.get_dummies(data=df,columns=columns_categorical)

	# Training model
	params = AdaBoostClassificationDSBaseModelParamsToMap(100,1.0)
	abc = ModelDSBaseWrapper('AB',df_num.values,df_y.values,[30,65,100],0.3,AdaBoostClassificationDSBaseModel,params,splitter=train_test_split)
	abc.train()

	# Collecting results
	lcabc = abc.getLearningCurves()
	score = abc.getScore()

	# Save model
	abc.save('models/fold' + fold_id)

	print('Training of fold ' + str(fold_id) + ' finalized!')

	return (score,lcabc)
