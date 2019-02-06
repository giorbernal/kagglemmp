import numpy as np
import pandas as pd
import sys
sys.path.append('/opt/dsbase')

from sklearn.model_selection import train_test_split
from ModelDSBase import ModelDSBaseWrapper
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModelParamsToMap
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModel

def train_test(fold_id, df):
	print('Initiating training of fold ' + str(fold_id) + ' ...')
	print('  size: ' + str(df.shape))
	out_path = 'models/fold' + str(fold_id) + "/test.sav.npy"
	np.save(out_path,np.array(['pepo','te','migo','micci']))
	print('Training of fold ' + str(fold_id) + ' finalized!')
	return (0.6455, np.zeros((2,3)))


def train(fold_id, df):

	print('Initiating training of fold ' + str(fold_id) + ' ...')

	out_path = 'models/fold' + str(fold_id)

	# Splitting label information
	df_y = df['HasDetections']
	df.drop(labels=['HasDetections'], axis=1, inplace=True)


	# Training model
	params = AdaBoostClassificationDSBaseModelParamsToMap(100,1.0)
	abc = ModelDSBaseWrapper('AB',df.values,df_y.values,[30,65,100],0.3,AdaBoostClassificationDSBaseModel,params,splitter=train_test_split)
	abc.train()

	# Collecting results
	lcabc = abc.getLearningCurves()
	score = abc.getScore()

	# Save model
	abc.save(out_path)

	print('Training of fold ' + str(fold_id) + ' finalized!')

	return (score,lcabc)

def saveColumnsToPartition(fold_id, df, columns_to_partition):

	out_path = 'models/fold' + str(fold_id)

	# Save columns partitioned at this fold
	for c in columns_to_partition:
		np.save(out_path + '/' + str(c) + '.sav.npy',df[c].unique())

