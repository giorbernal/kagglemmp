import numpy as np
import pandas as pd
import sys
sys.path.append('../dsbase/src/main')

from sklearn.model_selection import train_test_split
from ModelDSBase import ModelDSBaseWrapper
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModelParamsToMap
from AdaBoostClassificationDSBase import AdaBoostClassificationDSBaseModel
from utils.utils import getVector

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

def saveColumnsCategorical(fold_id, df, columns_categorical):

	out_path = 'models/fold' + str(fold_id)

	# Save columns partitioned at this fold
	for c in columns_categorical:
		np.save(out_path + '/' + str(c) + '.sav.npy',df[c].unique())

def loadColumnsCategorical(fold_id, df, columns_categorical):

	in_path = 'models/fold' + str(fold_id)
	df_aux = pd.DataFrame([list(map(lambda x: [x], row)) for row in df.values], columns=df.columns)

	# Save columns partitioned at this fold
	for c in columns_categorical:
		print('   column "' + c + '" transformation ...')
		vec = np.load('models/fold' + str(fold_id) + "/" + c + ".sav.npy")
		df_aux[c]=df_aux[c].apply(lambda x: getVector(x[0],vec))

	df_end = pd.DataFrame([np.concatenate(row) for row in df_aux.values])
	return df_end		
