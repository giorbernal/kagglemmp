import numpy as np
import pandas as pd

# if v is in the list, retrieves the one-hot vector, otherwise retrieve 0s vector
def getVector(v, list):
	ohv = pd.get_dummies(list)
	try:
		v = ohv[ohv[v] == 1]
		return v.values[0]
	except KeyError:
		return np.zeros(list.shape[0], dtype='uint8')


