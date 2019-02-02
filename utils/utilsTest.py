import numpy as np
import unittest

from utils import getVector

class utilsTest(unittest.TestCase):

	def testGetVector(self):

		list = np.array(['asd','wqe','345','zxc'])

		r = getVector('asd',list)
		self.assertTrue( (np.array([0, 1, 0, 0]) == r).all() )
		
		r = getVector('345',list)
		self.assertTrue( (np.array([1, 0, 0, 0]) == r).all() )

		r = getVector('wqe',list)
		self.assertTrue( (np.array([0, 0, 1, 0]) == r).all() )

		r = getVector('zxc',list)
		self.assertTrue( (np.array([0, 0, 0, 1]) == r).all() )

		r = getVector('aaa',list)
		self.assertTrue( (np.array([0, 0, 0, 0]) == r).all() )


if __name__ == '__main__':
    unittest.main()