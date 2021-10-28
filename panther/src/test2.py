import numpy as np
import py_panther2

if __name__ == '__main__':
	A = np.arange(10)

	print('A = \n',A)

	array = py_panther2.CustomVectorXd(A)

	print('array.mul(default) = \n'   ,array.mul()          )
	print('array.mul(factor=100) = \n',array.mul(factor=100))

	print('trans(A) = \n',py_panther2.trans(A))