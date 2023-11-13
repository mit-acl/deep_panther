import numpy as np

class TfMatrix():
	def __init__(self, T):
		self.T=T;
		##Debugging
		self.debug()
	def __mul__(self, data): #Overload multiplication operator
		if data.ndim==1:
			tmp=self.T@np.concatenate(data, np.array([1]))
			return tmp[0:3]
		elif data.ndim==2:
			# print("\ndata before=\n", data)
			#Apply the transformation matrix to each of the columns
			tmp=self.T@np.concatenate((data, np.ones((1, data.shape[1]))), axis=0)
			# print("\nusing T=\n", self.T)
			#self.debug()
			# print("\ndata after=\n", tmp)
			return tmp[0:3,:]
		else:
			raise NotImplementedError 
	def inv(self):
		return TfMatrix(np.linalg.inv(self.T))
	def debug(self):
		# print("=============DEBUGING INIT==========")
		R=self.T[0:3,0:3]
		# print("*****Using R=\n",R)
		# print("*****Using R@R.T=\n",R@R.T)
		np.testing.assert_allclose(R@R.T-np.identity(3), 0, atol=1e-07)
		np.testing.assert_allclose(self.T[3,:]-np.array([[0, 0, 0, 1]]), 0, atol=1e-07)
		# print("=============DEBUGING==========")
	def rot(self): #Rotational part
		return self.T[0:3,0:3]
	def translation(self): #Translational part
		return self.T[0:3,3].reshape(3,1)
