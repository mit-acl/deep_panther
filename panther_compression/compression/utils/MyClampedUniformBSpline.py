import numpy as np
from scipy.interpolate import BSpline

from .utils import generateKnotsForClampedUniformBspline

class MyClampedUniformBSpline():
	def __init__(self,t0, tf, deg, dim, num_seg, ctrl_pts, no_deriv=False):

		assert dim==ctrl_pts.shape[0]

		deg=int(deg)
		dim=int(dim)
		num_seg=int(num_seg)

		self.pos_bs=[]; #BSpline of all the coordinates
		if(deg>=1):
			self.vel_bs=[]; #BSpline of all the coordinates
		if(deg>=2):
			self.accel_bs=[]; #BSpline of all the coordinates
		if(deg>=3):
			self.jerk_bs=[]; #BSpline of all the coordinates
		self.deg=deg;
		self.num_seg=num_seg;
		self.dim=dim;
		self.knots=generateKnotsForClampedUniformBspline(t0, tf, deg, num_seg)

		###Debugging
		if (abs(tf-t0)<1e-5):
			print(f"t0={t0}, tf={tf}, deg={deg}, num_seg={num_seg}")
			print(f"self.knots={self.knots}")
		#######

		self.ctrl_pts=ctrl_pts;
		for i in range(dim):
			self.pos_bs.append( BSpline(self.knots, self.ctrl_pts[i,:], self.deg) )
			if(no_deriv==False):
				if(deg>=1):
					self.vel_bs.append( self.pos_bs[i].derivative(1) ); #BSpline of all the coordinates
				if(deg>=2):
					self.accel_bs.append( self.pos_bs[i].derivative(2) ); #BSpline of all the coordinates
				if(deg>=3):
					self.jerk_bs.append( self.pos_bs[i].derivative(3) ); #BSpline of all the coordinates

	def getPosT(self,t):
		result=np.empty((self.dim,1))
		for i in range(self.dim):
			result[i,0]=self.pos_bs[i](t)
		return result
		
	def getVelT(self,t):
		result=np.empty((self.dim,1))
		for i in range(self.dim):
			result[i,0]=self.vel_bs[i](t)
		return result

	def getAccelT(self,t):
		result=np.empty((self.dim,1))
		for i in range(self.dim):
			result[i,0]=self.accel_bs[i](t)
		return result

	def getJerkT(self,t):
		result=np.empty((self.dim,1))
		for i in range(self.dim):
			result[i,0]=self.jerk_bs[i](t)
		return result

	def getLastPos(self):
		result1=self.ctrl_pts[0:3,-1].reshape(self.dim,1)
		result2=self.getPosT(self.knots[-1])
		np.testing.assert_allclose(result1-result2, 0, atol=1e-07)
		return result1

	def getT0(self):
		return self.knots[0]

	def getTf(self):
		return self.knots[-1]

# 	def f_obs_f_traj_2f_ppSolOrGuess(self, f_traj): #pp stands for py_panther
# 		zero_state=State(np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((1,1)), np.zeros((1,1)))
# class State():
# 	def __init__(self, w_pos, w_vel, w_accel, w_yaw, yaw_dot):
# 		pos=np.zeros(3,1)
# 		vel=np.zeros(3,1)
# 		return self.f_trajAnd_w_State2w_ppSolOrGuess(f_traj, zero_state)
