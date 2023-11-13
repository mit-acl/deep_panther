import numpy as np

from colorama import Fore, Style

class State():
	def __init__(self, w_pos, w_vel, w_accel, w_yaw, yaw_dot):
		from .utils import posAccelYaw2TfMatrix
		assert w_pos.shape==(3,1)
		assert w_vel.shape==(3,1)
		assert w_accel.shape==(3,1)
		assert w_yaw.shape==(1,1)
		assert yaw_dot.shape==(1,1)
		self.w_pos = w_pos
		self.w_vel = w_vel
		self.w_accel = w_accel
		self.w_yaw = w_yaw
		self.yaw_dot = yaw_dot
		self.w_T_f= posAccelYaw2TfMatrix(self.w_pos, np.array([[0.0],[0.0], [0.0]]), w_yaw) #pos, accel, yaw
		ez=np.array([[0.0],[0.0],[1.0]]);
		np.testing.assert_allclose(self.w_T_f.T[0:3,2].reshape(3,1)-ez, 0, atol=1e-07)
		self.f_T_w= self.w_T_f.inv()
	def f_pos(self):
		return self.f_T_w*self.w_pos;
	def f_vel(self):
		f_vel=self.f_T_w.rot()@self.w_vel;
		# assert (np.linalg.norm(f_vel)-np.linalg.norm(self.w_vel)) == pytest.approx(0.0), f"f_vel={f_vel} (norm={np.linalg.norm(f_vel)}), w_vel={self.w_vel} (norm={np.linalg.norm(self.w_vel)}), f_R_w={self.f_T_w.rot()}, "
		return f_vel;
	def f_accel(self):
		self.f_T_w.debug();
		f_accel=self.f_T_w.rot()@self.w_accel;
		# assert (np.linalg.norm(f_accel)-np.linalg.norm(self.w_accel)) == pytest.approx(0.0), f"f_accel={f_accel} (norm={np.linalg.norm(f_accel)}), w_accel={self.w_accel} (norm={np.linalg.norm(self.w_accel)}), f_R_w={self.f_T_w.rot()}, " 
		return f_accel;
	def f_yaw(self):
		return np.array([[0.0]]);
	def print_w_frameHorizontal(self, msg_before=""):
		np.set_printoptions(precision=3, suppress=True)
		print(msg_before + "(In w frame)"+ \
		Fore.RED +f"pos, "+ \
		Fore.BLUE +f"vel, "+ \
		Fore.GREEN +f"accel, "+ \
		Fore.YELLOW +f"yaw, "+ \
		Fore.MAGENTA +f"dyaw: "+ \
		Fore.RED +f"{self.w_pos.T}"+Style.RESET_ALL+ \
		Fore.BLUE +f"{self.w_vel.T}"+Style.RESET_ALL+ \
		Fore.GREEN +f"{self.w_accel.T}"+Style.RESET_ALL+ \
		Fore.YELLOW +f"{self.w_yaw}"+Style.RESET_ALL+ \
		Fore.MAGENTA +f"{self.yaw_dot}"+Style.RESET_ALL)

	def print_f_frameHorizontal(self, msg_before=""):
		np.set_printoptions(precision=3, suppress=True)
		print(msg_before + "(In f frame)"+ \
		Fore.RED +f"pos, "+ \
		Fore.BLUE +f"vel, "+ \
		Fore.GREEN +f"accel, "+ \
		Fore.YELLOW +f"yaw, "+ \
		Fore.MAGENTA +f"dyaw: "+ \
		Fore.RED +f"{self.f_pos().T}"+Style.RESET_ALL+ \
		Fore.BLUE +f"{self.f_vel().T}"+Style.RESET_ALL+ \
		Fore.GREEN +f"{self.f_accel().T}"+Style.RESET_ALL+ \
		Fore.YELLOW +f"{self.f_yaw()}"+Style.RESET_ALL+ \
		Fore.MAGENTA +f"{self.yaw_dot}"+Style.RESET_ALL)