from compression.utils.other import ActionManager, posAccelYaw2TfMatrix, State, ObservationManager
import numpy as np

np.set_printoptions(linewidth=np.nan)

am=ActionManager();

action=am.getRandomAction()
print(action.shape)

# p0=np.zeros((3,1));
# v0=np.zeros((3,1));
# a0=np.zeros((3,1));

# y0=np.zeros((1,));
# y_dot0=np.zeros((1,));


p0=np.random.rand(3,1);
v0=np.random.rand(3,1);
a0=np.random.rand(3,1);

y0=np.random.rand(1,1);
y_dot0=np.random.rand(1,1);

print("p0= \n", p0)
print("v0= \n", v0)
print("a0= \n", a0)

print("y0= \n", y0)
print("y_dot0= \n", y_dot0)

print("action= \n", action)

my_state=State(p0, v0, a0, y0, y_dot0)

w_posBS, w_yawBS= am.f_actionAnd_w_State2wBS(action, my_state)



# print("\npos_ctrl_pts=\n", pos_ctrl_pts)
# print("\nyaw_ctrl_pts=\n", yaw_ctrl_pts)
# print("\ntotal_time=\n", total_time)

# print(action.	)


print("================================")

# posAccelYaw2TransfMatrix(p0, a0, y0)


om=ObservationManager();

obs=om.getRandomObservation();
print("Random Observation")
print(obs)
print("f_v =", om.getf_v(obs))
print("f_a =", om.getf_a(obs))
print("getyaw_dot =", om.getyaw_dot(obs))
print("getf_g =", om.getf_g(obs))
print("getObstacles =", om.getObstacles(obs))




print("================================")


p0=np.array([[0],[0],[0]]);
v0=np.array([[0],[0],[0]]);
a0=np.array([[0],[0],[0]]);
y0=np.array([[0]]);
y_dot0=np.array([[0]]);
my_state=State(p0, v0, a0, y0, y_dot0)

f_action=am.getRandomAction()

print("f_action 1= ", f_action)

w_SolOrGuess= am.f_actionAnd_w_State2w_ppSolOrGuess(f_action, my_state)

f_SolOrGuess=w_SolOrGuess; #Due to the fact that p0,v0,a0,y0,...=0

f_SolOrGuess.printInfo()

f_action_result=am.solOrGuess2action(f_SolOrGuess)

assert np.linalg.norm(f_action_result-f_action)<1e7, "Assertion failed"

print("should be zeros= ",f_action_result-f_action)



print("================================")

p0=np.array([[0],[0],[0]]);
v0=om.randomVel();
a0=om.randomAccel();
y0=om.randomYaw();
y_dot0=om.randomYdot();
my_wState=State(p0, v0, a0, y0, y_dot0)

f_action=am.getRandomAction();

w_posBS, w_yawBS=am.f_actionAnd_w_State2wBS(f_action, my_wState)

print(f"w_yawBS.getPosT(0.0)={w_yawBS.getPosT(0.0)}")
print(f"y0={y0}")
# assert w_yawBS.getPosT(0.0)==y0, "Assertion failed"
print("Test 1")
np.testing.assert_allclose(w_yawBS.getPosT(0.0)-y0, 0, atol=1e-05)


f_yaw_ctrl_pts=am.getYawCtrlPts(f_action);

print("w_Last pos=",w_yawBS.getLastPos())
print("f_yaw_ctrl_pts=",f_yaw_ctrl_pts)
np.testing.assert_allclose(w_yawBS.getLastPos()-(f_yaw_ctrl_pts[0,-1]+y0), 0, atol=1e-07)


print(f_yaw_ctrl_pts)
print(f_yaw_ctrl_pts+y0*np.ones(f_yaw_ctrl_pts.shape))
print(w_yawBS.ctrl_pts[0,2:-1].reshape(1,-1))
should_be_zeros= w_yawBS.ctrl_pts[0,2:-1].reshape(1,-1)-(f_yaw_ctrl_pts+y0*np.ones(f_yaw_ctrl_pts.shape))
print("Test 2")
print("should_be_zeros= ",should_be_zeros)
np.testing.assert_allclose(should_be_zeros, 0, atol=1e-07)



print(w_yawBS.ctrl_pts)
