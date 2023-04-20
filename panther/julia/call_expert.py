from imitation.algorithms import bc
from compression.utils.other import ObservationManager, ActionManager
from compression.policies.ExpertPolicy import ExpertPolicy
import numpy as np

def get_expert_action():

    om=ObservationManager()
    am=ActionManager()

    obs=np.array([[0.,     0.,     0.,     0.,     0.,     0.,     0.,     5.9999, 0.,     0., \
  2.5 ,   0.  ,   0. ,    2.5 ,   0. ,    0.  ,   2.5 ,   0.  ,   0. ,    2.5, \
  0. ,    0. ,    2.5 ,   0.,     0. ,    2.5 ,   0. ,    0.  ,   2.5 ,   0.,\
  0. ,    2.5 ,   0.,     0. ,    2.5  ,  0.  ,   0.,     2.5  ,  0.  ,   0.,\
  1.  ,   1. ,    1.    ]])

    obs_n=om.normalizeObservation(obs)
    expert_policy=ExpertPolicy()
    act = expert_policy.predict(obs_n)
    act = act[0]
    # np.set_printoptions(precision=3, suppress=True)
    # print(f"act= {act}")
    # print("====================ACTION EXPERT")
    # am.printAction(act)
    return obs, act