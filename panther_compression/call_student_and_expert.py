from imitation.algorithms import bc
from compression.utils.other import ObservationManager, ActionManager

from compression.policies.ExpertPolicy import ExpertPolicy
import numpy as np

if __name__ == "__main__":

    student_policy = bc.reconstruct_policy("/home/jtorde/Desktop/ws/src/panther_plus_plus/panther_compression/evals/tmp_dagger/2/intermediate_policy_170.pt") #final_policy.pt
    om=ObservationManager();
    am=ActionManager();


    obs=np.array([[0.,     0.,     0.,     0.,     0.,     0.,     0.,     5.9999, 0.,     0., \
  2.5 ,   0.  ,   0. ,    2.5 ,   0. ,    0.  ,   2.5 ,   0.  ,   0. ,    2.5, \
  0. ,    0. ,    2.5 ,   0.,     0. ,    2.5 ,   0. ,    0.  ,   2.5 ,   0.,\
  0. ,    2.5 ,   0.,     0. ,    2.5  ,  0.  ,   0.,     2.5  ,  0.  ,   0.,\
  1.  ,   1. ,    1.    ]])


    obs_n=om.normalizeObservation(obs);

    # obs_n=om.getRandomNormalizedObservation();



    action_student = student_policy.predict(obs_n, deterministic=True)
    action_student=action_student[0]
    # action_student=action_student.reshape(1,-1)

    expert_policy=ExpertPolicy();

    action_expert = expert_policy.predict(obs_n)
    action_expert=action_expert[0]

    np.set_printoptions(precision=3, suppress=True)


    print(f"action_student= {action_student}")
    print(f"action_expert= {action_expert}")


    print("====================ACTION STUDENT")
    am.printAction(action_student)


    print("====================ACTION EXPERT")
    am.printAction(action_expert)
