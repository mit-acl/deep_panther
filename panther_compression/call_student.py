from imitation.algorithms import bc
from compression.utils.other import ObservationManager

if __name__ == "__main__":

    student_policy = bc.reconstruct_policy("/home/jtorde/Desktop/ws/src/panther_plus_plus/panther_compression/evals/tmp_dagger/1/final_policy.pt")
    om=ObservationManager();

    obs=om.getRandomNormalizedObservation();
    action = student_policy.predict(obs, deterministic=True)

    print(action)