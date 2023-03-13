import pandas as pd
from imitation.data import types, rollout
from imitation.data.rollout import generate_trajectories, rollout_stats

def compute_success_rate(trajectories, eval_episodes, venv):
    nominal_ep_len = venv.get_attr("len_episode")[0]
    success = 0
    for traj in trajectories:
        if len(traj.obs) == int(nominal_ep_len + 1):
            success += 1
    return float(success)/float(eval_episodes)

def compute_success(trajectories, venv):
    try:
        nominal_ep_len = venv.get_attr("len_episode")[0]
    except: 
        nominal_ep_len = venv.len_episode
    success = []
    for traj in trajectories:
        if len(traj.obs) == int(nominal_ep_len + 1):
            success.append(True)
        else: 
            success.append(False)
    return success

def evaluate_policy(policy, venv, log_path, eval_episodes = 30):
    # venv.env_method(env_dist_f_call, (disturbance))
    trajectories = generate_trajectories(
        policy,
        venv,
        sample_until=rollout.make_min_episodes(eval_episodes), 
        deterministic_policy=True,
    )
    stats, descriptors = rollout_stats(trajectories)
    descriptors["success"] = compute_success(trajectories, venv)
    stats["success_rate"] = float(sum(descriptors["success"]))/float(eval_episodes)
    
    logs = pd.DataFrame(descriptors)
    # logs = logs.assign(disturbance = disturbance)
    # save logs 
    if log_path is not None:
        logs.to_pickle(log_path+".pkl")
    return stats