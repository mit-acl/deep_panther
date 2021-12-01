import gym
import os
import pandas as pd
from imitation.algorithms import bc, dagger
from imitation.data import types, rollout
from imitation.data.rollout import generate_trajectories, rollout_stats
from compression.policies.StudentPolicy import StudentPolicy
from compression.utils.eval import evaluate_policy, rollout_stats, compute_success
from compression.utils.other import ExpertDidntSucceed

def make_dagger_trainer(tmpdir, env, rampdown_rounds, batch_size):
    beta_schedule=dagger.LinearBetaSchedule(rampdown_rounds)
    return dagger.DAggerTrainer(
        env,
        tmpdir,
        beta_schedule,
        optimizer_kwargs=dict(lr=1e-3),
        policy_class = StudentPolicy,
        policy_kwargs = {"features_dim" : env.observation_space.shape[1]},
        batch_size= batch_size,
    )

def make_bc_trainer(tmpdir, env, batch_size):
    """Will make DAgger, but with a constant beta, set to 1 
    (always 100% prob of using expert)"""
    beta_schedule=dagger.AlwaysExpertBetaSchedule()
    return dagger.DAggerTrainer(
        env,
        tmpdir,
        beta_schedule,
        optimizer_kwargs=dict(lr=1e-3),
        policy_class = StudentPolicy,
        policy_kwargs = {"features_dim" : env.observation_space.shape[1]},
        batch_size= batch_size,
    )

#Trainer is the student
def train(trainer, expert, seed, n_traj_per_iter, n_epochs, log_path, save_full_policy_path, use_only_last_coll_ds):
    assert n_traj_per_iter > 0, "Number of trajectories per iter needs to be at least one!"
    assert n_epochs > 0, "Number of training epochs must be >= 0!"
    
    collector = trainer.get_trajectory_collector()
    # if not augm_sampling: 
    #     # Nominal trajectory collector
    #     collector = trainer.get_trajectory_collector()
    # else:  
    #     # Augmented collector
    #     collector= trainer.get_augmented_trajectory_collector(
    #                     n_extra_samples=expert.get_num_extra_samples(),
    #                     get_extra_samples_callback=expert.get_extra_states_actions)
    
    for _ in range(n_traj_per_iter):
        obs = collector.reset()
        done = False
        while not done:
            try: # Teacher may fail
                (expert_action,), act_infos = expert.predict(obs[None], deterministic=True)# Why OBS[None]??
            except ExpertDidntSucceed as e:
                # print("[Training] The following exception occurred: {}".format(e))
                print("[Training] Teacher unable to provide example. Concluding trajectory.")
                # print(f"[Training] Latest observation: {obs[None]}")
                collector.save_trajectory()
                done = True
            else: 
                # Executed if no exception occurs
                obs, _, done, _ = collector.step(expert_action, act_infos)
    
    # Add the collected rollout to the dataset and trains the classifier.
    #next_round_num = trainer.extend_and_update(n_epochs=n_epochs)
    next_round_num = trainer.selective_extend_and_update(n_epochs=n_epochs, use_only_last_coll_ds=use_only_last_coll_ds)

    # Use the round number to figure out stats of the trajectories we just collected.
    curr_rollout_stats, descriptors = rollout_stats(trainer.load_demos_at_round(next_round_num-1, augmented_demos=False))
    print("[Training] reward: {}, len: {}.".format(curr_rollout_stats["return_mean"], curr_rollout_stats["len_mean"]))

    # Store the policy obtained at this round
    trainer.save_policy(save_full_policy_path)
    print(f"[Training] Training completed. Policy saved to: {save_full_policy_path}.")

    # Save training logs
    descriptors["success"] = compute_success(trainer.load_demos_at_round(next_round_num-1, augmented_demos=False), trainer.env)
    logs = pd.DataFrame(descriptors)
    logs = logs.assign(disturbance = False) # TODO: Andrea: change this flag if disturbance is set to true in training environment
    # (maybe read flag from training environment itself)
    # save logs 
    if log_path is not None:
        logs.to_pickle(log_path+".pkl")
    
    return curr_rollout_stats