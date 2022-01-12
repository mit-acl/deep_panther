import gym
import os
import pandas as pd
from imitation.algorithms import bc, dagger
from imitation.data import types, rollout
from imitation.data.rollout import generate_trajectories, rollout_stats
from compression.policies.StudentPolicy import StudentPolicy
from compression.utils.eval import evaluate_policy, rollout_stats, compute_success
from compression.utils.other import ExpertDidntSucceed, ActionManager

def make_dagger_trainer(tmpdir, venv, rampdown_rounds, custom_logger, lr, batch_size):
    beta_schedule=dagger.LinearBetaSchedule(rampdown_rounds)

    am=ActionManager()


    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        optimizer_kwargs=dict(lr=lr),
        custom_logger=custom_logger,
        policy=StudentPolicy(observation_space=venv.observation_space, action_space=venv.action_space),
        batch_size=batch_size,
        traj_size_pos_ctrl_pts=am.traj_size_pos_ctrl_pts
    )

    return dagger.DAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        beta_schedule=beta_schedule,
        bc_trainer=bc_trainer,
        custom_logger=custom_logger,
    )

def make_bc_trainer(tmpdir, venv, custom_logger, lr, batch_size):
    """Will make DAgger, but with a constant beta, set to 1 
    (always 100% prob of using expert)"""
    beta_schedule=dagger.AlwaysExpertBetaSchedule()

    am=ActionManager()


    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        optimizer_kwargs=dict(lr=1e-3),
        custom_logger=custom_logger,
        policy=StudentPolicy(observation_space=venv.observation_space, action_space=venv.action_space),
        batch_size=batch_size,
        traj_size_pos_ctrl_pts=am.traj_size_pos_ctrl_pts
    )

    return dagger.DAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        beta_schedule=beta_schedule,
        bc_trainer=bc_trainer,
        custom_logger=custom_logger,
    )

#Trainer is the student
def train(trainer, expert, seed, n_traj_per_round, n_epochs, log_path, save_full_policy_path, use_only_last_coll_ds, only_collect_data):
    # assert n_traj_per_round > 0, "n_traj_per_round needs to be at least one!"
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
    

    for _ in range(n_traj_per_round):
        expert_succeeded_at_least_once=False;

        obs = collector.reset()
        done = False
        while not done:
            try: # Teacher may fail
                expert_action, act_infos = expert.predict(obs[None], deterministic=True)# Why OBS[None]??
            except ExpertDidntSucceed as e:
                # print("[Training] The following exception occurred: {}".format(e))
                # print(f"[Training] Latest observation: {obs[None]}")
                if(expert_succeeded_at_least_once):
                    done = True
                    print("[Training] Teacher unable to provide example. Concluding trajectory.")
                    collector.save_trajectory()
                else:
                    print("[Training] Teacher unable to provide first example. Resetting Collector and trying again.")
                    obs = collector.reset()
            else: 
                # Executed if no exception occurs
                obs, _, done, _ = collector.step([expert_action])#The [] were added by jtorde, 1/1/22 #act_infos
                expert_succeeded_at_least_once = True

    curr_rollout_stats=None
    if(only_collect_data==False):
        # Add the collected rollout to the dataset and trains the classifier.
        #next_round_num = trainer.extend_and_update(n_epochs=n_epochs)
        next_round_num = trainer.extend_and_update(dict(n_epochs=n_epochs, save_full_policy_path=save_full_policy_path))#use_only_last_coll_ds=use_only_last_coll_ds

        # Use the round number to figure out stats of the trajectories we just collected.
        curr_rollout_stats, descriptors = rollout_stats(trainer.load_demos_at_round(next_round_num-1, augmented_demos=False))
        print("[Training] reward: {}, len: {}.".format(curr_rollout_stats["return_mean"], curr_rollout_stats["len_mean"]))

        # Store the policy obtained at this round
        trainer.save_policy(save_full_policy_path)
        print(f"[Training] Training completed. Policy saved to: {save_full_policy_path}.")

        # Save training logs
        descriptors["success"] = compute_success(trainer.load_demos_at_round(next_round_num-1, augmented_demos=False), trainer.venv)
        logs = pd.DataFrame(descriptors)
        logs = logs.assign(disturbance = False) # TODO: Andrea: change this flag if disturbance is set to true in training environment
        # (maybe read flag from training environment itself)
        # save logs 
        if log_path is not None:
            logs.to_pickle(log_path+".pkl")
    
    return curr_rollout_stats