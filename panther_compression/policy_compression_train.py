# # POLICY FOR TUBE MPC with Dagger
#
# Train a policy using Dagger in the hovering drone environment.
import matplotlib.pyplot as plt
import pathlib
import os
import tempfile
import time
import argparse
import gym
import numpy as np
import pprint
from stable_baselines3.common import on_policy_algorithm
import torch
import random
from tqdm import trange

from colorama import init, Fore, Back, Style

from imitation.policies import serialize
from imitation.util import util, logger
from imitation.algorithms import bc

from compression.policies.ExpertPolicy import ExpertPolicy
from compression.utils.train import make_dagger_trainer, make_bc_trainer, train
from compression.utils.eval import evaluate_policy

from stable_baselines3.common.env_checker import check_env

def printInBoldBlue(data_string):
    print(Style.BRIGHT+Fore.BLUE+data_string+Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log_dir", type=str) # usually "log"
    parser.add_argument("--policy_dir", type=str) # usually "tmp"
    parser.add_argument("--planner-params", type=str) # Contains details on the tasks to be learnt (ref. trajectories)
    parser.add_argument("--use-DAgger", dest='on_policy_trainer', action='store_true') # Use DAgger when true, BC when false
    parser.add_argument("--use-BC", dest='on_policy_trainer', action='store_false')
    parser.set_defaults(on_policy_trainer=True) # Default will be to use DAgger

    parser.add_argument("--n_iters", default=20, type=int)
    parser.add_argument("--n_evals", default=5, type=int)
    parser.add_argument("--eval-ep-len", default=70, type=int)
    parser.add_argument("--train-ep-len", default=200, type=int)
    parser.add_argument("--use-only-last-collected-dataset", dest='use_only_last_coll_ds', action='store_true')
    parser.set_defaults(use_only_last_coll_ds=False)
    parser.add_argument("--n-traj-per-iter", default=1, type=int)
    # Method changes
    parser.add_argument("--no-train", dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument("--no-eval", dest='eval', action='store_false')
    parser.set_defaults(eval=True)
    parser.add_argument("--no-final-eval", dest='final_eval', action='store_false')
    parser.set_defaults(final_eval=True)
    # Dagger properties
    parser.add_argument("--dagger-beta", default=10, type=int)
       
    
    args = parser.parse_args()

    printInBoldBlue("---------------- Input Arguments: -----------------------")
    print("Trainer: {}.".format("DAgger" if args.on_policy_trainer ==True else "BC" ))
    print(f"Seed: {args.seed}, Log: {args.log_dir}, Num iters: {args.n_iters}")
    print(f"Eval episode len: {args.eval_ep_len}")
    print(f"DAgger Linear Beta: {args.dagger_beta}.")

    print(f"Num of trajectories per iteration: {args.n_traj_per_iter}.")
    print("---------------------------------------------------------")

    assert args.eval == True or args.train == True, "eval = True or train = True!"

    DATA_POLICY_PATH = os.path.join(args.policy_dir, str(args.seed))
    LOG_PATH = os.path.join(args.log_dir, str(args.seed))
    FINAL_POLICY_NAME = "final_policy.pt"

    # directly learn the policy params (with or without sampling extra states?)
    N_VEC = 1
    N_TRAJECTORIES = 1
    N_EPOCHS = 2           #WAS 50!! Num epochs for training.
    ENV_NAME = "my-environment-v1"
    assert N_VEC == 1, "Online N_VEC = 1 supported (environments cannot run in parallel)."

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    t0 = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create and set properties for TRAINING environment:
    printInBoldBlue("---------------- Making Environments: -------------------")
    print("[Train Env] Making training environment...")
    train_env = gym.make(ENV_NAME)

    train_env.seed(args.seed)
    train_env.action_space.seed(args.seed)
    train_env.set_len_ep(args.train_ep_len) 

    print(f"[Train Env] Ep. Len:  {train_env.get_len_ep()} [steps].")

    # Create and set properties for EVALUATION environment
    # TODO: Andrea: remove venv since it is not properly used, or implement it correctly. 
    print("[Test Env] Making test environment...")
    test_venv = util.make_vec_env(ENV_NAME, N_VEC)
    test_venv.seed(args.seed)

    test_venv.env_method("set_len_ep", (args.eval_ep_len)) # TODO: ANDREA: increase

    print("[Test Env] Ep. Len:  {} [steps].".format(test_venv.get_attr("len_episode")))

    
    printInBoldBlue("---------------- Making Learner Policy: -------------------")
    # Create learner policy
    if args.on_policy_trainer: 
        trainer = make_dagger_trainer(tmpdir=DATA_POLICY_PATH, venv=train_env, linear_beta=args.dagger_beta)
    else: 
        trainer = make_bc_trainer(tmpdir=DATA_POLICY_PATH, venv=train_env)

    printInBoldBlue("---------------- Making Expert Policy: --------------------")
    # Create expert policy 
    expert_policy = ExpertPolicy(observation_space=train_env.observation_space, action_space=train_env.action_space)

    # Init logging
    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    print( f"All Tensorboards and logging are being written inside {tempdir_path}/.")
    logger.configure(tempdir_path / "DAgger/",  format_strs=["log", "csv"])  # "stdout"

    printInBoldBlue("---------------- Preliminiary Evaluation: --------------------")

    # Evaluate student reward before training,

    #NOTES: args.n_evals is the number of trajectories collected in the environment
    #A trajectory is defined as a sequence of steps in the environment (until the environment returns done)
    #Hence, each trajectory usually contains the result of eval_ep_len timesteps (it may contains less if the environent returned done before) 
    #In other words, episodes in |evaluate_policy() is the number of trajectories
    #                            |the environment is the number of time steps


    pre_train_stats = evaluate_policy(trainer.policy, test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH+"/pre_train_no_dist")
    print("[Evaluation]Pre-training reward: {}, len: {}.".format(pre_train_stats["return_mean"], pre_train_stats["len_mean"]))

    # print("Exiting for now")
    # exit()

    # Evaluate the reward of the expert
    expert_stats = evaluate_policy( expert_policy, test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH+"/teacher_no_dist")
    print("[Evaluation] Expert reward: {}, len: {}.\n".format( expert_stats["return_mean"], expert_stats["len_mean"]))

    del expert_stats

    # Train and evaluate
    printInBoldBlue("---------------- Training Learner: --------------------")
    stats = {"training":list(), "eval_no_dist":list()}
    if args.on_policy_trainer == True:
        assert trainer.round_num == 0
    for i in trange(args.n_iters, desc="Iteration"):

        n_training_traj = int(i*args.n_traj_per_iter) # Note: we start to count from 0. e.g. policy_0 means that we used 1 
        policy_path = os.path.join(DATA_POLICY_PATH, "intermediate_policy_"+str(n_training_traj)+".pt") # Where to save curr policy
        log_dir = LOG_PATH + "/student/" + str(n_training_traj) + "/"                                  # Where to save eval logs
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Train for iteration
        if args.train:
            print(f"[Collector] Collecting iteration {i+1}/{args.n_iters}.")
            train_stats = train(trainer=trainer, expert=expert_policy, seed=args.seed, n_traj_per_iter=args.n_traj_per_iter, n_epochs=N_EPOCHS, 
                log_path=os.path.join(log_dir, "training"),  save_full_policy_path=policy_path, use_only_last_coll_ds=args.use_only_last_coll_ds)

        if args.eval:
            # Load saved policy
            student_policy = bc.reconstruct_policy(policy_path)

            # Evaluate
            log_path = os.path.join(log_dir, "no_dist") 
            eval_stats = evaluate_policy(student_policy, test_venv, eval_episodes=args.n_evals, log_path = log_path)
            print(Fore.BLUE+"[Evaluation] Iter.: {}, rwrd: {}, len: {}.\n".format(i+1, eval_stats["return_mean"], eval_stats["len_mean"])+Style.RESET_ALL)
            del eval_stats


    if args.train:
        # Store the final policy.
        save_full_policy_path = os.path.join(DATA_POLICY_PATH, FINAL_POLICY_NAME)
        trainer.save_policy(save_full_policy_path)
        print(f"[Trainer] Training completed. Policy saved to: {save_full_policy_path}.")


    if args.final_eval:
        printInBoldBlue("---------------- Evaluation After Training: --------------------")

        # Evaluate reward of student post-training
        post_train_stats = dict()

        # no disturbance
        post_train_stats = evaluate_policy(trainer.get_policy(), test_venv,eval_episodes=args.n_evals, log_path=LOG_PATH + "/post_train_no_dist" )
        print("[Complete] Reward: Pre: {}, Post: {}.".format( pre_train_stats["return_mean"], post_train_stats["return_mean"]))
        print("[Complete] Episode length: Pre: {}, Post: {}.".format( pre_train_stats["len_mean"], post_train_stats["len_mean"]))

        del pre_train_stats, post_train_stats

        # Load and evaluate the saved DAgger policy
        load_full_policy_path = os.path.join(DATA_POLICY_PATH, FINAL_POLICY_NAME)
        final_student_policy = bc.reconstruct_policy(load_full_policy_path)
        rwrd = evaluate_policy(final_student_policy, test_venv,eval_episodes=args.n_evals, log_path=None)
        print("[Evaluation Loaded Policy] Policy: {}, Reward: {}, Len: {}.".format(load_full_policy_path, rwrd["return_mean"], rwrd["len_mean"]))

        # Evaluate the reward of the expert as a sanity check
        expert_reward = evaluate_policy(expert_policy, test_venv, eval_episodes=args.n_evals, log_path=None)
        print("[Evaluation] Expert reward: {}\n".format(expert_reward))

        #fig, ax = plt.subplots()
        #plot_train_traj(ax, stats["training"])

    print("Elapsed time: {}".format(time.time() - t0))
