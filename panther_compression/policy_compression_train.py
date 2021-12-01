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
# import os
import subprocess


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
def printInBoldRed(data_string):
    print(Style.BRIGHT+Fore.RED+data_string+Style.RESET_ALL)
def printInBoldGreen(data_string):
    print(Style.BRIGHT+Fore.GREEN+data_string+Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--log_dir", type=str, default="evals/log_dagger") # usually "log"
    parser.add_argument("--policy_dir", type=str, default="evals/tmp_dagger") # usually "tmp"
    parser.add_argument("--planner-params", type=str) # Contains details on the tasks to be learnt (ref. trajectories)
    parser.add_argument("--use-DAgger", dest='on_policy_trainer', action='store_true') # Use DAgger when true, BC when false
    parser.add_argument("--use-BC", dest='on_policy_trainer', action='store_false')
    parser.set_defaults(on_policy_trainer=True) # Default will be to use DAgger

    parser.add_argument("--n_rounds", default=100, type=int) #was called n_iters before
    parser.add_argument("--n_evals", default=6, type=int)
    parser.add_argument("--eval_environment_max_steps", default=20, type=int)
    parser.add_argument("--train_environment_max_steps", default=20, type=int)
    parser.add_argument("--use_only_last_collected_dataset", dest='use_only_last_coll_ds', action='store_true')
    parser.set_defaults(use_only_last_coll_ds=False)
    parser.add_argument("--n_traj_per_iter", default=1, type=int)
    # Method changes
    parser.add_argument("--no_train", dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument("--no_eval", dest='eval', action='store_false')
    parser.set_defaults(eval=False)
    parser.add_argument("--no_init_and_final_eval", dest='init_and_final_eval', action='store_false')
    parser.set_defaults(init_and_final_eval=False)
    # Dagger properties
    parser.add_argument("--rampdown_rounds", default=100, type=int)
    
    args = parser.parse_args()

    printInBoldBlue("---------------- Input Arguments: -----------------------")
    print("Trainer: {}.".format("DAgger" if args.on_policy_trainer ==True else "BC" ))
    print(f"seed: {args.seed}, log_dir: {args.log_dir}, n_rounds: {args.n_rounds}")
    print(f"eval_environment_max_steps: {args.eval_environment_max_steps}")
    print(f"use_only_last_coll_ds: {args.use_only_last_coll_ds}")
    print(f"train: {args.train}, eval: {args.eval}, init_and_final_eval: {args.init_and_final_eval}")
    print(f"DAgger rampdown_rounds: {args.rampdown_rounds}.")

    print(f"Num of trajectories per iteration: {args.n_traj_per_iter}.")
    print("---------------------------------------------------------")


    #Remove previous directories
    os.system("rm -rf "+args.log_dir)
    os.system("rm -rf "+args.policy_dir)

    # np.set_printoptions(precision=3)

    assert args.rampdown_rounds<=args.n_rounds, f"Are you sure you wanna this? rampdown_rounds={args.rampdown_rounds}, n_rounds={args.n_rounds}"

    assert args.eval == True or args.train == True, "eval = True or train = True!"

    DATA_POLICY_PATH = os.path.join(args.policy_dir, str(args.seed))
    LOG_PATH = os.path.join(args.log_dir, str(args.seed))
    FINAL_POLICY_NAME = "final_policy.pt"

    # directly learn the policy params (with or without sampling extra states?)
    N_VEC = 1
    N_EPOCHS = 50           #WAS 50!! Num epochs for training.
    BATCH_SIZE = 32
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
    train_env.set_len_ep(args.train_environment_max_steps) 

    print(f"[Train Env] Ep. Len:  {train_env.get_len_ep()} [steps].")

    # Create and set properties for EVALUATION environment
    # TODO: Andrea: remove venv since it is not properly used, or implement it correctly. 
    print("[Test Env] Making test environment...")
    test_venv = util.make_vec_env(env_name=ENV_NAME, n_envs=N_VEC, seed=args.seed, parallel=False)
    test_venv.seed(args.seed)

    test_venv.env_method("set_len_ep", (args.eval_environment_max_steps)) # TODO: ANDREA: increase

    print("[Test Env] Ep. Len:  {} [steps].".format(test_venv.get_attr("len_episode")))

    
    printInBoldBlue("---------------- Making Learner Policy: -------------------")
    # Create learner policy
    if args.on_policy_trainer: 
        trainer = make_dagger_trainer(tmpdir=DATA_POLICY_PATH, env=train_env, rampdown_rounds=args.rampdown_rounds, batch_size=BATCH_SIZE)
    else: 
        trainer = make_bc_trainer(tmpdir=DATA_POLICY_PATH, env=train_env, batch_size=BATCH_SIZE)

    printInBoldBlue("---------------- Making Expert Policy: --------------------")
    # Create expert policy 
    expert_policy = ExpertPolicy()

    # Init logging
    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = LOG_PATH#"evals/log_tensorboard"#LOG_PATH#pathlib.Path(tempdir.name)
    print( f"All Tensorboards and logging are being written inside {tempdir_path}/.")
    logger.configure(tempdir_path,  format_strs=["log", "csv", "tensorboard"])  # "stdout"

    if(args.init_and_final_eval):
        printInBoldBlue("---------------- Preliminiary Evaluation: --------------------")

        #NOTES: args.n_evals is the number of trajectories collected in the environment
        #A trajectory is defined as a sequence of steps in the environment (until the environment returns done)
        #Hence, each trajectory usually contains the result of eval_environment_max_steps timesteps (it may contains less if the environent returned done before) 
        #In other words, episodes in |evaluate_policy() is the number of trajectories
        #                            |the environment is the number of time steps

        # Evaluate the reward of the expert
        expert_stats = evaluate_policy( expert_policy, test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH+"/teacher_no_dist")
        print("[Evaluation] Expert reward: {}, len: {}.\n".format( expert_stats["return_mean"], expert_stats["len_mean"]))

        #Debugging
        # exit();

        # Evaluate student reward before training,
        pre_train_stats = evaluate_policy(trainer.get_policy(), test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH+"/pre_train_no_dist")
        print("[Evaluation] Student reward: {}, len: {}.".format(pre_train_stats["return_mean"], pre_train_stats["len_mean"]))


        del expert_stats




    # Train and evaluate
    printInBoldBlue("---------------- Training Learner: --------------------")


    #Launch tensorboard visualization
    os.system("pkill -f tensorboard")
    proc1 = subprocess.Popen(["tensorboard","--logdir",LOG_PATH,"--bind_all"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    proc2 = subprocess.Popen(["google-chrome","http://jtorde-alienware-aurora-r8:6006/"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    # os.system("tensorboard --logdir "+args.log_dir +" --bind_all")
    # os.system("google-chrome http://jtorde-alienware-aurora-r8:6006/")  


    stats = {"training":list(), "eval_no_dist":list()}
    if args.on_policy_trainer == True:
        assert trainer.round_num == 0
    for i in trange(args.n_rounds, desc="Round"):

        #Create names for policies
        n_training_traj = int(i*args.n_traj_per_iter) # Note: we start to count from 0. e.g. policy_0 means that we used 1 
        policy_path = os.path.join(DATA_POLICY_PATH, "intermediate_policy_"+str(n_training_traj)+".pt") # Where to save curr policy
        log_dir = LOG_PATH + "/student/" + str(n_training_traj) + "/"                                  # Where to save eval logs
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Train for iteration
        if args.train:
            print(f"[Collector] Collecting round {i+1}/{args.n_rounds}.")
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


    if args.init_and_final_eval:
        printInBoldBlue("---------------- Evaluation After Training: --------------------")

        # Evaluate reward of student post-training
        post_train_stats = dict()

        # no disturbance
        post_train_stats = evaluate_policy(trainer.get_policy(), test_venv,eval_episodes=args.n_evals, log_path=LOG_PATH + "/post_train_no_dist" )
        print("[Complete] Reward: Pre: {}, Post: {}.".format( pre_train_stats["return_mean"], post_train_stats["return_mean"]))

        if(abs(pre_train_stats["return_mean"])>0):
            student_improvement=(post_train_stats["return_mean"]-pre_train_stats["return_mean"])/abs(pre_train_stats["return_mean"]);
            if(student_improvement>0):
                printInBoldGreen(f"Student improvement: {student_improvement*100}%")
            else:
                printInBoldRed(f"Student improvement: {student_improvement*100}%")
        
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
