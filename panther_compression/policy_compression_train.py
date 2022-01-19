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
from compression.utils.train import make_dagger_trainer, make_bc_trainer, train, make_simple_dagger_trainer
from compression.utils.eval import evaluate_policy

from compression.utils.other import getNumOfEnv


from stable_baselines3.common.env_checker import check_env

#################### Coloring of the python errors, https://stackoverflow.com/a/52797444/6057617
import sys
from IPython.core import ultratb
###########################

###
from joblib import Parallel, delayed
import multiprocessing
##

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

    parser.add_argument("--n_rounds", default=5000, type=int) #was called n_iters before
    parser.add_argument("--n_evals", default=1, type=int)
    parser.add_argument("--test_environment_max_steps", default=1, type=int)
    parser.add_argument("--train_environment_max_steps", default=1, type=int)
    parser.add_argument("--use_only_last_collected_dataset", dest='use_only_last_coll_ds', action='store_true')
    parser.set_defaults(use_only_last_coll_ds=False)
    parser.add_argument("--n_traj_per_round", default=9, type=int) #This is PER environment
    # Method changes
    parser.add_argument("--no_train", dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument("--no_eval", dest='eval', action='store_false')
    parser.set_defaults(eval=False)
    parser.add_argument("--no_init_and_final_eval", dest='init_and_final_eval', action='store_false')
    parser.set_defaults(init_and_final_eval=False)
    # Dagger properties
    parser.add_argument("--rampdown_rounds", default=1e6, type=int)
    

    args = parser.parse_args()
    
    only_collect_data=False
    train_only_supervised=False
    reuse_previous_samples=False

    record_bag=False
    launch_tensorboard=True
    verbose_python_errors=False
    batch_size = 8
    N_EPOCHS = 150           #WAS 50!! Num epochs for training.
    lr=1e-3
    weight_prob=0.005
    num_envs = getNumOfEnv()


    if(only_collect_data==True):
        train_only_supervised=False
        launch_tensorboard=False

    if(train_only_supervised==True):
        reuse_previous_samples=True
        only_collect_data=False  


    if(train_only_supervised==True):
        num_envs=1
        demos_dir=args.policy_dir+"/2/demos/"

        #This places all the demos in the round-000 folder
        os.system("find "+demos_dir+" -type f -print0 | xargs -0 mv -t "+demos_dir)
        os.system("rm -rf "+demos_dir+"/round*")
        os.system("mkdir "+demos_dir+"/round-000")
        os.system("mv "+demos_dir+"/*.npz "+demos_dir+"/round-000")
        #########

        #Find max round in demos folder
        max_round = max([int(s.replace("round-", "")) for s in os.listdir(demos_dir)])

        args.n_rounds=max_round+1; #It will use the demonstrations of these folders
        args.n_traj_per_round=0

    if(args.train_environment_max_steps>1 and only_collect_data==True):
        printInBoldRed("Note that DAgger will not be used (since we are only collecting data)")

    os.system("find "+args.policy_dir+" -type f -name '*.pt' -delete") #Delete the policies
    if(reuse_previous_samples==False):
        os.system("rm -rf "+args.log_dir)
        os.system("rm -rf "+args.policy_dir)

    if(record_bag==True):
        os.system("rm training*.bag")

    if(verbose_python_errors==False):
        mode='Plain'
    else:
        mode='Verbose'

    ################## Coloring of the python errors, https://stackoverflow.com/a/52797444/6057617
    sys.excepthook = ultratb.FormattedTB(mode=mode, color_scheme='Linux', call_pdb=False)
    ####################

    def my_func(thread_count):
        time.sleep(thread_count) #To avoid the RuntimeError: CUDA error: out of memory
        printInBoldBlue(f"---------------- Thread {thread_count}: -----------------------")
        printInBoldBlue("---------------- Input Arguments: -----------------------")
        print("Trainer: {}.".format("DAgger" if args.on_policy_trainer ==True else "BC" ))
        print(f"seed: {args.seed}, log_dir: {args.log_dir}, n_rounds: {args.n_rounds}")
        print(f"test_environment_max_steps: {args.test_environment_max_steps}")
        print(f"use_only_last_coll_ds: {args.use_only_last_coll_ds}")
        print(f"train: {args.train}, eval: {args.eval}, init_and_final_eval: {args.init_and_final_eval}")
        print(f"DAgger rampdown_rounds: {args.rampdown_rounds}.")

        print(f"Num of trajectories per round: {args.n_traj_per_round}.")
        print("---------------------------------------------------------")


        #Remove previous directories
        # os.system("rm -rf "+args.log_dir)
        # os.system("rm -rf "+args.policy_dir)

        # np.set_printoptions(precision=3)

        # assert args.rampdown_rounds<=args.n_rounds, f"Are you sure you wanna this? rampdown_rounds={args.rampdown_rounds}, n_rounds={args.n_rounds}"

        assert args.eval == True or args.train == True, "eval = True or train = True!"

        DATA_POLICY_PATH = os.path.join(args.policy_dir, str(args.seed))
        LOG_PATH = os.path.join(args.log_dir, str(args.seed))
        FINAL_POLICY_NAME = "final_policy.pt"

        ENV_NAME = "my-environment-v1"

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        t0 = time.time()

        torch.manual_seed(args.seed+thread_count)
        np.random.seed(args.seed+thread_count)
        random.seed(args.seed+thread_count)

        # Create and set properties for TRAINING environment:
        printInBoldBlue("---------------- Making Environments: -------------------")
        print("[Train Env] Making training environment...")
        # train_env = gym.make(ENV_NAME)
        # train_env.seed(args.seed)
        # train_env.action_space.seed(args.seed)
        # train_env.set_len_ep(args.train_environment_max_steps) 
        # if(record_bag):
        #     train_env.startRecordBag("training.bag")
        # print(f"[Train Env] Ep. Len:  {train_env.get_len_ep()} [steps].")


        train_venv = util.make_vec_env(env_name=ENV_NAME, n_envs=num_envs, seed=args.seed, parallel=False)
        train_venv.seed(args.seed)
        train_venv.env_method("set_len_ep", (args.train_environment_max_steps)) 
        print("[Train Env] Ep. Len:  {} [steps].".format(train_venv.get_attr("len_episode")))

        if(record_bag):
            train_venv.env_method("startRecordBag", ("training"+str(thread_count)+".bag")) 


        # Create and set properties for EVALUATION environment
        # TODO: Andrea: remove venv since it is not properly used, or implement it correctly. 
        print("[Test Env] Making test environment...")
        test_venv = util.make_vec_env(env_name=ENV_NAME, n_envs=num_envs, seed=args.seed, parallel=False)
        test_venv.seed(args.seed)
        test_venv.env_method("set_len_ep", (args.test_environment_max_steps)) 
        print("[Test Env] Ep. Len:  {} [steps].".format(test_venv.get_attr("len_episode")))
        # test_venv = gym.make(ENV_NAME)
        # test_venv.seed(args.seed)
        # test_venv.action_space.seed(args.seed)
        # test_venv.set_len_ep(args.test_environment_max_steps)
        # print(f"[Train Env] Ep. Len:  {test_venv.get_len_ep()} [steps].")

        # Init logging
        tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
        tempdir_path = LOG_PATH#"evals/log_tensorboard"#LOG_PATH#pathlib.Path(tempdir.name)
        print( f"All Tensorboards and logging are being written inside {tempdir_path}/.")
        custom_logger=logger.configure(tempdir_path,  format_strs=["log", "csv", "tensorboard"])  # "stdout"

        printInBoldBlue("---------------- Making Expert Policy: --------------------")
        # Create expert policy 
        expert_policy = ExpertPolicy()

        printInBoldBlue("---------------- Making Learner Policy: -------------------")
        # Create learner policy
        if args.on_policy_trainer: 
            trainer = make_simple_dagger_trainer(tmpdir=DATA_POLICY_PATH, venv=train_venv, rampdown_rounds=args.rampdown_rounds, custom_logger=custom_logger, lr=lr, batch_size=batch_size, weight_prob=weight_prob, expert_policy=expert_policy) 
        else: 
            trainer = make_bc_trainer(tmpdir=DATA_POLICY_PATH, venv=train_venv, custom_logger=custom_logger, lr=lr, batch_size=batch_size, weight_prob=weight_prob)





        if(args.init_and_final_eval):
            printInBoldBlue("---------------- Preliminiary Evaluation: --------------------")

            test_venv.env_method("changeConstantObstacleAndGtermPos", gterm_pos=np.array([[6.0],[0.0],[1.0]]), obstacle_pos=np.array([[2.0],[0.0],[1.0]])) 

            #NOTES: args.n_evals is the number of trajectories collected in the environment
            #A trajectory is defined as a sequence of steps in the environment (until the environment returns done)
            #Hence, each trajectory usually contains the result of test_environment_max_steps timesteps (it may contains less if the environent returned done before) 
            #In other words, episodes in |evaluate_policy() is the number of trajectories
            #                            |the environment is the number of time steps

            # Evaluate the reward of the expert
            expert_stats = evaluate_policy( expert_policy, test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH+"/teacher")
            print("[Evaluation] Expert reward: {}, len: {}.\n".format( expert_stats["return_mean"], expert_stats["len_mean"]))


            # Evaluate student reward before training,
            pre_train_stats = evaluate_policy(trainer.get_policy(), test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH+"/student_pre_train")
            print("[Evaluation] Student reward: {}, len: {}.".format(pre_train_stats["return_mean"], pre_train_stats["len_mean"]))


            del expert_stats




        # Train and evaluate
        printInBoldBlue("---------------- Training Learner: --------------------")


        # # #Launch tensorboard visualization
        if(launch_tensorboard==True):
            os.system("pkill -f tensorboard")
            proc1 = subprocess.Popen(["tensorboard","--logdir",LOG_PATH,"--bind_all"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            proc2 = subprocess.Popen(["google-chrome","http://jtorde-alienware-aurora-r8:6006/"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # # os.system("tensorboard --logdir "+args.log_dir +" --bind_all")
        # # os.system("google-chrome http://jtorde-alienware-aurora-r8:6006/")  


        stats = {"training":list(), "eval_no_dist":list()}
        if args.on_policy_trainer == True:
            assert trainer.round_num == 0


        policy_path = os.path.join(DATA_POLICY_PATH, "intermediate_policy.pt") # Where to save curr policy
        trainer.train(n_rounds=args.n_rounds, n_traj_per_round=args.n_traj_per_round, only_collect_data=only_collect_data, bc_train_kwargs=dict(n_epochs=N_EPOCHS, save_full_policy_path=policy_path))


        # for i in trange(args.n_rounds, desc="Round"):

        #     #Create names for policies
        #     n_training_traj = int(i*args.n_traj_per_round) # Note: we start to count from 0. e.g. policy_0 means that we used 1 
        #     policy_path = os.path.join(DATA_POLICY_PATH, "intermediate_policy_round"+str(i)+".pt") # Where to save curr policy
        #     log_path_student_n = LOG_PATH + "/student/" + str(n_training_traj) + "/"                                  # Where to save eval logs
        #     if not os.path.exists(log_path_student_n):
        #         os.makedirs(log_path_student_n)

        #     # Train for iteration
        #     if args.train:
        #         print(f"[Collector] Collecting round {i+1}/{args.n_rounds}.")
        #         # train_stats = train(trainer=trainer, expert=expert_policy, seed=args.seed, n_traj_per_round=args.n_traj_per_round, n_epochs=N_EPOCHS, 
        #         #     log_path=os.path.join(log_path_student_n, "training"),  save_full_policy_path=policy_path, use_only_last_coll_ds=args.use_only_last_coll_ds, only_collect_data=only_collect_data)


        #     if args.eval:
        #         # Load saved policy
        #         student_policy = bc.reconstruct_policy(policy_path)

        #         # Evaluate
        #         eval_stats = evaluate_policy(student_policy, test_venv, eval_episodes=args.n_evals, log_path = log_path_student_n + "student_after_training_iteration")
        #         printInBoldGreen("[Evaluation] Iter.: {}, rwrd: {}, len: {}.\n".format(i+1, eval_stats["return_mean"], eval_stats["len_mean"]))
        #         del eval_stats


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
            post_train_stats = evaluate_policy(trainer.get_policy(), test_venv,eval_episodes=args.n_evals, log_path=LOG_PATH + "/student_post_train" )
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

    # if(only_collect_data==True):
    #     num_jobs=1#multiprocessing.cpu_count();
    #     results = Parallel(n_jobs=num_jobs)(map(delayed(my_func), list(range(num_jobs)))) #multiprocessing.cpu_count()
    # else:
    my_func(0);
