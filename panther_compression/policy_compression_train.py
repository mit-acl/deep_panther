
##
##  This file trains and evaluates student policy  
##

import matplotlib.pyplot as plt
import sys
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
import subprocess
from colorama import init, Fore, Back, Style
from imitation.policies import serialize
from imitation.util import util, logger
from imitation.algorithms import bc
from compression.policies.ExpertPolicy import ExpertPolicy
from compression.utils.train import make_dagger_trainer, make_bc_trainer, make_simple_dagger_trainer
from compression.utils.eval import evaluate_policy
from compression.utils.other import readPANTHERparams
from stable_baselines3.common.env_checker import check_env
from joblib import Parallel, delayed
import multiprocessing
from IPython.core import ultratb

##
## Coloring of the python errors, https://stackoverflow.com/a/52797444/6057617
##


def printInBoldBlue(data_string):
    print(Style.BRIGHT+Fore.BLUE+data_string+Style.RESET_ALL)
def printInBoldRed(data_string):
    print(Style.BRIGHT+Fore.RED+data_string+Style.RESET_ALL)
def printInBoldGreen(data_string):
    print(Style.BRIGHT+Fore.GREEN+data_string+Style.RESET_ALL)

##
## See https://stackoverflow.com/a/43357954/6057617
##

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

##
## https://stackoverflow.com/a/16801605/6057617
##

def single_true(iterable):
    i = iter(iterable)
    return any(i) and not any(i)

##
## Main
##

if __name__ == "__main__":

    ###
    ### Parameters
    ###

    use_test_run_params = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=str, default="")
    args = parser.parse_args()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--log_dir", type=str, default=args.home_dir+"evals/log_dagger") # usually "log"
    parser.add_argument("--policy_dir", type=str, default=args.home_dir+"evals/tmp_dagger") # usually "tmp"
    parser.add_argument("--evaluation_data_dir", type=str, default=args.home_dir+"evals/evalations") # usually "tmp"
    parser.add_argument("--planner-params", type=str) # Contains details on the tasks to be learnt (ref. trajectories)
    parser.add_argument("--use-DAgger", dest='on_policy_trainer', action='store_true') # Use DAgger when true, BC when false
    parser.add_argument("--use-BC", dest='on_policy_trainer', action='store_false')
    parser.set_defaults(on_policy_trainer=True) # Default will be to use DAgger
    if use_test_run_params:
        parser.add_argument("--n_rounds", default=1, type=int) 
        parser.add_argument("--total_demos_per_round", default=10, type=int)
    else:
        parser.add_argument("--n_rounds", default=50, type=int)
        parser.add_argument("--total_demos_per_round", default=256*5, type=int) 
    parser.add_argument("--rampdown_rounds", default=5, type=int) # Dagger properties
    parser.add_argument("--n_evals", default=100, type=int)
    parser.add_argument("--train_environment_max_steps", default=100, type=int)
    parser.add_argument("--test_environment_max_steps", default=50, type=int)
    
    ##
    ## Method changes
    ##

    parser.add_argument("--train", dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument("--no_eval", dest='eval', action='store_false')
    parser.set_defaults(eval=False)
    parser.add_argument("--init_eval", dest='init_eval', action='store_false')
    parser.set_defaults(init_eval=False)
    parser.add_argument("--final_eval", dest='final_eval', action='store_true')
    parser.set_defaults(final_eval=True)
    parser.add_argument("--use_only_last_collected_dataset", dest='use_only_last_coll_ds', action='store_true')
    parser.set_defaults(use_only_last_coll_ds=False)
    parser.add_argument("--evaluation_data_collection", dest='evaluation_data_collection', action='store_true')
    parser.set_defaults(evaluation_data_collection=True)

    ##
    ## Loss calculation
    ##

    parser.add_argument("--only_test_loss", type=str2bool, default=False)
    parser.add_argument("--type_loss", type=str, default="Hung") 

    ##
    ## NN
    ##

    parser.add_argument("--epsilon_RWTA", type=float, default=0.05)
    net_arch = [1024, 1024, 1024, 1024]

    args = parser.parse_args()

    ##
    ## Print Loss params info
    ##

    printInBoldBlue("----------------------- Loss Params Info: -----------------------")
    printInBoldRed(f"only_test_loss={args.only_test_loss}")
    printInBoldRed(f"type_loss={args.type_loss}")
    printInBoldRed(f"epsilon_RWTA={args.epsilon_RWTA}")

    ##
    ## Parameters
    ##

    params=readPANTHERparams()

    # use one zero beta in DAagger? if False, it will be LinearBetaSchedule()
    use_one_zero_beta = False

    # when you want to collect data and not train student
    only_collect_data = True

    # when you want to train student only from existing data
    train_only_supervised = False

    # use the existing data?
    reuse_previous_samples = False

    # reuse the latest_policy?
    reuse_latest_policy = False

    # record bags?
    record_bag = True

    # use tensorboard?
    launch_tensorboard=False

    # verbose python errors?
    verbose_python_errors=False

    # batch size
    batch_size = 256 if not use_test_run_params else 5

    # evaluation batch size
    evaluation_data_size = 100 if not use_test_run_params else 1

    # reset evaluation data
    reset_evaluation_data = True

    # epoch size
    N_EPOCHS = 50

    # use learning rate schedule?
    use_lr_scheduler = True

    # constant learning rate (if use_lr_scheduler is False)
    lr=1e-3

    # probably not used
    weight_prob=0.005

    # number of environments
    num_envs = 1

    # log stats after every log_interval batches.
    log_interval=200

    assert args.total_demos_per_round>=batch_size #If not, round_{k+1} will train on the same dataset as round_{k} (until enough rounds are taken to generate a new batch of demos)

    ##
    ## 18.337 specific parameters
    ##

    # do performance review?
    do_performance_review = True


    os.system("rm -rf "+args.log_dir) #Delete the logs
    
    if(only_collect_data==True):
        train_only_supervised=False
        launch_tensorboard=False

    if reset_evaluation_data:
        evals_dir=args.evaluation_data_dir+"/2/demos"
        os.system("rm -rf "+evals_dir+"/round*")
        os.system("mkdir "+evals_dir+"/round-000")

    if(train_only_supervised==True):
        reuse_previous_samples=True
        only_collect_data=False 
        log_interval=15 
        num_envs=1
        demos_dir=args.policy_dir+"/2/demos/"

        ##
        ## This places all the demos in the round-000 folder
        ##

        os.system("find "+demos_dir+" -type f -print0 | xargs -0 mv -t "+demos_dir)
        os.system("rm -rf "+demos_dir+"/round*")
        os.system("mkdir "+demos_dir+"/round-000")
        os.system("mv "+demos_dir+"/*.npz "+demos_dir+"/round-000")

        ##
        ## Find max round in demos folder
        ##

        max_round = max([int(s.replace("round-", "")) for s in os.listdir(demos_dir)])
        args.n_rounds=max_round+1; #It will use the demonstrations of these folders
        args.total_demos_per_round=0

    if(args.only_test_loss):
        batch_size=1
        N_EPOCHS=1
        log_interval=1

    if(args.train_environment_max_steps>1 and only_collect_data==True):
        printInBoldRed("Note that DAgger will not be used (since we are only collecting data)")

    if(args.only_test_loss==False and reuse_latest_policy == False):
        os.system("find "+args.policy_dir+" -type f -name '*.pt' -delete") #Delete the policies

    if(reuse_previous_samples==False):
        os.system("rm -rf "+args.policy_dir) #Delete the demos

    if(record_bag==True):
        os.system("rm training*.bag")

    if(verbose_python_errors==False):
        mode='Plain'
    else:
        mode='Verbose'

    ##
    ## Coloring of the python errors, https://stackoverflow.com/a/52797444/6057617
    ##

    sys.excepthook = ultratb.FormattedTB(mode=mode, color_scheme='Linux', call_pdb=False)

    ##
    ## my_func defintion
    ##

    def my_func(thread_count):

        ##
        ## To avoid the RuntimeError: CUDA error: out of memory
        ##

        time.sleep(thread_count)

        ##
        ## Print
        ##

        printInBoldBlue("----------------------- Input Arguments: -----------------------")
        print("Trainer: {}.".format("DAgger" if args.on_policy_trainer ==True else "BC" ))
        print(f"seed: {args.seed}, log_dir: {args.log_dir}, n_rounds: {args.n_rounds}")
        print(f"train_environment_max_steps: {args.train_environment_max_steps}")
        print(f"test_environment_max_steps: {args.test_environment_max_steps}")
        print(f"use_only_last_coll_ds: {args.use_only_last_coll_ds}")
        print(f"DAgger rampdown_rounds: {args.rampdown_rounds}.")
        print(f"total_demos_per_round: {args.total_demos_per_round}.")

        # assert args.eval == True or args.train == True, "eval = True or train = True!"

        ##
        ## Params
        ##

        DATA_POLICY_PATH = os.path.join(args.policy_dir, str(args.seed))
        EVALUATION_DATA_POLICY_PATH = os.path.join(args.evaluation_data_dir, str(args.seed))
        LOG_PATH = os.path.join(args.log_dir, str(args.seed))
        FINAL_POLICY_NAME = "final_policy.pt"
        ENV_NAME = "my-environment-v1"

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)

        if not os.path.exists(EVALUATION_DATA_POLICY_PATH):
            os.makedirs(EVALUATION_DATA_POLICY_PATH)

        t0 = time.time()

        ##
        ## Seeds
        ##

        torch.manual_seed(args.seed+thread_count)
        np.random.seed(args.seed+thread_count)
        random.seed(args.seed+thread_count)

        ##
        ## Create and set properties for TRAINING environment:
        ##

        printInBoldBlue("----------------------- Making Environments: -------------------")
        train_venv = util.make_vec_env(env_name=ENV_NAME, n_envs=num_envs, seed=args.seed, parallel=False)#Note that parallel applies to the environment step, not to the expert step
        train_venv.seed(args.seed)
        train_venv.env_method("set_len_ep", (args.train_environment_max_steps))
        print("[Train Env] Ep. Len:  {} [steps].".format(train_venv.get_attr("len_episode")))

        for i in range(num_envs):
            train_venv.env_method("setID", i, indices=[i]) 

        if(record_bag):
            for i in range(num_envs):
                train_venv.env_method("startRecordBag", indices=[i]) 

        if (args.init_eval or args.final_eval or args.eval):
            # Create and set properties for EVALUATION environment
            print("[Test Env] Making test environment...")
            test_venv = util.make_vec_env(env_name=ENV_NAME, n_envs=num_envs, seed=args.seed, parallel=False)#Note that parallel applies to the environment step, not to the expert step
            test_venv.seed(args.seed)
            test_venv.env_method("set_len_ep", (args.test_environment_max_steps)) 
            print("[Test Env] Ep. Len:  {} [steps].".format(test_venv.get_attr("len_episode")))

            for i in range(num_envs):
                test_venv.env_method("setID", i, indices=[i]) 

        # Init logging
        tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
        tempdir_path = LOG_PATH#"evals/log_tensorboard"#LOG_PATH#pathlib.Path(tempdir.name)
        print( f"All Tensorboards and logging are being written inside {tempdir_path}/.")
        custom_logger=logger.configure(tempdir_path,  format_strs=["log", "csv", "tensorboard"])

        ##
        ## Create expert policy 
        ##

        printInBoldBlue("----------------------- Making Expert Policy: --------------------")
        expert_policy = ExpertPolicy()

        ##
        ## Create student policy
        ##

        printInBoldBlue("----------------------- Making Student Policy: -------------------")
        trainer = make_simple_dagger_trainer(tmpdir=DATA_POLICY_PATH, eval_dir=EVALUATION_DATA_POLICY_PATH, venv=train_venv, rampdown_rounds=args.rampdown_rounds, custom_logger=custom_logger, lr=lr, use_lr_scheduler=use_lr_scheduler, batch_size=batch_size, \
            evaluation_data_size=evaluation_data_size, weight_prob=weight_prob, expert_policy=expert_policy, type_loss=args.type_loss, only_test_loss=args.only_test_loss, epsilon_RWTA=args.epsilon_RWTA, net_arch=net_arch, reuse_latest_policy=reuse_latest_policy, use_lstm=params["use_lstm"], use_one_zero_beta=use_one_zero_beta)

        ##
        ## Create policy for evaluation data set
        ##

        printInBoldBlue("----------------------- Making Policy for Evaluation: -------------------")
        evaluation_trainer = make_simple_dagger_trainer(tmpdir=EVALUATION_DATA_POLICY_PATH, eval_dir=EVALUATION_DATA_POLICY_PATH, venv=train_venv, rampdown_rounds=args.rampdown_rounds, 
                                                        custom_logger=None, lr=lr, use_lr_scheduler=use_lr_scheduler, batch_size=batch_size, 
                                                            evaluation_data_size=evaluation_data_size, weight_prob=weight_prob, expert_policy=expert_policy, 
                                                            type_loss=args.type_loss, only_test_loss=args.only_test_loss, epsilon_RWTA=args.epsilon_RWTA, net_arch=net_arch, 
                                                            reuse_latest_policy=reuse_latest_policy, use_lstm=params["use_lstm"], use_one_zero_beta=True)

        ##
        ## Collect evaluation data
        ##

        if args.evaluation_data_collection:
            printInBoldBlue("----------------------- Collecting Evaluation Data: --------------------")
            evaluation_policy_path = os.path.join(EVALUATION_DATA_POLICY_PATH, "evaluation_policy.pt") # Where to save curr policy
            evaluation_trainer.train(n_rounds=1, total_demos_per_round=evaluation_data_size, only_collect_data=True, 
                                     bc_train_kwargs=dict(n_epochs=N_EPOCHS, save_full_policy_path=evaluation_policy_path, 
                                                          log_interval=log_interval))
            
        ##
        ## Preliminiary evaluation
        ##

        if not do_performance_review:

            if args.init_eval:
                printInBoldBlue("----------------------- Preliminary Evaluation: --------------------")

                test_venv.env_method("changeConstantObstacleAndGtermPos", gterm_pos=np.array([[6.0],[0.0],[1.0]]), obstacle_pos=np.array([[2.0],[0.0],[1.0]])) 

                #NOTES: args.n_evals is the number of trajectories collected in the environment
                #A trajectory is defined as a sequence of steps in the environment (until the environment returns done)
                #Hence, each trajectory usually contains the result of test_environment_max_steps timesteps (it may contains less if the environent returned done before) 
                #In other words, episodes in |evaluate_policy() is the number of trajectories
                #                            |the environment is the number of time steps

                ##
                ## Evaluate the reward of the expert
                ##

                expert_stats = evaluate_policy(expert_policy, test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH+"/teacher")
                print("[Evaluation] Expert reward: {}, len: {}.\n".format( expert_stats["return_mean"], expert_stats["len_mean"]))

                ##
                ## Evaluate student reward before training,
                ##

                pre_train_stats = evaluate_policy(trainer.policy, test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH+"/student_pre_train")
                print("[Evaluation] Student reward: {}, len: {}.".format(pre_train_stats["return_mean"], pre_train_stats["len_mean"]))

                del expert_stats


        ##
        ## Train and evaluate
        ##

        printInBoldBlue("----------------------- Training Student: --------------------")

        ##
        ## Launch tensorboard visualization
        ##

        if(launch_tensorboard==True):
            os.system("pkill -f tensorboard")
            proc1 = subprocess.Popen(["tensorboard","--logdir",LOG_PATH,"--bind_all"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        ##
        ## Train
        ##

        if not do_performance_review:
            if args.train:
                stats = {"training":list(), "eval_no_dist":list()}
                if args.on_policy_trainer == True:
                    assert trainer.round_num == 0

                policy_path = os.path.join(DATA_POLICY_PATH, "intermediate_policy.pt") # Where to save curr policy
                trainer.train(n_rounds=args.n_rounds, total_demos_per_round=args.total_demos_per_round, only_collect_data=only_collect_data, bc_train_kwargs=dict(n_epochs=N_EPOCHS, save_full_policy_path=policy_path, log_interval=log_interval))
        else:
            stats = {"training":list(), "eval_no_dist":list()}
            if args.on_policy_trainer == True:
                assert trainer.round_num == 0

            elapsed_times = []
            for i in range(0, 2):
                policy_path = os.path.join(DATA_POLICY_PATH, "intermediate_policy.pt") # Where to save curr policy
                start_time = time.time()
                trainer.train(n_rounds=args.n_rounds, total_demos_per_round=10, only_collect_data=only_collect_data, bc_train_kwargs=dict(n_epochs=N_EPOCHS, save_full_policy_path=policy_path, log_interval=log_interval))
                end_time = time.time()
                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
            
            print("Average time: {}".format(np.mean(elapsed_times)))
            # exit()
        # ##
        # ## Store the final policy.
        # ##

        # if args.train:
        #     save_full_policy_path = os.path.join(DATA_POLICY_PATH, FINAL_POLICY_NAME)
        #     trainer.save_policy(save_full_policy_path)
        #     print(f"[Trainer] Training completed. Policy saved to: {save_full_policy_path}.")

        # ##
        # ## Evaluation 
        # ##

        # if args.final_eval:
        #     printInBoldBlue("----------------------- Evaluation After Training: --------------------")

        #     ##
        #     ## Evaluate reward of student post-training
        #     ##

        #     post_train_stats = dict()

        #     ##
        #     ## Note: no disturbance
        #     ##

        #     ##
        #     ## Print
        #     ##

        #     if args.init_eval:

        #         post_train_stats = evaluate_policy(trainer.policy, test_venv, eval_episodes=args.n_evals, log_path=LOG_PATH + "/student_post_train" )
        #         print("[Complete] Reward: Pre: {}, Post: {}.".format( pre_train_stats["return_mean"], post_train_stats["return_mean"]))

        #         printInBoldBlue("----------------------- Improvement: --------------------")

        #         if(abs(pre_train_stats["return_mean"])>0):
        #             student_improvement=(post_train_stats["return_mean"]-pre_train_stats["return_mean"])/abs(pre_train_stats["return_mean"])
        #             if(student_improvement>0):
        #                 printInBoldGreen(f"Student improvement: {student_improvement*100}%")
        #             else:
        #                 printInBoldRed(f"Student improvement: {student_improvement*100}%")
                
        #         print("[Complete] Episode length: Pre: {}, Post: {}.".format( pre_train_stats["len_mean"], post_train_stats["len_mean"]))

        #         ##
        #         ## Clean up
        #         ##

        #         del pre_train_stats, post_train_stats

        #     ##
        #     ## Load and evaluate the saved DAgger policy
        #     ##

        #     load_full_policy_path = os.path.join(DATA_POLICY_PATH, FINAL_POLICY_NAME)
        #     final_student_policy = bc.reconstruct_policy(load_full_policy_path)
        #     rwrd = evaluate_policy(final_student_policy, test_venv, eval_episodes=args.n_evals, log_path=None)

        #     ##
        #     ## Evaluate the reward of the expert as a sanity check
        #     ##

        #     expert_reward = evaluate_policy(expert_policy, test_venv, eval_episodes=args.n_evals, log_path=None)

        #     ##
        #     ## Print
        #     ##

        #     print("\n")
        #     printInBoldRed("----------------------- TEST RESULTS: --------------------")
        #     print("\n")
            
        #     print("[Test] Student Policy: Avg. Cost: {}, Success Rate: {}".format(-rwrd["return_mean"], rwrd["success_rate"]))
        #     print("[Test] Expert Policy: Avg. Cost: {}, Success Rate: {}".format(-expert_reward["return_mean"], expert_reward["success_rate"]))

        #     ##
        #     ## Plot
        #     ##

        #     #fig, ax = plt.subplots()
        #     #plot_train_traj(ax, stats["training"])

        # print("Elapsed time: {}".format(time.time() - t0))

    my_func(0)
