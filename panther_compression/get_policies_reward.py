
import os
from colorama import init, Fore, Back, Style
from imitation.algorithms import bc
from compression.utils.eval import evaluate_policy, evaluate_policy_for_benchmark
from compression.policies.ExpertPolicy import ExpertPolicy
from imitation.util import util, logger

def printInBoldBlue(data_string):
    print(Style.BRIGHT+Fore.BLUE+data_string+Style.RESET_ALL)
def printInBoldRed(data_string):
    print(Style.BRIGHT+Fore.RED+data_string+Style.RESET_ALL)
def printInBoldGreen(data_string):
    print(Style.BRIGHT+Fore.GREEN+data_string+Style.RESET_ALL)

##
## Build test environment
##

ENV_NAME = "my-environment-v1"
num_envs = 16
seed = 2
test_environment_max_steps = 50

test_venv = util.make_vec_env(env_name=ENV_NAME, n_envs=num_envs, seed=seed, parallel=False) #Note that parallel applies to the environment step, not to the expert step
test_venv.seed(seed)
test_venv.env_method("set_len_ep", (test_environment_max_steps)) 

##
## Load and evaluate the student policy
##

printInBoldRed("----------------------- Student Evaluation: --------------------")

# DATA_POLICY_PATH = "evals/tmp_dagger/2"
DATA_POLICY_PATH = "trained_policies/policies"
FINAL_POLICY_NAME = "test4.pt"
n_demos = 100

load_full_policy_path = os.path.join(DATA_POLICY_PATH, FINAL_POLICY_NAME)
student_policy = bc.reconstruct_policy(load_full_policy_path)
student_stats = evaluate_policy_for_benchmark(student_policy, test_venv, eval_episodes=n_demos, log_path=None)

##
## Evaluate the reward of the expert
##

printInBoldRed("----------------------- Expert Evaluation: --------------------")

expert_policy = ExpertPolicy()
expert_stats = evaluate_policy_for_benchmark(expert_policy, test_venv, eval_episodes=n_demos, log_path=None)

##
## Print
##

print("\n")
printInBoldRed("----------------------- TEST RESULTS: --------------------")
print("\n")

string = "{}:\n Avg. Cost: {},\n Computation Time: {} ms, \n Success Rate: {}%,\n Obst. Avoidance Failure Rate: {}%,\n Trans Dyn. Limit Failure Rate: {}%,\n Yaw Dyn. Limit Failure Rate: {}%\n"

print(string.format( "Student", -round(student_stats["return_mean"],2), round(student_stats["mean_computation_time"],4), round(student_stats["success_rate"],2), round(student_stats["obs_avoidance_failure_rate"],2), round(student_stats["trans_dyn_limit_failure_rate"],2), round(student_stats["yaw_dyn_limit_failure_rate"],2)))
print(string.format( "Expert", -round(expert_stats["return_mean"],2), round(expert_stats["mean_computation_time"],2), round(expert_stats["success_rate"],2), round(expert_stats["obs_avoidance_failure_rate"],2), round(expert_stats["trans_dyn_limit_failure_rate"],2), round(expert_stats["yaw_dyn_limit_failure_rate"],2)))