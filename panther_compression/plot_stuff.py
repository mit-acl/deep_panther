import pandas as pd
import numpy as np
from compression.utils.load import load_student_logs, load_teacher_logs
from compression.utils.plot import plot_logs

seeds = [2] #"/home/andrea/policy_compression/results"
# Note: results generated using beta=100 and 32x32 network. 
n_rounds=10
teacher_logs = load_teacher_logs(seeds = seeds, log_prefix = "evals/log_dagger", n_rounds = n_rounds, agent_name="expert")
dagger_logs = load_student_logs(seeds = seeds, log_prefix = "evals/log_dagger", n_rounds = n_rounds, agent_name="student", n_traj_per_round=10)
# dagger_dr_logs = load_student_logs(seeds = seeds, log_prefix = "log_dagger_with_dr", n_iterations = 25, agent_name="baseline (DAgger + DR)")
# dagger_augm_logs  = load_student_logs(seeds = seeds, log_prefix = "log_dagger_augm_vertexes", n_iterations = 25, agent_name="proposed (DAgger + Augm))")
# logs = pd.concat([teacher_logs, dagger_logs, dagger_dr_logs, dagger_augm_logs], axis = 0, ignore_index=True)
logs = pd.concat([teacher_logs, dagger_logs], axis = 0, ignore_index=True)

plot_logs(logs, True)