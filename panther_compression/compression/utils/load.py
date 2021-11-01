import os
import copy
import pandas as pd
import numpy as np

def load_student_logs(seeds, log_prefix, n_iterations, agent_name="student", n_traj_per_iter = 1):
    # agent_name: change according to type of algorithm. E.g. Dagger, Augmented Dagger, ...
    # Load student during training
    student_eval = []
    for it in range(n_iterations):
        it = it*n_traj_per_iter
        logs_at_iter = []
        for seed in seeds:  
            path_to_file = os.path.join(log_prefix, str(seed), "student", str(it))
            logs_at_iter.append(pd.read_pickle(os.path.join(path_to_file, "no_dist.pkl")).assign(training = False))
            logs_at_iter.append(pd.read_pickle(os.path.join(path_to_file, "with_dist.pkl")).assign(training = False))
            logs_at_iter.append(pd.read_pickle(os.path.join(path_to_file, "training.pkl")).assign(training = True))
        student_eval.append(pd.concat(logs_at_iter, axis=0, ignore_index=True).assign(iteration = it+1))
    student_eval_logs = pd.concat(student_eval, axis=0, ignore_index=True)

    # Load student before training
    student_before = []
    for seed in seeds:
        student_before.append(pd.read_pickle(os.path.join(log_prefix, str(seed), "pre_train_no_dist.pkl")).assign(training = False))
        student_before.append(pd.read_pickle(os.path.join(log_prefix, str(seed), "pre_train_with_dist.pkl")).assign(training = False))
    student_before_logs = pd.concat(student_before, axis=0, ignore_index=True).assign(iteration = 0)

    # Combine student before and during training
    student_logs = pd.concat([student_eval_logs, student_before_logs], axis=0, ignore_index=True)
    student_logs["success"] = student_logs["success"].astype(float)
    student_logs = student_logs.assign(agent=agent_name)
    return student_logs

def load_teacher_logs(seeds, log_prefix, n_iterations, agent_name="teacher"):
    # Load teacher 
    teacher = []
    for seed in seeds: 
        teacher.append(pd.read_pickle(os.path.join(log_prefix, str(seed), "teacher_no_dist.pkl")).assign(training = False))
        teacher.append(pd.read_pickle(os.path.join(log_prefix, str(seed), "teacher_with_dist.pkl")).assign(training = False))
    teacher_logs = pd.concat(teacher, axis=0, ignore_index=True)    
    # Repeat these numbers for every iteration (is there a better way?)
    all_teacher_logs = []
    for it in range(n_iterations+1):
        teacher_logs_copy = copy.deepcopy(teacher_logs)
        all_teacher_logs.append(teacher_logs_copy.assign(iteration=it))
    teacher_logs = pd.concat(all_teacher_logs, axis=0, ignore_index=True)

    teacher_logs["success"] = teacher_logs["success"].astype(float)
    teacher_logs = teacher_logs.assign(agent=agent_name)
    return teacher_logs