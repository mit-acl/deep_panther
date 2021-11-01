import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({
    "font.size": 11,
    "text.usetex": True,
    "font.family": "serif",
    "axes.formatter.limits": [0,3],
    "font.sans-serif": ["Helvetica"]})

ci=95

# https://seaborn.pydata.org/tutorial/relational.html#relational-tutorial
def plot_logs(input_df, show_training = False):

    # df: student logs
    if show_training:
        fig, axs = plt.subplots(3, 3, sharex=True, sharey='row', figsize=(21, 12))
    else: 
        fig, axs = plt.subplots(3, 2, sharex=True, sharey='row', figsize=(16, 12))

    eval_df = input_df.loc[input_df["training"]==False]
    training_df = input_df.loc[input_df["training"]==True]
    axs[0, 0].set_title("Source Env. (Without Disturbance)")
    sns.lineplot(ax = axs[0, 0], x="iteration", y="return", hue="agent", estimator='mean', ci=ci, data =eval_df.loc[eval_df["disturbance"] == False])
    axs[0, 0].set(xlabel='Num. demonstrations used for training', ylabel='Return')

    axs[0, 1].set_title("Target Env. (With Disturbance)")
    sns.lineplot(ax = axs[0, 1], x="iteration", y="return", hue="agent", estimator='mean', ci=ci, data =eval_df.loc[eval_df["disturbance"] == True])
    axs[0, 1].set(xlabel='Num. demonstrations used for training', ylabel='Return')

    axs[1, 0].set_title("Source Env. (Without Disturbance)")
    sns.lineplot(ax = axs[1, 0], x="iteration", y="len", hue="agent", estimator='mean', ci=ci, data =eval_df.loc[eval_df["disturbance"] == False])
    axs[1, 0].set(xlabel='Num. demonstrations used for training', ylabel='Episode Length')

    axs[1, 1].set_title("Target Env. (With Disturbance)")
    sns.lineplot(ax = axs[1, 1], x="iteration", y="len", hue="agent", estimator='mean', ci=ci, data =eval_df.loc[eval_df["disturbance"] == True])
    axs[1, 1].set(xlabel='Num. demonstrations used for training', ylabel='Episode Length')

    axs[2, 0].set_title("Source Env. (Without Disturbance)")
    sns.lineplot(ax = axs[2, 0], x="iteration", y="success", hue="agent", estimator='mean', ci=ci, data =eval_df.loc[eval_df["disturbance"] == False])
    axs[2, 0].set(xlabel='Num. demonstrations used for training', ylabel="Success Rate")

    axs[2, 1].set_title("Target Env. (With Disturbance)")
    sns.lineplot(ax = axs[2, 1], x="iteration", y="success", hue="agent", estimator='mean', ci=ci, data =eval_df.loc[eval_df["disturbance"] == True])
    axs[2, 1].set(xlabel='Num. demonstrations used for training', ylabel='Success Rate')


    if show_training: 
        axs[0, 2].set_title("Training Env.")
        sns.lineplot(ax = axs[0, 2], x="iteration", y="return", hue="agent", estimator='mean', ci=ci, data=training_df)
        axs[0, 2].set(xlabel='Num. demonstrations used for training', ylabel='Return')

        axs[1, 2].set_title("Training Env.")
        sns.lineplot(ax = axs[1, 2], x="iteration", y="len", hue="agent", estimator='mean', ci=ci, data=training_df)
        axs[1, 2].set(xlabel='Num. demonstrations used for training', ylabel='Episode Length')

        axs[2, 2].set_title("Training Env.")
        sns.lineplot(ax = axs[2, 2], x="iteration", y="success", hue="agent", estimator='mean', ci=ci, data =training_df)
        axs[2, 2].set(xlabel='Num. demonstrations used for training', ylabel="Success Rate")

def plot_lean_logs(full_df):
    # full_df: student logs during training/evaluation
    df = full_df.loc[full_df["training"]==False]
    fig, axs = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(12, 6.7))
    axs[0, 0].set_title("Source Env. (Without Disturbance)")
    sns.lineplot(ax = axs[0, 0], x="iteration", y="success", hue="agent", estimator='mean', ci=ci, data =df.loc[df["disturbance"] == False])
    axs[0, 0].set(xlabel='Num. demonstrations used for training', ylabel="Robustness (Success Rate)")

    axs[0, 1].set_title("Target Env. (With Disturbance)")
    sns.lineplot(ax = axs[0, 1], x="iteration", y="success", hue="agent", estimator='mean', ci=ci, data =df.loc[df["disturbance"] == True])
    axs[0, 1].set(xlabel='Num. demonstrations used for training', ylabel='Robustness (Success Rate)')

    axs[1, 0].set_title("Source Env. (Without Disturbance)")
    sns.lineplot(ax = axs[1, 0], x="iteration", y="return", hue="agent", estimator='mean', ci=ci, data =df.loc[df["disturbance"] == False])
    axs[1, 0].set(xlabel='Num. demonstrations used for training', ylabel='Performance (MPC Cost)')

    axs[1, 1].set_title("Target Env. (With Disturbance)")
    sns.lineplot(ax = axs[1, 1], x="iteration", y="return", hue="agent", estimator='mean', ci=ci, data =df.loc[df["disturbance"] == True])
    axs[1, 1].set(xlabel='Num. demonstrations used for training', ylabel='Performance (MPC Cost)')
    plt.tight_layout()

def plot_robustness_at_convergence(full_logs, from_iter, to_iter, fig_name="robustness_at_convergence.pdf", perturbation_name="Disturbance"):
    plot_logs = copy.deepcopy(full_logs[~full_logs["training"]])
    sns.set_palette(colors)    
    title = "Robustness Evaluation At Convergence"
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), sharey=True, sharex=True)
    sns.barplot(x="disturbance", y="success", hue="agent", data=plot_logs[(plot_logs["iteration"]>=from_iter) & (plot_logs["iteration"]<=to_iter)], ax=axs, ci=ci, alpha=1.0)
    axs.set_ylabel("Avg. Stage Cost")
    axs.set_xlabel("Environment")
    axs.set_ylim([0.9, 1.0])
    axs.set_xticklabels([f"Source Domain \n(No {perturbation_name})", f"Target Domain \n(With {perturbation_name})"])
    axs.set_title(title)
    fig.tight_layout()

def plot_perf_at_convergence(full_logs, from_iter, to_iter, fig_name="abs_perf_at_convergence.pdf", separate_x_and_u = True, perturbation_name="Disturbance"):
    eval_logs = full_logs[~full_logs["training"]]
    eval_logs = copy.deepcopy(eval_logs)
    eval_logs["return"] = -eval_logs["return"]
    plot_logs = eval_logs
    sns.set_palette(colors)    
    title = "Performance Evaluation At Convergence"
    if not separate_x_and_u: 
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), sharey=True, sharex=True)
        sns.barplot(x="disturbance", y="return", hue="agent", data=plot_logs[(plot_logs["iteration"]>=from_iter) & (plot_logs["iteration"]<=to_iter)], ax=axs, ci=ci, alpha=1.0)
        axs.set_ylabel("Avg. Stage Cost")
        axs.set_xlabel("Environment")
        axs.set_xticklabels([f"Source Domain \n(No {perturbation_name})", f"Target Domain \n(With {perturbation_name})"])
        axs.set_title(title)
        fig.tight_layout()
    else: 
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9.0, 3.5), sharey=True)
        fig.subplots_adjust(bottom=0.38)  # create some space below the plots by increasing the bottom-value
        
        sns.barplot(x="disturbance", y="stage_x", hue="agent", data=plot_logs[(plot_logs["iteration"]>=from_iter) & (plot_logs["iteration"]<=to_iter)], ax=axs[0], ci=ci, alpha=1.0)
        
        axs[0].set_ylabel("Actuation Cost")
        axs[0].set_xticklabels([f"Source Domain \n(No {perturbation_name})", f"Target Domain \n(With {perturbation_name})"])
        axs[0].set_xlabel("Environment")
        axs[0].get_legend().remove()
        sns.barplot(x="disturbance", y="stage_u", hue="agent", data=plot_logs[(plot_logs["iteration"]>=from_iter) & (plot_logs["iteration"]<=to_iter)], ax=axs[1], ci=ci, alpha=1.0)
        axs[1].set_ylabel("Tracking Error Cost")
        axs[1].set_xticklabels([f"Source Domain \n(No {perturbation_name})", f"Target Domain \n(With {perturbation_name})"])
        axs[1].set_xlabel("Environment")
        axs[1].get_legend().remove()
        fig.suptitle(title)

        #legend = axs[0].legend(frameon=True, framealpha=0.6, loc=3, )
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, borderaxespad=0.)
    plt.savefig(fig_name, dpi=1000)


def plot_normalized_perf_at_convergence(full_logs, from_iter, to_iter, fig_name="rel_perf_at_convergence.pdf", perturbation_name="Disturbance"):
    
    eval_logs = full_logs[~full_logs["training"]]
    eval_logs = copy.deepcopy(eval_logs)
    eval_logs["return"] = -eval_logs["return"]
    
    # Normalize
    expert_logs  = eval_logs[eval_logs["agent"].str.contains("expert")]
    non_expert_logs = eval_logs[~eval_logs["agent"].str.contains("expert")]

    # Normalize non-expert logs by expert logs, per iteration number and disturbance
    expert_summary = expert_logs[~expert_logs["training"]].groupby(["disturbance", "iteration"]).mean()
    expert_summary.columns = [str(col) + '_expert' for col in expert_summary.columns]
    expert_summary = expert_summary.reset_index()
    
    non_exp_summary = non_expert_logs.groupby(["agent", "disturbance", "iteration"]).mean()
    full_expert_summary = pd.DataFrame()
    for agent_name in non_expert_logs["agent"].unique():
        full_expert_summary = pd.concat([full_expert_summary, expert_summary.assign(agent=agent_name)], axis=0, ignore_index=True)
    log_summary = pd.merge(non_exp_summary, full_expert_summary, how="right", on=["agent", "disturbance", "iteration"]) # This merge looses 
    log_summary["rel_stage_x"] = np.abs(log_summary["stage_x"] - log_summary["stage_x_expert"])/log_summary["stage_x_expert"]*100.0 #/log_summary["stage_x_expert"]*100.0
    log_summary["rel_stage_u"] = np.abs(log_summary["stage_u"] - log_summary["stage_u_expert"])/log_summary["stage_u_expert"]*100.0 #/log_summary["stage_u_expert"]*100.0
    plot_logs = log_summary.reset_index()
        
    title = "Performance Difference From Expert at Convergence"
    sns.set_palette(colors[1:])
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7.0, 3.5), sharey=True)
    fig.subplots_adjust(bottom=0.42)  # create some space below the plots by increasing the bottom-value
    sns.barplot(x="disturbance", y="rel_stage_x", hue="agent", data=plot_logs[(plot_logs["iteration"]>=from_iter) & (plot_logs["iteration"]<=to_iter)], ax=axs[0], ci=ci, alpha=1.0)
    axs[0].set_ylabel("Actuation [\%]")
    axs[0].set_xticklabels([f"Source Domain \n(No {perturbation_name})", f"Target Domain \n(With {perturbation_name})"])
    axs[0].set_xlabel("Environment")
    axs[0].set_yscale('log')
    axs[0].get_legend().remove()

    sns.barplot(x="disturbance", y="rel_stage_u", hue="agent", data=plot_logs[(plot_logs["iteration"]>=from_iter) & (plot_logs["iteration"]<=to_iter)], ax=axs[1], ci=ci, alpha=1.0)
    axs[1].set_ylabel("Tracking Error [\%]")
    axs[1].set_xticklabels([f"Source Domain \n(No {perturbation_name})", f"Target Domain \n(With {perturbation_name})"])
    axs[1].set_xlabel("Environment")
    axs[1].get_legend().remove()
    axs[1].set_yscale('log')
    fig.suptitle(title)

    #legend = axs[0].legend(frameon=True, framealpha=0.6, loc=3, )
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, borderaxespad=0.)
    plt.savefig(fig_name, dpi=1000)


def plot_safety_during_training(full_logs, fig_name="training_success_rate.pdf"):
    training_logs = full_logs[full_logs["training"]==True]
    sns.set_palette(colors[1:])
    fig, axs = plt.subplots(1, 1, figsize=(5.0, 3.5))
    sns.barplot(y = "agent", x = "success", data =training_logs, ax=axs, estimator=lambda x: sum(x==1)*1.0/len(x)*100)
    axs.set_xlabel("Safe Trajectory [\%]")
    axs.set_ylabel("Agent")
    #axs.set_xticklabels(["Source Domain \n(No Disturbance)", "Target Domain \n(With Disturbance)"])
    axs.set_title("Safety During Training")

    fig.tight_layout()
    plt.savefig(fig_name, dpi=1000)

def plot_robustness(logs, figname="robustness.pdf", perturbation_name="Disturbance"):
    logs = copy.deepcopy(logs[~logs["training"]])
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10.5,3.8))
    fig.subplots_adjust(bottom=0.32)
    sns.set_palette(colors)
    #plt.rcParams.update({'font.size': 11})
    axs[0].set_title(f"Source Domain (No {perturbation_name})")
    sns.lineplot(ax = axs[0], x="iteration", y="success", hue="agent", estimator='mean', ci=ci, data =logs.loc[logs["disturbance"] == False])
    axs[0].set(xlabel='Num. Demonstrations Used for Training', ylabel="Success Rate")
    axs[0].get_legend().remove()
    axs[0].set_xlim(left=0.0)
    axs[0].set_ylim(bottom=0.0)

    axs[1].set_title(f"Target Domain (With {perturbation_name})")
    sns.lineplot(ax = axs[1], x="iteration", y="success", hue="agent", estimator='mean', ci=ci, data =logs.loc[logs["disturbance"] == True])
    axs[1].set(xlabel='Num. Demonstrations Used for Training', ylabel='Success Rate')
    axs[1].get_legend().remove()
    axs[1].set_xlim(left=0.0)
    axs[1].set_ylim(bottom=0.0)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, borderaxespad=0.)
    #plt.tight_layout()
    plt.savefig(figname, dpi=1000)

def plot_performance(logs, figname="performance.pdf", perturbation_name="Disturbance"):
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10.5,3.8))
    fig.subplots_adjust(bottom=0.32)
    sns.set_palette(colors)
    logs = copy.deepcopy(logs[~logs["training"]])
    #plt.rcParams.update({'font.size': 11})
    axs[0].set_title(f"Source Domain (No {perturbation_name})")
    sns.lineplot(ax = axs[0], x="iteration", y="return", hue="agent", estimator='mean', ci=ci, data =logs.loc[logs["disturbance"] == False])
    axs[0].set(xlabel='Num. Demonstrations Used for Training', ylabel="MPC Stage Cost")
    axs[0].get_legend().remove()
    axs[0].set_xlim(left=0.0)
    #axs[0].ticklabel_format(axis='y',useMathText=True)
    #axs[0].set_ylim(bottom=0.0)

    axs[1].set_title(f"Target Domain (With {perturbation_name})")
    sns.lineplot(ax = axs[1], x="iteration", y="return", hue="agent", estimator='mean', ci=ci, data =logs.loc[logs["disturbance"] == True])
    axs[1].set(xlabel='Num. Demonstrations Used for Training', ylabel='MPC Stage Cost')
    axs[1].get_legend().remove()
    axs[1].set_xlim(left=0.0)
    #axs[1].ticklabel_format(axis='y',scilimits=(0,1))
    #axs[1].set_ylim(bottom=0.0)
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, borderaxespad=0.)
    #plt.tight_layout()
    plt.savefig(figname, dpi=1000)