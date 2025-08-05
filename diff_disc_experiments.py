import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import mdp_algms
import task_structure
import helper

# %% functions


def gen_data_diff_discounts(states, actions, horizon, reward_unit,
                            reward_shirk, beta, discount_factor_reward,
                            discount_factor_cost, efficacy, effort_work,
                            n_trials, states_no):
    """
    function to generate a trajectory of state and action sequences given
    parameters and reward, transition models of the diff-disc model
    """

    reward_func = task_structure.reward_immediate(
        states, actions, reward_shirk, reward_unit)

    effort_func = task_structure.effort(states, actions, effort_work)

    reward_func_last = np.zeros(len(states))
    effort_func_last = np.zeros(len(states))

    T = task_structure.T_binomial(states, actions, efficacy)

    V_opt_full, policy_opt_full, Q_values_full = (
        mdp_algms.find_optimal_policy_diff_discount_factors(
            states, actions, horizon, discount_factor_reward,
            discount_factor_cost, reward_func, effort_func, reward_func_last,
            effort_func_last, T)
    )

    # effective Q_values for the agent
    effective_Q = []
    for i_s in range(len(states)):
        Q_s_temp = []
        for i in range(horizon):
            Q_s_temp.append(Q_values_full[horizon-1-i][i_s][:, i])
        effective_Q.append(np.array(Q_s_temp).T)

    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, effective_Q, actions, initial_state,
            horizon, states, T, beta)
        data.append(s)

    return data

# %% constants


STATES_NO = 20+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)
# actions = no. of units to complete in each state
# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]
HORIZON = 15  # no. of weeks for task
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)
REWARD_UNIT = 1  # reward per unit
BETA = 10  # softmax beta for diff-disc model
N_TRIALS = 20  # no. of trajectories
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

DISOCUNT_FACTOR_REWARD = 0.9
DISCOUNT_FACTOR_COST = 0.6

# %% explore param regime

rewards_unit = [0.5, 1, 1.5, 2, 2.5]


fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)

plt.figure()

delay_mn = []
delay_sem = []
completion_rate = []
for reward_unit in rewards_unit:

    trajectories = gen_data_diff_discounts(
        STATES, ACTIONS, HORIZON, reward_unit, REWARD_SHIRK, BETA,
        DISOCUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, EFFICACY, EFFORT_WORK,
        N_TRIALS, STATES_NO)

    delays = helper.time_to_finish(trajectories, STATES_NO)
    delay_mn.append(np.nanmean(delays))
    delay_sem.append(sem(delays, nan_policy='omit'))
    completions = helper.did_it_finish(trajectories, STATES_NO)
    completion_rate.append(np.nanmean(completions))

ax1.errorbar(rewards_unit, delay_mn, yerr=delay_sem, linewidth=3,
             marker='o', linestyle='--')

ax2.plot(rewards_unit, completion_rate, linewidth=3, marker='o',
         linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel('reward for completion')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
ax1.set_title(r'$\gamma_r$ = 0.9, $\gamma_c$=0.6')
ax1.set_xticks(rewards_unit)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel('reward for completion', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks(rewards_unit)

for reward_unit in rewards_unit:

    trajectories = gen_data_diff_discounts(
        STATES, ACTIONS, HORIZON, reward_unit, REWARD_SHIRK, BETA,
        DISOCUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, EFFICACY, EFFORT_WORK,
        N_TRIALS, STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    helper.plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
    plt.title(f'reward={np.round(reward_unit,2)}',
              fontsize=24)
    sns.despine()
    plt.show()

# %% experimental manipulation
