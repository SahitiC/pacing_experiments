import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
import mdp_algms
import task_structure
import helper
from cycler import cycler
import matplotlib as mpl
mpl.rcParams['font.size'] = 20
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2

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
N_TRIALS = 1000  # no. of trajectories
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

DISCOUNT_FACTOR_REWARD = 0.9
DISCOUNT_FACTOR_COST = 0.6

# %% explore param regime

rewards_unit = [0.5, 1, 1.5, 2, 4]


fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)
fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=300)


delay_mn = []
delay_sem = []
completion_rate = []
completion_num = []
for reward_unit in rewards_unit:

    trajectories = gen_data_diff_discounts(
        STATES, ACTIONS, HORIZON, reward_unit, REWARD_SHIRK, BETA,
        DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, EFFICACY, EFFORT_WORK,
        N_TRIALS, STATES_NO)

    delays = helper.time_to_finish(trajectories, STATES_NO)
    delay_mn.append(np.nanmean(delays))
    delay_sem.append(sem(delays, nan_policy='omit'))
    completions, compl_number = helper.did_it_finish(trajectories, STATES_NO)
    completion_rate.append(np.nanmean(completions))
    completion_num.append(np.nanmean(compl_number))

ax1.errorbar(rewards_unit, delay_mn, yerr=delay_sem, linewidth=3,
             marker='o', linestyle='--')

ax2.plot(rewards_unit, completion_rate, linewidth=3, marker='o',
         linestyle='--')

ax3.plot(rewards_unit, completion_num, linewidth=3, marker='o',
         linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel('reward for completion')
ax1.set_yticks([0, 5, 10, 15])
# ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=18,
#            loc='upper center', columnspacing=0.5)
ax1.set_title(
    rf'$\gamma_r$ = {DISCOUNT_FACTOR_REWARD}, $\gamma_c$={DISCOUNT_FACTOR_COST}')
ax1.set_xticks(rewards_unit)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate')
ax2.set_xlabel('reward for completion')
ax2.set_yticks([0, 1])
ax2.set_xticks(rewards_unit)

sns.despine(ax=ax3)
ax3.set_ylabel('Avg units \n completed')
ax3.set_xlabel('reward for completion')
ax3.set_yticks([0, 10, 20])
ax3.set_xticks(rewards_unit)

for reward_unit in rewards_unit:

    trajectories = gen_data_diff_discounts(
        STATES, ACTIONS, HORIZON, reward_unit, REWARD_SHIRK, BETA,
        DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, EFFICACY, EFFORT_WORK,
        N_TRIALS, STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    helper.plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10,
                             ylim=STATES_NO, xticks=[0, 7, 15],
                             yticks=[0, 10, 20])
    plt.title(f'reward={np.round(reward_unit,2)}')
    sns.despine()
    plt.show()

# %% vary discounts

cmap_blues = plt.get_cmap('Blues')
discounts_reward = [0.5, 0.7, 0.8, 0.9, 0.95]
discounts_cost = np.linspace(0.4, 1, 10)
cycle_colors = cycler('color',
                      cmap_blues(np.linspace(0.3, 1, 5)))
reward_unit = 0.5

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
fig3, ax3 = plt.subplots(figsize=(4, 3), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)
ax3.set_prop_cycle(cycle_colors)

plt.figure()
for discount_factor_reward in discounts_reward:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    completion_num = []
    for discount_factor_cost in discounts_cost:

        trajectories = gen_data_diff_discounts(
            STATES, ACTIONS, HORIZON, reward_unit, REWARD_SHIRK, BETA,
            discount_factor_reward, discount_factor_cost, EFFICACY,
            EFFORT_WORK, N_TRIALS, STATES_NO)

        delays = helper.time_to_finish(trajectories, STATES_NO)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions, compl_number = helper.did_it_finish(trajectories,
                                                         STATES_NO)
        completion_rate.append(np.nanmean(completions))
        completion_num.append(np.nanmean(compl_number))

    ax1.errorbar(discounts_cost, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor_reward}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

    ax3.plot(completion_num, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel(r'$\gamma_{c}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=5, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.08, 1.00, r'$\gamma_{r}$', ha='center', va='center')


sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate')
ax2.set_xlabel(r'$\gamma_{c}$')
ax2.set_yticks([0, 1])
ax2.set_xticks([])
ax2.set_ylim(-1, 1.5)

sns.despine(ax=ax3)
ax3.set_ylabel('Avg units \n completed')
ax3.set_xlabel(r'$\gamma_{c}$')
ax3.set_yticks([0, 10, 20])
ax3.set_xticks([])
ax3.set_ylim(-1, STATES_NO)

# %% recovery : minimum sample required to recover params and model
# fit basic model
