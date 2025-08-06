# what if manipulation didnt work, people still percieve reward to come at a delay
# what would the manipulations predict in this case?
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

# %%


def gen_data_basic(states, actions, horizon, reward_unit, reward_shirk, beta,
                   discount_factor, efficacy, effort_work, n_trials,
                   states_no):
    """
    function to generate a trajectory of state and action sequences given
    parameters and reward, transition models of the basic model
    """

    # get reward function
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    # reward delivered at the end of the semester
    total_reward_func_last = task_structure.reward_final_no_thr(
        states, reward_unit, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    # get tranistions
    T = task_structure.T_binomial(states, actions, efficacy)

    # get policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # generate data - forward runs
    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, actions, initial_state,
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
REWARD_UNIT = 4  # reward per unit
BETA = 10  # softmax beta for diff-disc model
N_TRIALS = 1000  # no. of trajectories
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3

DISCOUNT_FACTOR = 0.7

# %%
# delays and completion rates with efficacy

discounts = np.linspace(0.4, 1, 10)
cmap_oranges = plt.get_cmap('Oranges')
cycle_colors = cycler('color',
                      cmap_oranges(np.linspace(0.4, 1, 2)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
fig3, ax3 = plt.subplots(figsize=(4, 3), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)
ax3.set_prop_cycle(cycle_colors)


for effort_work in [-2.5, -0.3]:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    completion_num = []
    for discount_factor in discounts:

        trajectories = gen_data_basic(
            STATES, ACTIONS, HORIZON, REWARD_UNIT, REWARD_SHIRK, BETA,
            discount_factor, EFFICACY, effort_work, N_TRIALS, STATES_NO)

        delays = helper.time_to_finish(trajectories, STATES_NO)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions, compl_number = helper.did_it_finish(
            trajectories, STATES_NO)
        completion_rate.append(np.nanmean(completions))
        completion_num.append(np.nanmean(compl_number))

    ax1.errorbar(discounts, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{effort_work}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

    ax3.plot(completion_num, linewidth=3, marker='o', linestyle='--')

    sns.despine(ax=ax1)
    ax1.set_ylabel('Avg. time to \n complete task')
    ax1.set_xlabel(r'$\gamma$')
    ax1.set_yticks([0, 5, 10, 15])
    ax1.legend(bbox_to_anchor=(0.4, 1.25), ncol=5, frameon=False, fontsize=18,
               loc='upper center', columnspacing=0.5)
    fig1.text(0.1, 1.00, 'Effort', ha='center', va='center')

    sns.despine(ax=ax2)
    ax2.set_ylabel('Completion rate', fontsize=20)
    ax2.set_xlabel(r'$\gamma$', fontsize=20)
    ax2.set_yticks([0, 1])
    ax2.set_xticks([])

    sns.despine(ax=ax3)
    ax3.set_ylabel('Avg units \n completed')
    ax3.set_xlabel(r'$\gamma$')
    ax3.set_yticks([0, 10, 20])
    ax3.set_xticks([])
    ax3.set_ylim(-1, STATES_NO)


discount_factor = 0.8
for effort_work in [-2.5, -0.3]:

    trajectories = gen_data_basic(
        STATES, ACTIONS, HORIZON, REWARD_UNIT, REWARD_SHIRK, BETA,
        discount_factor, EFFICACY, effort_work, N_TRIALS, STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    helper.plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10,
                             ylim=STATES_NO, xticks=[0, 7, 15],
                             yticks=[0, 10, 20])
    plt.title(f'effort={np.round(effort_work,2)}')
    sns.despine()
    plt.show()
