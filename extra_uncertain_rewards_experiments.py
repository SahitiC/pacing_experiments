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


def gen_data_no_commitment(states, actions_base, horizon, reward_unit,
                           reward_shirk, beta, p_stay_low,
                           p_stay_high, discount_factor, efficacy, effort_work,
                           reward_interest, n_trials, states_no):
    """
    function to generate a trajectory of state and action sequences given 
    parameters and reward, transition models of the no commitment model
    """

    states_no = len(states)

    # reward for completion
    reward_func_base = task_structure.reward_immediate(
        states[:int(states_no/2)], actions_base, 0, reward_unit)

    # immediate interest rewards
    reward_func_interest = task_structure.reward_immediate(
        states[:int(states_no/2)], actions_base, 0, reward_interest)

    # effort costs
    effort_func = task_structure.effort(states[:int(states_no/2)],
                                        actions_base, effort_work)

    # total reward for low reward state = reward_base + effort
    total_reward_func_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = reward_func_base[state_current] + effort_func[state_current]
        # replicate rewards for high reward states
        total_reward_func_low.append(np.block([temp, temp]))

    # total reward for high reward state = reward_base+interest rewards+effort
    total_reward_func_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = (reward_func_base[state_current]
                + reward_func_interest[state_current]
                + effort_func[state_current])
        total_reward_func_high.append(np.block([temp, temp]))

    total_reward_func = []
    total_reward_func.extend(total_reward_func_low)
    total_reward_func.extend(total_reward_func_high)

    total_reward_func_last = np.zeros(len(states))

    # tranistion matrix based on efficacy and stay-switch probabilities
    T_partial = task_structure.T_binomial(states[:int(states_no/2)],
                                          actions_base, efficacy)
    T_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([p_stay_low * T_partial[state_current],
                         (1 - p_stay_low) * T_partial[state_current]])
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_low.append(temp)

    T_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([(1 - p_stay_high) * T_partial[state_current],
                         p_stay_high * T_partial[state_current]])
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_high.append(temp)

    T = []
    T.extend(T_low)
    T.extend(T_high)

    # optimal policy based on task structure
    actions_all = actions_base.copy()
    # same actions available for low and high reward states: so repeat
    actions_all.extend(actions_base)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions_all, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            task_structure.softmax_policy, Q_values, actions_all,
            initial_state, horizon, states, T, beta)
        s_unit = np.where(s > states_no/2 - 1, s-states_no/2, s)
        data.append(s_unit.astype(int))

    return data


# %%
STATES_NO = (20+1) * 2
STATES = np.arange((20+1) * 2)
ACTIONS_BASE = [np.arange(21-i) for i in range(21)]
REWARD_UNIT = 4.0
REWARD_SHIRK = 0.1
EFFORT_WORK = -0.3
BETA = 10
HORIZON = 15  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)
N_TRIALS = 1000
INTEREST_STATES = np.array([0, 1])
P_STAY_LOW = 0.95
P_STAY_HIGH = 0.05
REWARD_INTEREST = 4.0

# %%

rewards_interest = np.linspace(0.0, 20, 10)
discounts = [0.5, 0.6, 0.8, 0.95, 1]
cmap_blues = plt.get_cmap('Blues')
cycle_colors = cycler('color',
                      cmap_blues(np.linspace(0.3, 1, 5)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
fig3, ax3 = plt.subplots(figsize=(4, 3), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)
ax3.set_prop_cycle(cycle_colors)

for discount_factor in discounts:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    completion_num = []
    for reward_interest in rewards_interest:

        trajectories = gen_data_no_commitment(
            STATES, ACTIONS_BASE, HORIZON, REWARD_UNIT, REWARD_SHIRK,
            BETA, P_STAY_LOW, P_STAY_HIGH, discount_factor, EFFICACY,
            EFFORT_WORK, reward_interest, N_TRIALS, STATES_NO)

        delays = helper.time_to_finish(trajectories, STATES_NO/2)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions, compl_number = helper.did_it_finish(trajectories,
                                                         STATES_NO/2)
        completion_rate.append(np.nanmean(completions))
        completion_num.append(np.nanmean(compl_number))

    ax1.errorbar(rewards_interest, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

    ax3.plot(completion_num, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel(r'$r_{interest}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=5, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(-0.05, 0.95, r'$\gamma$', ha='center', va='center')


sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel(r'$r_{interest}$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])

sns.despine(ax=ax3)
ax3.set_ylabel('Avg units \n completed')
ax3.set_xlabel(r'$r_{interest}$')
ax3.set_yticks([0, 10, 20])
ax3.set_xticks([])
ax3.set_ylim(-1, STATES_NO/2)


discount_factor = 0.5
for reward_interest in [0., 6.]:

    trajectories = gen_data_no_commitment(
        STATES, ACTIONS_BASE, HORIZON, REWARD_UNIT, REWARD_SHIRK,
        BETA, P_STAY_LOW, P_STAY_HIGH, discount_factor, EFFICACY,
        EFFORT_WORK, reward_interest, N_TRIALS, STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    helper.plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10,
                             ylim=STATES_NO/2, xticks=[0, 7, 15],
                             yticks=[0, 10, 20])
    plt.title(f'r_interest={np.round(reward_interest,2)}')
    sns.despine()
    plt.show()
