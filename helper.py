import numpy as np
import matplotlib.pyplot as plt
import task_structure


def time_to_finish(trajectories, states_no):
    """
    find when all units arre completed for each trajectory
    (of work) inputted; if threshold is never reached, returns NaN
    """

    times = []
    trajectories = np.array(trajectories)

    for trajectory in trajectories[:, 1:]:

        if trajectory[-1] == states_no-1:
            times.append(np.where(trajectory >= states_no-1)[0][0])
        else:
            times.append(np.nan)

    return times


def did_it_finish(trajectories, states_no):
    """
    find if all units have been completed for each trajectory inputted
    """

    completed = []
    number_completed = []
    trajectories = np.array(trajectories)

    for trajectory in trajectories[:, 1:]:

        number_completed.append(trajectory[-1])
        if trajectory[-1] == states_no-1:
            completed.append(1)
        else:
            completed.append(0)

    return completed, number_completed


def plot_trajectories(trajectories, color, lwidth_mean, lwidth_sample,
                      number_samples, ylim, xticks=[], yticks=[], ):
    """
    plot input trajectories
    """
    mean = np.mean(trajectories, axis=0)

    plt.plot(mean, color=color, linewidth=lwidth_mean)
    for i in range(number_samples):
        plt.plot(trajectories[i], color=color,
                 linewidth=lwidth_sample, linestyle='dashed')
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.ylim(-1, ylim)
    plt.xlabel('time')
    plt.ylabel('Units of work \n completed')


def calculate_likelihood(data, Q_values, beta, T, actions):
    """
    calculate likelihood of data under model given optimal Q_values, beta,
    transitions and actions available
    """
    nllkhd = 0

    for i_trial in range(len(data)):

        for i_time in range(len(data[i_trial])-1):

            partial = 0
            # enumerate over all posible actions for the observed state
            for i_a, action in enumerate(actions[data[i_trial][i_time]]):

                partial += (
                    task_structure.softmax_policy(
                        Q_values[data[i_trial][i_time]]
                        [:, i_time], beta)[action]
                    * T[data[i_trial][i_time]][action][
                        data[i_trial][i_time+1]])

            nllkhd = nllkhd - np.log(partial)

    return nllkhd


def calculate_likelihood_interest_rewards(data, Q_values, beta, T, p_stay,
                                          actions, interest_states):
    """
    calculate likelihood of data under interest reward model given 
    optimal Q_values, beta, transitions, probability of staying in low and
    high states, and actions available
    """
    nllkhd = 0

    for i_trial in range(len(data)):

        # marginal prob of interest rewards at very first time step
        p_interest = np.zeros(len(interest_states))
        for i_a, action in enumerate(actions[data[i_trial][0]]):

            p_interest += (p_stay[0, :]  # assume 1st interest state = 0 (low)
                           * task_structure.softmax_policy(
                               Q_values[0][data[i_trial][0]]
                [:, 0], beta)[action]
                * T[data[i_trial][0]][action][data[i_trial][1]])

        # marginal prob for rest of time steps
        for i_time in range(1, len(data[i_trial])-1):

            partial = np.zeros(len(interest_states))

            # enumerate over all possible interest states
            for i_i, interest_state in enumerate(interest_states):

                # enumerate over all possible actions for the observed state
                for i_a, action in enumerate(actions[data[i_trial][i_time]]):

                    partial += (
                        p_stay[interest_state, :]
                        * task_structure.softmax_policy(
                            Q_values[interest_state]
                            [data[i_trial][i_time]]
                            [:, i_time], beta)[action]
                        * T[data[i_trial][i_time]][action][
                            data[i_trial][i_time+1]]
                        * p_interest[interest_state])

            # the above calculation results in a marginal prob over the (two)
            # possible interest states, which'll be added up in next iteration
            p_interest = partial

        # final prob is over the two interest states, so must be added up
        nllkhd = nllkhd - np.log(np.sum(p_interest))

    return nllkhd
