import numpy as np
import matplotlib.pyplot as plt


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
    trajectories = np.array(trajectories)

    for trajectory in trajectories[:, 1:]:

        if trajectory[-1] == states_no-1:
            completed.append(1)
        else:
            completed.append(0)

    return completed


def plot_trajectories(trajectories, color, lwidth_mean, lwidth_sample,
                      number_samples):
    """
    plot input trajectories
    """
    mean = np.mean(trajectories, axis=0)

    plt.plot(mean, color=color, linewidth=lwidth_mean)
    for i in range(number_samples):
        plt.plot(trajectories[i], color=color,
                 linewidth=lwidth_sample, linestyle='dashed')
    plt.xticks([0, 7, 15])
    plt.yticks([0, 10, 20])
    plt.ylim(-1, 21)
    plt.xlabel('time')
    plt.ylabel('Units of work \n completed')
