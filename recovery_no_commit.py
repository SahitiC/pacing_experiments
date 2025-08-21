import numpy as np
import concurrent.futures
import extra_uncertain_rewards_experiments
import likelihoods

# %%


def model_recovery(inputs):
    """
    fit all models to trajectories generated from known parameters, find best
    model and parameters
    """

    models = ['basic', 'diff_discounts', 'no_commit']
    # model and params to generate data
    params = inputs[0]
    data = extra_uncertain_rewards_experiments.gen_data_no_commitment(*params)

    # fit each model to data and recover params
    result_basic = likelihoods.maximum_likelihood_estimate_basic(
        extra_uncertain_rewards_experiments.STATES_BASE,
        extra_uncertain_rewards_experiments.ACTIONS_BASE,
        extra_uncertain_rewards_experiments.HORIZON,
        extra_uncertain_rewards_experiments.REWARD_UNIT,
        extra_uncertain_rewards_experiments.REWARD_SHIRK,
        extra_uncertain_rewards_experiments.BETA,
        extra_uncertain_rewards_experiments.EFFORT_WORK,
        int(extra_uncertain_rewards_experiments.STATES_NO/2),
        data)

    result_diff_disc = likelihoods.maximum_likelihood_estimate_diff_discounts(
        extra_uncertain_rewards_experiments.STATES_BASE,
        extra_uncertain_rewards_experiments.ACTIONS_BASE,
        extra_uncertain_rewards_experiments.HORIZON,
        extra_uncertain_rewards_experiments.REWARD_UNIT,
        extra_uncertain_rewards_experiments.REWARD_SHIRK,
        extra_uncertain_rewards_experiments.BETA,
        extra_uncertain_rewards_experiments.EFFORT_WORK,
        int(extra_uncertain_rewards_experiments.STATES_NO/2),
        data)

    result_no_commit = likelihoods.maximum_likelihood_estimate_no_commitment(
        extra_uncertain_rewards_experiments.STATES,
        extra_uncertain_rewards_experiments.INTEREST_STATES,
        extra_uncertain_rewards_experiments.ACTIONS_BASE,
        extra_uncertain_rewards_experiments.HORIZON,
        extra_uncertain_rewards_experiments.REWARD_UNIT,
        extra_uncertain_rewards_experiments.REWARD_SHIRK,
        extra_uncertain_rewards_experiments.BETA,
        extra_uncertain_rewards_experiments.P_STAY_LOW,
        extra_uncertain_rewards_experiments.P_STAY_HIGH,
        extra_uncertain_rewards_experiments.EFFORT_WORK,
        extra_uncertain_rewards_experiments.REWARD_INTEREST,
        extra_uncertain_rewards_experiments.STATES_NO,
        data)

    print('done')

    results = [result_basic, result_diff_disc, result_no_commit]

    # find best model, what are the parameters?
    nllkhds = [result_basic.fun, result_diff_disc.fun, result_no_commit.fun]
    params = [result_basic.x, result_diff_disc.x, result_no_commit.x]

    return [nllkhds, params]


if __name__ == "__main__":

    np.random.seed(0)

    # generate data
    N = 500  # no. of param sets per model type to recover
    N_TRIALS = 30  # no. of trials
    free_param_no = [2, 3, 2]  # no. of free params basic and diff disc

    # generate iterable list of input params and models
    input_lst = []

    for i in range(N):
        # generate random parameters
        discount_factor = np.random.uniform(0.6, 1)
        efficacy = np.random.uniform(0.35, 1)

        input_lst.append([[
            extra_uncertain_rewards_experiments.STATES,
            extra_uncertain_rewards_experiments.ACTIONS_BASE,
            extra_uncertain_rewards_experiments.HORIZON,
            extra_uncertain_rewards_experiments.REWARD_UNIT,
            extra_uncertain_rewards_experiments.REWARD_SHIRK,
            extra_uncertain_rewards_experiments.BETA,
            extra_uncertain_rewards_experiments.P_STAY_LOW,
            extra_uncertain_rewards_experiments.P_STAY_HIGH,
            discount_factor, efficacy,
            extra_uncertain_rewards_experiments.EFFORT_WORK,
            extra_uncertain_rewards_experiments.REWARD_INTEREST,
            N_TRIALS,
            extra_uncertain_rewards_experiments.STATES_NO], 2])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_lst = executor.map(model_recovery, input_lst)

    result_lst = [*result_lst]
    result = np.array(result_lst, dtype=object)
    np.save('data/result_recovery.npy', result)

    input_lst = [*input_lst]
    inputs = np.array(input_lst, dtype=object)
    np.save('data/inputs_recovery.npy', inputs)
