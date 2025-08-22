import numpy as np
import concurrent.futures
import diff_disc_experiments
import likelihoods

# %%


def model_recovery(inputs):
    """
    fit all models to trajectories generated from known parameters, find best
    model and parameters
    """

    models = ['basic', 'diff_discounts']
    # model and params to generate data
    params = inputs[0]
    data = diff_disc_experiments.gen_data_diff_discounts(*params)

    # fit each model to data and recover params
    result_basic = likelihoods.maximum_likelihood_estimate_basic(
        diff_disc_experiments.STATES, diff_disc_experiments.ACTIONS,
        diff_disc_experiments.HORIZON, diff_disc_experiments.REWARD_UNIT,
        diff_disc_experiments.REWARD_SHIRK,  diff_disc_experiments.BETA,
        diff_disc_experiments.EFFORT_WORK, diff_disc_experiments.STATES_NO,
        data)

    result_diff_disc = likelihoods.maximum_likelihood_estimate_diff_discounts(
        diff_disc_experiments.STATES, diff_disc_experiments.ACTIONS,
        diff_disc_experiments.HORIZON, diff_disc_experiments.REWARD_UNIT,
        diff_disc_experiments.REWARD_SHIRK,  diff_disc_experiments.BETA,
        diff_disc_experiments.EFFORT_WORK, diff_disc_experiments.STATES_NO,
        data)

    results = [result_basic, result_diff_disc]

    # find best model, what are the parameters?
    nllkhds = [result_basic.fun, result_diff_disc.fun]
    params = [result_basic.x, result_diff_disc.x]

    return [nllkhds, params]


if __name__ == "__main__":

    np.random.seed(0)

    # generate data
    N = 500  # no. of param sets per model type to recover
    N_TRIALS = 30  # no. of trials
    free_param_no = [2, 3]  # no. of free params basic and diff disc

    # generate iterable list of input params and models
    input_lst = []

    for i in range(N):
        # generate random parameters
        discount_factor_reward = np.random.uniform(0.2, 1)
        # sample only those < disc_reward-0.1
        discount_factor_cost = np.random.uniform(0.2,
                                                 discount_factor_reward-0.15)
        efficacy = np.random.uniform(0.35, 1)

        input_lst.append([[
            diff_disc_experiments.STATES, diff_disc_experiments.ACTIONS,
            diff_disc_experiments.HORIZON, diff_disc_experiments.REWARD_UNIT,
            diff_disc_experiments.REWARD_SHIRK, diff_disc_experiments.BETA,
            discount_factor_reward, discount_factor_cost, efficacy,
            diff_disc_experiments.EFFORT_WORK, N_TRIALS,
            diff_disc_experiments.STATES_NO], 1])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_lst = executor.map(model_recovery, input_lst)

    result_lst = [*result_lst]
    result = np.array(result_lst, dtype=object)
    np.save('data/result_recovery.npy', result)

    input_lst = [*input_lst]
    inputs = np.array(input_lst, dtype=object)
    np.save('data/inputs_recovery.npy', inputs)
