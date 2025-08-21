import task_structure
import mdp_algms
import helper
import numpy as np
from scipy.optimize import minimize


def likelihood_basic_model(x, states, actions, horizon, reward_unit,
                           reward_shirk, beta, effort_work, states_no, data):
    """
    implement likelihood calculation for immediate basic model
    """

    discount_factor = x[0]
    efficacy = x[1]

    # define task structure
    reward_func = task_structure.reward_immediate(
        states, actions, reward_shirk, reward_unit)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = np.zeros(len(states))

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = helper.calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_diff_discounts_model(
        x, states, actions, horizon, reward_unit, reward_shirk, beta,
        effort_work, states_no, data):
    """
    implement likelihood calculation for diff discount model
    """

    discount_factor_reward = x[0]
    discount_factor_cost = x[1]
    efficacy = x[2]

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
            effort_func_last, T))

    # effective Q_values for the agent
    effective_Q = []
    for i_s in range(len(states)):
        Q_s_temp = []
        for i in range(horizon):
            Q_s_temp.append(Q_values_full[horizon-1-i][i_s][:, i])
        effective_Q.append(np.array(Q_s_temp).T)

    nllkhd = helper.calculate_likelihood(data, effective_Q, beta, T, actions)

    return nllkhd


def likelihood_no_commitment_model(
        x, states, interest_states, actions_base, horizon, reward_unit,
        reward_shirk, beta, p_stay_low, p_stay_high, effort_work,
        reward_interest, states_no, data):
    """
    implement likelihood calculation for no commit model
    """

    discount_factor = x[0]
    efficacy = x[1]

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
        # assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_low.append(temp)

    T_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([(1 - p_stay_high) * T_partial[state_current],
                         p_stay_high * T_partial[state_current]])
        # assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
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

    # adapt some quantities for llkhd calculating function
    p_stay = np.array([[p_stay_low, 1-p_stay_low],
                       [1-p_stay_high, p_stay_high]])

    Q_values_unstacked = np.array([Q_values[:int(states_no/2)],
                                   Q_values[int(states_no/2):]])

    nllkhd = helper.calculate_likelihood_interest_rewards(
        data, Q_values_unstacked, beta, T_partial, p_stay, actions_base,
        interest_states)

    return nllkhd


def maximum_likelihood_estimate_basic(states, actions, horizon, reward_unit,
                                      reward_shirk, beta, effort_work,
                                      states_no, data,
                                      true_params=None, initial_real=0,
                                      verbose=0):
    """
    maximise likelihood of data under basic model parameters using 
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in 
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_basic_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_unit, reward_shirk, beta,
                                      effort_work, states_no, data),
                                bounds=((0, 1), (0, 1)))
        nllkhd = likelihood_basic_model(
            final_result.x, states, actions, horizon, reward_unit,
            reward_shirk, beta, effort_work, states_no, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor, efficacy,"
                  f" effort_work = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(20):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_basic_model,
                          x0=[discount_factor, efficacy],
                          args=(states, actions, horizon,
                                reward_unit, reward_shirk, beta,
                                effort_work, states_no, data),
                          bounds=((0, 1), (0, 1)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_basic_model(
            result.x, states, actions, horizon, reward_unit, reward_shirk,
            beta, effort_work, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print(
                    "current estimate for discount_factor, efficacy,"
                    f" = {final_result.x} "
                    f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_diff_discounts(
        states, actions, horizon, reward_unit, reward_shirk,
        beta, effort_work, states_no, data, true_params=None,
        initial_real=0, verbose=0):
    """
    maximise likelihood of data under diff-disc model parameters using
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_diff_discounts_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_unit, reward_shirk, beta,
                                      effort_work, states_no, data),
                                bounds=((0, 1), (0, 1), (0, 1)))
        nllkhd = likelihood_diff_discounts_model(
            final_result.x, states, actions, horizon, reward_unit,
            reward_shirk, beta, effort_work, states_no, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor_reward, "
                  f"discount_factor_cost, reward_shirk, effort_work, "
                  f"efficacy = {final_result.x}"
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(20):

        # set initial value for params (random draws)
        discount_factor_reward = np.random.uniform(0, 1)
        discount_factor_cost = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_diff_discounts_model,
                          x0=[discount_factor_reward, discount_factor_cost,
                              efficacy],
                          args=(states, actions, horizon,
                                reward_unit, reward_shirk, beta,
                                effort_work, states_no, data),
                          bounds=((0, 1), (0, 1), (0, 1)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_diff_discounts_model(
            result.x, states, actions, horizon, reward_unit,
            reward_shirk, beta, effort_work, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print("with initial point = true param "
                      "current estimate for discount_factor_reward, "
                      f"discount_factor_cost, reward_shirk, effort_work, "
                      f"efficacy = {final_result.x}"
                      f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_no_commitment(
        states, interest_states, actions_base, horizon, reward_unit,
        reward_shirk, beta, p_stay_low, p_stay_high, effort_work,
        reward_interest, states_no, data, true_params=None, initial_real=0,
        verbose=0):
    """
    maximise likelihood of data under bno commit model parameters using
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_no_commitment_model,
                                x0=true_params,
                                args=(states, interest_states, actions_base,
                                      horizon, reward_unit, reward_shirk,
                                      beta, p_stay_low, p_stay_high,
                                      effort_work, reward_interest, states_no,
                                      data),
                                bounds=((0, 1), (0, 1)))
        # method='Powell')
        nllkhd = likelihood_no_commitment_model(
            final_result.x, states, interest_states, actions_base, horizon,
            reward_unit, reward_shirk, beta, p_stay_low, p_stay_high,
            effort_work, reward_interest, states_no, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor,"
                  f" efficacy = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    for i in range(20):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_no_commitment_model,
                          x0=[discount_factor, efficacy],
                          args=(states, interest_states, actions_base,
                                horizon, reward_unit, reward_shirk,
                                beta, p_stay_low, p_stay_high,
                                effort_work, reward_interest, states_no,
                                data),
                          bounds=((0, 1), (0, 1)), method='Powell')

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_no_commitment_model(
            result.x, states, interest_states, actions_base, horizon,
            reward_unit, reward_shirk, beta, p_stay_low, p_stay_high,
            effort_work, reward_interest, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print("with initial point = true param "
                      "current estimate for discount_factor "
                      f"efficacy = {final_result.x}"
                      f"with neg log likelihood = {nllkhd}")

    return final_result
