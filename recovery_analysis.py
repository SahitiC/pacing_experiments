import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2

# %%


def choose_best_model(fits, free_param_no):
    """
    choose best model based on AIC, what were the fitted parameters for this
    model?
    """

    likelihoods, params = fits[0], fits[1]
    model_recovered = np.argmin(
        2*np.array(likelihoods) + np.array(free_param_no) * 2)
    params_recovered = params[model_recovered]

    return [params_recovered, model_recovered]


# %%
result_recovery_diff_disc = np.load('result_recovery_diff_disc.npy',
                                    allow_pickle=True)
result_recovery_no_commit = np.load('result_recovery_no_commit.npy',
                                    allow_pickle=True)

input_recovery_diff_disc = np.load('inputs_recovery_diff_disc.npy',
                                   allow_pickle=True)
input_recovery_no_commit = np.load('inputs_recovery_no_commit.npy',
                                   allow_pickle=True)

# %% diff disc

# model recovery
models_recovered_diff_disc = []
for i in range(len(result_recovery_diff_disc)):

    models_recovered_diff_disc.append(
        choose_best_model(result_recovery_diff_disc[i, :, :], 3))
models_recovered_diff_disc = np.array(models_recovered_diff_disc, dtype=object)
model_recovery_counts_diff_disc = np.unique_counts(
    models_recovered_diff_disc[:, 1])

# params recovery
params_recovered_diff_disc = np.stack(result_recovery_diff_disc[:, 1, 1])

all_params = np.array(input_recovery_diff_disc[:, 0])
params_input_diff_disc = []
for i in range(len(all_params)):
    params_input_diff_disc.append(all_params[i][-6:-3])
params_input_diff_disc = np.stack(params_input_diff_disc)

pars = [r'$\gamma_r$', r'$\gamma_c$', r'$\eta$']
for par in range(3):
    plt.figure(figsize=(4, 4))
    plt.scatter(params_input_diff_disc[:, par],
                params_recovered_diff_disc[:, par])
    plt.plot(np.arange(0.4, 1.1, 0.1), np.arange(0.4, 1.1, 0.1), color='k')
    plt.xlabel(f'true {pars[par]}')
    plt.ylabel(f'recovered {pars[par]}')


# %% no commit

models_recovered_no_commit = []
for i in range(len(result_recovery_no_commit)):

    models_recovered_no_commit.append(
        choose_best_model(result_recovery_no_commit[i, :, :], 3))
models_recovered_no_commit = np.array(models_recovered_no_commit, dtype=object)
model_recovery_counts_no_commit = np.unique_counts(
    models_recovered_no_commit[:, 1])

# params recovery
params_recovered_no_commit = np.stack(result_recovery_no_commit[:, 1, 2])

all_params = np.array(input_recovery_no_commit[:, 0])
params_input_no_commit = []
for i in range(len(all_params)):
    params_input_no_commit.append(all_params[i][-6:-4])
params_input_no_commit = np.stack(params_input_no_commit)
mask = params_recovered_no_commit[:, 0] == params_recovered_no_commit[0, 0]

pars = [r'$\gamma$', r'$\eta$']
for par in range(2):
    plt.figure(figsize=(4, 4))
    plt.scatter(params_input_no_commit[:, par],
                params_recovered_no_commit[:, par])
    plt.plot(np.arange(0.4, 1.1, 0.1), np.arange(0.4, 1.1, 0.1), color='k')
    plt.xlabel(f'true {pars[par]}')
    plt.ylabel(f'recovered {pars[par]}')
