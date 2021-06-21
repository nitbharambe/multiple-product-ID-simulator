import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import rc


def create_process(df, column, time):
    # fill NaN with the average
    df[column] = df[column].fillna(df[column].mean())

    # indices of the process
    indices_time = pd.unique(df[time])

    # initial state for each quarter (marginal distribution)
    mu_0 = dict()
    sigma_0 = dict()

    for q in indices_time:
        i_q = df[df[time] == q]
        mu_0[q] = i_q[column].mean()
        sigma_0[q] = i_q[column].std()

    # transition probability distribution
    a_q = dict()
    b_q = dict()
    sigma_q = dict()

    for q, q_plus in zip(indices_time, np.roll(indices_time, -1)):
        i_q = df[df[time] == q].reset_index()
        i_q_plus = df[df[time] == q_plus].reset_index()

        # the time series may not have the same number of samples
        min_len = min(i_q[column].size, i_q_plus[column].size)

        # linear combination of the transition model
        cov_mat = np.cov(i_q[column].head(min_len), i_q_plus[column].head(min_len))
        a_q[q_plus] = cov_mat[0, 0] / cov_mat[0, 1]
        b_q[q_plus] = (i_q_plus[column] - a_q[q_plus] * i_q[column].mean()).mean()

        # std deviation of the transition probability
        sigma_q[q_plus] = np.sqrt((i_q_plus[column].head(min_len) -
                                   a_q[q_plus] * i_q[column].head(min_len) - b_q[q_plus]).pow(2).mean())

    # average of the transition probability after convergence (linear system)
    a = 1 - np.roll(np.diag([a_q[q] for q in indices_time]), -1, axis=1)
    b = np.array([b_q[q] for q in indices_time])
    mu_q = {q: mu for q, mu in zip(indices_time, np.linalg.solve(a, b))}

    return mu_0, sigma_0, mu_q, sigma_q, a_q, b_q


def plot_process(mu_q, sigma_q, legend_x, legend_y, save_name=None):
    q = np.array(list(mu_q.keys()))
    mu = np.array(list(mu_q.values()))
    sigma = np.array(list(sigma_q.values()))

    plt.figure()
    rc('text', usetex=True)

    lower = mu - sigma / 2
    upper = mu + sigma / 2
    plt.plot(q, mu, color='green')
    plt.fill_between(q, lower, upper, facecolor='green', alpha=0.3)

    plt.ylabel(legend_y, fontsize=16)
    plt.xlabel(legend_x, fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name + '.pdf')
