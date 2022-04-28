import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from absl import flags

from config import get_config

FLAGS = flags.FLAGS
flags.DEFINE_boolean('central_included', True, 'central link included or not')


def data_analyzer(file1, file2, label_name, config, max_tm_idx=30, save_plot=False):
    df = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    mlu_idx = [1, 3, 5, 7, 9, 11]
    delay_idx = [13, 14, 15, 16, 17, 19]

    avg_mlu = [np.mean(df[mlu_idx[0]][:max_tm_idx].to_numpy()), np.mean(df2[mlu_idx[0]][:max_tm_idx].to_numpy()),
               np.mean(df[mlu_idx[1]][:max_tm_idx].to_numpy()),
               np.mean(df[mlu_idx[2]][:max_tm_idx].to_numpy()), np.mean(df[mlu_idx[3]][:max_tm_idx].to_numpy()),
               np.mean(df[mlu_idx[4]][:max_tm_idx].to_numpy()), np.mean(df[mlu_idx[5]][:max_tm_idx].to_numpy())]

    avg_delay = [np.mean(df[delay_idx[0]][:max_tm_idx].to_numpy()), np.mean(df2[delay_idx[0]][:max_tm_idx].to_numpy()),
                 np.mean(df[delay_idx[1]][:max_tm_idx].to_numpy()),
                 np.mean(df[delay_idx[2]][:max_tm_idx].to_numpy()), np.mean(df[delay_idx[3]][:max_tm_idx].to_numpy()),
                 np.mean(df[delay_idx[4]][:max_tm_idx].to_numpy()), np.mean(df[delay_idx[5]][:max_tm_idx].to_numpy())]

    print('#*#*#* Schemes: [CrFRO, CrFRO, TopK-Critical, TopK-Cum-Centrality, TopK-Centralized, TopK, ECMP] *#*#*#')

    print(avg_mlu)
    print(avg_delay)


def pr_plot(file1, file2, scheme='mlu', label_name=None, config=None, save_plot=False):
    fig_size = (20, 6)
    fig = plt.figure(figsize=fig_size)
    df = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)
    if scheme == 'mlu':
        idx = [1, 3, 5, 7, 9, 11]
        y_label = r'$\mathrm{{PR_U}}$'
    if scheme == 'delay':
        idx = [13, 14, 15, 16, 17, 19]
        y_label = r'$\mathrm{{PR_\Omega}}$'
    Y1, Y2, Y3, Y4, Y5, Y6 = [df[idx[i]].to_numpy() for i in range(len(idx))]
    Y1_2 = df2[idx[0]].to_numpy()

    max_idx = 30
    Y1, Y1_2, Y2, Y3, Y4, Y5, Y6 = Y1[:max_idx], Y1_2[:max_idx], Y2[:max_idx], Y3[:max_idx], \
                                   Y4[:max_idx], Y5[:max_idx], Y6[:max_idx]

    x = np.array([i + 1 for i in range(len(Y1))])
    label_font_size = 13
    title_font_size = 15
    marker = ['r|-', 'b*-', 'y^-', 'm_-', 'g>-', 'cx-']
    # marker = ['r', 'g', 'm', 'b', 'y', 'c', 'gray']
    # marker = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']
    # marker = ['r|-', 'g*-', 'm^-', 'bx-', 'y+-', 'c.-', 'gray']
    markersize = 5
    light_weigt = 0.8
    plt.plot(x, Y1, marker[0], markersize=markersize, linewidth=light_weigt, label=label_name[0])
    plt.plot(x, Y1_2, marker[1], markersize=markersize, linewidth=light_weigt, label=label_name[1])
    plt.plot(x, Y3, marker[2], markersize=markersize, linewidth=light_weigt, label=label_name[3])
    plt.plot(x, Y5, marker[3], markersize=markersize, linewidth=light_weigt, label=label_name[5])
    plt.plot(x, Y6, marker[4], markersize=markersize, linewidth=light_weigt, label=label_name[6])

    # axes[i, j].plot(x, y2, marker[2], markersize=markersize, label=label_name[2])
    # axes[i, j].plot(x, y3, marker[3], markersize=markersize, label=label_name[3])
    # axes[i, j].plot(x, y4, marker[4], markersize=markersize, label=label_name[4])
    # axes[i, j].plot(x, y5, marker[5], markersize=markersize, label=label_name[5])
    # axes[i, j].plot(x, y6, marker[6], markersize=markersize, label=label_name[6])
    plt.legend(loc='lower right')
    plt.xlabel("Traffic Matrix index", weight="bold", fontsize=label_font_size)
    plt.xlim(0, len(Y1))
    plt.ylabel(y_label, weight="bold", fontsize=label_font_size)

    if scheme == 'mlu':
        plt.ylim(0.5, 1.005)
        plt.title(
            r'Link Load Balancing Performance ($\mathrm{{PR_U}}$) among different rerouting schemes base on test traffic matrix (' + str(
                len(Y1)) + ' TMs)',
            fontsize=title_font_size)
        plt.show()
        if save_plot:
            if config.method == 'actor_critic':
                fig.savefig(
                    os.getcwd() + '/result/img/week_details_all-schemes-curve-ac-pr-mlu-week-' + label_name[0] + '.png',
                    format='png')
            if config.method == 'pure_policy':
                fig.savefig(
                    os.getcwd() + '/result/img/week_details_all-schemes-curve-pg-pr-mlu-week-' + label_name[0] + '-' +
                    label_name[
                        1] + '.png',
                    format='png')

    if scheme == 'delay':
        plt.ylim(0.2, 0.95)
        plt.title(r'End-to-End Delay Performance ($\mathrm{{PR_\Omega}}$) among different rerouting schemes base on test traffic matrix (' + str(
            len(Y1)) + ' TMs)',
                  fontsize=title_font_size)
        plt.show()
        if save_plot:
            if config.method == 'actor_critic':
                fig.savefig(os.getcwd() + '/result/img/week_details_all-schemes-curve-ac-pr-delay-week-' + label_name[
                    0] + '.png',
                            format='png')
            if config.method == 'pure_policy':
                fig.savefig(
                    os.getcwd() + '/result/img/week_details_all-schemes-curve-pg-pr-delay-week-' + label_name[0] + '-' +
                    label_name[
                        1] + '.png',
                    format='png')

if __name__ == '__main__':
    config = get_config(FLAGS) or FLAGS

    file1 = 'result/csv/result-pure-policy-ebone-baseline-ckpt46.csv'
    file2 = 'result/csv/result-pure-policy-ebone-baseline-ckpt46.csv'

    label_name = ['CFR-RL-Ebone', 'CFR-RL-Ebone', 'TopK Critical', 'TopK Cum-Centrality', 'TopK Centralized', 'TopK',
                  'ECMP']
    label_name[0] = label_name[0] + '-' + str(config.max_moves)
    label_name[1] = label_name[1] + '-' + str(config.max_moves)
    save_plot = False
    cdf_plot = False
    day_list = [1, 2, 3, 5]
    data_analyzer(file1, file2, label_name=label_name, config=config, save_plot=save_plot)
    pr_plot(file1, file2, scheme='mlu', label_name=label_name, config=config, save_plot=save_plot)
    pr_plot(file1, file2, scheme='delay', label_name=label_name, config=config, save_plot=save_plot)
