import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from absl import flags

from config import get_config

FLAGS = flags.FLAGS
flags.DEFINE_boolean('central_included', True, 'central link included or not')


def data_analyzer(file, label_name, config):
    df = pd.read_csv(file, header=None)

    max_tm_idx = 288 * 6  # 288 * 7 (one week) by default
    mlu_idx = [1, 3, 5, 7, 9, 11]
    delay_idx = [13, 14, 15, 16, 17, 19]

    avg_mlu = [np.mean(df[mlu_idx[0]][:max_tm_idx].to_numpy()), np.mean(df[mlu_idx[1]][:max_tm_idx].to_numpy()),
               np.mean(df[mlu_idx[2]][:max_tm_idx].to_numpy()), np.mean(df[mlu_idx[3]][:max_tm_idx].to_numpy()),
               np.mean(df[mlu_idx[4]][:max_tm_idx].to_numpy()), np.mean(df[mlu_idx[5]][:max_tm_idx].to_numpy())]

    avg_delay = [np.mean(df[delay_idx[0]][:max_tm_idx].to_numpy()), np.mean(df[delay_idx[1]][:max_tm_idx].to_numpy()),
                 np.mean(df[delay_idx[2]][:max_tm_idx].to_numpy()), np.mean(df[delay_idx[3]][:max_tm_idx].to_numpy()),
                 np.mean(df[delay_idx[4]][:max_tm_idx].to_numpy()), np.mean(df[delay_idx[5]][:max_tm_idx].to_numpy())]

    print('#*#*#* Schemes: [DRL-Policy, Critical-TopK, TopK-Critical, TopK-Centralized, TopK, ECMP] *#*#*#')

    days = 7
    s = [i * 288 for i in range(days)]
    e = [(i + 1) * 288 - 1 for i in range(days)]

    day_avg_mlu_save = []
    day_avg_delay_save = []

    for i in range(days):
        df_segment = df[s[i]:e[i]]

        day_avg_mlu = [np.mean(df_segment[mlu_idx[0]].to_numpy()), np.mean(df_segment[mlu_idx[1]].to_numpy()),
                       np.mean(df_segment[mlu_idx[2]].to_numpy()), np.mean(df_segment[mlu_idx[3]].to_numpy()),
                       np.mean(df_segment[mlu_idx[4]].to_numpy()), np.mean(df_segment[mlu_idx[5]].to_numpy())]

        day_avg_delay = [np.mean(df_segment[delay_idx[0]].to_numpy()), np.mean(df_segment[delay_idx[1]].to_numpy()),
                         np.mean(df_segment[delay_idx[2]].to_numpy()), np.mean(df_segment[delay_idx[3]].to_numpy()),
                         np.mean(df_segment[delay_idx[4]].to_numpy()), np.mean(df_segment[delay_idx[5]].to_numpy())]

        day_avg_mlu_save.append(day_avg_mlu)
        day_avg_delay_save.append(day_avg_delay)

        print('\n*Day{} AVG MLU: '.format(i + 1), day_avg_mlu)
        print('*Day{} AVG DELAY:  '.format(i + 1), day_avg_delay)

    print('\n*Average load balancing performance ratio among different schemes in one week *\n', avg_mlu)
    print('\n*Average end-to-end delay performance ratio among different schemes in one week*\n', avg_delay)

    def pr_bar_plot_week(metric_name, avg_mlu=None, avg_delay=None, label_name=None):
        label_name = label_name
        fig_size = (8, 6)
        plt.figure(figsize=fig_size)
        plt.rcParams['font.family'] = 'sans-serif'
        if metric_name == "mlu":
            data = avg_mlu
        if metric_name == "delay":
            data = avg_delay
        plt.bar(label_name, data, width=0.5)
        if metric_name == "mlu":
            plt.title(
                r'Average load balancing performance ratio ($\mathrm{{PR_U}}$) among different schemes in one week',
                fontsize=10)
            plt.ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=10)
            plt.ylim(0.6, 1)
            plt.legend()
        if metric_name == "delay":
            plt.title(
                r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
                fontsize=10)
            plt.ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=10)
            plt.ylim(0.6, 0.9)
            plt.legend()
        plt.show()

    def pr_bar_plot_week_integrate(avg_mlu=None, avg_delay=None, label_name=None):
        fig_size = (18, 6)
        fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=False, gridspec_kw={'width_ratios': [1, 1]})
        plt.rcParams['font.family'] = 'sans-serif'

        data1 = avg_mlu
        data2 = avg_delay
        width = 0.5
        axes[0].bar(label_name, data1, width=width)
        axes[1].bar(label_name, data2, width=width)

        axes[0].set_title(
            r'Average load balancing performance ratio ($\mathrm{{PR_U}}$) among different schemes in one week',
            fontsize=10)
        axes[0].set_ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=12)
        axes[0].set_ylim(0.6, 1)
        axes[0].tick_params(axis='x', rotation=20)

        axes[1].set_title(
            r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
            fontsize=10)
        axes[1].set_ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=12)
        axes[1].set_ylim(0.3, 0.86)
        axes[1].tick_params(axis='x', rotation=20)
        plt.show()
        if config.method == 'actor-critic':
            fig.savefig(os.getcwd() + '/result/img/ac-pr-mlu-delay-' + label_name[0] + '.png', format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/pp-pr-mlu-delay-' + label_name[0] + '.png', format='png')

    def pr_bar_plot_day(metric_name, day_avg_mlu_save=None, day_avg_delay_save=None, label_name=None, days=7):
        figsize = (15, 8)
        fig = plt.figure(figsize=figsize)
        plt.rcParams['font.family'] = 'sans-serif'
        if metric_name == "mlu":
            day_avg_mlu_save = day_avg_mlu_save[:days]
            data = np.array(day_avg_mlu_save).transpose()
        if metric_name == "delay":
            day_avg_delay_save = day_avg_delay_save[:days]
            data = np.array(day_avg_delay_save).transpose()

        xvals, yvals, zvals, ivals, jvals, kvals = [data[i].tolist() for i in range(6)]
        N = len(xvals)
        ind = np.arange(N)
        width = 0.15
        zorder = 3

        # color_scheme = ['#b24745FF', '#00A1D5FF', '#DF8F44FF', '#374E55FF', '#79AF97FF', '#80796BFF']

        color_scheme = ['#BB0021FF', '#088B45FF', '#3B4992FF', '#631879FF', '#008280FF', '#808180FF']
        plt.bar(ind, xvals, width, color=color_scheme[0], label=label_name[0], zorder=zorder)
        plt.bar(ind + width, yvals, width, color=color_scheme[1], label=label_name[1], zorder=zorder)
        plt.bar(ind + width * 2, zvals, width, color=color_scheme[2], label=label_name[2], zorder=zorder)
        plt.bar(ind + width * 3, ivals, width, color=color_scheme[3], label=label_name[3], zorder=zorder)
        plt.bar(ind + width * 4, jvals, width, color=color_scheme[4], label=label_name[4], zorder=zorder)
        plt.bar(ind + width * 5, kvals, width, color=color_scheme[5], label=label_name[5], zorder=zorder)

        plt.xlabel("Day of a week", loc="center", fontsize=12)
        plt.xticks(ind + width, [i + 1 for i in range(len(xvals))], fontsize=12)

        if metric_name == "mlu":
            plt.title(
                r'Average load balancing performance ratio ($\mathrm{{PR_U}}$) among different schemes in one week',
                fontsize=15)
            plt.ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=16)
            plt.ylim(0.6, 1)
            plt.legend(loc='upper right')
            plt.show()
            if config.method == 'actor-critic':
                fig.savefig(os.getcwd() + '/result/img/ac-day-pr-mlu-' + label_name[0] + '.png', format='png')
            if config.method == 'pure_policy':
                fig.savefig(os.getcwd() + '/result/img/pp-day-pr-mlu-' + label_name[0] + '.png', format='png')

        if metric_name == "delay":
            plt.title(
                r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
                fontsize=15)
            plt.ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=16)
            plt.ylim(0.35, 0.9)
            plt.legend(loc='upper right')
            plt.show()
            if config.method == 'actor-critic':
                fig.savefig(os.getcwd() + '/result/img/ac-day-pr-delay-' + label_name[0] + '.png', format='png')
            if config.method == 'pure_policy':
                fig.savefig(os.getcwd() + '/result/img/pp-day-pr-delay-' + label_name[0] + '.png', format='png')

    if False:
        pr_bar_plot_week('mlu', avg_mlu=avg_mlu)
        pr_bar_plot_week('delay', avg_delay=avg_delay)

    pr_bar_plot_week_integrate(avg_mlu=avg_mlu, avg_delay=avg_delay, label_name=label_name)
    pr_bar_plot_day('mlu', day_avg_mlu_save=day_avg_mlu_save, label_name=label_name, days=5)
    pr_bar_plot_day('delay', day_avg_delay_save=day_avg_delay_save, label_name=label_name, days=5)


def pr_plot(file, scheme='mlu', label_name=None, day=1, week=False, config=None):
    figsize = (18, 6)
    fig = plt.figure(figsize=figsize)
    df = pd.read_csv(file, header=None)

    if scheme == 'mlu':
        idx = [1, 3, 5, 7, 9, 11]
        y_label = r'$\mathrm{{PR_U}}$'
    if scheme == 'delay':
        idx = [13, 14, 15, 16, 17, 19]
        y_label = r'$\mathrm{{PR_\Omega}}$'
    y1, y2, y3, y4, y5, y6 = [df[idx[i]].to_numpy() for i in range(len(idx))]

    days = 7
    s = [i * 288 for i in range(days)]
    e = [(i + 1) * 288 - 1 for i in range(days)]

    x = np.array([i + 1 for i in range(e[day - 1] - s[day - 1])])
    y1 = y1[s[day - 1]:e[day - 1]]
    y2 = y2[s[day - 1]:e[day - 1]]
    y3 = y3[s[day - 1]:e[day - 1]]
    y4 = y4[s[day - 1]:e[day - 1]]
    y5 = y5[s[day - 1]:e[day - 1]]
    y6 = y6[s[day - 1]:e[day - 1]]

    # plt.plot(x, y1, 'r--',  x, y2, 'b*', x, y3, 'g+', x, y4, 'c')
    plt.plot(x, y1, 'r|-', label=label_name[0])
    plt.plot(x, y2, 'b*-', label=label_name[1])
    plt.plot(x, y3, 'y^-', label=label_name[2])
    plt.plot(x, y4, 'm_-', label=label_name[3])
    plt.plot(x, y5, 'g>-', label=label_name[4])
    plt.plot(x, y6, 'cx-', label=label_name[5])
    plt.legend(loc='lower right')
    plt.xlabel("Traffic Matrix index")
    plt.xlim(0, 288)
    # plt.ylim(0.3, 1)
    plt.ylabel(y_label)

    if scheme == 'mlu':
        plt.title("Load balancing performance ratio with traffic matrices from Day {}".format(day))
        plt.show()
        if config.method == 'actor_critic':
            fig.savefig(os.getcwd() + '/result/img/ac-curve-pr-mlu-day{}-'.format(day) + label_name[0] + '.png',
                        format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/pp-curve-pr-mlu-day{}-'.format(day) + label_name[0] + '.png',
                        format='png')

    if scheme == 'delay':
        plt.title("End-to-end delay performance ratio with traffic matrices from Day {}".format(day))
        plt.show()
        if config.method == 'actor_critic':
            fig.savefig(os.getcwd() + '/result/img/ac-curve-pr-delay-day{}-'.format(day) + label_name[0] + '.png',
                        format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/pp-curve-pr-delay-day{}-'.format(day) + label_name[0] + '.png',
                        format='png')


def pr_multi_plot(file, scheme='mlu', label_name=None, week=False, config=None, day_list=None):
    fig_size = (18, 16)
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axis = [(0, 0), (0, 1), (1, 0), (1, 1)]
    df = pd.read_csv(file, header=None)
    if scheme == 'mlu':
        idx = [1, 3, 5, 7, 9, 11]
        y_label = r'$\mathrm{{PR_U}}$'
    if scheme == 'delay':
        idx = [13, 14, 15, 16, 17, 19]
        y_label = r'$\mathrm{{PR_\Omega}}$'
    Y1, Y2, Y3, Y4, Y5, Y6 = [df[idx[i]].to_numpy() for i in range(len(idx))]

    days = 7
    s = [i * 288 for i in range(days)]
    e = [(i + 1) * 288 - 1 for i in range(days)]

    if day_list is None:
        day_list = [2, 3, 5, 6]
    for day_idx, day in enumerate(day_list):
        x = np.array([i + 1 for i in range(e[day - 1] - s[day - 1])])
        y1 = Y1[s[day - 1]:e[day - 1]]
        y2 = Y2[s[day - 1]:e[day - 1]]
        y3 = Y3[s[day - 1]:e[day - 1]]
        y4 = Y4[s[day - 1]:e[day - 1]]
        y5 = Y5[s[day - 1]:e[day - 1]]
        y6 = Y6[s[day - 1]:e[day - 1]]

        i, j = axis[day_idx]
        # marker = ['r|-', 'b*-', 'y^-', 'm_-', 'g>-', 'cx-']
        marker = ['r', 'b', 'y', 'm', 'g', 'c']
        axes[i, j].plot(x, y1, marker[0], label=label_name[0])
        axes[i, j].plot(x, y2, marker[1], label=label_name[1])
        axes[i, j].plot(x, y3, marker[2], label=label_name[2])
        axes[i, j].plot(x, y4, marker[3], label=label_name[3])
        axes[i, j].plot(x, y5, marker[4], label=label_name[4])
        axes[i, j].plot(x, y6, marker[5], label=label_name[5])
        axes[i, j].legend(loc='lower right')
        axes[i, j].set_xlabel("Traffic Matrix index")
        axes[i, j].set_xlim(0, 288)
        # plt.ylim(0.3, 1)
        axes[i, j].set_ylabel(y_label)
        axes[i, j].set_title('Day {}'.format(day))

    if scheme == 'mlu':
        plt.show()
        if config.method == 'actor_critic':
            fig.savefig(os.getcwd() + '/result/img/curve-ac-pr-mlu-days-' + label_name[0] + '.png',
                        format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/curve-pg-pr-mlu-days-' + label_name[0] + '.png',
                        format='png')

    if scheme == 'delay':
        plt.show()
        if config.method == 'actor_critic':
            fig.savefig(os.getcwd() + '/result/img/curve-ac-pr-delay-days-' + label_name[0] + '.png',
                        format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/curve-pg-pr-delay-days-' + label_name[0] + '.png',
                        format='png')


def cdf_plot(file, scheme='mlu', label_name=None, day=3, week=False, config=None):
    figsize = (18, 6)
    fig = plt.figure(figsize=figsize)
    df = pd.read_csv(file, header=None)
    if scheme == 'mlu':
        idx = [1, 3, 5, 7, 9, 11]
        x_label = r'$\mathrm{{PR_U}}$'
    if scheme == 'delay':
        idx = [13, 14, 15, 16, 17, 19]
        x_label = r'$\mathrm{{PR_\Omega}}$'

    if week:
        data1, data2, data3, data4, data5, data6 = [df[idx[i]].to_numpy() for i in range(len(idx))]  # One week
    else:
        days = 7
        s = [i * 288 for i in range(days)]
        e = [(i + 1) * 288 - 1 for i in range(days)]
        data1, data2, data3, data4, data5, data6 = [df[idx[i]][s[day - 1]:e[day - 1]].to_numpy() for i in
                                                    range(len(idx))]

    def app(data):
        count, bins_count = np.histogram(data, bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        return cdf, bins_count

    for idx, data in enumerate([data1, data2, data3, data4, data5, data6]):
        cdf, bins_count = app(data)
        plt.plot(bins_count[1:], cdf, label=label_name[idx])
    plt.title('Day {}'.format(day))
    plt.xlabel(x_label)
    plt.ylabel('CDF')
    plt.xlim(0.2, 1)
    plt.ylim(0.1, 1)
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()

    if scheme == 'mlu':
        if config.method == 'actor_critic':
            fig.savefig(os.getcwd() + '/result/img/ac-cdf-pr-mlu-day{}-'.format(day) + label_name[0] + '.png',
                        format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/pp-cdf-pr-mlu-day{}-'.format(day) + label_name[0] + '.png',
                        format='png')
    if scheme == 'delay':
        if config.method == 'actor_critic':
            fig.savefig(os.getcwd() + '/result/img/ac-cdf-pr-delay-day{}-'.format(day) + label_name[0] + '.png',
                        format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/pp-cdf-pr-delay-day{}-'.format(day) + label_name[0] + '.png',
                        format='png')


def cdf_multi_plot(file, scheme='mlu', label_name=None,  config=None, day_list=None):
    fig_size = (18, 16)
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    df = pd.read_csv(file, header=None)
    if scheme == 'mlu':
        idx = [1, 3, 5, 7, 9, 11]
        x_label = r'$\mathrm{{PR_U}}$'
    if scheme == 'delay':
        idx = [13, 14, 15, 16, 17, 19]
        x_label = r'$\mathrm{{PR_\Omega}}$'
    days = 7
    s = [i * 288 for i in range(days)]
    e = [(i + 1) * 288 - 1 for i in range(days)]

    if day_list is None:
        day_list = [2, 3, 5, 6]

    def app(data):
        count, bins_count = np.histogram(data, bins=10)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        return cdf, bins_count

    axis = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for day_idx, day in enumerate(day_list):
        data1, data2, data3, data4, data5, data6 \
            = [df[idx[i]][s[day - 1]:e[day - 1]].to_numpy() for i in range(len(idx))]
        for data_idx, data in enumerate([data1, data2, data3, data4, data5, data6]):
            cdf, bins_count = app(data)
            i, j = axis[day_idx]
            axes[i, j].plot(bins_count[1:], cdf, label=label_name[data_idx])
            axes[i, j].set_xlabel(x_label)
            axes[i, j].set_ylabel('CDF')
            axes[i, j].set_xlim(0.2, 1)
            axes[i, j].set_ylim(0.1, 1)
            axes[i, j].set_title('Day {}'.format(day))
            axes[i, j].legend(loc='upper left')
            axes[i, j].grid()

    if scheme == 'mlu':
        # plt.title("Load balancing performance ratio with traffic matrices")
        plt.show()
        if config.method == 'actor_critic':
            fig.savefig(os.getcwd() + '/result/img/cdf-ac-pr-mlu-days-' + label_name[0] + '.png',
                        format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/cdf-pg-pr-mlu-days-' + label_name[0] + '.png',
                        format='png')

    if scheme == 'delay':
        plt.title("End-to-end delay performance ratio with traffic matrices")
        plt.show()
        if config.method == 'actor_critic':
            fig.savefig(os.getcwd() + '/result/img/cdf-ac-pr-delay-days-' + label_name[0] + '.png',
                        format='png')
        if config.method == 'pure_policy':
            fig.savefig(os.getcwd() + '/result/img/cdf-pg-pr-delay-days-' + label_name[0] + '.png',
                        format='png')


if __name__ == '__main__':
    config = get_config(FLAGS) or FLAGS

    # file = 'result/csv/result-pure-policy-baseline-ckpt37.csv'
    # file = 'result/csv/result-pure-policy-alpha-ckpt43.csv'
    # file = 'result/csv/result-pure-policy-alpha+-ckpt36.csv'
    # file = 'result/csv/result-pure-policy-alpha++-ckpt32-adam.csv'
    # file = 'result/csv/result-pure-policy-alpha++-ckpt29.csv'

    # file = 'result/csv/result-actor-critic-baseline-ckpt7.csv'
    # file = 'result/csv/result-actor-critic-alpha-ckpt10.csv'
    # file = 'result/csv/result-actor-critic-alpha+-ckpt13.csv'  # reward max
    # file = 'result/csv/result-actor-critic-alpha+-ckpt48.csv' # value loss min
    # file = 'result/csv/result-actor-critic-beta++ckpt27.csv'
    # file = 'result/csv/result-actor-critic-beta++++ckpt9.csv'
    # file = 'result/csv/result-actor-critic-delta-ckpt5.csv'

    # file = 'result/csv/result-pure-policy-alpha-lastK-centralized-sample.csv'
    # file = 'result/csv/result-pure-policy-alpha-lastK-centralized-sample-0.5scaleK.csv'
    # file = 'result/csv/result-pure-policy-alpha-lastK-sample.csv'
    # file = 'result/csv/result-pure-policy-alpha+-lastK-sample-0.5scaleK.csv'
    # file = 'result/csv/result-pure-policy-alpha+-lastK-centralized-sample-0.5scaleK.csv'

    file = '../result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.25scaleK.csv'

    # label_name = ['_', 'TopK Critical', 'Centralized-TopK', 'TopK-Centralized', 'TopK', 'ECMP']
    label_name = ['_', 'TopK Critical', 'TopK Cum-Centrality', 'TopK Centralized', 'TopK', 'ECMP']

    # label_name[0] = config.scheme.title()  # Be careful !
    # label_name[0] = config.scheme.title() + 'LKC--0.5K'  # Last K Centralized
    # label_name[0] = config.scheme.title() + 'LK--0.5K'  # Last K

    if config.scheme == 'alpha':
        label_name[0] = 'CFRO-V1'
    if config.scheme == 'alpha+':
        label_name[0] = 'CFRO-V2'

    data_analyzer(file, label_name=label_name, config=config)
    pr_multi_plot(file, scheme='mlu', label_name=label_name, config=config)
    pr_multi_plot(file, scheme='delay', label_name=label_name, config=config)
    cdf_multi_plot(file, scheme='mlu', label_name=label_name, config=config)
    cdf_multi_plot(file, scheme='delay', label_name=label_name, config=config)

    exit(1)
    pr_plot(file, scheme='mlu', label_name=label_name, day=3, config=config)
    pr_plot(file, scheme='delay', label_name=label_name, day=3, config=config)
