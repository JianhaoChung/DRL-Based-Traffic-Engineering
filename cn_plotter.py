import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from absl import flags

from config import get_config

FLAGS = flags.FLAGS
flags.DEFINE_boolean('central_included', True, 'central link included or not')


def data_analyzer(file1, file2, label_name, config, save_plot=False):
    df = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    max_tm_idx = 288 * 6  # 288 * 7 (one week) by default
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

    days = 7
    s = [i * 288 for i in range(days)]
    e = [(i + 1) * 288 - 1 for i in range(days)]

    day_avg_mlu_save = []
    day_avg_delay_save = []

    for i in range(days):
        df_segment = df[s[i]:e[i]]
        df2_segment = df2[s[i]:e[i]]

        day_avg_mlu = [np.mean(df_segment[mlu_idx[0]].to_numpy()), np.mean(df2_segment[mlu_idx[0]].to_numpy()),
                       np.mean(df_segment[mlu_idx[1]].to_numpy()),
                       np.mean(df_segment[mlu_idx[2]].to_numpy()), np.mean(df_segment[mlu_idx[3]].to_numpy()),
                       np.mean(df_segment[mlu_idx[4]].to_numpy()), np.mean(df_segment[mlu_idx[5]].to_numpy())]

        day_avg_delay = [np.mean(df_segment[delay_idx[0]].to_numpy()), np.mean(df2_segment[delay_idx[0]].to_numpy()),
                         np.mean(df_segment[delay_idx[1]].to_numpy()),
                         np.mean(df_segment[delay_idx[2]].to_numpy()), np.mean(df_segment[delay_idx[3]].to_numpy()),
                         np.mean(df_segment[delay_idx[4]].to_numpy()), np.mean(df_segment[delay_idx[5]].to_numpy())]

        day_avg_mlu_save.append(day_avg_mlu)
        day_avg_delay_save.append(day_avg_delay)

        print('\n*Day{} AVG MLU: '.format(i + 1), day_avg_mlu)
        print('*Day{} AVG DELAY:  '.format(i + 1), day_avg_delay)

    print('\n*Average load balancing performance ratio among different schemes in one week *\n', avg_mlu)
    print('\n*Average end-to-end delay performance ratio among different schemes in one week*\n', avg_delay)

    def pr_bar_plot_week_integrate(avg_mlu=None, avg_delay=None, label_name=None):
        # fig_size = (18, 6.5)
        fig_size = (15, 6)
        fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=False, gridspec_kw={'width_ratios': [1, 1]})
        plt.rcParams['font.family'] = 'sans-serif'

        avg_mlu = [avg_mlu[0], avg_mlu[1], avg_mlu[3], avg_mlu[-2], avg_mlu[-1]]
        avg_delay = [avg_delay[0], avg_delay[1], avg_delay[3], avg_delay[-2], avg_delay[-1]]
        label_name = [label_name[0], label_name[1], label_name[3], label_name[-2], label_name[-1]]
        width = 0.3

        data1 = avg_mlu
        data2 = avg_delay

        # color = ['#be5629', '#056faf',  '#4e9595',  '#f4cc66', '#808180FF']
        # std_err = [0.005, 0.02, 0.025, 0.025, 0.02]
        # error_params = dict(elinewidth=1, ecolor=color[-1], capsize=5)  # 设置误差标记参数
        # axes[0].bar(label_name, data1, width=width, yerr=std_err, error_kw=error_params, color=color)
        # axes[1].bar(label_name, data2, width=width, yerr=std_err, error_kw=error_params, color=color)

        axes[0].bar(label_name, data1, width=width)
        axes[1].bar(label_name, data2, width=width)

        axes[0].set_title(
            r'Average load balancing performance ratio ($\mathrm{{PR_U}}$) among different schemes in one week',
            fontsize=10)
        axes[0].set_ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=12)
        axes[0].set_ylim(0.5, 1)
        axes[0].tick_params(axis='x', rotation=10)

        axes[1].set_title(
            r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
            fontsize=10)
        axes[1].set_ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=12)
        axes[1].set_ylim(0.3, 0.9)
        axes[1].tick_params(axis='x', rotation=10)
        plt.show()
        # if config.method == 'actor-critic':
        #     fig.savefig(os.getcwd() + '/result/img/ac-pr-mlu-delay-' + label_name[0] + '.png', format='png')
        # if config.method == 'pure_policy':
        #     fig.savefig(os.getcwd() + '/result/img/pp-pr-mlu-delay-' + label_name[0] + '.png', format='png')

    def pr_bar_plot_day(metric_name, day_avg_mlu_save=None, day_avg_delay_save=None, label_name=None, days=7,
                        save_plot=save_plot):
        figsize = (15, 6)
        fig = plt.figure(figsize=figsize)
        plt.rcParams['font.family'] = 'sans-serif'
        if metric_name == "mlu":
            day_avg_mlu_save = day_avg_mlu_save[:days]
            data = np.array(day_avg_mlu_save).transpose()
        if metric_name == "delay":
            day_avg_delay_save = day_avg_delay_save[:days]
            data = np.array(day_avg_delay_save).transpose()

        xvals, xxvals, yvals, zvals, ivals, jvals, kvals = [data[i].tolist() for i in range(7)]
        N = len(xvals)
        ind = np.arange(N)
        width = 0.12
        bar_width = width * 0.95
        zorder = 3

        # color_scheme = ['#b24745FF', '#00A1D5FF', '#DF8F44FF', '#374E55FF', '#79AF97FF', '#80796BFF']
        # color_scheme = ['#BB0021FF', '#A20056FF', '#088B45FF', '#3B4992FF', '#631879FF', '#008280FF', '#808180FF']
        # color_scheme = ['#be5629', '#056faf',  '#b79561', '#003c4b', '#758837', '#f4cc66',  '#808180FF']
        color_scheme = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']
        # color_scheme = ['#be5629', '#056faf',  '#84574d', '#6e94ce', '#4e9595', '#f4cc66',  '#808180FF']
        # color_scheme = ['#be5629', '#003c4b',   '#b79561','#056faf', '#758837', '#f4cc66',  '#808180FF']
        plt.bar(ind, xvals, bar_width, color=color_scheme[0], label=label_name[0], zorder=zorder)
        plt.bar(ind + width, xxvals, bar_width, color=color_scheme[1], label=label_name[1], zorder=zorder)
        plt.bar(ind + width * 2, zvals, bar_width, color=color_scheme[3], label=label_name[3], zorder=zorder)
        plt.bar(ind + width * 3, jvals, bar_width, color=color_scheme[5], label=label_name[5], zorder=zorder)
        plt.bar(ind + width * 4, kvals, bar_width, color=color_scheme[6], label=label_name[6], zorder=zorder)

        # plt.bar(ind + width * 2, yvals, bar_width, color=color_scheme[2], label=label_name[2], zorder=zorder)
        # plt.bar(ind + width * 3, zvals, bar_width, color=color_scheme[3], label=label_name[3], zorder=zorder)
        # plt.bar(ind + width * 4, ivals, bar_width, color=color_scheme[4], label=label_name[4], zorder=zorder)
        # plt.bar(ind + width * 5, jvals, bar_width, color=color_scheme[5], label=label_name[5], zorder=zorder)
        # plt.bar(ind + width * 6, kvals, bar_width, color=color_scheme[6], label=label_name[6], zorder=zorder)

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
            if save_plot:
                if config.method == 'actor-critic':
                    fig.savefig(os.getcwd() + '/result/img/week-ac-avg-mlu-' + label_name[0] + '.png', format='png')
                if config.method == 'pure_policy':
                    fig.savefig(
                        os.getcwd() + '/result/img/week-pg-avg-mlu-' + label_name[0] + '-' + label_name[1] + '.png',
                        format='png')

        if metric_name == "delay":
            plt.title(
                r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
                fontsize=15)
            plt.ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=16)
            plt.ylim(0.35, 0.9)
            plt.legend(loc='upper right')
            plt.show()
            if save_plot:
                if config.method == 'actor-critic':
                    fig.savefig(os.getcwd() + '/result/img/week-ac-avg-delay-' + label_name[0] + '.png', format='png')
                if config.method == 'pure_policy':
                    fig.savefig(
                        os.getcwd() + '/result/img/week-pg-avg-delay-' + label_name[0] + '-' + label_name[1] + '.png',
                        format='png')

    pr_bar_plot_week_integrate(avg_mlu=avg_mlu, avg_delay=avg_delay, label_name=label_name)
    pr_bar_plot_day('mlu', day_avg_mlu_save=day_avg_mlu_save, label_name=label_name, days=7)
    pr_bar_plot_day('delay', day_avg_delay_save=day_avg_delay_save, label_name=label_name, days=7)


def pr_week_plot(file1, file2, scheme='mlu', label_name=None, config=None, save_plot=False):
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

    max_idx = 288 * 7
    Y1, Y1_2, Y2, Y3, Y4, Y5, Y6 = Y1[:max_idx], Y1_2[:max_idx], Y2[:max_idx], Y3[:max_idx], Y4[:max_idx], Y5[
                                                                                                           :max_idx], Y6[
                                                                                                                      :max_idx]

    x = np.array([i + 1 for i in range(len(Y1))])
    label_font_size = 13
    title_font_size = 15
    # marker = ['r|-', 'b*-', 'y^-', 'm_-', 'g>-', 'cx-']
    marker = ['r', 'g', 'm', 'b', 'y', 'c', 'gray']
    # marker = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']
    # marker = ['r|-', 'g*-', 'm^-', 'bx-', 'y+-', 'c.-', 'gray']
    markersize = 2
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
            r'Link Load Balancing Performance ($\mathrm{{PR_U}}$) among different rerouting schemes in on week (' + str(
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
        plt.title(r'End-to-End Delay Performance ($\mathrm{{PR_\Omega}}$) among different rerouting in on week (' + str(
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


def pr_multi_plot(file1, file2, scheme='mlu', label_name=None, config=None, day_list=None, save_plot=False):
    # fig_size = (18, 16)
    fig_size = (20, 18)
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axis = [(0, 0), (0, 1), (1, 0), (1, 1)]
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

    days = 7
    s = [i * 288 for i in range(days)]
    e = [(i + 1) * 288 - 1 for i in range(days)]

    if day_list is None:
        day_list = [2, 3, 5, 6]
    for day_idx, day in enumerate(day_list):
        x = np.array([i + 1 for i in range(e[day - 1] - s[day - 1])])
        y1 = Y1[s[day - 1]:e[day - 1]]
        y1_2 = Y1_2[s[day - 1]:e[day - 1]]
        y2 = Y2[s[day - 1]:e[day - 1]]
        y3 = Y3[s[day - 1]:e[day - 1]]
        y4 = Y4[s[day - 1]:e[day - 1]]
        y5 = Y5[s[day - 1]:e[day - 1]]
        y6 = Y6[s[day - 1]:e[day - 1]]

        i, j = axis[day_idx]
        # marker = ['r|-', 'b*-', 'y^-', 'm_-', 'g>-', 'cx-']
        marker = ['r', 'g', 'm', 'b', 'y', 'c', 'gray']
        # marker = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']

        # marker = ['r|-', 'g*-', 'm^-', 'bx-', 'y+-', 'c.-', 'gray']
        light_weigt = 0.8
        markersize = 5
        axes[i, j].plot(x, y1, marker[0], markersize=markersize, linewidth=light_weigt, label=label_name[0])
        axes[i, j].plot(x, y1_2, marker[1], markersize=markersize, linewidth=light_weigt, label=label_name[1])
        axes[i, j].plot(x, y3, marker[2], markersize=markersize, linewidth=light_weigt, label=label_name[3])
        axes[i, j].plot(x, y5, marker[3], markersize=markersize, linewidth=light_weigt, label=label_name[5])
        axes[i, j].plot(x, y6, marker[4], markersize=markersize, linewidth=light_weigt, label=label_name[6])

        # axes[i, j].plot(x, y2, marker[2], markersize=markersize, label=label_name[2])
        # axes[i, j].plot(x, y3, marker[3], markersize=markersize, label=label_name[3])
        # axes[i, j].plot(x, y4, marker[4], markersize=markersize, label=label_name[4])
        # axes[i, j].plot(x, y5, marker[5], markersize=markersize, label=label_name[5])
        # axes[i, j].plot(x, y6, marker[6], markersize=markersize, label=label_name[6])
        axes[i, j].legend(loc='lower right')
        axes[i, j].set_xlabel("Traffic Matrix index")
        axes[i, j].set_xlim(0, 288)
        axes[i, j].set_ylim(0.3, 1.01)
        axes[i, j].set_ylabel(y_label)
        axes[i, j].set_title('Day {}'.format(day))

    if scheme == 'mlu':
        plt.show()
        if save_plot:
            if config.method == 'actor_critic':
                fig.savefig(os.getcwd() + '/result/img/all-schemes-curve-ac-pr-mlu-days-' + label_name[0] + '.png',
                            format='png')
            if config.method == 'pure_policy':
                fig.savefig(
                    os.getcwd() + '/result/img/all-schemes-curve-pg-pr-mlu-days-' + label_name[0] + '-' + label_name[
                        1] + '.png',
                    format='png')

    if scheme == 'delay':
        plt.ylim(0.3, 0.95)
        plt.show()
        if save_plot:
            if config.method == 'actor_critic':
                fig.savefig(os.getcwd() + '/result/img/all-schemes-curve-ac-pr-delay-days-' + label_name[0] + '.png',
                            format='png')
            if config.method == 'pure_policy':
                fig.savefig(
                    os.getcwd() + '/result/img/all-schemes-curve-pg-pr-delay-days-' + label_name[0] + '-' + label_name[
                        1] + '.png',
                    format='png')


def cdf_multi_plot(file1, file2, scheme='mlu', label_name=None, config=None, day_list=None, save_plot=False):
    fig_size = (18, 16)
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    df = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)
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
        data1_2 = df2[idx[0]][s[day - 1]:e[day - 1]].to_numpy()

        for data_idx, data in enumerate([data1, data1_2, data2, data3, data4, data5, data6]):
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
        if save_plot:
            if config.method == 'actor_critic':
                fig.savefig(os.getcwd() + '/result/img/all-schemes-cdf-ac-pr-mlu-days-' + label_name[0] + '.png',
                            format='png')
            if config.method == 'pure_policy':
                fig.savefig(
                    os.getcwd() + '/result/img/all-schemes-cdf-pg-pr-mlu-days-' + label_name[0] + '-' + label_name[
                        1] + '.png',
                    format='png')

    if scheme == 'delay':
        plt.show()
        if save_plot:
            if config.method == 'actor_critic':
                fig.savefig(os.getcwd() + '/result/img/all-schemes-cdf-ac-pr-delay-days-' + label_name[0] + '.png',
                            format='png')
            if config.method == 'pure_policy':
                fig.savefig(
                    os.getcwd() + '/result/img/all-schemes-cdf-pg-pr-delay-days-' + label_name[0] + '-' + label_name[
                        1] + '.png',
                    format='png')


def curve_plot(metric='mlu'):
    figsize = (10, 6)
    fig = plt.figure(figsize=figsize)
    if metric == 'mlu':
        k1 = [0.93268, 0.99502, 0.99786, 0.99846]  # CrFRO MLU
        k2 = [0.93833, 0.99345, 0.99765, 0.99857]  # CrFRO MLU
        y_label = r'$\mathrm{{PR_U}}$'
        title = r'The performance of load balancing ($\mathrm{{PR_U}}$) between scheme CrFRO and CrFRO'
        save_path = '/result/img/K-percent-MLU.png'
    if metric == 'delay':
        k1 = [0.82392, 0.842304, 0.827948, 0.81655]  # CrFRO Delay
        k2 = [0.81247, 0.84496, 0.82564, 0.82423]  # CrFRO Delay
        y_label = r'$\mathrm{{PR_\Omega}}$'
        title = r'The performance of end-to-end delay ($\mathrm{{PR_\Omega}}$) between scheme CrFRO and CrFRO'
        save_path = '/result/img/K-percent-dealy.png'

    x = [5, 10, 15, 20]  # 点的横坐标
    color = ['#be5629', '#f4cc66', '#056faf']
    plt.plot(x, k1, 's-', color=color[0], label="CrFRO")
    plt.plot(x, k2, 'o-', color=color[2], label="CrFRO")
    plt.xlabel("Percentage of all OD flows to be re-route(%)")
    plt.xticks([5, 10, 15, 20])
    plt.ylabel(y_label)
    # plt.grid(axis='y')
    plt.legend(loc="best")
    plt.title(title)
    plt.show()
    fig.savefig(os.getcwd() + save_path, format='png')


def time_bar():
    figsize = (10, 5)
    fig = plt.figure(figsize=figsize)
    width = 0.2
    bar_width = width * 0.95
    color_scheme = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']
    X = ['10', '15']
    CeFRO = [380, 305]
    CrFRO = [398, 333]
    CFRRL = [388, 600]

    X_axis = np.arange(len(X))

    plt.bar(X_axis, CeFRO, bar_width, color=color_scheme[0], label='CrFRO')
    plt.bar(X_axis + width, CrFRO, bar_width, color=color_scheme[1], label='CrFRO')
    plt.bar(X_axis + 2 * width, CFRRL, bar_width, color=color_scheme[-2], label='CFR-RL')

    plt.xticks([0.22, 1.22], X)
    plt.xlabel("Pencentage of all OD flows to be re-route(%)", fontsize=13)
    plt.ylabel("Time Used(mins)", fontsize=13)
    plt.title("Bset convegence time of different smart re-rotue schemes with different re-routed OD flows  ",
              fontsize=13)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    config = get_config(FLAGS) or FLAGS

    # curve_plot(metric='mlu')
    # curve_plot(metric='delay')

    # file = 'result/csv/result-pure-policy-baseline-ckpt37.csv'
    # file = 'result/csv/result-pure-policy-alpha-lastK-centralized-sample.csv'
    # file = 'result/csv/result-pure-policy-alpha-lastK-sample.csv'
    # file = 'result/csv/result-pure-policy-alpha+-lastK-sample-0.5scaleK.csv'
    # file1 = 'result/csv/result-pure-policy-alpha-lastK-centralized-sample-0.5scaleK.csv'
    # file1 = 'result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.5scaleK.csv'

    # file1 = 'result/csv/result-pure-policy-baseline-ckpt74.csv'
    # file2 = 'result/csv/result-pure-policy-baseline-ckpt93-maxMoves15.csv'

    file1 = 'result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.25scaleK.csv'
    # file1 = 'result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.4scaleK.csv'
    # file1 = 'result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.2scaleK.csv'
    # file1 = 'result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.25scaleK-maxMoves5.csv'
    # file1 = 'result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.25scaleK-maxMoves13.csv'
    # file1 = 'result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.25scaleK-maxMoves15.csv'
    # file1 = 'result/csv/result-pure-policy-alpha-update-lastK-centralized-sample-0.25scaleK-maxMoves20.csv'

    file2 = 'result/csv/result-pure-policy-baseline-ckpt74.csv'
    # file2 = 'result/csv/result-pure-policy-baseline-ckpt93-maxMoves15.csv'

    label_name = ['CeFRO-0.25k', 'CFR-RL', 'TopK Critical', 'TopK Cum-Centrality', 'TopK Centralized', 'TopK', 'ECMP']
    label_name[0] = label_name[0] + '-' + str(config.max_moves)
    label_name[1] = label_name[1] + '-' + str(config.max_moves)
    save_plot = False
    cdf_plot = False
    day_list = [1, 2, 3, 5]
    data_analyzer(file1, file2, label_name=label_name, config=config, save_plot=save_plot)
    pr_week_plot(file1, file2, scheme='mlu', label_name=label_name, config=config, save_plot=save_plot)
    pr_week_plot(file1, file2, scheme='delay', label_name=label_name, config=config, save_plot=save_plot)
    pr_multi_plot(file1, file2, scheme='mlu', label_name=label_name, config=config, day_list=day_list,
                  save_plot=save_plot)
    pr_multi_plot(file1, file2, scheme='delay', label_name=label_name, config=config, day_list=day_list,
                  save_plot=save_plot)
    if cdf_plot:
        cdf_multi_plot(file1, file2, scheme='mlu', label_name=label_name, config=config, day_list=day_list,
                       save_plot=save_plot)
        cdf_multi_plot(file1, file2, scheme='delay', label_name=label_name, config=config, day_list=day_list,
                       save_plot=save_plot)
