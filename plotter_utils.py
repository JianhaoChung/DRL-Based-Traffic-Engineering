import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from absl import flags
from config import get_config

FLAGS = flags.FLAGS
flags.DEFINE_boolean('central_included', True, 'central link included or not')


def data_analyzer(file1, file2, label_name, save_plot=False, avg_week_plot=True, avg_day_plot=True, days=6):
    df = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    max_tm_idx = 288 * days  # 288 * 7 (one week) by default
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

    print('#*#*#* Schemes: [PKE-DRL, CFR-RL, TopK-Critical, TopK-Cum-Centrality, TopK-Centralized, TopK, ECMP] *#*#*#')

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

        fig_size = (18, 6)
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
            r'Average load balancing performance ratio ($\mathrm{{PR_U}}$) among different rerouting schemes ' + '\n' + ' in Abilene dataset',
            fontsize=10)
        axes[0].set_ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=12)
        axes[0].set_ylim(0.5, 1)
        axes[0].tick_params(axis='x', rotation=10)

        axes[1].set_title(
            r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different rerouting schemes ' + '\n' + ' in Abilene dataset',
            fontsize=10)
        axes[1].set_ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=12)
        axes[1].set_ylim(0.3, 0.9)
        axes[1].tick_params(axis='x', rotation=10)
        plt.show()

        fig.savefig(os.getcwd() + '/result/img/avg_week_pr_plot.png', format='png')
        fig.savefig(os.getcwd() + '/result/img/eps/avg_week_pr_plot.eps', format='eps')  # for latex vetor diagram

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

        color_scheme = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']
        plt.bar(ind, xvals, bar_width, color=color_scheme[0], label=label_name[0], zorder=zorder)
        plt.bar(ind + width, xxvals, bar_width, color=color_scheme[1], label=label_name[1], zorder=zorder)
        plt.bar(ind + width * 2, zvals, bar_width, color=color_scheme[3], label=label_name[3], zorder=zorder)
        plt.bar(ind + width * 3, jvals, bar_width, color=color_scheme[5], label=label_name[5], zorder=zorder)
        plt.bar(ind + width * 4, kvals, bar_width, color=color_scheme[6], label=label_name[6], zorder=zorder)

        plt.xlabel("Day of a week", loc="center", fontsize=12)
        plt.xticks(ind + width, [i + 1 for i in range(len(xvals))], fontsize=12)

        if metric_name == "mlu":
            plt.title(
                r'Average load balancing performance ratio ($\mathrm{{PR_U}}$) among different schemes in one week',
                fontsize=12)
            plt.ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=16)
            plt.ylim(0.6, 1)
            plt.legend(loc='upper right')
            plt.show()
            if save_plot:
                fig.savefig(os.getcwd() + '/result/img/avg-day-mlu.png', format='png')
                # fig.savefig(os.getcwd() + '/result/img/avg-day-mlu.eps', format='eps') # for latex

        if metric_name == "delay":
            plt.title(
                r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
                fontsize=12)
            plt.ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=16)
            plt.ylim(0.35, 0.9)
            plt.legend(loc='upper right')
            plt.show()
            if save_plot:
                fig.savefig(os.getcwd() + '/result/img/avg-day-delay.png', format='png')
                # fig.savefig(os.getcwd() + '/result/img/avg-day-delay.eps', format='eps') # for latex

    if avg_week_plot:
        pr_bar_plot_week_integrate(avg_mlu=avg_mlu, avg_delay=avg_delay, label_name=label_name)
    if avg_day_plot:
        pr_bar_plot_day('mlu', day_avg_mlu_save=day_avg_mlu_save, label_name=label_name, days=6)
        pr_bar_plot_day('delay', day_avg_delay_save=day_avg_delay_save, label_name=label_name, days=6)


def pr_week_detail_plot(file1, file2, scheme='mlu', label_name=None, save_plot=False, days=6):
    fig_size = (25, 6)
    label_font_size = 10
    title_font_size = 12
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

    max_idx = 288 * days
    Y1, Y1_2, Y2, Y3, Y4, Y5, Y6 = Y1[:max_idx], Y1_2[:max_idx], Y2[:max_idx], \
                                   Y3[:max_idx], Y4[:max_idx], Y5[:max_idx], Y6[:max_idx]

    x = np.array([i + 1 for i in range(len(Y1))])

    # marker = ['r|-', 'b*-', 'y^-', 'm_-', 'g>-', 'cx-']
    marker = ['#0073C2', '#A73030', '#EFC000', '#4A6990', '#868686', '#f4cc66', '#EFC000', '#808180FF', '#79AF97']
    markersize = 2
    light_weigt = 0.6
    plt.plot(x, Y1, marker[-1], markersize=markersize, linewidth=light_weigt, label=label_name[0])
    plt.plot(x, Y1_2, marker[1], markersize=markersize, linewidth=light_weigt, label=label_name[1])
    plt.plot(x, Y3, marker[2], markersize=markersize, linewidth=light_weigt, label=label_name[3])
    plt.plot(x, Y5, marker[3], markersize=markersize, linewidth=light_weigt, label=label_name[5])
    plt.plot(x, Y6, marker[4], markersize=markersize, linewidth=light_weigt, label=label_name[6])

    plt.legend(loc='lower right')
    plt.xlabel("Traffic Matrix index", fontsize=label_font_size)
    plt.xlim(0, len(Y1))
    plt.ylabel(y_label, weight="bold", fontsize=label_font_size)

    if scheme == 'mlu':
        plt.ylim(0.5, 1.005)
        plt.title(
            r'Load balancing performance ratio ($\mathrm{{PR_U}}$) among different rerouting schemes in Abilene dataset (' + str(
                len(Y1)) + ' TMs)',
            fontsize=title_font_size)
        plt.show()

    if scheme == 'delay':
        plt.ylim(0.2, 0.95)
        plt.title(
            r'End-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different rerouting schemes in Abilene dataset (' + str(
                len(Y1)) + ' TMs)',
            fontsize=title_font_size)
        plt.show()

    if save_plot:
        fig.savefig(os.getcwd() + '/result/img/week_details_{}.png'.format(scheme), format='png')
        fig.savefig(os.getcwd() + '/result/img/eps/week_details_{}.eps'.format(scheme), format='eps')


def pr_multi_plot(file1, file2, scheme='mlu', label_name=None, day_list=None, save_plot=False):
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
        marker = ['r', 'g', 'm', 'b', 'y', 'c', 'gray']
        light_weigt = 0.8
        markersize = 5
        axes[i, j].plot(x, y1, marker[0], markersize=markersize, linewidth=light_weigt, label=label_name[0])
        axes[i, j].plot(x, y1_2, marker[1], markersize=markersize, linewidth=light_weigt, label=label_name[1])
        axes[i, j].plot(x, y3, marker[2], markersize=markersize, linewidth=light_weigt, label=label_name[3])
        axes[i, j].plot(x, y5, marker[4], markersize=markersize, linewidth=light_weigt, label=label_name[5])
        axes[i, j].plot(x, y6, marker[-1], markersize=markersize, linewidth=light_weigt, label=label_name[6])

        axes[i, j].legend(loc='lower right')
        axes[i, j].set_xlabel("Traffic Matrix index")
        axes[i, j].set_xlim(0, 288)
        axes[i, j].set_ylim(0.3, 1.01)
        axes[i, j].set_ylabel(y_label)
        axes[i, j].set_title('Day {}'.format(day))
        if scheme == 'delay':
            plt.ylim(0.3, 0.95)
        plt.show()
        if save_plot:
            fig.savefig(os.getcwd() + '/result/img/pr_{}_multi_days.png'.format(scheme), format='png')
            fig.savefig(os.getcwd() + '/result/img/eps/pr_{}_multi_days.eps'.format(scheme), format='eps')


def abilene_data(model=None, max_moves=10, pke_drl_idx=None):
    if model == 'PKE-DRL':
        if max_moves == 10:
            pke_drl_result = ['result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.2scaleK_maxMoves10.csv',
                              'result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.25_scaleK_maxMoves10.csv',
                              'result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.4scaleK_maxMoves10.csv',
                              'result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.5scaleK_maxMoves10.csv'
                              ]
        if max_moves == 15:
            pke_drl_result = ['result/csv/result_pure_policy_conv_Abilene_alpha_update_maxMoves15.csv',
                              'result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.2scaleK_maxMoves15.csv',
                              'result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.25scaleK_maxMoves15.csv',
                              'result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.4scaleK_maxMoves15.csv',
                              'result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.5scaleK_maxMoves15.csv'
                              ]
        if max_moves == 20:
            pke_drl_result = ['result/csv/result_pure_policy_conv_Abilene_alpha_update_lastK_centralized_sample_0.25scaleK_maxMoves20.csv']

        return pke_drl_result[pke_drl_idx]

    if model == 'CFR-RL':
        if max_moves == 10:
            cfr_rl_result = 'result/csv/result_pure_policy_conv_Abilene_baseline_maxMoves10_ckpt74.csv'
        if max_moves == 15:
            cfr_rl_result = 'result/csv/result_pure_policy_conv_Abilene_baseline_maxMoves15.csv'
        if max_moves == 20:
            cfr_rl_result = 'result/csv/result_pure_policy_conv_Abilene_baseline_maxMoves20.csv'

        return cfr_rl_result


def ebone_data(model=None, max_moves=10):
    if model == 'PKE-DRL':
        if max_moves == 10:
            pke_drl_result = 'result/csv/result_pure_policy_conv_Ebone_alpha_update_lastK_centralized_sample_0.25scaleK_maxMoves10.csv'
        if max_moves == 15:
            pke_drl_result = 'result/csv/result_pure_policy_conv_Ebone_alpha_update_lastK_centralized_sample_0.25scaleK_maxMoves15.csv'
        if max_moves == 20:
            pke_drl_result = 'result/csv/result_pure_policy_conv_Ebone_alpha_update_lastK_centralized_sample_0.25scaleK_maxMoves20.csv'
        return pke_drl_result

    if model == 'CFR-RL':
        if max_moves == 10:
            cfr_rl_result = 'result/csv/result_pure_policy_conv_Ebone_baseline_maxMoves10.csv'
        if max_moves == 15:
            cfr_rl_result = 'result/csv/result_pure_policy_conv_Ebone_baseline_maxMoves15.csv'
        if max_moves == 20:
            cfr_rl_result = 'result/csv/result_pure_policy_conv_Ebone_baseline_maxMoves20.csv'
        return cfr_rl_result

    # file1 = 'result/csv/result_pure_policy_conv_Ebone_alpha_update_lastK_centralized_sample_0.25scaleK_maxMoves10_new.csv'
    # file2 = 'result/csv/result_pure_policy_conv_Ebone_baseline_maxMoves10_new.csv'


def pr_plot_abilene(label_name=None, save_plot=False):

    avg_mlu_p10 = [0.9950249904829205, 0.9942723496690274, 0.9506918908566959, 0.9592661821993126, 0.957341811085134,
                   0.8784579441315484, 0.6774799910966899]
    avg_delay_p10 = [0.8423043413059632, 0.8446440719681698, 0.7978797445196646, 0.8060997609634855, 0.8274249675268561,
                     0.7117787583829724, 0.5043758020673198]

    avg_mlu_p15 = [0.9978671575045642, 0.9975918254855103, 0.9741628785722924, 0.9701071425519396, 0.9579092840065208,
                   0.9400014085811664, 0.6774799910966899]
    avg_delay_p15 = [0.8279481036673891, 0.8296127198139992, 0.7916017487913771, 0.793938386350183, 0.8006170389820378,
                     0.7589892753158913, 0.5043758020673198]

    avg_mlu_p20 = [0.9984664571247994, 0.9982207719037961, 0.9864062285741213, 0.9774525642335924, 0.9693157419437542, 0.9684277578433246, 0.6774799910966899]
    avg_delay_p20 = [0.8165517056168974, 0.8154938744984617, 0.7856863590303923, 0.7868262529474731, 0.8177119350994488, 0.7554193272919896, 0.5043758020673198]

    fig_size = (18, 6.5)
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=False, gridspec_kw={'width_ratios': [1, 1]})
    plt.rcParams['font.family'] = 'sans-serif'

    avg_mlu_p10 = [avg_mlu_p10[0], avg_mlu_p10[1], avg_mlu_p10[3], avg_mlu_p10[-2], avg_mlu_p10[-1]]
    avg_delay_p10 = [avg_delay_p10[0], avg_delay_p10[1], avg_delay_p10[3], avg_delay_p10[-2], avg_delay_p10[-1]]

    avg_mlu_p15 = [avg_mlu_p15[0], avg_mlu_p15[1], avg_mlu_p15[3], avg_mlu_p15[-2], avg_mlu_p15[-1]]
    avg_delay_p15 = [avg_delay_p15[0], avg_delay_p15[1], avg_delay_p15[3], avg_delay_p15[-2], avg_delay_p15[-1]]

    avg_mlu_p20 = [avg_mlu_p20[0], avg_mlu_p20[1], avg_mlu_p20[3], avg_mlu_p20[-2], avg_mlu_p20[-1]]
    avg_delay_p20 = [avg_delay_p20[0], avg_delay_p20[1], avg_delay_p20[3], avg_delay_p20[-2], avg_delay_p15[-1]]

    label_name = [label_name[0], label_name[1], label_name[3], label_name[-2], label_name[-1]]

    width = 0.2
    bar_width = width * 0.95

    X_axis = np.arange(len(label_name))

    color_scheme = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']

    axes[0].bar(X_axis, avg_mlu_p10, bar_width, color=color_scheme[0], label='10%')
    axes[0].bar(X_axis + width, avg_mlu_p15, bar_width, color=color_scheme[1], label='15%')
    axes[0].bar(X_axis + 2 * width, avg_mlu_p20, bar_width, color=color_scheme[5], label='20%')
    axes[0].set_xticks(np.arange(5) + 0.15)
    axes[0].set_xticklabels(label_name)

    axes[1].bar(X_axis, avg_delay_p10, bar_width, color=color_scheme[0], label='10%')
    axes[1].bar(X_axis + width, avg_delay_p15, bar_width, color=color_scheme[1], label='15%')
    axes[1].bar(X_axis + 2 * width, avg_delay_p20, bar_width, color=color_scheme[5], label='20%')
    axes[1].set_xticks(np.arange(5) + 0.15)
    axes[1].set_xticklabels(label_name)

    axes[0].set_ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=12)
    axes[0].set_ylim(0.5, 1)
    axes[0].tick_params(axis='x', rotation=10)
    axes[0].legend()

    axes[1].set_ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=12)
    axes[1].set_ylim(0.3, 0.9)
    axes[1].tick_params(axis='x', rotation=10)
    axes[1].legend()
    plt.show()

    if save_plot:
        fig.savefig(os.getcwd() + '/result/img/pr_abilene_maxmoves.png', format='png')
        fig.savefig(os.getcwd() + '/result/img/eps/pr_abilene_maxmoves.png', format='eps')


def pr_plot_ebone(label_name=None, save_plot=False):
    avg_mlu_p10 = [0.9056523728685594, 0.9101181743049621, 0.8363944274810661, 0.6478968876080679, 0.6701661600437573,
                   0.7083088300527904, 0.5514805420906973]
    avg_delay_p10 = [0.8942038387724837, 0.8788793944458101, 0.8233073365239438, 0.7082302833266743, 0.7864387091496052,
                     0.7670693836826263, 0.6018860610692244]

    avg_mlu_p15 = [0.9994672508241744, 0.9956259497694926, 0.889453375487612, 0.6791347139813715, 0.7063108839235823,
                   0.7803012254723772, 0.5514805420906973]
    avg_delay_p15 = [0.874655332730042, 0.8732619157475815, 0.8306033674725836, 0.6955506909239392, 0.7329244588080907,
                     0.7725107858578288, 0.6018860610692244]

    avg_mlu_p20 = [0.9999452722452786, 1.0000000099075164, 0.9552565264542654, 0.705269041330125, 0.7396612036937912, 0.8312208720255662, 0.5514805420906973]
    avg_delay_p20 = [0.8731026782772542, 0.8705796773749591, 0.8206036847463571, 0.7116194450211883, 0.749580092653247, 0.7714145445126336, 0.6018860610692244]

    fig_size = (18, 6.5)
    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharey=False, gridspec_kw={'width_ratios': [1, 1]})
    plt.rcParams['font.family'] = 'sans-serif'

    avg_mlu_p10 = [avg_mlu_p10[0], avg_mlu_p10[1], avg_mlu_p10[3], avg_mlu_p10[-2], avg_mlu_p10[-1]]
    avg_delay_p10 = [avg_delay_p10[0], avg_delay_p10[1], avg_delay_p10[3], avg_delay_p10[-2], avg_delay_p10[-1]]

    avg_mlu_p15 = [avg_mlu_p15[0], avg_mlu_p15[1], avg_mlu_p15[3], avg_mlu_p15[-2], avg_mlu_p15[-1]]
    avg_delay_p15 = [avg_delay_p15[0], avg_delay_p15[1], avg_delay_p15[3], avg_delay_p15[-2], avg_delay_p15[-1]]

    avg_mlu_p20 = [avg_mlu_p20[0], avg_mlu_p20[1], avg_mlu_p20[3], avg_mlu_p20[-2], avg_mlu_p20[-1]]
    avg_delay_p20 = [avg_delay_p20[0], avg_delay_p20[1], avg_delay_p20[3], avg_delay_p20[-2], avg_delay_p20[-1]]

    label_name = [label_name[0], label_name[1], label_name[3], label_name[-2], label_name[-1]]

    width = 0.2
    bar_width = width * 0.95

    X_axis = np.arange(len(label_name))

    color_scheme = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']

    axes[0].bar(X_axis, avg_mlu_p10, bar_width, color=color_scheme[0], label='10%')
    axes[0].bar(X_axis + width, avg_mlu_p15, bar_width, color=color_scheme[1], label='15%')
    axes[0].bar(X_axis + 2*width, avg_mlu_p20, bar_width, color=color_scheme[5], label='20%')
    axes[0].set_xticks(np.arange(5) + 0.15)
    axes[0].set_xticklabels(label_name)

    axes[1].bar(X_axis, avg_delay_p10, bar_width, color=color_scheme[0], label='10%')
    axes[1].bar(X_axis + width, avg_delay_p15, bar_width, color=color_scheme[1], label='15%')
    axes[1].bar(X_axis + 2 * width, avg_delay_p20, bar_width, color=color_scheme[5], label='20%')
    axes[1].set_xticks(np.arange(5) + 0.15)
    axes[1].set_xticklabels(label_name)

    axes[0].set_ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=12)
    axes[0].set_ylim(0.5, 1)
    axes[0].tick_params(axis='x', rotation=10)
    axes[0].legend()

    axes[1].set_ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=12)
    axes[1].set_ylim(0.3, 0.9)
    axes[1].tick_params(axis='x', rotation=10)
    axes[1].legend()
    plt.show()

    if save_plot:
        fig.savefig(os.getcwd() + '/result/img/pr_ebone_maxmoves.png', format='png')
        fig.savefig(os.getcwd() + '/result/img/eps/pr_ebone_maxmoves.eps', format='eps')


def flow_sampling_effect_pr_plot(metric='mlu', save_plot=False):
    figsize = (13, 6)
    fig = plt.figure(figsize=figsize)
    fontsize = 12
    if metric == 'mlu':
        k = [0.99246,
             0.9932432720881841,
             0.9950249904829205,
             0.993769993415826,
             0.9939921483280257]
        k2 = [0.9976456283569746,
              0.997816974350677,
              0.9978671575045642,
              0.9977656988337685,
              0.9979200536934301
              ]
        y_label = r'$\mathrm{{PR_U}}$'
        # title = r'Load balancing performance ratio ($\mathrm{{PR_U}}$) of PKE-DRL with different mixed sampling flows'
        title = r'Load balancing performance ratio ($\mathrm{{PR_U}}$) of PKE-DRL when mixed sampling different ratio of TopK central flows'

    if metric == 'delay':
        k = [0,
             0.8423043413059632,
             0.8419957835510535,
             0.8442336107805992,
             0.84222
             ]
        k2 = []
        label = "Delay"
        y_label = r'$\mathrm{{PR_\Omega}}$'
        title = r'End-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) of PKE-DRL with different hybird sampling OD flows'

    x = [5, 10, 15, 20, 25]
    color = ['#be5629', '#f4cc66', '#056faf']
    plt.plot(x, k, 's-', color=color[0], label='10%')
    plt.plot(x, k2, 's-', color=color[2], label="15%")

    # plt.xlabel("Different sampling ratio between TopK central flows and LastM central flows", fontsize=fontsize)
    # plt.xticks([5, 10, 15, 20, 25], ['4:1', '3:1', '3:2', '1:1', '1:0'])

    plt.xlabel("Different sampling ratio of TopK central flows in mixed sampling method", fontsize=fontsize)
    plt.xticks([5, 10, 15, 20, 25], ['100%', '80%', '75%', '60%', '50%'])
    plt.ylabel(y_label, fontsize=fontsize)
    # plt.grid(axis='y')
    plt.legend(loc="best")
    plt.title(title, fontsize=fontsize)
    plt.show()
    if save_plot:
        fig.savefig(os.getcwd() + '/result/img/K-sample-{}.png'.format(metric), format='png')
        fig.savefig(os.getcwd() + '/result/img/eps/K-sample-{}.eps'.format(metric), format='eps')


def convergence_time_plot(topo='Abilene', save_plot=False):
    figsize = (13, 6)
    fig = plt.figure(figsize=figsize)
    width = 0.2
    bar_width = width * 0.95
    color_scheme = ['#be5629', '#056faf', '#b79561', '#4e9595', '#758837', '#f4cc66', '#808180FF']
    X = ['10', '15', '20']
    if topo == 'Abilene':
        PKEDRL = [380, 305, 554]
        CFRRL = [388, 600, 761]
    if topo == 'Ebone':
        PKEDRL = [2118, 4230, 4500]
        CFRRL = [2804, 4569, 5790]

    X_axis = np.arange(len(X))

    plt.bar(X_axis, PKEDRL, bar_width, color=color_scheme[0], label='PKE-DRL')
    plt.bar(X_axis + width, CFRRL, bar_width, color=color_scheme[1], label='CFR-RL')

    plt.xticks([0.1, 1.1, 2.1], X)
    plt.xlabel("Pencentage of all OD flows to be re-route(%)", fontsize=13)
    plt.ylabel("Time Used(mins)", fontsize=13)
    plt.title(
        "Convergence time of PKE-DRL and CFR-RL when rerouting different ratio of flows for {} Network".format(topo),
        fontsize=10)
    plt.legend()
    plt.show()
    if save_plot:
        fig.savefig(os.getcwd() + '/result/img/convergence_time_{}.png'.format(topo), format='png')
        fig.savefig(os.getcwd() + '/result/img/eps/convergence_time_{}.eps'.format(topo), format='eps')


def ebone_data_analyzer(file1, file2, label_name, config, max_tm_idx=30, save_plot=False):
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

    print('#*#*#* Schemes: [PKE-DRL, CFR-RL, TopK-Critical, TopK-Cum-Centrality, TopK-Centralized, TopK, ECMP] *#*#*#')
    print(avg_mlu)
    print(avg_delay)

    def pr_plot(file1, file2, scheme='mlu', label_name=None, config=None, save_plot=False):
        # fig_size = (20, 6)
        fig_size = (10, 5)
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
        label_font_size = 10
        title_font_size = 10
        marker = ['r|-', 'b*-', 'y^-', 'm_-', 'g>-', 'cx-']
        markersize = 2.5
        light_weigt = 0.6
        plt.plot(x, Y1, marker[0], markersize=markersize, linewidth=light_weigt, label=label_name[0])
        plt.plot(x, Y1_2, marker[1], markersize=markersize, linewidth=light_weigt, label=label_name[1])
        plt.plot(x, Y3, marker[2], markersize=markersize, linewidth=light_weigt, label=label_name[3])
        plt.plot(x, Y5, marker[3], markersize=markersize, linewidth=light_weigt, label=label_name[5])
        plt.plot(x, Y6, marker[4], markersize=markersize, linewidth=light_weigt, label=label_name[6])

        plt.legend(loc='lower right')
        # plt.xlabel("Traffic Matrix index", weight="bold", fontsize=label_font_size)
        plt.xlabel("Traffic Matrix index", fontsize=label_font_size)
        plt.xlim(0, len(Y1))
        plt.ylabel(y_label, weight="bold", fontsize=label_font_size)
        plt.grid(axis='y')
        plt.axvline(x=15, color='k', linewidth=0.5)
        if scheme == 'mlu':
            plt.ylim(0.2, 1.005)
            plt.title(r'Exponential TM                                          Uniform TM    ',
                      fontsize=title_font_size)

            plt.show()

        if scheme == 'delay':
            plt.ylim(0.2, 1)

            plt.title(r'Exponential TM                                          Uniform TM    ',
                      fontsize=title_font_size)
            plt.show()

        if save_plot:
            fig.savefig(os.getcwd() + '/result/img/ebone_{}_pr_details_ebone.png'.format(scheme), format='png')
            fig.savefig(os.getcwd() + '/result/img/eps/ebone_{}_pr_details_ebone.eps'.format(scheme), format='eps')

    pr_plot(file1, file2, scheme='mlu', label_name=label_name, config=config, save_plot=save_plot)
    pr_plot(file1, file2, scheme='delay', label_name=label_name, config=config, save_plot=save_plot)


if __name__ == '__main__':
    config = get_config(FLAGS) or FLAGS

    file1 = abilene_data(model='PKE-DRL', max_moves=10, pke_drl_idx=0)
    file2 = abilene_data(model='CFR-RL', max_moves=10)

    save_plot = False
    daily_performance_4_frames = False

    if os.path.exists(file1) and os.path.exists(file2):

        data_analyzer(file1, file2, label_name=config.label_name, config=config, save_plot=save_plot,
                      avg_week_plot=True, avg_day_plot=False)
        pr_week_detail_plot(file1, file2, scheme='mlu', label_name=config.label_name, config=config,
                            save_plot=save_plot)
        pr_week_detail_plot(file1, file2, scheme='delay', label_name=config.label_name, config=config,
                            save_plot=save_plot)

        if daily_performance_4_frames:
            day_list = [1, 2, 3, 5]
            pr_multi_plot(file1, file2, scheme='mlu', label_name=config.label_name, config=config, day_list=day_list,
                          save_plot=save_plot)
            pr_multi_plot(file1, file2, scheme='delay', label_name=config.label_name, config=config, day_list=day_list,
                          save_plot=save_plot)
