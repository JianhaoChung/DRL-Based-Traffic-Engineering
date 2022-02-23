import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('central_included', True, 'central link included or not')


def data_analyzer(file, label_name):
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
        fig.savefig(os.getcwd() + '/result/img/pr-mlu-delay-' + label_name[0] + '.png', format='png')

    def pr_bar_plot_day(metric_name, day_avg_mlu_save=None, day_avg_delay_save=None, label_name=None):
        figsize = (15, 8)
        fig = plt.figure(figsize=figsize)
        plt.rcParams['font.family'] = 'sans-serif'

        if metric_name == "mlu":
            data = np.array(day_avg_mlu_save).transpose()
        if metric_name == "delay":
            data = np.array(day_avg_delay_save).transpose()

        xvals, yvals, zvals, ivals, jvals, kvals = [data[i].tolist() for i in range(6)]
        N = len(xvals)
        ind = np.arange(N)
        width = 0.15
        zorder = 3

        plt.bar(ind, xvals, width, color='orangered', label=label_name[0], zorder=zorder)
        plt.bar(ind + width, yvals, width, color='y', label=label_name[1], zorder=zorder)
        plt.bar(ind + width * 2, zvals, width, color='cyan', label=label_name[2], zorder=zorder)
        plt.bar(ind + width * 3, ivals, width, color='royalblue', label=label_name[3], zorder=zorder)
        plt.bar(ind + width * 4, jvals, width, color='violet', label=label_name[4], zorder=zorder)
        plt.bar(ind + width * 5, kvals, width, color='b', label=label_name[5], zorder=zorder)

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
            fig.savefig(os.getcwd() + '/result/img/day-pr-mlu-' + label_name[0] + '.png', format='png')

        if metric_name == "delay":
            plt.title(
                r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
                fontsize=15)
            plt.ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=16)
            plt.ylim(0.6, 0.9)
            plt.legend(loc='upper right')
            plt.show()
            fig.savefig(os.getcwd() + '/result/img/day-pr-delay-' + label_name[0] + '.png', format='png')

    if False:
        pr_bar_plot_week('mlu', avg_mlu=avg_mlu)
        pr_bar_plot_week('delay', avg_delay=avg_delay)

    pr_bar_plot_week_integrate(avg_mlu=avg_mlu, avg_delay=avg_delay, label_name=label_name)
    pr_bar_plot_day('mlu', day_avg_mlu_save=day_avg_mlu_save, label_name=label_name)
    pr_bar_plot_day('delay', day_avg_delay_save=day_avg_delay_save, label_name=label_name)


def pr_plot(file, scheme='mlu', label_name=None, day=1, week=False):
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
        fig.savefig(os.getcwd() + '/result/img/curve-pr-mlu-day{}-'.format(day) + label_name[0] + '.png', format='png')
    if scheme == 'delay':
        plt.title("End-to-end delay performance ratio with traffic matrices from Day {}".format(day))
        plt.show()
        fig.savefig(os.getcwd() + '/result/img/curve-pr-delay-day{}-'.format(day) + label_name[0] + '.png',
                    format='png')


def cdf_plot(file, scheme=None):
    df = pd.read_csv(file, header=None)

    if scheme == 'mlu':
        idx = [1, 3, 5, 7]
        x_label = r'$\mathrm{{PR_U}}$'
    if scheme == 'delay':
        idx = [9, 10, 11, 12, 13]
        x_label = r'$\mathrm{{PR_\Omega}}$'

    data = df[idx[0]].to_numpy()
    data1 = df[idx[1]].to_numpy()
    data2 = df[idx[2]].to_numpy()
    data3 = df[idx[3]].to_numpy()

    # getting data of the histogram
    count, bins_count = np.histogram(data, bins=10)
    count1, bins_count1 = np.histogram(data1, bins=10)
    count2, bins_count2 = np.histogram(data2, bins=10)
    count3, bins_count3 = np.histogram(data3, bins=10)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    pdf1 = count / sum(count1)
    pdf2 = count / sum(count2)
    pdf3 = count / sum(count3)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    cdf1 = np.cumsum(pdf1)
    cdf2 = np.cumsum(pdf2)
    cdf3 = np.cumsum(pdf3)

    # plotting PDF and CDF

    # plt.plot(bins_count[1:], pdf, color="red", label="PDF")

    plt.plot(bins_count[1:], cdf, label="Our Method")

    plt.plot(bins_count1[1:], cdf1, label="Top-K Critical")
    plt.plot(bins_count2[1:], cdf2, label="Top-K")
    plt.plot(bins_count3[1:], cdf3, label="ECMP")
    plt.legend()
    plt.ylim(0, 1)
    plt.xlabel(x_label)
    plt.ylabel("CDF")
    plt.show()


def cdf_plot_v2(file, scheme=None):
    df = pd.read_csv(file, header=None)

    np.random.seed(19680801)

    mu = 200
    sigma = 25
    n_bins = 50

    x = df[1].to_numpy()
    x1 = df[3].to_numpy()
    x2 = df[5].to_numpy()
    x3 = df[7].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(x, n_bins, density=True, histtype='step',
                               cumulative=True, label='Our Method')
    n, bins, patches = ax.hist(x1, n_bins, density=True, histtype='step',
                               cumulative=True, label='Top-K Critical')
    n, bins, patches = ax.hist(x2, n_bins, density=True, histtype='step',
                               cumulative=True, label='Top-K')
    n, binss, patches = ax.hist(x3, n_bins, density=True, histtype='step',
                                cumulative=True, label='ECMP')

    # Add a line showing the expected distribution.
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    y = y.cumsum()
    y /= y[-1]

    y1 = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
          np.exp(-0.5 * (1 / sigma * (binss - mu)) ** 2))
    y1 = y1.cumsum()
    y1 /= y1[-1]

    ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')
    ax.plot(binss, y1, 'k--', linewidth=1.5, label='Theoretical2')

    # Overlay a reversed cumulative histogram.
    # ax.hist(x, bins=bins, density=True, histtype='step', cumulative=-1,
    #         label='Reversed emp.')

    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_title('Cumulative step histograms')
    ax.set_xlabel(r'$\mathrm{{PR_U}}$')
    ax.set_ylabel('CDF')
    plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.show()


def cdf_plot_v3(file=None):
    df = pd.read_csv(file, header=None)
    x = df[1].to_numpy()
    print(np.max(x), np.min(x))
    y = 0.25 * np.exp((-x ** 2) / 2)

    y = y / (np.sum(x * y))
    cdf = np.cumsum(y * x)

    plt.plot(x, y, label="pdf")
    plt.plot(x, cdf, label="cdf")
    plt.xlabel("X")
    plt.ylabel("Probability Values")
    plt.title("CDF for continuous distribution")
    plt.xlim(0.6, 1)
    plt.legend()
    plt.show()


def cdf_plot_v5(file, scheme='mlu', label_name=None, day=3, week=False):
    figsize = (12, 6)
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
        fig.savefig(os.getcwd() + '/result/img/cdf-pr-mlu-day{}-'.format(day) + label_name[0] + '.png', format='png')
    if scheme == 'delay':
        fig.savefig(os.getcwd() + '/result/img/cdf-pr-delay-day{}-'.format(day) + label_name[0] + '.png', format='png')


if __name__ == '__main__':
    # file = 'result/result-actor-critic-baseline-ckpt7.csv'

    # file = 'result/csv/result-actor-critic-alpha-ckpt10.csv'

    file = 'result/csv/result-actor-critic-alpha+-ckpt13.csv'

    # file = 'result/csv/result-actor-critic-beta++ckpt27.csv'

    # file = 'result/csv/result-actor-critic-beta++++ckpt9.csv'

    # file = 'result/csv/result-actor-critic-delta-ckpt5.csv'

    # file = 'result/result-pure-policy-baseline.csv'
    # file = 'result/result-pure-policy-alpha+.csv'

    label_name = ['Baseline', 'TopK Critical', 'Centralized-TopK', 'TopK-Centralized', 'TopK', 'ECMP']
    our_method = 'Alpha+'
    label_name[0] = our_method

    data_analyzer(file, label_name=label_name)

    pr_plot(file, scheme='mlu', label_name=label_name, day=3)
    pr_plot(file, scheme='delay', label_name=label_name, day=3)

    cdf_plot_v5(file, scheme='mlu', label_name=label_name, day=5, week=False)
    cdf_plot_v5(file, scheme='delay', label_name=label_name, day=5, week=False)
    exit(1)
    # cdf_plot_v2(file)
    # cdf_plot_v3(file)
    cdf_plot(file, scheme='mlu')
    cdf_plot(file, scheme='delay')
