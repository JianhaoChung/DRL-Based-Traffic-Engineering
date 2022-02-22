import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('central_included', True, 'central link included or not')


def data_analyzer(file, label_name, central=False):
    df = pd.read_csv(file, header=None)
    # print(df.shape, df.info(), df.describe())
    mlu_idx = [1, 3, 5, 7]
    delay_idx = [11, 12, 13, 16]
    max_tm_idx = 288 * 6  # 288 * 7 (one week) by default
    if central:
        mlu_idx = [1, 3, 5, 7, 9]
        delay_idx = [11, 12, 13, 14, 16]

    avg_mlu = [np.mean(df[mlu_idx[0]][:max_tm_idx].to_numpy()), np.mean(df[mlu_idx[1]][:max_tm_idx].to_numpy()),
               np.mean(df[mlu_idx[2]][:max_tm_idx].to_numpy()), np.mean(df[mlu_idx[3]][:max_tm_idx].to_numpy())]

    avg_delay = [np.mean(df[delay_idx[0]][:max_tm_idx].to_numpy()), np.mean(df[delay_idx[1]][:max_tm_idx].to_numpy()),
                 np.mean(df[delay_idx[2]][:max_tm_idx].to_numpy()), np.mean(df[delay_idx[3]][:max_tm_idx].to_numpy())]
    if central:
        avg_delay.append(np.mean(df[mlu_idx[-1]][:max_tm_idx].to_numpy()))
        avg_mlu.append(np.mean(df[mlu_idx[-1]][:max_tm_idx].to_numpy()))
    if central:
        print('#*#*#* Schemes: [DRL-Policy, TopK-Critical, TopK-Centralized, TopK, ECMP] *#*#*#')
    else:
        print('#*#*#* Schemes: [DRL-Policy, TopK-Critical, TopK, ECMP] *#*#*#')

    days = 7
    s = [i * 288 for i in range(days)]
    e = [(i + 1) * 288 - 1 for i in range(days)]

    day_avg_mlu_save = []
    day_avg_delay_save = []

    for i in range(days):
        df_segment = df[s[i]:e[i]]

        day_avg_mlu = [np.mean(df_segment[mlu_idx[0]].to_numpy()), np.mean(df_segment[mlu_idx[1]].to_numpy()),
                       np.mean(df_segment[mlu_idx[2]].to_numpy()), np.mean(df_segment[mlu_idx[3]].to_numpy())]

        day_avg_delay = [np.mean(df_segment[delay_idx[0]].to_numpy()), np.mean(df_segment[delay_idx[1]].to_numpy()),
                         np.mean(df_segment[delay_idx[2]].to_numpy()), np.mean(df_segment[delay_idx[3]].to_numpy())]
        if central:
            day_avg_delay.append(np.mean(df_segment[mlu_idx[-1]].to_numpy()))
            day_avg_mlu.append(np.mean(df_segment[mlu_idx[-1]].to_numpy()))

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
        axes[1].set_title(
            r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
            fontsize=10)
        axes[1].set_ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=12)
        axes[1].set_ylim(0.6, 0.85)
        plt.show()
        fig.savefig(os.getcwd() + '/result/img/pr-mlu-delay-'+ label_name[0] + '.png', format='png')

    def pr_bar_plot_day(metric_name, day_avg_mlu_save=None, day_avg_delay_save=None, label_name=None):
        figsize = (15, 8)
        fig = plt.figure(figsize=figsize)
        plt.rcParams['font.family'] = 'sans-serif'

        if metric_name == "mlu":
            data = np.array(day_avg_mlu_save).transpose()
        if metric_name == "delay":
            data = np.array(day_avg_delay_save).transpose()

        xvals, yvals, zvals, mvals, nvals = [data[i].tolist() for i in range(5)]
        N = len(xvals)
        ind = np.arange(N)
        width = 0.15
        zorder = 3

        plt.bar(ind, xvals, width, color='orangered', label=label_name[0], zorder=zorder)
        plt.bar(ind + width, yvals, width, color='y', label=label_name[1], zorder=zorder)
        plt.bar(ind + width * 2, zvals, width, color='cyan', label=label_name[2], zorder=zorder)
        plt.bar(ind + width * 3, mvals, width, color='royalblue', label=label_name[3], zorder=zorder)
        plt.bar(ind + width * 4, nvals, width, color='violet', label=label_name[4], zorder=zorder)

        plt.xlabel("Days", loc="center", fontweight='bold', fontsize=16)
        plt.xticks(ind + width, [i + 1 for i in range(len(xvals))], fontsize=13)

        if metric_name == "mlu":
            plt.title(
                r'Average load balancing performance ratio ($\mathrm{{PR_U}}$) among different schemes in one week',
                fontsize=15)
            plt.ylabel(r'$\mathrm{{PR_U}}$', weight="bold", fontsize=16)
            plt.ylim(0.6, 1)
            plt.legend()
            plt.show()
            fig.savefig(os.getcwd() + '/result/img/day-pr-mlu-' + label_name[0] + '.png', format='png')

        if metric_name == "delay":
            plt.title(
                r'Average end-to-end delay performance ratio ($\mathrm{{PR_\Omega}}$) among different schemes in one week',
                fontsize=15)
            plt.ylabel(r'$\mathrm{{PR_\Omega}}$', weight="bold", fontsize=16)
            plt.ylim(0.6, 0.9)
            plt.legend()
            plt.show()
            fig.savefig(os.getcwd() + '/result/img/day-pr-delay-' + label_name[0] + '.png', format='png')

    if False:
        pr_bar_plot_week('mlu', avg_mlu=avg_mlu)
        pr_bar_plot_week('delay', avg_delay=avg_delay)

    pr_bar_plot_week_integrate(avg_mlu=avg_mlu, avg_delay=avg_delay, label_name=label_name)
    pr_bar_plot_day('mlu', day_avg_mlu_save=day_avg_mlu_save, label_name=label_name)
    pr_bar_plot_day('delay', day_avg_delay_save=day_avg_delay_save, label_name=label_name)


def pr_plot(file, scheme='mlu'):
    df = pd.read_csv(file, header=None)

    if scheme == 'mlu':
        idx = [1, 3, 5, 7]
        y_label = r'$\mathrm{{PR_U}}$'
    if scheme == 'delay':
        idx = [10, 11, 12, 13]
        y_label = r'$\mathrm{{PR_\Omega}}$'

    y1 = df[idx[0]].to_numpy()
    y2 = df[idx[1]].to_numpy()
    y3 = df[idx[2]].to_numpy()
    y4 = df[idx[3]].to_numpy()

    day = 3
    tm_idx = [288 * (i + 1) for i in range(7)]
    start = tm_idx[day - 1] + 1
    end = tm_idx[day]

    x = np.array([i + 1 for i in range(end - start)])
    y1 = y1[start:end]
    y2 = y2[start:end]
    y3 = y3[start:end]
    y4 = y4[start:end]

    # plt.plot(x, y1, 'r--',  x, y2, 'b*', x, y3, 'g+', x, y4, 'c')
    plt.plot(x, y1, 'r--', label="Our Method")
    plt.plot(x, y2, 'b*', label="Top-K Critical")
    plt.plot(x, y3, 'g+', label="Top-K")
    plt.plot(x, y4, 'c', label="ECMP")
    plt.legend()
    plt.title("Curve plotted using the given points")
    plt.xlabel("Traffic Matrix index")
    plt.ylabel(y_label)
    plt.show()


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
    n, bins, patches = ax.hist(x3, n_bins, density=True, histtype='step',
                               cumulative=True, label='ECMP')

    # Add a line showing the expected distribution.
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    #      np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    # y = y.cumsum()
    # y /= y[-1]
    #
    # ax.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')

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


if __name__ == '__main__':
    # file = 'result/result-actor-critic-baseline.csv'
    # file = 'result/result-actor-critic-baseline-ckpt7.csv'

    # file = 'result/result-actor-critic-alpha+.csv'
    # file = 'result/result-actor-critic-beta.csv'
    # file = 'result/result-actor-critic-beta+.csv'

    # file = 'result/result-actor-critic-beta++.csv' # *
    # file = 'result/result-actor-critic-beta++ckpt27.csv'
    file = 'result/result-actor-critic-beta++ckpt91.csv'

    # file = 'result/result-actor-critic-beta++++.csv'
    # file = 'result/result-actor-critic-beta++++ckpt9.csv'

    # file = 'result/result-actor-critic-debug++.csv'

    # file = 'result/result-pure-policy-baseline.csv'
    # file = 'result/result-pure-policy-alpha+.csv'

    label_name = ['Baseline', 'TopK Critical', 'TopK Centralized', 'TopK', 'ECMP']
    our_method = 'Method Beta++'
    label_name[0] = our_method

    data_analyzer(file, central=True, label_name=label_name)
    exit(1)

    cdf_plot_v2(file)
    cdf_plot(file, scheme='mlu')
    cdf_plot(file, scheme='delay')

    pr_plot(file, scheme='mlu')
    pr_plot(file, scheme='delay')
