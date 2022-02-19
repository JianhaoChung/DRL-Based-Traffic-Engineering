import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_analyzer(file):
    df = pd.read_csv(file, header=None)

    # print(df.shape, df.info(), df.describe())

    mlu_idx = [1, 3, 5, 7]
    delay_idx = [10, 11, 12, 13]
    print('*Average load balancing performance ratio among different schemes*')
    print(np.mean(df[mlu_idx[0]].to_numpy()), np.mean(df[mlu_idx[1]].to_numpy()),
          np.mean(df[mlu_idx[2]].to_numpy()), np.mean(df[mlu_idx[3]].to_numpy()))

    print('\n*Average end-to-end delay performance ratio among different schemes*')
    print(np.mean(df[delay_idx[0]].to_numpy()), np.mean(df[delay_idx[1]].to_numpy()),
          np.mean(df[delay_idx[2]].to_numpy()), np.mean(df[delay_idx[3]].to_numpy()))


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
    file = 'result/result-actor-critic-baseline.csv'

    # file = 'result/result-actor-critic-alpha+.csv'
    # file = 'result/result-actor-critic-beta.csv'

    # file = 'result/result-actor-critic-debug++.csv'

    # file = 'result/result-pure-policy-baseline.csv'
    # file = 'result/result-pure-policy-alpha+.csv'

    data_analyzer(file)
    cdf_plot_v2(file)
    # exit(1)
    cdf_plot(file, scheme='mlu')
    cdf_plot(file, scheme='delay')

    pr_plot(file, scheme='mlu')
    pr_plot(file, scheme='delay')
