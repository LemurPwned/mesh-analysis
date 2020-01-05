import os

import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def laplacian_plot(filename='./eigenvals.npy'):
    eigenvals = np.load(filename)
    fig, ax = plt.subplots()
    eigenvals = np.abs(eigenvals[eigenvals <= (np.mean(eigenvals) + 2*np.std(eigenvals))])
    plot_histogram(ax, eigenvals, fit=True)
    plt.legend()
    plt.show()

def KL_divergence(bins, H1, H2):
    KL_sum = 0
    for b in bins:
        if H2[b] and H1[b]:
            KL_sum += H1[b] * np.log2(H1[b] / H2[b])
    return KL_sum

def hist_cmp(bins, H1, H2):
    hsum = np.sum(np.minimum(H1, H2))
    return hsum/np.sum(H1)

def compare_histograms(f1, f2, fn=hist_cmp):
    rgb1 = np.load(f1)
    rgb2 = np.load(f2)

    for i in range(3):
        rgb1_norm = np.sum(rgb1[:, i])
        rgb2_norm = np.sum(rgb1[:, i])
        kl = fn(range(rgb1.shape[0]), rgb1[:, i] / rgb1_norm,
                           rgb2[:, i] / rgb2_norm)
        invkl = fn(range(rgb1.shape[0]), rgb2[:, i] / rgb2_norm,
                              rgb1[:, i] / rgb1_norm)
        print(kl, invkl)


def rbg_hist(filename):
    mesh_rgb = np.load(filename)
    fig, ax = plt.subplots()

    nbins = 50
    for i, col in enumerate(['r', 'g', 'b']):
        plot_histogram(ax, mesh_rgb[:, i], color=col, stats=False)
    plt.title(f'RGB distribution histogram')
    plt.xlabel('RGB value')
    plt.ylabel('Value count')
    plt.legend()
    plt.show()


def gaussian_fn(x, *p):
    A, mean, sigma = p
    return A * np.exp(-(x - mean)**2 / (2. * sigma**2))


def plot_histogram(ax, hist_data, bins=50, color='g', stats=True, fit=True):
    hist, bins = np.histogram(hist_data, bins=bins, density=True)
    bin_cntr = (bins[:-1] + bins[1:]) / 2
    width = 0.9 * (bins[1] - bins[0])

    """
    Fit histogram
    """
    params_found = False
    if fit:
        p0 = [1., 0, 1.]
        try:
            coeff, var_matrix = curve_fit(gaussian_fn, bin_cntr, hist, p0=p0)
            print("A: {} mean: {} std: {}".format(*coeff))
            generated_gaussian = [gaussian_fn(x, *coeff) for x in bin_cntr]
            params_found = True 
        except RuntimeError:
            params_found = False 
    if stats:
        mean_quality = np.mean(hist_data)
        std_quality = np.around(np.std(hist_data), decimals=2)

    ax.bar(bin_cntr,
            hist,
            align='center',
            facecolor=color,
            width=width,
            alpha=0.75)
    ax.set_ylim([0, max(hist)+0.1])
    if fit and params_found:
        ax.plot(bin_cntr, generated_gaussian, label='KDE', c='m')
    if stats:
        ax.axvline(x=mean_quality, linestyle='--', label='Mean', c='orange')
        ax.axvline(x=mean_quality + std_quality,
                    linestyle='--',
                    label=f'+std: {std_quality}',
                    c='gray')
        if (mean_quality - std_quality) > 0:
            ax.axvline(x=mean_quality - std_quality,
                        linestyle='-.',
                        label=f'-std: {std_quality}',
                        c='gray')

def plot_quality(filename):
    basename = os.path.basename(filename).replace('.npy', '')
    qualities = np.load(filename)
    quality_type = basename.split('_')[-1]
    fig, ax = plt.subplots()
    plot_histogram(ax, qualities)

    plt.title(f'Mesh {quality_type} histogram')
    plt.xlabel('Quality bins')
    plt.ylabel('Quality Count')
    plt.legend()
    # plt.show()
    plt.savefig(f'meshes/histograms/qualities/MeshQual_{basename}_{quality_type}.png')



quality_dir = 'meshes/mesh_data/quality_data'
filenames = os.listdir(quality_dir)
for filename in filenames:
    fn = os.path.join(quality_dir, filename)
    plot_quality(fn)