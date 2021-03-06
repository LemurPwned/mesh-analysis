import os
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict


def quadruple_plot(fs, title):
    d = [np.load(f) for f in fs]
    colors = ['aqua', 'lime', 'darkorange', 'fuchsia']
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    c = 0
    for i in range(2):
        for j in range(2):
            bsn = os.path.basename(fs[c]).replace('.npy', '').replace(
                '_', ' ').replace('vertex quality', ' ')
            if 'principal' in bsn:
                plot_histogram(axs[i, j],
                               d[c],
                               bins=50,
                               color=colors[c],
                               density=False)
            else:
                plot_histogram(axs[i, j],
                               d[c],
                               bins=50,
                               color=colors[c],
                               density=False)
            axs[i, j].set_title(bsn)
            c += 1
    for ax in axs.flat:
        ax.set(xlabel='', ylabel='Curvature value count')
        # ax.label_outer()

    fig.suptitle(title, y=1)
    plt.tight_layout()
    plt.savefig("Vertex curvature")
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:


def laplacian_plot(filename='./eigenvals.npy'):
    eigenvals = np.load(filename)
    fig, ax = plt.subplots()
    eigenvals = np.abs(
        eigenvals[eigenvals <= (np.mean(eigenvals) + 2 * np.std(eigenvals))])
    plot_histogram(ax, eigenvals, fit=True)
    plt.legend()
    plt.show()


def KL_divergence(bins, H1, H2):
    KL_sum = 0
    for b in range(len(bins)):
        if H2[b] and H1[b]:
            KL_sum += H1[b] * np.log2(H1[b] / H2[b])
    return KL_sum


def jensen_shannon_divergence(bins, H1, H2):
    M = 0.5 * (H1 + H2)
    jsd = 0.5*KL_divergence(bins, H1, M) + \
           0.5*KL_divergence(bins, M, H2)
    return jsd


def hist_cmp(bins, H1, H2):
    hsum = np.sum(np.minimum(H1, H2))
    return hsum / np.sum(H1)


def compare_histograms(f1, f2, fn=hist_cmp):
    rgb1 = np.load(f1)
    rgb2 = np.load(f2)
    fname = fn.__name__
    metric_list = {fname: []}
    color_map = {1: 'r', 2: 'g', 3: 'b'}
    for i in range(3):
        rgb1_norm = np.sum(rgb1[:, i])
        rgb2_norm = np.sum(rgb1[:, i])
        kl = fn(range(rgb1.shape[0]), rgb1[:, i] / rgb1_norm,
                rgb2[:, i] / rgb2_norm)
        invkl = fn(range(rgb1.shape[0]), rgb2[:, i] / rgb2_norm,
                   rgb1[:, i] / rgb1_norm)
        print(kl, invkl)
        metric_list[fname].append(kl)
        metric_list[fname].append(invkl)
    return metric_list, fname


def rbg_hist(filename):
    mesh_rgb = np.load(filename)
    fig, ax = plt.subplots()

    nbins = 50
    for i, col in enumerate(['r', 'g', 'b']):
        plot_histogram(ax, mesh_rgb[:, i], color=col, stats=False)
    basename = os.path.basename(filename).replace('.npy', '')
    plt.title(f'RGB distribution histogram for {basename}')
    plt.xlabel('RGB value')
    plt.ylabel('Value count')
    plt.legend()

    # plt.show()
    plt.savefig(f'meshes/histograms/comparisons/RGB_dist_{basename}.png')


def gaussian_fn(x, *p):
    A, mean, sigma = p
    return A * np.exp(-(x - mean)**2 / (2. * sigma**2))


def plot_histogram(ax,
                   hist_data,
                   bins=50,
                   color='g',
                   stats=True,
                   fit=True,
                   density=True):
    hist, bins = np.histogram(hist_data, bins=bins, density=density)
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
    ax.set_ylim([0, max(hist) + 0.1])
    if fit and params_found:
        ax.plot(bin_cntr, generated_gaussian, label='KDE', c='darkgray')
    if stats:
        ax.axvline(x=mean_quality, linestyle='--', label='Mean', c='k')
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

    plt.title(f'Mesh {quality_type} histogram for {basename}')
    plt.xlabel('Quality bins')
    plt.ylabel('Quality Count')
    plt.legend()
    plt.savefig(
        f'meshes/histograms/qualities/MeshQual_{basename}_{quality_type}.png')


def aggregate_hist_plot():
    quality_dir = 'meshes/mesh_data/quality_data'
    filenames = os.listdir(quality_dir)
    for filename in filenames:
        fn = os.path.join(quality_dir, filename)
        plot_quality(fn)


def compare_qualities():
    qualities_dir = 'meshes/mesh_data/quality_data'
    filenames = [
        os.path.join(qualities_dir, fn) for fn in os.listdir(qualities_dir)
        if fn.endswith('.npy')
    ]
    metrics = defaultdict(list)
    min_bin = 0
    max_bin = 1.1
    for quality_type in [
            'areas', 'aspect_ratio', 'min_angles', 'qualities', ' skewness'
    ]:
        quality_fns = list(filter(lambda x: quality_type in x, filenames))
        for element in itertools.permutations(quality_fns, r=2):
            if element[0] != element[1]:
                print(f"Processing {element}")
                e1 = np.load(element[0])
                e2 = np.load(element[1])

                h1, be1 = np.histogram(e1, bins=50, range=(min_bin, max_bin))
                h2, be2 = np.histogram(e2, bins=50, range=(min_bin, max_bin))
                b1 = (be1[:-1] + be1[1:]) / 2  # centres
                b2 = (be2[:-1] + be2[1:]) / 2
                assert len(b1) == len(b2) == len(h1) == len(h2)
                assert np.array_equal(b1, b2)

                # entropy
                dp = b1[1] - b1[0]
                area1 = np.sum(h1) * dp
                area2 = np.sum(h2) * dp
                p1 = h1 / area1
                p2 = h2 / area2
                entropy1 = -np.nansum((h1 * dp) * np.log2(p1))
                entropy2 = -np.nansum((h2 * dp) * np.log2(p2))

                for i, e in enumerate([e1, e2]):
                    for metrics_fn in [np.mean, np.std, np.std, np.median]:
                        metrics[
                            f"{metrics_fn.__name__.capitalize()} H{i}"] = metrics_fn(
                                e)

                bnames = [
                    os.path.basename(el.replace('.npy', '')) for el in element
                ]
                metrics['H1'].append(bnames[0])
                metrics['H2'].append(bnames[1])
                metrics['H1 entropy'] = entropy1
                metrics['H2 entropy'] = entropy2
                metrics['KL divergence'].append(KL_divergence(b1, h1, h2))
                metrics['Intersection'].append(hist_cmp(b1, h1, h2))
                metrics['JSD'].append(jensen_shannon_divergence(b1, h1, h2))
                metrics['Quality type'].append(quality_type)
            else:
                print("Dupe! Skipping")
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(f'{qualities_dir}/QualityMetrics.csv')


def calculate_aggregate_metrics():
    base_colors_curvature_dir = 'meshes/mesh_data/curvature_comp/base'
    filenames = [
        os.path.join(base_colors_curvature_dir, fn)
        for fn in os.listdir(base_colors_curvature_dir)
    ]
    metrics = defaultdict(list)
    for element in itertools.combinations(filenames, r=2):
        if element[0] != element[1]:
            print(f"Processing {element}")

            m1, fname1 = compare_histograms(*element, fn=hist_cmp)
            m2, fname2 = compare_histograms(*element, fn=KL_divergence)
            sfilenames = []
            assert len(m1[fname1]) == len(m2[fname2])
            element = tuple(
                [os.path.basename(e).replace('.npy', '') for e in element])
            for i in range(int(len(m1[fname1]) / 2)):
                # accomodate for inverse hist comparison
                sfilenames.append('_vs_'.join(element))
                sfilenames.append('_vs_'.join(reversed(element)))
            metrics['elements'].extend(sfilenames)
            metrics[fname1].extend(m1[fname1])
            metrics[fname2].extend(m2[fname2])
        else:
            print("Dupe! Skipping")
    df = pd.DataFrame.from_dict(metrics)
    df.to_csv('meshes/mesh_data/curvature_comp/results.csv')


def plot_curvatures():
    base_colors_curvature_dir = 'meshes/mesh_data/curvature_comp/pymesh_curv'
    filenames = [
        os.path.join(base_colors_curvature_dir, fn)
        for fn in os.listdir(base_colors_curvature_dir)
    ]
    density = True
    for filename in filenames:
        curv = np.load(filename)
        bname = os.path.basename(filename.replace('.npy', ''))
        splits = bname.split('_')
        fig, ax = plt.subplots()
        curv = curv[~np.isnan(curv)]
        curv = curv[~np.isposinf(curv)]

        # limit curv
        cmean = np.mean(curv)
        std_ = np.std(curv)

        curv = curv[(curv <= (cmean + 2 * std_))
                    & (curv >= (cmean - 2 * std_))]

        print(curv.shape)
        plot_histogram(ax, curv, density=density)
        plt.title(f"{splits[-1].capitalize()} curvature for {splits[0]}")
        plt.xlabel(f'{splits[-1].capitalize()} bins')
        plt.ylabel('Curvature values count')
        plt.legend()
        plt.savefig(
            f'meshes/histograms/comparisons/CurvHist_{splits[0]}_{splits[-1]}_den_{density}.png'
        )


compare_qualities()
# calculate_aggregate_metrics()
# # plot_curvatures()
# quadruple_plot([
#     'meshes/mesh_data/vertex_quality/sphere/curvature_gauss_normal_vertex_quality.npy',
#     'meshes/mesh_data/vertex_quality/sphere/curvature_gauss_principal_vertex_quality.npy',
#     'meshes/mesh_data/vertex_quality/sphere/curvature_gauss_pseudoinverse_vertex_quality.npy',
#     'meshes/mesh_data/vertex_quality/sphere/curvature_gauss_taubin_apr_vertex_quality.npy'
# ], "Random mesh quality")

# quadruple_plot([
#     'meshes/mesh_data/vertex_quality/sphere/curvature_mean_normal_vertex_quality.npy',
#     'meshes/mesh_data/vertex_quality/sphere/curvature_mean_principal_vertex_quality.npy',
#     'meshes/mesh_data/vertex_quality/sphere/curvature_mean_pseudoinverse_vertex_quality.npy',
#     'meshes/mesh_data/vertex_quality/sphere/curvature_mean_taubin_apr_vertex_quality.npy'
# ], "Random mesh quality")
