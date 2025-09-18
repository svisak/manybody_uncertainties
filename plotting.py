# Standard libraries
import os
import pathlib
import warnings

# 3rd party
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import numpy as np
import prettyplease

# Local
import bayesian
import inout

def figsave(fig, fname):
    '''Convenience function to easily save figures.'''
    print(f'Saving figure to {fname}')
    directory = os.path.dirname(fname)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, bbox_inches='tight')

def get_color(order, light=False):
    '''Color specifications for various quantities.'''
    if order == 'exp':
        return 'xkcd:black'
    elif order == 'exact':
        return 'xkcd:light gray'
    order = inout.get_order(order, enforce_str=True) # Make sure order is in MBPT(x) format
    if order == 'MBPT(2)':
        return 'xkcd:golden' if not light else 'xkcd:light yellow'
    elif order == 'MBPT(3)':
        return 'xkcd:electric blue' if not light else 'xkcd:light blue'
    elif order == 'MBPT(4)':
        return 'xkcd:black' if not light else 'xkcd:light gray'

def custom_line(color, linestyle=None, lw=None, marker=None, mec=None, mew=None):
    '''Get lines to be used in legends.'''
    if lw is None:
        return Line2D([0], [0], color=color, linestyle=linestyle, marker=marker, mec=mec, mew=mew)
    else:
        return Line2D([0], [0], color=color, linestyle=linestyle, lw=lw, mec=mec, mew=mew)

def custom_patch(color, linestyle=None, alpha=1.0, lw=None):
    '''Filled legend entries.'''
    return mpatches.Patch(color=color, alpha=alpha, lw=4)

def interval_handle(color1, color2, lw=4):
    '''Legend handle for error bands.'''
    return (custom_patch(color1, lw=lw), custom_line(color2, lw=lw))

def ticksetup(ax, only_y=False):
    '''Add minor ticks and right+top ticks.'''
    axis = 'y' if only_y else 'both'
    ax.tick_params(axis=axis, which='both', direction='in')
    ax.minorticks_on()
    if not only_y:
        ax2 = ax.secondary_xaxis('top')
        ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
        ax2.minorticks_on()
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticks(ax.get_xticks(minor=True), minor=True)
    ax3 = ax.secondary_yaxis('right')
    ax3.tick_params(axis='y', which='both', direction='in', labelright=False)
    ax3.minorticks_on()
    ax3.set_yticks(ax.get_yticks())
    ax3.set_yticks(ax.get_yticks(minor=True), minor=True)

def plot_predictions(rng, data, samples, output_dir, orders):
    '''
    This function plots predictions by calling plot_energy with various inputs,
    and provides nice things like colors, labels, markers etc.
    '''
    markers = ['o'] + [None] * len(orders)
    labels = ['IMSRG'] + orders
    colors = [get_color('exact')]
    colors += [(get_color(o, light=True), get_color(o)) for o in orders]
    imsrg = inout.get_array(data, 'IMSRG')
    A = inout.get_array(data, 'A')
    ms = 5
    mew = 0.4
    mec = 'black'
    fig, ax = plot_energy(data, imsrg/A, colors[0], marker=markers[0], ms=ms, mew=mew, mec=mec)
    for i, order in enumerate(orders):
        predictions = bayesian.truncation_error(samples, data, order, rng)
        plot_energy(data, predictions, colors[i+1], ax=ax)

    # Legend
    line1 = custom_line(colors[0], linestyle='', marker=markers[0], mec=mec, mew=mew)
    lines = [line1]
    for i in range(len(orders)):
        lines.append(interval_handle(*colors[i+1]))
    ax.legend(lines, labels, fancybox=False, loc='upper right')
    interaction = inout.get_common_value(data, 'interaction')
    path = f'{output_dir}/fig'
    figsave(fig, f'{path}/ppd_{interaction}.pdf')

def plot_energy(data, predictions, colors, quantiles=[0.05,0.16,0.84,0.95], marker=None, ms=None, mew=None, mec=None, ax=None):
    '''
    Plots the energy of a range of nuclei.
    If predictions is a 1d array (i.e. no uncertainty) then just plot the value.
    If it's a 2d array this indicates that each row is a sample; compute quantiles and fill.
    '''
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(5,3.5))
        ax = fig.add_subplot(111)

    pretty_nuclei = inout.get_array(data, 'pretty_label')
    n_data = len(pretty_nuclei)
    x = np.arange(1, n_data+1)

    handle = None
    if predictions.ndim <= 1:
        # Not plotting a range, just a single value
        ax.plot(x, predictions, marker=marker, ls='', markersize=ms, markeredgewidth=mew, markerfacecolor=colors, markeredgecolor=mec, zorder=10)
    else:
        include = ~np.isnan(predictions).any(axis=0)
        filtered_predictions = predictions[:, include]

        # Plot error bands
        lw = 0.1
        alpha = 0.90
        q = np.quantile(filtered_predictions, quantiles, axis=0)
        lowest = q[0,:]
        low = q[1,:]
        high = q[2,:]
        highest = q[3,:]
        tmp_x = x[include]
        if len(tmp_x) == 1:
            # Happens if only one nucleus is predicted, fill_between needs an x range to be visible
            tmp_x = [tmp_x[0]-0.3, tmp_x[0]+0.3]
            lowest = lowest * np.ones(2)
            low = low * np.ones(2)
            high = high * np.ones(2)
            highest = highest * np.ones(2)
        ax.fill_between(tmp_x, lowest, highest, color=colors[0], lw=lw, alpha=alpha)
        ax.fill_between(tmp_x, low, high, color=colors[1], lw=lw, alpha=alpha)
        ax.plot(tmp_x, lowest, color='xkcd:dark gray', lw=lw)
        ax.plot(tmp_x, low, color='xkcd:dark gray', lw=lw)
        ax.plot(tmp_x, high, color='xkcd:dark gray', lw=lw)
        ax.plot(tmp_x, highest, color='xkcd:dark gray', lw=lw)

    # Add interaction label
    interaction = inout.get_common_value(data, 'interaction')
    pretty_interaction = inout.get_common_value(data, 'pretty_interaction')
    ax.text(0.03, 0.05, pretty_interaction, transform=ax.transAxes)

    # Axis limits
    lim = (-10, -6)
    pad = 0.3
    ax.set_xlim(1-pad, n_data+pad)

    # Ticks and labels
    ax.set_ylabel('$E / A$ (MeV)')
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_nuclei)
    step = 1 if lim[1]-lim[0] <= 5 else 2
    yticks = np.arange(np.round(lim[0]), np.round(lim[1])+1, step)
    ax.set_yticks(yticks)
    ax.set_ylim(lim)
    ticksetup(ax, only_y=True)
    ax.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax.get_xaxis().minorticks_off() # Requires matplotlib >= 3.9 I think
    return fig, ax

def empirical_coverage_setup(num_axes):
    '''Set up a figure for weather plots. Plotting is then done with the plot_weather function.'''
    xsize = 4.8 * (num_axes // 2 + 1)
    ysize = 3.0
    fig = plt.figure(figsize=(xsize, ysize))
    axes = [fig.add_subplot(1,num_axes,i+1) for i in range(num_axes)]

    for ax in axes:
        # Add diagonal line
        ax.plot([0,100], [0,100], color='xkcd:black', ls='--')

        # Labels, limits, ticks
        ax.set_xlim(0,100)
        ax.set_ylim(0,100)
        ax.set_xlabel(r'$p \times 100\%$')
        ticksetup(ax)
        ax.set_aspect('equal')

    axes[0].set_ylabel(r'Empirical coverage (\%)')
    if num_axes > 1:
        fig.subplots_adjust(wspace=0, hspace=0)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    return fig, axes

def plot_empirical_coverage(ax, ps, ci_limits, empirical_coverage, orders, ls='-', pretty_interaction=None, legend=True):
    '''Plot empirical coverages in the ax axes.'''
    num_orders = len(orders)
    assert(num_orders == empirical_coverage.shape[1])

    # Confidence intervals
    ax.fill_between(100*ps, 100*ci_limits[0,:], 100*ci_limits[3,:], color='xkcd:light gray', alpha=1.0)
    ax.fill_between(100*ps, 100*ci_limits[1,:], 100*ci_limits[2,:], color='xkcd:gray', alpha=1.0)

    # Plot empirical coverages
    for i in range(num_orders):
        color = get_color(orders[i], light=False)
        label = orders[i]
        lw = 1.4
        ax.plot(100*ps, 100*empirical_coverage[:,i], color=color, ls=ls, lw=lw, label=label)

    if legend:
        ax.legend(loc='lower right', fancybox=False)
    if pretty_interaction is not None:
        ax.set_title(pretty_interaction)

def gamma2_prior(output_dir, color='xkcd:black'):
    '''Create a plot for the gamma^2 prior.'''
    xmin = 0
    xmax = 10
    x = np.linspace(xmin, xmax, 500)

    fig = plt.figure(figsize=(4.8,3.0))
    ax = fig.add_subplot(111)

    variants = ['lower', 'higher', 'default']
    linestyles = ['-', '--', '-.']
    alphas = [1.0, 0.5, 0.5]
    for v in variants:
        ig, alpha0, beta0 = bayesian.prior_gamma2(variant=v)
        label = rf'($\alpha,\beta$) = (${alpha0},{beta0}$)'
        y = ig.pdf(x)
        ls = linestyles.pop(-1)
        ax.plot(x, y, label=label, color=color, ls=ls, alpha=alphas.pop(-1))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0, top=0.56)
    yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel(r'$\bar{\gamma}^2$')
    ax.set_ylabel(r'pr$(\bar{\gamma}^2 | I)$')
    ax.legend(fancybox=False)
    ticksetup(ax)
    path = f'{output_dir}/fig'
    figsave(fig, f'{path}/gamma2_prior.pdf')

def R_data(interaction, data, output_dir, orders):
    '''Plot MBPT ratios. This function is just for information, the output is not publication worthy.'''
    fig = plt.figure(figsize=(4.8,3.0))
    ax = fig.add_subplot(111)
    A = inout.get_array(data, 'A')
    for order in orders:
        R = bayesian.compute_MBPT_ratio(data, order)
        ax.plot(A, R, color=get_color(order), marker='.', ls='', label=order)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('$A$')
    ax.set_ylabel('$R$')
    ax.legend(fancybox=False)
    ticksetup(ax)
    path = f'{output_dir}/fig'
    figsave(fig, f'{path}/R_{interaction}.pdf')

def R_data_paper(interaction, data, output_dir, orders):
    '''Create Fig. 1 in the paper.'''
    matter_idx = []
    for i, d in enumerate(data):
        if d.get('matter_type') is not None:
            matter_idx.append(i)
    if len(matter_idx) != 2:
        print('nuclear matter not included, not producing the R data plot (paper version)')
        return
    fig = plt.figure(figsize=(4.8,3.0))
    ax = fig.add_subplot(111)
    A = inout.get_array(data, 'A')
    x_SNM = [230, 245]
    xticks = [0, 50, 100, 150, 200] + x_SNM
    xticks_minor = [i for i in range(10,220,10) if i not in xticks]
    xticklabels = [None, '$A=50$', '$A=100$', '$A=150$', '$A=200$', '$n = 0.5n_0$', '$n = 1.0n_0$']
    for order in orders:
        R = bayesian.compute_MBPT_ratio(data, order)
        ax.plot(A, R, color=get_color(order), marker='.', ls='', label=order, ms=9, markeredgecolor='black', mew=0.5)
        ax.plot(x_SNM, R[matter_idx], color=get_color(order), marker='.', ls='', ms=9, markeredgecolor='black', mew=0.5)
    ax.axvline(220, ls='--', color='xkcd:gray')
    ax.set_xticks(xticks)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(xticklabels)
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    ax.text(-0.04, -0.18, r'$A=14$', rotation=45, transform=ax.transAxes)
    ax.yaxis.minorticks_on()
    ax.set_xlim(left=0, right=x_SNM[-1]+10)
    ax.set_ylim(0, 1)
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylabel('$E^{(k)} / E^{(k-1)}$')
    ax.tick_params(axis='both', which='both', direction='in', labeltop=False)
    ax.legend(fancybox=False)
    pretty_interaction = inout.get_common_value(data, 'pretty_interaction')
    ax.text(0.06, 0.68, pretty_interaction, ha='left', transform=ax.transAxes)

    ax.text(0.35, 0.9, 'Finite nuclei', transform=ax.transAxes)
    ax.text(0.89, 0.9, 'SNM', transform=ax.transAxes)
    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', which='both', direction='in', labeltop=False)
    ax2.set_xticks(xticks)
    ax2.set_xticks(xticks_minor, minor=True)

    ax3 = ax.secondary_yaxis('right')
    ax3.tick_params(axis='y', which='both', direction='in', labelright=False)
    ax3.minorticks_on()
    path = f'{output_dir}/fig'
    figsave(fig, f'{path}/R_{interaction}_paper.pdf')

def R_data_nuclear_matter(output_dir, interaction):
    '''Plot MBPT ratios for nuclear matter (Fig. 7 in the paper).'''
    if interaction != 'EM1.8_2.0':
        warnings.warn(f'Not computing nuclear matter R because interaction is not EM1.8_2.0')
        return

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4.8,4.8))
    for i, matter_type in enumerate(['PNM', 'SNM']):
        ax = axes[i]

        # Read data
        fname = inout.get_fname_matter(interaction, matter_type)
        f = h5py.File(fname, 'r')
        n = f.get('n')[...]
        hf = f.get('HF')[...]
        mbpt2 = f.get('mbpt2')[...]
        mbpt3 = f.get('mbpt3')[...]
        f.close()

        # Plot the ratios
        r1 = np.abs(mbpt2 / hf)
        r2 = np.abs(mbpt3 / mbpt2)

        ax.plot(n, r1, label='MBPT(2)', color=get_color('MBPT(2)'))
        ax.plot(n, r2, label='MBPT(3)', color=get_color('MBPT(3)'))
        ax.set_xlabel('$n$ [fm$^{-3}]$')
        ax.set_xlim(0.0, 0.32)
        ax.set_ylim(bottom=0)
        ax.text(0.56, 0.82, matter_type, transform=ax.transAxes)
        ax.legend(fancybox=False, loc='upper right')

        # Labels etc
        ylabel = '$E^{(k)} / E^{(k-1)}$'
        ax.set_ylabel(ylabel)
        ticksetup(ax)

    fig.subplots_adjust(hspace=0, wspace=0)
    path = f'{output_dir}/fig'
    figsave(fig, f'{path}/R_NM_{interaction}.pdf')

def joint_R_gamma2_posterior(samples, data, output_dir, Rvariant, gvariant):
    '''Create a corner plot for the (R, gamma^2) posterior.'''
    R_limits = (0, 0.5)
    gamma2_limits = (0, 10)

    # Plot the posterior
    labels = [r'$R$', r'$\bar{\gamma}^2$']
    limits = [R_limits, gamma2_limits]
    color = 'xkcd:dark periwinkle'
    fig, axes = prettyplease.corner(samples, bins=100, bins2d=40, limits=limits, color=color, histtype='accented', labels=labels, n_ticks=4, title_loc='center', return_axes=True, alpha1d=0.88, minorticks=True, n_uncertainty_digits=2)

    prior_color = 'xkcd:black'
    accent_color = 'xkcd:light gray'
    prior_lw = 1.3
    # Plot R prior
    R_prior, _, _ = bayesian.prior_R(variant=Rvariant)
    x = np.linspace(R_limits[0], R_limits[1], 500)
    y = R_prior.pdf(x)
    axes[0,0].plot(x, y, color=prior_color, lw=prior_lw, zorder=-5)

    # Plot gamma2 prior
    gamma2_prior, _, _ = bayesian.prior_gamma2(variant=gvariant)
    x = np.linspace(gamma2_limits[0], gamma2_limits[1], 500)
    y = gamma2_prior.pdf(x)
    axes[1,1].plot(x, y, color=prior_color, lw=prior_lw, zorder=-5)

    # Plot joint prior
    n_gridpoints = 100
    Rs = np.linspace(R_limits[0], R_limits[1], n_gridpoints)
    gamma2s = np.linspace(gamma2_limits[0], gamma2_limits[1], n_gridpoints)
    R_grid, gamma2_grid = np.meshgrid(Rs, gamma2s)

    pdf = np.zeros((n_gridpoints,n_gridpoints))
    for R_idx, R in enumerate(Rs):
        for gamma2_idx, gamma2 in enumerate(gamma2s):
            pdf[gamma2_idx][R_idx] = R_prior.pdf(R) * gamma2_prior.pdf(gamma2)
    axes[1,0].contour(R_grid, gamma2_grid, pdf, 10, colors=accent_color, linewidths=0.2, zorder=-4)
    cmap = LinearSegmentedColormap.from_list("density_cmap", colors=['xkcd:white', prior_color])
    axes[1,0].contourf(R_grid, gamma2_grid, pdf, 10, cmap=cmap, zorder=-5)

    # Add title and save
    pretty_interaction = inout.get_common_value(data, 'pretty_interaction')
    axes[0,1].text(0.5, 0.5, pretty_interaction, ha='center', transform=axes[0,1].transAxes, bbox=dict(facecolor='none', edgecolor='black', pad=10.0, lw=0.8))
    interaction = inout.get_common_value(data, 'interaction')
    path = f'{output_dir}/fig'
    figsave(fig, f'{path}/Rgamma2posterior_{interaction}.pdf')
