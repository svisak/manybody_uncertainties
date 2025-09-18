# Standard libraries
import argparse

# 3rd party
import h5py
import matplotlib.pyplot as plt

# Local
import plotting

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--interactions', nargs='+')
parser.add_argument('-dir', '--directory', type=str)
args = parser.parse_args()

interactions = args.interactions if args.interactions is not None else ['EM1.8_2.0', 'DNNLOgo', 'EM7.5']
n = len(interactions)
directory = args.directory if args.directory is not None else 'output'

fig, axes = plotting.empirical_coverage_setup(n)

# Plot the empirical coverages to the figure
legend = True # Legend in first column
for i, interaction in enumerate(interactions):
    ax = axes[i]
    fname = f'{directory}/empirical_coverage/{interaction}.h5'
    f = h5py.File(fname, 'r')
    ps = f.get('ps')[...]
    ci_limits = f.get('ci_limits')[...]
    empirical_coverage = f.get('empirical_coverage')[...]
    pretty_interaction = f.attrs.get('pretty_interaction')
    orders = f.attrs.get('orders')
    plotting.plot_empirical_coverage(ax, ps, ci_limits, empirical_coverage, orders, pretty_interaction=pretty_interaction, legend=legend)
    legend = False # No legend in the other columns

fig.subplots_adjust(wspace=0, hspace=0)
plotting.figsave(fig, f'{directory}/fig/empirical_coverage.pdf')
