# Standard libraries
import argparse
import pathlib
import sys
import warnings

# 3rd party
import numpy as np

# Local
import bayesian
import inout
import plotting

'''
# Explanation of input arguments:
Mandatory:
-m / -d / -e: Choose which interaction to use, 1.8/2.0 (EM), Delta-N2LOgo or 1.8/2.0 (EM7.5)

Optional:
-o: which orders to use. Given as a list of numbers, for example: -o 2 3 (which is the default). "2" and "3" indicate MBPT(2) and MBPT(3), respectively. The default indicates that the HF-MBPT(2) and MBPT(2)-MBPT(3) differences are used to infer the truncation error. This option also determines which orders are plotted in the PPD and empirical coverage figures, but this can be changed by manually changing what order list is sent to those plotting functions.
-nm: Which nuclear matter types to include, e.g. -nm SNM PNM. Defaults to empty.
-c: Do not sample, instead use the chain provided via "-c path/to/chain.npy"
-n: number of samples to use during MCMC sampling. Defaults to 1,000,000. Shorter runs like 100,000 also give good results.
-nucl: Which nuclei to use in the inference. Defaults to "-nucl O16 Ca48 Sn132".
-gvar: Which gamma^2 prior to use, the options are "default", "higher", and "lower".
-Rvar: Which R prior to use, the options are "uniform" (default), "mid", "higher", and "lower".
-dir: The output directory. Defaults to "output". This directory will contain subdirectories like "fig" and "chains", containing figures and MCMC chains.
'''

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dnnlogo', action='store_true')
parser.add_argument('-m', '--magic', action='store_true')
parser.add_argument('-e', '--em75', action='store_true')
parser.add_argument('-o', '--orders', nargs='+')
parser.add_argument('-nm', '--nuclear_matter', nargs='+')
parser.add_argument('-c', '--chain', type=str)
parser.add_argument('-n', '--num_samples', type=int)
parser.add_argument('-nucl', '--inference_nuclei', nargs='+')
parser.add_argument('-gvar', '--gamma2variant', type=str)
parser.add_argument('-Rvar', '--Rvariant', type=str)
parser.add_argument('-dir', '--output_dir', type=str)

args = parser.parse_args()

# Check for incompatible input arguments
assert(not (args.chain is not None and args.num_samples is not None))

# Set the interaction and orders variables
interaction = None
if args.dnnlogo:
    interaction = 'DNNLOgo'
elif args.magic:
    interaction = 'EM1.8_2.0'
elif args.em75:
    interaction = 'EM7.5'

orders = ['MBPT(2)', 'MBPT(3)']
if args.orders is not None:
    orders = [f'MBPT({x})' for x in args.orders]

# Choose which nuclei to use in the inference
inference_nuclei = ['O16', 'Ca48', 'Sn132'] # Default
if args.inference_nuclei is not None:
    inference_nuclei = args.inference_nuclei

# Default priors
if args.Rvariant is None:
    args.Rvariant = 'uniform'
if args.gamma2variant is None:
    args.gamma2variant = 'default'

# Directory to store chains and figures
output_dir = args.output_dir if args.output_dir is not None else 'output'

# Which matter types to include, if any
matter_types = args.nuclear_matter if args.nuclear_matter is not None else []
for matter_type in matter_types:
    for tmp in [0.5, 1.0]:
        inference_nuclei.append(f'{matter_type}_{tmp}n0')

# Number of samples to use in inference
num_samples = 1000000 if args.num_samples is None else args.num_samples

# Print argument information
print(f'Interaction: {interaction}')
print(f'Orders:', end=' ')
[print(order, end=' ') for order in orders]
print()
print(f'Nuclei used in inference:', end=' ')
[print(nucleus, end=' ') for nucleus in inference_nuclei]
print()
print(f'Nuclear matter: {[x for x in matter_types]}')
print(f'Output directory: {output_dir}')
print(f'MCMC chain: {args.chain}')
print(f'Number of samples: {num_samples}')
print(f'R prior variant: {args.Rvariant}')
print(f'g2 prior variant: {args.gamma2variant}')

# Random number generator
rng = np.random.default_rng()

# ============================ Plots requiring no data ===================================
# Plot gamma2 prior
plotting.gamma2_prior(output_dir)

# ============================ Plots/samplings requiring data ===================================
if interaction is None:
    print('No interaction provided, exiting')
    sys.exit(0)
# Read data
data = inout.read_data(interaction, matter_types=matter_types)

# Plot R data
plotting.R_data(interaction, data, output_dir, orders=orders)

# Plot R nuclear matter
plotting.R_data_nuclear_matter(output_dir, interaction)

# Sample R-gamma2, but only if no MCMC chain has been provided
samples = None
if args.chain is not None:
    samples = np.load(args.chain)
else:
    inference_data = inout.filter_nuclei(data, inference_nuclei)
    print('Using data in inference:', end=' ')
    inout.print_nuclei(inference_data)
    samples = bayesian.sample(num_samples, inference_data, interaction, output_dir, orders=orders, Rvariant=args.Rvariant, gvariant=args.gamma2variant)

# Plot joint distribution
plotting.joint_R_gamma2_posterior(samples, data, output_dir, Rvariant=args.Rvariant, gvariant=args.gamma2variant)

# Plot scaled predictions
ppd_nuclei = ['O16', 'O24', 'Ca40', 'Ca48', 'Ni56', 'Ni78', 'Sn100', 'Sn132', 'Pb208']
prediction_data = inout.filter_nuclei(data, ppd_nuclei)
print('Plotting PPDs for', end=' ')
inout.print_nuclei(prediction_data)
plotting.plot_predictions(rng, prediction_data, samples, output_dir, orders)

# Calculate empirical coverage. (Plotting is done with a separate script)
data_no_matter = [d for d in data if d.get('matter_type') is None]
validation_data = inout.get_array(data_no_matter, 'IMSRG')
bayesian.empirical_coverage(rng, validation_data, samples, data_no_matter, orders, output_dir)
