#!/bin/bash

echo "Reproducing all figures in the paper."
echo "The main results will be saved to output/fig/, and the results with different priors to output/lower/fig/ and output/higher/fig/"
echo "To speed up the inference you can add \"-n 100000\" to the arguments for \"run.py\" in this script."
echo "This will decrease the number of MCMC samples from 1,000,000 to 100,000."
echo ""

# Create Figure 1
echo "Generating Fig. 1"
python fig_1.py
echo ""

# Create results for 1.8/2.0 (EM)
echo "Generating results for the 1.8/2.0 (EM) interaction"
python run.py -m
echo ""

# Create results for Delta-N2LOgo
echo "Generating results for the Delta-N2LOgo interaction"
python run.py -d
echo ""

# Create results for 1.8/2.0 (EM7.5)
echo "Generating results for the 1.8/2.0 (EM7.5) interaction"
python run.py -e
echo ""

# Plot empirical coverages
# Note that this requires "run.py" to have generated the empirical coverages first
echo "Plotting empirical coverages"
python plot_empirical_coverages.py
echo ""

# Results with different priors for gamma^2
echo "Generating results for the 1.8/2.0 (EM) interaction with the \"lower\" gamma^2 prior variant"
python run.py -m -gvar lower -dir output/lower
python plot_empirical_coverages.py -i EM1.8_2.0 -dir output/lower
echo ""

echo "Generating results for the 1.8/2.0 (EM) interaction with the \"higher\" gamma^2 prior variant"
python run.py -m -gvar higher -dir output/higher
python plot_empirical_coverages.py -i EM1.8_2.0 -dir output/higher
