# A Bayesian approach for many-body uncertainties in nuclear structure: Many-body perturbation theory for finite nuclei

**Isak Svensson<sup>1,2,3</sup>, Alexander Tichai<sup>1,2,3</sup>, Kai Hebeler<sup>1,2,3</sup>, and Achim Schwenk<sup>1,2,3</sup>**

**<sup>1</sup>Technische Universität Darmstadt, Department of Physics, 64289 Darmstadt, Germany**

**<sup>2</sup>ExtreMe Matter Institute EMMI, GSI Helmholtzzentrum für Schwerionenforschung GmbH, 64291 Darmstadt, Germany**

**<sup>3</sup>Max-Planck-Institut für Kernphysik, Saupfercheckweg 1, 69117 Heidelberg, Germany**

## Introduction and requirements
This repository contains all code and data used to generate the results in [Svensson et al. (arXiv:2507.09079)](https://arxiv.org/abs/2507.09079). Some Python libraries are required to run the code. They are:

- `emcee`
- `h5py`
- `matplotlib`
- `mendeleev`
- `numpy`
- `prettyplease`
- `scipy`

These can all be installed with `pip`.

## Generating results
To produce all figures in the paper, simply execute the `analysis.sh` script. This will result in a full reproduction of our results, from reading in data to sampling the posteriors to creating the plots. The script repeatedly executes the `run.py` script with different inputs as well as the `fig_1.py` and `plot_empirical_coverages.py` scripts. By default we use a high number of MCMC samples, and the full runtime is therefore of the order of one hour in total. This can be reduced considerably, while still getting reasonable results, by decreasing the number of samples.

`run.py` is the main user-facing script and is typically invoked like `python run.py -m`, where `-m` indicates the 1.8/2.0 (EM) interaction. The argument can be changed to `-d` for $\Delta$-N<sup>2</sup>LO<sub>GO</sub> or `-e` for 1.8/2.0 (EM7.5). Additional optional arguments are explained in `run.py`. Use `-n` to set the number of MCMC samples, for example to 100000 (the default is 1 million).

The input MBPT data is located in the `data` folder, where the `.csv` files contain results for finite nuclei for the three different chiral interactions, the 1.8/2.0 (EM), $\Delta$-N<sup>2</sup>LO<sub>GO</sub>, and 1.8/2.0 (EM7.5) interactions. The `.h5` files contain MBPT results for symmetric nuclear matter (SNM) and pure neutron matter (PNM), calculated with 1.8/2.0 (EM).

`run.py` calls functions in additional Python files which actually perform the analysis. These are `inout.py` (input and output), `bayesian.py` (calculations and statistical methods), and `plotting.py` (figures).

## Acknowledgements
We thank R.J. Furnstahl and Z. Li for useful discussions, P. Arthuis and M. Heinz for sharing MBPT and IMSRG results, T. Miyagi for sharing MBPT(4) results, and Y. Dietz and F. Alp for sharing MBPT results for nuclear matter. This work was supported in part by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant Agreements No. 101020842 and No. 101162059).
