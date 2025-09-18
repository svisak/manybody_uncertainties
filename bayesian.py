# Standard libraries
import pathlib

# 3rd party
import emcee
import h5py
import numpy as np
from scipy.stats import norm, invgamma, uniform, beta

# Local
import inout

def sample(n, data, interaction, output_dir, orders, burn_in=0.1, Rvariant='uniform', gvariant='default'):
    '''Sample the (R, gamma^2) posterior with emcee.'''
    # Set up number of walkers etc.
    # orders is the MBPT orders used in the inference
    n_walkers = 10
    n_dim = 2
    n = n // n_walkers
    print(f'Sampling with n_samples = {n}, n_walkers = {n_walkers}, n_dim = {n_dim}')
    print('Orders used in sampling:', end=' ')
    [print(order, end=' ') for order in orders]
    print()

    # Starting positions drawn from the priors
    p0_R = prior_R(variant=Rvariant)[0].rvs(size=(n_walkers,1))
    p0_gamma2 = prior_gamma2()[0].rvs(size=(n_walkers,1))
    p0 = np.hstack((p0_R, p0_gamma2))

    # Sample
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnposterior_R_gamma2, args=[data, orders, Rvariant, gvariant])
    sampler.run_mcmc(p0, n)

    # Discard the first samples to avoid biasing from the starting position
    discard = int(burn_in * n)
    samples = sampler.get_chain(discard=discard)
    samples = samples.reshape((-1, 2))

    # Save result and return
    path = f'{output_dir}/chains'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    ofile = f'{path}/{interaction}.npy'
    np.save(ofile, samples)
    print(f'Saving MCMC chain to {ofile}')
    acc_rate = np.mean(sampler.acceptance_fraction)
    print(f'Mean acceptance fraction: {acc_rate:.3f}')
    return samples

def lnposterior_R_gamma2(params, data, orders, Rvariant, gvariant):
    '''
    Compute the joint (log) posterior for (R, gamma^2) according to Equations (22)-(26).
    params: [R, gamma^2]
    '''
    R = params[0]
    gamma2 = params[1]
    if R <= 0:
        return -np.inf
    gamma_data = compute_gamma(data, R, orders, stack=True)
    gamma2_post, alpha, beta = posterior_gamma2(gamma_data, variant=gvariant)
    val1 = gamma2_post.logpdf(gamma2)
    val2 = lnposterior_R(R, gamma_data, alpha, beta, orders, Rvariant)
    return val1 + val2

def posterior_gamma2(gamma, return_dist=True, variant='default'):
    '''Computes the posterior for gamma^2 based on the gamma data.'''
    alpha0, beta0 = prior_gamma2(return_dist=False, variant=variant)
    alpha, beta = update_invgamma_hyperparameters(alpha0, beta0, gamma)
    if return_dist:
        ig = inverse_gamma_distribution(alpha, beta)
        return ig, alpha, beta
    else:
        return alpha, beta

def lnposterior_R(R, gamma, alpha, beta, orders, Rvariant):
    '''
    Marginal posterior for R. Equivalent to Eq. (26) but using a different parametrization.
    Based on Eq. (12) in Wesolowski et al, PRC, 2021.
    alpha, beta are the updated hyperparameters for the gamma2 distribution
    '''
    pr = prior_R(variant=Rvariant)[0]
    nu, tau = ig_to_invchi2(alpha, beta)
    n_obs = gamma.shape[1]
    val = pr.logpdf(R)
    n = [inout.get_order(i, enforce_int=True) for i in orders]
    val = val - nu * np.log(tau) - n_obs * np.sum(n)*np.log(R)
    return val

def prior_gamma2(return_dist=True, variant='default'):
    '''
    Priors for gamma^2. Can return either just the hyperparameters,
    or the hyperparameters and the distribution object.
    '''
    alpha0 = None
    beta0 = None
    if variant == 'default':
        alpha0 = 1
        beta0 = 1
    elif variant == 'higher':
        # Less restrictive of higher values, more of lower
        alpha0 = 1
        beta0 = 5
    elif variant == 'lower':
        # Vice versa
        alpha0 = 0.5
        beta0 = 0.5
    if return_dist:
        ig = inverse_gamma_distribution(alpha0, beta0)
        return ig, alpha0, beta0
    else:
        return alpha0, beta0

def prior_R(return_dist=True, variant='uniform'):
    '''
    Priors for R. Includes uniform and IG priors, where uniform is our default.
    As for the gamma^2 prior, whether to return the distribution object is optional.
    '''

    # Uniform (uninformative) prior
    if variant == 'uniform':
        a = 0
        b = 2
        if return_dist:
            return uniform(a,b), a, b
        else:
            return a, b

    # Inverse-gamma priors
    if variant == 'mid':
        alpha0 = 2.0
        beta0 = 0.5
    elif variant == 'higher':
        # Less restrictive of higher values, more of lower
        alpha0 = 4
        beta0 = 2
    elif variant == 'lower':
        # Vice versa
        alpha0 = 3.0
        beta0 = 0.3
    if return_dist:
        ig = inverse_gamma_distribution(alpha0, beta0)
        return ig, alpha0, beta0
    else:
        return alpha0, beta0

def ig_to_invchi2(alpha, beta):
    '''Convert between inverse-gamma and scaled inverse chi-squared distributions.'''
    nu = 2 * alpha
    tau2 = 2 * beta / nu
    tau = np.sqrt(tau2)
    return (nu, tau)

def inverse_gamma_distribution(alpha, beta):
    '''Returns a scipy invgamma distribution, just added this to simplify the syntax.'''
    return invgamma(a=alpha, scale=beta)

def update_invgamma_hyperparameters(alpha0, beta0, gammas):
    '''
    Compute the posterior hyperparameters for gamma^2 given
    the prior hyperparameters and the gamma data.
    '''
    gammas = gammas.reshape((-1,))
    n = len(gammas)
    alpha = alpha0 + n / 2
    beta = beta0 + gammas@gammas / 2
    return (alpha, beta)

def truncation_error(samples, data, order, rng, scale=True, include_total=True):
    '''
    Given a list of samples of (R, gamma^2), this function goes through the list
    and draws a random truncation error using each sample.
    If include_total is True, the raw MBPT calculation is included in the prediction
    (otherwise it's just a truncation error).
    If scale is True, the result is scaled by the mass number A.
    '''
    order = inout.get_order(order, enforce_int=True)
    A = inout.get_array(data, 'A') # For scaling
    scale_factor = 1/A if scale else np.ones_like(A)
    n_samples = len(samples)
    refval = reference_value(data)
    predictions = np.empty((n_samples, len(refval)))
    n_div = 0
    for i in range(n_samples):
        R = samples[i, 0]
        gamma2 = samples[i, 1]
        gamma = np.sqrt(gamma2)
        std = None
        if R < 1:
            std = scale_factor * refval * gamma * R**(order+1) / np.sqrt(1-R**2)
            std = np.abs(std)
        else:
            # We have a divergent series, so draw a truncation error using the
            # largest standard deviation supported by numpy's normal distribution.
            # Also keeep track of how many divergent samples we have.
            std = np.sqrt(np.finfo(np.float64).max)
            n_div += 1
        val = rng.normal(0, std)
        if include_total:
            val = val + scale_factor * compute_E_total(data, order)
        predictions[i,:] = val
    print(f'Samples with R >= 1: {n_div}')
    return predictions

def empirical_coverage(rng, validation_data, samples, data, orders, output_dir=None):
    '''
    Calculate the empirical coverage of the MBPT predictions compared to the validation data.
    returns: the independent p values, the limits of the confidence intervals, and the empirical coverage.
    '''
    # Note that validation_data is just a numpy array, whereas data is a list of dictionaries
    n_isotopes = len(validation_data)
    ps = np.arange(0.00, 1.01, 0.05)

    # Confidence intervals, see Furnstahl PRC 2015
    N = n_isotopes
    ci_limits = np.empty((4, len(ps)))
    for i, p in enumerate(ps):
        n = int(round(p*N))
        a = n + 1
        b = N - n + 1
        ci_limits[1,i], ci_limits[2,i] = beta.interval(0.68, a, b)
        ci_limits[0,i], ci_limits[3,i] = beta.interval(0.95, a, b)

    empirical_coverage = np.empty((len(ps), len(orders)))
    for i, order in enumerate(orders):
        predictions = truncation_error(samples, data, order, rng, scale=False)
        assert(predictions.shape[1] == n_isotopes)
        for j, p in enumerate(ps):
            quantiles = [0.50-p/2, 0.50+p/2]
            q = np.quantile(predictions, quantiles, axis=0)
            hits = 0
            for k in range(n_isotopes):
                val = validation_data[k]
                if val >= q[0,k] and val <= q[1,k]:
                    hits += 1
            empirical_coverage[j, i] = hits / n_isotopes

    # Save the calculated values in HDF5 format
    if output_dir is not None:
        path = f'{output_dir}/empirical_coverage'
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        interaction = inout.get_common_value(data, 'interaction')
        pretty_interaction = inout.get_common_value(data, 'pretty_interaction')
        fname = f'{path}/{interaction}.h5'
        f = h5py.File(fname, 'w')
        f.create_dataset('ps', data=ps)
        f.create_dataset('ci_limits', data=ci_limits)
        f.create_dataset('empirical_coverage', data=empirical_coverage)
        f.attrs['pretty_interaction'] = pretty_interaction
        f.attrs['orders'] = orders
        f.close()
        print(f'Wrote empirical coverage information to {fname}')

    return (ps, ci_limits, empirical_coverage)

def reference_value(data):
    '''Get reference values. For nuclei, this is 8*A.'''
    A = inout.get_array(data, 'A')
    matter_types = inout.get_array(data, 'matter_type')
    refvals = []
    for i in range(len(A)):
        if matter_types[i] is None:
            # Finite nucleus
            refvals.append(-8*A[i]) # 8 MeV per nucleon
        else:
            # Nuclear matter
            n = data[i].get('n')
            val = get_matter_refval(matter_types[i], n)
            refvals.append(val)
    return refvals

def get_matter_refval(matter_type, n):
    '''Get reference value for nuclear matter.'''
    if matter_type == 'PNM':
        return refscale_PNM(n)
    elif matter_type == 'SNM':
        return refscale_SNM(n)
    else:
        raise ValueError(f'Unknown matter type {matter_type}')

# Reference values for nuclear matter
def k_F0_SNM():
    return 1.333

def k_F0_PNM():
    return 1.680

def n_to_k_SNM(n, g=4):
    return (n * 6*np.pi**2 / g)**(1/3)

def k_to_n_SNM(k_F0, g=4):
    return g * k_F0**3 / (6*np.pi**2)

def n_to_k_PNM(n, g=2):
    return (n * 6*np.pi**2 / g)**(1/3)

def k_to_n_PNM(k_F0, g=2):
    return g * k_F0**3 / (6*np.pi**2)

def refscale_SNM(n):
    k_F = n_to_k_SNM(n)
    return 16 * (k_F / k_F0_SNM())**2

def refscale_PNM(n):
    k_F = n_to_k_PNM(n)
    return 16 * (k_F / k_F0_PNM())**2

def compute_gamma(data, R, orders, stack=False):
    '''Computes the gamma coefficients according to Eq. (17) in the paper.'''
    refval = reference_value(data)
    if type(orders) != list:
        orders = [orders]
    gamma = np.empty(0)
    orders_int = [inout.get_order(order, enforce_int=True) for order in orders]
    for order in orders_int:
        diff = inout.get_energy(data, order) # The nth order correction is already listed in the data, so no need to compute the difference
        tmp = diff / refval / R**order
        gamma = np.concatenate((gamma, tmp))
    if stack:
        n_orders = len(orders)
        gamma = gamma.reshape((n_orders, -1))
    return gamma

def compute_MBPT_ratio(data, orders, stack=False):
    '''Computes raw energy ratios, e.g. MBPT(3)/MBPT(2). Used for plotting only.'''
    assert(type(orders[0]) == str)
    lower_energy = np.empty(0) # Lower order
    higher_energy = np.empty(0) # Higher order
    if type(orders) == str:
        orders = [orders]
    for order in orders:
        higher_order = inout.get_order(order, enforce_int=True)
        lower_order = higher_order - 1
        tmp1 = inout.get_energy(data, lower_order)
        tmp2 = inout.get_energy(data, higher_order)
        lower_energy = np.concatenate((lower_energy, tmp1))
        higher_energy = np.concatenate((higher_energy, tmp2))
    if stack:
        n_orders = len(orders)
        lower_energy = lower_energy.reshape((n_orders, -1))
        higher_energy = higher_energy.reshape((n_orders, -1))
    if None in higher_energy:
        # May happen if MBPT(4) is included, since we have less data there
        higher_energy = np.array(higher_energy, dtype=float)
    ratio = higher_energy / lower_energy
    val = np.abs(ratio)
    return val

def compute_E_total(data, order):
    '''Sum up all contributions to get the total energy.'''
    total = inout.get_energy(data, 'EHF')
    k_min = inout.get_order('MBPT(2)')
    k_max = inout.get_order(order, enforce_int=True)
    for i in range(k_min, k_max+1):
        energy = inout.get_energy(data, i)
        if None in energy:
            # May happen if MBPT(4) is included, since we have less data there
            energy = np.array(energy, dtype=float)
        total = total + energy
    return total
