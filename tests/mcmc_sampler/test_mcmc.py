from model import model
import numpy as np
import os
from mcmcpy.mcmc.mcmc_sampler import MCMCSampler


def test_model():
    """
    Initial test to ensure model has not been altered.
    """
    # set up ground truth
    a = 2
    b = 3.5
    x = [1, 2, 3, 4, 5, 6]
    y = np.array(x) * a + b

    # instance model / evaluate
    m = model(x)
    y_model = m.evaluate({'a': a, 'b': b})

    # test
    np.testing.assert_equal(y, y_model)


def test_mcmc_fit():
    """
    Tests the fit method of the mcmc class.
    """
    # set up ground truth
    a = 2
    b = 3.5
    x = [1, 2, 3, 4, 5, 6]
    y = np.array(x) * a + b

    # instance model
    m = model(x)

    # define priors for params
    params = {'a': ['Uniform', -50.0, 50.0],
              'b': ['Uniform', -50.0, 50.0]}

    # initial param guess
    params0 = {'a': 0.0, 'b': 0.0}

    # instance prognosis class
    mcmc = MCMCSampler(data=y, model=m, params=params)

    # LSQ fit
    params_opt, ssq_opt = mcmc.fit(params0, repeats=2, opt_method='L-BFGS-B')

    # test params
    params_true = [a, b]
    params_fit = [params_opt['a'], params_opt['b']]
    np.testing.assert_array_almost_equal(params_true, params_fit, 4)

    # test output
    y_opt = m.evaluate(params_opt)
    np.testing.assert_array_almost_equal(y, y_opt, 5)


def test_mcmc_standard():
    """
    Test parameter estimation using standard (non-adaptive) step method.
    """
    # set up ground truth / instance model
    a = 2
    b = 3.5
    x = np.arange(200)
    m = model(x)
    std_dev = 0.01
    yn = m.generate_noisy_data({'a': a, 'b': b}, std_dev)

    # define priors for params
    params = {'a': ['Uniform', -50.0, 50.0],
              'b': ['Uniform', -50.0, 50.0]}

    # instance prognosis class
    mcmc = MCMCSampler(data=yn, model=m, params=params)

    # LSQ fit for get good starting point
    params_opt, ssq_opt = ({'a': 2.00052, 'b': 3.499}, 5.494477e-05)

    # generate pymc model
    mcmc.generate_pymc_model(q0=params_opt, ssq0=ssq_opt)

    # sample
    N = 1E4
    burn = 2E3
    mcmc.sample(N, burn, step_method='metropolis',
                scales={'a': 0.015, 'b': 0.015})

    # extract chain traces 
    a_trace = mcmc.MCMC.trace('a')[:]
    b_trace = mcmc.MCMC.trace('b')[:]
    std_dev_trace = mcmc.MCMC.trace('std_dev')[:]

    # test
    np.testing.assert_almost_equal(np.mean(a_trace), a, 2)
    np.testing.assert_almost_equal(np.mean(b_trace), b, 2)

    # remove unwanted files
    os.remove('mcmc.p')


def test_mcmc_adaptive():
    """
    Test parameter estimation using standard (non-adaptive) step method.
    """
    # set up ground truth / instance model
    a = 2
    b = 3.5
    x = np.arange(200)
    m = model(x)
    std_dev = 0.01
    yn = m.generate_noisy_data({'a': a, 'b': b}, std_dev)

    # define priors for params
    params = {'a': ['Uniform', -50.0, 50.0],
              'b': ['Uniform', -50.0, 50.0]}

    # initial param guess
    params0 = {'a': 0.0, 'b': 0.0}
    

    # instance prognosis class
    mcmc = MCMCSampler(data=yn, model=m, params=params)

    # LSQ fit for get good starting point
    params_opt, ssq_opt = ({'a': 2.00052, 'b': 3.499}, 5.494477e-05)

    # generate pymc model
    mcmc.generate_pymc_model(q0=params_opt, ssq0=ssq_opt)

    # sample
    N = 1E4
    burn = 2E3
    mcmc.sample(N, burn, step_method='adaptive', interval=300,
                delay=burn/2, scales={'a': 0.015, 'b': 0.015})

    # extract chain traces
    a_trace = mcmc.MCMC.trace('a')[:]
    b_trace = mcmc.MCMC.trace('b')[:]
    std_dev_trace = mcmc.MCMC.trace('std_dev')[:]
    
    # make sure std_dev is actually being sampled
    if len(set(std_dev_trace)) < 2:
        raise ValueError('std_dev trace contains only one unique value!')

    # test
    np.testing.assert_almost_equal(a, np.mean(a_trace), 2)
    np.testing.assert_almost_equal(b, np.mean(b_trace), 2)
    np.testing.assert_almost_equal(std_dev, np.mean(std_dev_trace), 2)

    # remove unwanted files
    os.remove('mcmc.p')


