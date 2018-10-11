import numpy as np
import pymc
from model import model
from mcmcpy.mcmc.mcmc_sampler import MCMCSampler
from mock_db import mock_db

# NOTE: temp
import matplotlib.pyplot as plt
from sensitivity.approx_cov import calc_mcmcCov

import warnings
warnings.filterwarnings("ignore")

def test_prior():
    '''
    Test parameter estimation using a non-standard multivariate prior estimated
    from samples using gaussian kernel-density estimation.
    '''
    # define number of samples to generate for prior
    n = 5000
    
    # define true parameters
    a = 2.5
    b = 1.0
    
    # define x range to return values from model
    x = [1, 2, 3, 4, 5, 6]
    
    # define standard deviation of measurement errors
    std_dev = 0.02
    
    # define sampling args (num samples, burn in, step method)
    N = 1E4
    burn = 2E3
    step_method = 'dram'
    
    # -----------------------------------------------------------------

    # generate uniform joint distribution (noninformative prior)
    params = {'a': ['Uniform', -50.0, 50.0],
              'b': ['Uniform', -50.0, 50.0]}
    
    # generate multivariate parameter joint distribution (prior)
    mean=[2.48, 1.2]
    cov = np.array([[1E-3, -2E-3], [-2E-3, 2E-2]])
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=n)
    
    # instance model
    m = model(x)
    
    # initial guess
    params0 = {'a': 0.0, 'b': 0.0}
    
    # generate data from true data
    data = m.evaluate({'a': a, 'b': b})
    data_noise = data+np.random.normal(0, std_dev, data.shape)
    
    ## instance mcmc for noninformative prior case
    #mcmc_ni = MCMC(data=data_noise, model=m, params=params)
    #q_opt, ssq_opt = mcmc_ni.fit(params0)
    #mcmc_ni.generate_pymc_model(q0=q_opt, std_dev0=std_dev, fix_var=True)
    #mcmc_ni.sample(N, burn, step_method)
    
    # build mock db using prior
    prior_db = mock_db(samples, ['a', 'b'])
    
    # convert samples to prior distribution in pymc
    params_prior = {'joint_prior': ['KDE', prior_db, ['a', 'b']]}
    
    # instance mcmc_ni with prior
    mcmc = MCMCSampler(data=data_noise, model=m, params=params_prior)
    mcmc.generate_pymc_model(std_dev0=std_dev, fix_var=True)
    mcmc.sample(N, burn, step_method)
    
    # extract chain traces 
    a_trace = mcmc.MCMC.trace('a')[:]
    b_trace = mcmc.MCMC.trace('b')[:]

    # test
    np.testing.assert_almost_equal(a, np.mean(a_trace), 1)
    np.testing.assert_almost_equal(b, np.mean(b_trace), 1)
    
    ## plot for visual
    #a_mean = np.mean(mcmc_ni.db.trace('a')[:])
    #b_mean = np.mean(mcmc_ni.db.trace('b')[:])
    #mean = [a_mean, b_mean]
    #cov = calc_mcmcCov(mcmc_ni.db, ['a', 'b'])[0]
    #samples_post = np.random.multivariate_normal(mean=mean, cov=cov, size=n)

    #a2_mean = np.mean(mcmc.db.trace('a')[:])
    #b2_mean = np.mean(mcmc.db.trace('b')[:])
    #mean2 = [a2_mean, b2_mean]
    #cov2 = calc_mcmcCov(mcmc.db, ['a', 'b'])[0]
    #samples_post2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=n)


    #fig = plt.figure()
    #ax = fig.add_subplot(111) 
    #ax.plot(samples[:, 0], samples[:, 1], 'ro', alpha=0.5)
    #ax.plot(samples_post[:, 0], samples_post[:, 1], 'go', alpha=0.5)
    #ax.plot(samples_post2[:, 0], samples_post2[:, 1], 'bo', alpha=0.5)
    #plt.show()
