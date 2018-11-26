MCMCPy - **M**arkov **C**hain **M**onte **C**arlo **S**ampling with **Py**thon 
==========================================================================
Python module for uncertainty quantification using a Markov chain Monte
Carlo sampler.

MCMCPy is a wrapper around the popular PyMC package for Python 2.7. The purpose
of the MCMCPy module is to (1) standardize the format of the input and output of
the underlying PyMC code and (2) reduce the inherent complexity of PyMC by pre-
defining a statistical model of a commonly-used form. The MCMCPy module was originally
released as part of the SMCPy code (https://github.com/nasa/SMCPy), but, in some
cases, it is possible to isolate MCMCPy and use it directly without calling SMCPy's
primary module.
 
To operate MCMCPy, the user supplies a computational model built in Python 2.7,
defines prior distributions for each of the model parameters to be estimated, and
provides data to be used for calibration. These are roughly the same steps required
to operate SMCPy. Markov chain Monte Carlo sampling can be conducted with ease
through instantiation of the MCMCSampler class and a call to the sample () method.
The output of this process is an approximation of the parameter posterior probability
distribution conditioned on the data provided.

==========================================================================
