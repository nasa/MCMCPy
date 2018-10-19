class MockDatabase():
    '''
    Mimicks a pymc database (primarily for the access to a trace object by
    parameter name via the trace method). This can be useful, for example,
    when trying to use mcmcpy plotting functions on the fly.
    '''
    def __init__(self, param_dict):
        self.params = param_dict
        self.trace_names = [param_dict.keys()]

    def trace(self, key):
        return self.params[key]



class TraceSampler():
    '''
    Allows for sampling from mcmc database
    '''

    def __init__(self, mcmc_fname, backend):
        self._pymc_backend = self._find_backend(backend)
        self._db = backend.load(mcmc_fname)
        self.param_names = self._db.trace_names
        self.samples = self._extract_sample_array()


    def _find_backend(self, backend):
        if backend == 'hdf5':
            backend = pymc.database.hdf5
        elif backend == 'pickle':
            backend = pymc.database.pickle
        elif backend == 'sqlite':
            backend = pymc.database.sqlite
        else:
            raise ValueError('Backend %s is not supported.' % backend)
        return backend


    def _extract_sample_array(self,):
        samples = [self._db.trace(key)[:] for key in self.param_names]
        samples = np.array(samples).transpose()
        return samples


    def sample(self, num_samples):
        '''
        Randomly samples num_samples parameter vectors from the trace stored in
        the mcmc database that was passed to class __init__().
        '''
        num_avail_samples = len(self.samples)
        if num_samples > num_avail_samples:
            sample_indices = np.random.randit(0, num_avail_samples)
        return self.samples(sample_indices)



