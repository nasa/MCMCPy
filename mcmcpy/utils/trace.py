'''
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
 
'''
import numpy as np
import pymc


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
        self._db = self._pymc_backend.load(mcmc_fname)
        self.param_names = self._db.trace_names[0]
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
        the mcmc database that was passed to class __init__(). Returns a list
        with length equal to num_samples; each element of list is a parameter
        dictionary.
        '''
        num_avail_samples = len(self.samples)
        if num_samples < num_avail_samples:
            indices = np.random.randint(0, num_avail_samples, num_samples)
        else:
            raise ValueError('num_samples > available samples.')
        samples = self.samples[indices]
        samples = self._convert_sample_array_to_dictionaries(samples)
        return samples


    def _convert_sample_array_to_dictionaries(self, samples):
        conv_samples = []
        for s in samples:
            conv_samples.append({k: s[i] for i, k in enumerate(self.param_names)})
        return conv_samples
