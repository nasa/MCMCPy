
class mock_db():
    '''
    Class used to reformulate a sample set as a pymc database object (db).
    Pass samples with shape [# samples, # params] and param_names corresponds
    to columns.
    '''
    def __init__(self, samples, param_names):
        chain = dict()
        for i, sample in enumerate(samples.transpose()):
            chain[param_names[i]] = sample
        self.chain = chain

    def trace(self, param_name):
        return self.chain[param_name]
