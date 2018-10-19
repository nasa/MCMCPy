class MockDatabase():

    def __init__(self, param_dict):
        self.params = param_dict
        self.trace_names = [param_dict.keys()]

    def trace(self, key):
        return self.params[key]

