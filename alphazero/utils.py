class dotdict(dict):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError
        return self[name]

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        data = super().copy()
        return self.__class__(data)


def get_iter_file(iteration: int):
    return f'iteration-{iteration:04d}.pkl'
