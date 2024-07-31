from typing import Dict, Any
from copy import deepcopy
from pyterrier.java._init import before_init, started

_CONFIGS = {}

class Configuration:
    def __init__(self, name):
        self.name = name

    def get(self, key):
        return deepcopy(_CONFIGS[self.name][key])

    @before_init
    def set(self, key, value):
        self(**{key: value})

    def append(self, key, value):
        res = self.get(key)
        res.append(value)
        self(**{key: res})

    def __getitem__(self, key):
        return self.get(key)

    @before_init
    def __setitem__(self, key, value):
        self.set(key, value)

    def __call__(self, **settings: Any):
        if started() and any(settings):
            raise RuntimeError('You cannot change java settings after java has started')
        for key, value in settings.items():
            if key not in _CONFIGS[self.name]:
                raise AttributeError(f'{key!r} not defined as a java setting for {self.name!r}')
            _CONFIGS[self.name][key] = value
        return deepcopy(_CONFIGS[self.name])


def register(name, config: Dict[str, Any]):
    assert name not in _CONFIGS
    _CONFIGS[name] = deepcopy(config)
    return Configuration(name)


configure = register('pyterrier.java', {
    'jars': [],
    'options': [],
    'mem': None,
    'log_level': 'WARN',
    'redirect_io': True,
})
