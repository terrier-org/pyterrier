from typing import Dict, Any
from copy import deepcopy
import pyterrier.java

_CONFIGS = {}

def register(name, config: Dict[str, Any]):
    assert name not in _CONFIGS
    _CONFIGS[name] = deepcopy(config)

    @pyterrier.java.before_init()
    def _configure(**settings: Any):
        for key, value in settings.items():
            if key not in _CONFIGS[name]:
                raise AttributeError(f'{key!r} not defined as a java setting for {name!r}')
            _CONFIGS[name][name] = value
        return deepcopy(_CONFIGS[name])

    return _configure

configure = register('pyterrier.java', {
    'jars': [],
    'options': [],
    'mem': None,
    'log_level': 'WARN',
})

def get_configs():
    return deepcopy(_CONFIGS)

def set_configs(configs):
    for key, value in configs:
        _CONFIGS[key] = value
