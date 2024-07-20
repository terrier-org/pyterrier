from functools import wraps
from typing import Callable

import pyterrier as pt


_terrier_version = None

def set_terrier_version(version=None):
    """
    Sets the terrier version to use.
    version(str): Which version of Terrier to use. Default is `None`.
     * If None, it will find the newest Terrier version released in MavenCentral and download it.
     * If `"snapshot"`, will download the latest build from `Jitpack <https://jitpack.io/>`_.
    """
    global _terrier_version
    if started():
        raise RuntimeError('pt.java.set_terrier_version is not available after pt.java.init() is called.')
    _terrier_version = version


def _legacy_init(jnius_config):
    """
    Calls pt.init if it hasn't already been called.
    """
    if not pt.started():
        pt.init(version=_terrier_version)


@pt.utils.once()
def init() -> None:
    # TODO: if we make java optional some day, should check here that it's installed. E.g.,
    # if find_spec('jnius_config') is None:
    #     warnings.warn('pyterrier[java] not installed; no need to run pt.java.init()')
    #     return

    import jnius_config
    for entry_point in pt.utils.entry_points('pyterrier.java.init'):
        _init = entry_point.load()
        _init(jnius_config)
    import jnius


def started() -> bool:
    """
    Returns True if pt.java.init() has been called. Otherwise False.
    """
    return init.called()


def required(raise_on_not_started: bool = False) -> Callable:
    """
    Wraps a function that requires the Java Virtual Machine to be started before running. If the JVM has not yet
    been started, it runs pt.java.init() or raises an error, depending on the value of raise_on_not_started.
    """
    def _required(fn: Callable) -> Callable:
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            if not started():
                if raise_on_not_started:
                    raise RuntimeError(f'You need to call pt.java.init() required before you can call {fn}')
                else:
                    init()
            return fn(*args, **kwargs)
        return _wrapper
    return _required


# jnius wrappers:

@required(raise_on_not_started=True)
def autoclass(*args, **kwargs):
    """
    Wraps jnius.autoclass once java has started. Raises an error if called before pt.java.init() is called.
    """
    import jnius
    return jnius.autoclass(*args, **kwargs)


@required(raise_on_not_started=True)
def cast(*args, **kwargs):
    """
    Wraps jnius.cast once java has started. Raises an error if called before pt.java.init() is called.
    """
    import jnius
    return jnius.cast(*args, **kwargs)
