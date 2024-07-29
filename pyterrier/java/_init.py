import os
from warnings import warn
from functools import wraps
from typing import Dict, Any, Tuple, Callable, Optional, Union
from copy import deepcopy
import pyterrier as pt
import pyterrier.java.config


_started = False


def _post_init(jnius):
    pt.HOME_DIR = pt.io.pyterrier_home()
    pt.autoclass = jnius.autoclass
    pt.cast = jnius.cast

    jnius.protocol_map['java.util.Map$Entry'] = {
        '__getitem__' : _mapentry_getitem,
        '__iter__' : lambda self: iter([self.getKey(), self.getValue()]),
        '__len__' : lambda self: 2
    }


# Map$Entry can be decoded like a tuple
def _mapentry_getitem(self, i):
    if i == 0:
        return self.getKey()
    if i == 1:
        return self.getValue()
    raise IndexError()


@pt.utils.once()
def init() -> None:
    global _started
    # TODO: if we make java optional some day, should check here that it's installed. E.g.,
    # if find_spec('jnius_config') is None:
    #     warnings.warn('pyterrier[java] not installed; no need to run pt.java.init()')
    #     return

    if pt.utils.is_windows():
        if "JAVA_HOME" in os.environ:
            java_home =  os.environ["JAVA_HOME"]
            fix = f'{java_home}\\jre\\bin\\server\\;{java_home}\\jre\\bin\\client\\;{java_home}\\bin\\server\\'
            os.environ["PATH"] = os.environ["PATH"] + ";" + fix

    import jnius_config

    for entry_point in pt.utils.entry_points('pyterrier.java.pre_init'):
        _pre_init = entry_point.load()
        _pre_init(jnius_config)

    if pt.java.configure['mem'] is not None:
        jnius_config.add_options('-Xmx' + str(pt.java.configure['mem']) + 'm')

    for opt in pt.java.configure['options']:
        jnius_config.add_options(opt)

    for jar in pt.java.configure['jars']:
        jnius_config.add_classpath(jar)

    import jnius
    _started = True

    pt.java.set_log_level(pt.java.configure['log_level'])
    if pt.java.configure['redirect_io']:
        pt.java.redirect_stdouterr()

    java_version = pt.java.J.System.getProperty("java.version")
    if java_version.startswith("1.") or java_version.startswith("9."):
        raise RuntimeError(f"Pyterrier requires Java 11 or newer, we only found Java version {java_version};"
            + " install a more recent Java, or change os.environ['JAVA_HOME'] to point to the proper Java installation")

    for entry_point in pt.utils.entry_points('pyterrier.java.post_init'):
        _post_init = entry_point.load()
        _post_init(jnius)


def started() -> bool:
    """
    Returns True if pt.java.init() has been called. Otherwise False.
    """
    return _started


def required(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    """
    Requires the Java Virtual Machine to be started. If the JVM has not yet been started, it runs pt.java.init().

    Can be used as either a standalone function or a function decorator.
    """
    if fn is None:
        return required(pt.utils.noop)()

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if not started():
            init()
        return fn(*args, **kwargs)
    return _wrapper


def required_raise(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    """
    Similar to `pt.java.required`, but raises an error if called before pt.java.init().
    """
    if fn is None:
        return required_raise(pt.utils.noop)()

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if not started():
            raise RuntimeError(f'You need to call pt.java.init() required before you can call {fn}')
        return fn(*args, **kwargs)
    return _wrapper


def before_init(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    """
    If the JVM has already started, an error is raised.

    Can be used as either a standalone function or a function decorator.
    """
    if fn is None:
        return before_init(pt.utils.noop)()

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if started():
            raise RuntimeError(f'You need to call pt.java.init() required before you can call {fn}')
        return fn(*args, **kwargs)
    return _wrapper


def parallel_init(started: bool, configs: Dict[str, Dict[str, Any]]) -> None:
    if started:
        if not pt.java.started():
            warn(f'Starting java parallel with configs {configs}')
            pyterrier.java.config._CONFIGS = configs
            init()
        else:
            warn("Avoiding reinit of PyTerrier")


def parallel_init_args() -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    return (
        started(),
        deepcopy(pyterrier.java.config._CONFIGS),
    )


@required_raise
def autoclass(*args, **kwargs):
    """
    Wraps jnius.autoclass once java has started. Raises an error if called before pt.java.init() is called.
    """
    import jnius
    return jnius.autoclass(*args, **kwargs)


@required_raise
def cast(*args, **kwargs):
    """
    Wraps jnius.cast once java has started. Raises an error if called before pt.java.init() is called.
    """
    import jnius
    return jnius.cast(*args, **kwargs)
