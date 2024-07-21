from functools import wraps
import os
from typing import Callable, Optional
from pyterrier import mavenresolver

import pyterrier as pt


_started = False


def _legacy_post_init(jnius):
    pt.HOME_DIR = pt.io.pyterrier_home()
    pt.properties = autoclass('java.util.Properties')()
    pt.ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')
    pt.autoclass = jnius.autoclass
    pt.cast = jnius.cast


@pt.utils.once()
def init() -> None:
    global _started
    # TODO: if we make java optional some day, should check here that it's installed. E.g.,
    # if find_spec('jnius_config') is None:
    #     warnings.warn('pyterrier[java] not installed; no need to run pt.java.init()')
    #     return

    if pt.is_windows():
        if "JAVA_HOME" in os.environ:
            java_home =  os.environ["JAVA_HOME"]
            fix = f'{java_home}\\jre\\bin\\server\\;{java_home}\\jre\\bin\\client\\;{java_home}\\bin\\server\\'
            os.environ["PATH"] = os.environ["PATH"] + ";" + fix

    import jnius_config

    for entry_point in pt.utils.entry_points('pyterrier.java.pre_init'):
        _pre_init = entry_point.load()
        _pre_init(jnius_config)

    if _mem is not None:
        jnius_config.add_options('-Xmx' + str(_mem) + 'm')

    for opt in _options:
        jnius_config.add_options(opt)

    for jar in _jars:
        jnius_config.add_classpath(jar)

    import jnius
    _started = True

    set_log_level(_log_level)

    java_version = autoclass("java.lang.System").getProperty("java.version")
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


def before_init() -> Callable:
    """
    Wraps a function that can only be run before the Java Virtual Machine has started. If the JVM has already started,
    an error is raised.
    """
    def _before_init(fn: Callable) -> Callable:
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            if started():
                raise RuntimeError(f'You need to call pt.java.init() required before you can call {fn}')
            return fn(*args, **kwargs)
        return _wrapper
    return _before_init


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


# Settings


_jars = []
_options = []
_mem = None
_log_level = 'WARN'

@before_init()
def add_jar(jar_path):
    _jars.append(jar_path)


@before_init()
def add_package(org_name: str = None, package_name: str = None, version: str = None, file_type='jar'):
    if version is None or version == 'snapshot':
        version = mavenresolver.latest_version_num(org_name, package_name)
    file_name = mavenresolver.downloadfile(org_name, package_name, version, pt.io.pyterrier_home(), file_type)
    add_jar(file_name)


@before_init()
def set_memory_limit(mem: Optional[float]):
    global _mem
    _mem = mem


@before_init()
def add_option(option: str):
    _options.append(option)


def set_log_level(level):
    """
        Set the logging level. The following string values are allowed, corresponding
        to Java logging levels:
        
         - `'ERROR'`: only show error messages
         - `'WARN'`: only show warnings and error messages (default)
         - `'INFO'`: show information, warnings and error messages
         - `'DEBUG'`: show debugging, information, warnings and error messages
        
        Unlike other java settings, this can be changed either before or after init() has been called.
    """
    global _log_level
    _log_level = level
    if started():
        autoclass("org.terrier.python.PTUtils").setLogLevel(level, None)
