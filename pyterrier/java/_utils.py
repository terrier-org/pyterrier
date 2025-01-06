import sys
import warnings
from functools import wraps
from typing import Dict, Any, Tuple, Callable, Optional, Union, TypeVar
from copy import deepcopy
import pyterrier as pt


_started = False
_configs = {}


# ----------------------------------------------------------
# Decorators
# ----------------------------------------------------------
# These functions wrap functions/classes to enforce certain
# behavior regarding Java. For instance @pt.java.required
# automatically starts java before it's invoked (if it's not
# already started).
# ----------------------------------------------------------

T = TypeVar("T", bound=Callable[..., Any])

@pt.utils.pre_invocation_decorator
def required(fn: T) -> None:
    """
    Requires the Java Virtual Machine to be started. If the JVM has not yet been started, it runs pt.java.init().

    Can be used as a function/class @decorator. When used as a class decorator, it
    is applied to all methods defined by the class.
    """
    if not _started:
        trigger = fn.__qualname__ if hasattr(fn, '__qualname__') else fn.__name__
        _init(trigger=trigger)

def required_raise(fn: T) -> T:
    """
    Similar to `pt.java.required`, but raises an error if called before pt.java.init().
    """
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if not started():
            raise RuntimeError(f'You need to call pt.java.init() required before you can call {fn}')
        return fn(*args, **kwargs)
    return _wrapper


def before_init(fn: T) -> T:
    """
    If the JVM has already started, an error is raised.
    """
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        if started():
            raise RuntimeError(f'You can only call {fn} before either you start using java or call pt.java.init()')
        return fn(*args, **kwargs)
    return _wrapper


# ----------------------------------------------------------
# Jnius Wrappers
# ----------------------------------------------------------
# These functions wrap jnius to make sure that java is
# running before they're called. Doing it this way allows
# functions to import them before java is loaded.
# ----------------------------------------------------------

@required_raise
def autoclass(*args, **kwargs):
    """
    Wraps jnius.autoclass once java has started. Raises an error if called before pt.java.init() is called.
    """
    import jnius # noqa: PT100
    return jnius.autoclass(*args, **kwargs) # noqa: PT100


@required_raise
def cast(*args, **kwargs):
    """
    Wraps jnius.cast once java has started. Raises an error if called before pt.java.init() is called.
    """
    import jnius # noqa: PT100
    return jnius.cast(*args, **kwargs) # noqa: PT100


# ----------------------------------------------------------
# Init
# ----------------------------------------------------------
# This function (along with legacy_init) loads all modules
# registered via pyterrier.java.init entry points and starts
# the JVM.
# ----------------------------------------------------------

def init() -> None:
    _init()


@pt.utils.once()
def _init(trigger=None):
    global _started
    # TODO: if we make java optional some day, should check here that it's installed. E.g.,
    # if find_spec('jnius_config') is None:
    #     warnings.warn('pyterrier[java] not installed; no need to run pt.java.init()')
    #     return

    # TODO: what about errors during init? What happens to _started? Etc.

    initializers = []
    for entry_point in pt.utils.entry_points('pyterrier.java.init'):
        initalizer = entry_point.load()()
        if initalizer.condition():
            initializers.append((entry_point.name, initalizer))

    if len(initializers) == 0:
        raise RuntimeError('No Java initializers found. This is likely a configuration issue with the package. '
                           'If installed using `pip install -e .` or `python setup.py develop`, try reinstalling.')

    # sort by priority
    initializers = sorted(initializers, key=lambda i: i[1].priority())

    import jnius_config

    # run pre-initialization setup
    for _, initializer in initializers:
        initializer.pre_init(jnius_config)

    import jnius # noqa: PT100 
    _started = True

    # run post-initialization setup
    for _, initializer in initializers:
        initializer.post_init(jnius)

    # build "Java started" message
    message = []
    if trigger:
        message.append(f'Java started (triggered by {trigger}) and loaded: ')
    else:
        message.append('Java started and loaded: ')
    for name, initializer in initializers:
        msg = initializer.message()
        if msg is None:
            message.append(name)
        else:
            message.append(f'{name} [{msg}]')
    sys.stderr.write(message[0] + ', '.join(message[1:]) + '\n')


@before_init
def legacy_init(version=None, mem=None, packages=[], jvm_opts=[], redirect_io=True, logging='WARN', home_dir=None, boot_packages=[], tqdm=None, no_download=False, helper_version = None):
    """
    Function that can be called before Terrier classes and methods can be used.
    Loads the Terrier .jar file and imports classes. Also finds the correct version of Terrier to download if no version is specified.

    Args:
        version(str): Which version of Terrier to download. Default is `None`.

         * If None, find the newest Terrier released version in MavenCentral and download it.
         * If `"snapshot"`, will download the latest build from `Jitpack <https://jitpack.io/>`_.

        mem(str): Maximum memory allocated for the Java virtual machine heap in MB. Corresponds to java `-Xmx` commandline argument. Default is 1/4 of physical memory.
        boot_packages(list(str)): Extra maven package coordinates files to load before starting Java. Default=`[]`. There is more information about loading packages in the `Terrier documentation <https://github.com/terrier-org/terrier-core/blob/5.x/doc/terrier_develop.md>`_
        packages(list(str)): Extra maven package coordinates files to load, using the Terrier classloader. Default=`[]`. See also `boot_packages` above.
        jvm_opts(list(str)): Extra options to pass to the JVM. Default=`[]`. For instance, you may enable Java assertions by setting `jvm_opts=['-ea']`
        redirect_io(boolean): If True, the Java `System.out` and `System.err` will be redirected to Pythons sys.out and sys.err. Default=True.
        logging(str): the logging level to use:

         * Can be one of `'INFO'`, `'DEBUG'`, `'TRACE'`, `'WARN'`, `'ERROR'`. The latter is the quietest.
         * Default is `'WARN'`.

        home_dir(str): the home directory to use. Default to PYTERRIER_HOME environment variable.
        tqdm: The `tqdm <https://tqdm.github.io/>`_ instance to use for progress bars within PyTerrier. Defaults to tqdm.tqdm. Available options are `'tqdm'`, `'auto'` or `'notebook'`.
        helper_version(str): Which version of the helper.

   
    **Locating the Terrier .jar file:** PyTerrier is not tied to a specific version of Terrier and will automatically locate and download a recent Terrier .jar file. However, inevitably, some functionalities will require more recent Terrier versions. 
    
     * If set, PyTerrier uses the `version` init kwarg to determine the .jar file to look for.
     * If the `version` init kwarg is not set, Terrier will query MavenCentral to determine the latest Terrier release.
     * If `version` is set to `"snapshot"`, the latest .jar file build derived from the `Terrier Github repository <https://github.com/terrier-org/terrier-core/>`_ will be downloaded from `Jitpack <https://jitpack.io/>`_.
     * Otherwise the local (`~/.mvn`) and MavenCentral repositories are searched for the jar file at the given version.
    In this way, the default setting is to download the latest release of Terrier from MavenCentral. The user is also able to use a locally installed copy in their private Maven repository, or track the latest build of Terrier from Jitpack.
    
    If you wish to run PyTerrier in an offline enviroment, you should ensure that the "terrier-assemblies-{your version}-jar-with-dependencies.jar" and "terrier-python-helper-{your helper version}.jar"
    are in the  "~/.pyterrier" (if they are not present, they will be downloaded the first time). Then you should set their versions when calling ``init()`` function. For example:
    ``pt.init(version = 5.5, helper_version = "0.0.6")``.
    """

    deprecated_calls = []

    # Set the corresponding options
    if mem is not None:
        pt.java.set_memory_limit(mem)
        deprecated_calls.append(f'pt.java.set_memory_limit({mem!r})')

    if not redirect_io:
        pt.java.set_redirect_io(redirect_io)
        deprecated_calls.append(f'pt.java.set_redirect_io({redirect_io!r})')

    if logging != 'WARN':
        pt.java.set_log_level(logging)
        deprecated_calls.append(f'pt.java.set_log_level({logging!r})')

    for package in boot_packages:
        # format: org:package:version:filetype (where version and filetype are optional)
        pkg_split = package.split(':')
        pkg_string = ", ".join(f'{w!r}' for w in pkg_split)
        pt.java.add_package(*pkg_split) 
        deprecated_calls.append(f'pt.java.add_package({pkg_string})')

    for opt in jvm_opts:
        pt.java.add_option(opt)
        deprecated_calls.append(f'pt.java.add_option({opt!r})')

    if version is not None:
        pt.terrier.set_version(version)
        deprecated_calls.append(f'pt.terrier.set_version({version!r})')

    if helper_version is not None:
        pt.terrier.set_helper_version(helper_version)
        deprecated_calls.append(f'pt.terrier.set_helper_version({helper_version!r})')

    if tqdm is not None:
        pt.utils.set_tqdm(tqdm)
        deprecated_calls.append(f'pt.utils.set_tqdm({tqdm!r})')

    if no_download:
        pt.java.mavenresolver.offline()
        deprecated_calls.append('pt.java.mavenresolver.offline()')

    pt.java.init()
    deprecated_calls.append('pt.java.init() # optional, forces java initialisation')

    # Import other java packages
    if packages:
        pkgs_string = ",".join(packages)
        pt.terrier.set_property("terrier.mvn.coords", pkgs_string)
        deprecated_calls.append(f'pt.terrier.set_property("terrier.mvn.coords", {pkgs_string!r})')

    # Warning to give new initialization
    deprecated_message = 'Call to deprecated method pt.init(). Deprecated since version 0.11.0.'
    if len(deprecated_calls) > 1: # called setup other than pt.java.init()
        deprecated_message = deprecated_message + '\nThe following code will have the same effect:'
    else: # only called pt.java.init()
        deprecated_message = deprecated_message + '\njava is now started automatically with default settings. To force initialisation early, run:'
    deprecated_message = '\n'.join([deprecated_message] + deprecated_calls)
    with warnings.catch_warnings():
        warnings.simplefilter('once', DeprecationWarning) # DeprecationWarning hidden by default, @deprecated does this to show the messages
        warnings.warn(deprecated_message, category=DeprecationWarning, stacklevel=3) # stacklevel=3 prints wherever this call comes from, rather than here, see @deprecated source


def started() -> bool:
    """
    Returns True if pt.java.init() has been called. Otherwise False.
    """
    return _started


class JavaInitializer:
    """
    A `JavaInitializer` manages the initilization of a module that uses java components. The two main methods are
    `pre_init` and `post_init`, which perform configuration before and after the JVM has started, respectively.
    """

    def priority(self) -> int:
        """
        Returns the priority of this initializer. A lower priority is executed first.
        """
        return 0

    def condition(self) -> bool:
        """
        Returns True if the initializer should be run. Otherwise False.
        """
        return True

    def pre_init(self, jnius_config) -> None:
        """
        Called before the JVM is started. `jnius_config` is the `jnius_config` module, whic can be used to configure
        java, such as by adding jars to the classpath.
        """
        pass

    def post_init(self, jnius) -> None:
        """
        Called after the JVM has started. `jnius` is the `jnius` module, which can be used to interact with java.
        """
        pass

    def message(self) -> Optional[str]:
        """
        Returns a message to be displayed after the JVM has started alongside the name of the entry point. If None,
        only the entry point name will be displayed.
        """
        return None


# ----------------------------------------------------------
# Parallel
# ----------------------------------------------------------
# These functions are for working in parallel mode, e.g.,
# with multiprocessing. They help restarting and configure
# the JVM the same way it was when it was started in the
# parent process
# ----------------------------------------------------------

def parallel_init(started: bool, configs: Dict[str, Dict[str, Any]]) -> None:
    global _configs
    if started:
        if not pt.java.started():
            _configs = configs
            _init(trigger='parallel_init')
        else:
            warnings.warn(
                "Avoiding reinit of PyTerrier")


def parallel_init_args() -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    return (
        started(),
        deepcopy(_configs),
    )


# ----------------------------------------------------------
# Configuration Utils
# ----------------------------------------------------------
# We need a global store of all java-related configurations
# so that when running in parallel, we can set everying back
# up the same way it started. These utils help manage this
# global configuration.
# ----------------------------------------------------------

class Configuration:
    def __init__(self, name):
        self.name = name

    def get(self, key):
        return deepcopy(_configs[self.name][key])

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
            if key not in _configs[self.name]:
                raise AttributeError(f'{key!r} not defined as a java setting for {self.name!r}')
            _configs[self.name][key] = value
        return deepcopy(_configs[self.name])


def register_config(name, config: Dict[str, Any]):
    assert name not in _configs
    _configs[name] = deepcopy(config)
    return Configuration(name)


# ----------------------------------------------------------
# Java Classes
# ----------------------------------------------------------
# This class enables the lazy loading of java classes. It
# helps avoid needing a ton of autoclass() statements to
# pre-load Java classes.
# ----------------------------------------------------------

class JavaClasses:
    def __init__(self, **mapping: Union[str, Callable[[], str]]):
        self._mapping = mapping
        self._cache : Dict[str, Callable]= {}

    def __dir__(self):
        return list(self._mapping.keys())

    @required_raise
    def __getattr__(self, key: str) -> Any:
        if key not in self._mapping:
            return AttributeError(f'{self} has no attribute {key!r}')
        if key not in self._cache:
            clz = self._mapping[key]
            if callable(clz):
                clz = clz()
            self._cache[key] = pt.java.autoclass(clz)
        return self._cache[key]
