import sys
import inspect
import os
from warnings import warn
from functools import wraps
from typing import Dict, Any, Tuple, Callable, Optional, Union
from copy import deepcopy
import pyterrier as pt
import pyterrier.java.config


_started = False


def started() -> bool:
    """
    Returns True if pt.java.init() has been called. Otherwise False.
    """
    return _started


@pt.utils.pre_invocation_decorator
def required(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    """
    Requires the Java Virtual Machine to be started. If the JVM has not yet been started, it runs pt.java.init().

    Can be used as either a standalone function or a function/class @decorator. When used as a class decorator, it
    is applied to all methods defined by the class.
    """
    if not started():
        init()


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
            raise RuntimeError(f'You can only call {fn} before either you start using java or call pt.java.init()')
        return fn(*args, **kwargs)
    return _wrapper


@required_raise
def _post_init(jnius):
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

    import jnius # noqa: PT100 
    _started = True

    pt.java.set_log_level(pt.java.configure['log_level'])
    if pt.java.configure['redirect_io']:
        pt.java.redirect_stdouterr()

    java_version = pt.java.J.System.getProperty("java.version")
    if java_version.startswith("1.") or java_version.startswith("9."):
        raise RuntimeError(f"Pyterrier requires Java 11 or newer, we only found Java version {java_version};"
            + " install a more recent Java, or change os.environ['JAVA_HOME'] to point to the proper Java installation")

    message = []
    for entry_point in pt.utils.entry_points('pyterrier.java.post_init'):
        _post_init = entry_point.load()
        msg = _post_init(jnius)
        if msg is None:
            message.append(entry_point.name)
        elif msg is False:
            pass
        else:
            message.append(f'{entry_point.name} [{msg}]')
    sys.stderr.write('Java started with: ' + ', '.join(message) + '\n')


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
    import jnius # noqa: PT100
    return jnius.autoclass(*args, **kwargs) # noqa: PT100


@required_raise
def cast(*args, **kwargs):
    """
    Wraps jnius.cast once java has started. Raises an error if called before pt.java.init() is called.
    """
    import jnius # noqa: PT100
    return jnius.cast(*args, **kwargs) # noqa: PT100


@before_init
def legacy_init(version=None, mem=None, packages=[], jvm_opts=[], redirect_io=True, logging='WARN', home_dir=None, boot_packages=[], tqdm=None, no_download=False,helper_version = None):
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

    # Set the corresponding options
    pt.java.set_memory_limit(mem)
    pt.java.set_redirect_io(redirect_io)
    pt.java.set_log_level(logging)
    for package in boot_packages:
        pt.java.add_package(*package.split(':')) # format: org:package:version:filetype (where version and filetype are optional)
    for opt in jvm_opts:
        pt.java.add_option(opt)
    pt.terrier.set_version(version)
    pt.terrier.set_helper_version(helper_version)
    if tqdm is not None:
        pt.utils.set_tqdm(tqdm)
    if no_download:
        pt.java.mavenresolver.offline()

    pt.java.init()

    # Import other java packages
    if packages:
        pkgs_string = ",".join(packages)
        pt.terrier.set_property("terrier.mvn.coords", pkgs_string)
