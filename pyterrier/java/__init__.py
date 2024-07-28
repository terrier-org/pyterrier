from typing import Optional
import pyterrier as pt
from pyterrier.java import mavenresolver
from pyterrier.java._init import init, started, parallel_init, parallel_init_args, required, before_init, autoclass, cast
from pyterrier.java import config
from pyterrier.java.config import configure


@before_init()
def add_jar(jar_path):
    configure.append('jars', jar_path)


@before_init()
def add_package(org_name: str = None, package_name: str = None, version: str = None, file_type='jar'):
    if version is None or version == 'snapshot':
        version = mavenresolver.latest_version_num(org_name, package_name)
    file_name = mavenresolver.downloadfile(org_name, package_name, version, pt.io.pyterrier_home(), file_type)
    add_jar(file_name)


@before_init()
def set_memory_limit(mem: Optional[float]):
    configure(mem=mem)


@before_init()
def add_option(option: str):
    configure.append('options', option)


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
    if not started():
        configure(log_level=level)
    else:
        J.PTUtils.setLogLevel(level, None)


# Utils
from pyterrier.java.utils import redirect_stdouterr, bytebuffer_to_array


# Classes

from pyterrier.java.utils import JavaClasses

J = JavaClasses({
    'ArrayList': 'java.util.ArrayList',
    'Properties': 'java.util.Properties',
    'ApplicationSetup': 'org.terrier.utility.ApplicationSetup',
    'PTUtils': 'org.terrier.python.PTUtils',
    'System': 'java.lang.System',
})
