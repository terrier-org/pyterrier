import os
import inspect
from glob import glob
from typing import Callable, Optional, Union
from functools import wraps
import pyterrier as pt


configure = pt.java.config.register('pyterrier.anserini', {
    'version': None,
})


def _pre_init(jnius_config):
    if not is_installed():
        # pyserini not installed, do nothing
        return

    if configure['version'] is None:
        jar = _get_pyserini_jar()
    else:
        # download and use the anserini version specified by the user
        jar = pt.java.mavenresolver.get_package_jar('io.anserini', "anserini", configure['version'], 'fatjar')

    if jar is None:
        raise RuntimeError('Could not find anserini jar')
    else:
        jnius_config.add_classpath(jar)


def _post_init(jnius):
    if not is_installed():
        # pyserini not installed, do nothing
        return

    # Temporarily disable the configure_classpath while pyserini is init'd, otherwise it will try to reconfigure jnius
    import pyserini.setup
    _configure_classpath = pyserini.setup.configure_classpath
    try:
        pyserini.setup.configure_classpath = pt.utils.noop
        import pyserini.search.lucene # load the package
    finally:
        pyserini.setup.configure_classpath = _configure_classpath


def is_installed():
    try:
        import pyserini.setup # try to load
    except ImportError:
        return False
    return True


def _get_pyserini_jar() -> Optional[str]:
    # find the anserini jar distributed with pyserini
    # Adapted from pyserini/setup.py and pyserini/pyclass.py
    import pyserini.setup
    jar_root = os.path.join(os.path.split(pyserini.setup.__file__)[0], 'resources/jars/')
    paths = glob(os.path.join(jar_root, 'anserini-*-fatjar.jar'))
    if not paths:
        return None
    latest_jar = max(paths, key=os.path.getctime)
    return latest_jar

@pt.java.before_init
def set_version(version: str):
    configure['version'] = version


def required(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    """
    Requires pyserini to be installed (raises error if not installed). If the JVM has not yet been started, it runs
    pt.java.init(), too, similar to pt.java.required().

    Can be used as either a standalone function or a function/class @decorator. When used as a class decorator, it
    is applied to all methods defined by the class.
    """
    if fn is None: # standalone call
        return required(pt.utils.noop)()

    if isinstance(fn, type): # wrapping a class
        for name, value in pt.utils.get_class_methods(fn):
            setattr(fn, name, required(value))
        return fn

    else: # wrapping a function
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            if not is_installed():
                raise RuntimeError('pyserini required to use pyterrier.anserini. `pip install pyserini` and try again.')
            if not pt.java.started():
                pt.java.init()
            return fn(*args, **kwargs)
        return _wrapper


J = pt.java.JavaClasses({
    'ClassicSimilarity': 'org.apache.lucene.search.similarities.ClassicSimilarity',
    'BM25Similarity': 'org.apache.lucene.search.similarities.BM25Similarity',
    'LMDirichletSimilarity': 'org.apache.lucene.search.similarities.LMDirichletSimilarity',
    'IndexReaderUtils': 'io.anserini.index.IndexReaderUtils',
})
