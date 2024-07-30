import sys
from functools import wraps
from packaging.version import Version
from typing import Optional, Union, List, Callable
import pyterrier as pt

TERRIER_PKG = "org.terrier"

_SAVED_FNS = []

_properties = None

configure = pt.java.config.register('pt.terrier.java', {
    'terrier_version': None,
    'helper_version': None,
    'boot_packages': [],
    'force_download': True,
    'prf_version': None,
})


@pt.java.before_init
def set_version(version: Optional[str] = None):
    configure['terrier_version'] = version


@pt.java.before_init
def set_helper_version(version: Optional[str] = None):
    configure['helper_version'] = version


@pt.java.before_init
def enable_prf(version: Optional[str] = '-SNAPSHOT'):
    configure['prf_version'] = version


@pt.utils.pre_invocation_decorator
def prf_required(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    """
    Requires prf to be enabled (raises error if not enabled).

    Can be used as either a standalone function or a function/class @decorator. When used as a class decorator, it
    is applied to all methods defined by the class.
    """
    if not configure['prf_version']:
        raise RuntimeError('you need to call pt.terrier.enable_prf() before java is loaded to use this function.')


def _pre_init(jnius_config):
    """
    Download Terrier's jar file for the given version at the given file_path
    Called by pt.init()

    Args:
        file_path(str): Where to download
        terrier_version(str): Which version of Terrier - None is latest release; "snapshot" uses Jitpack to download a build of the current Github 5.x branch.
        helper_version(str): Which version of the helper - None is latest
    """
    # If version is not specified, find newest and download it
    if configure['terrier_version'] is None:
        terrier_version = pt.java.mavenresolver.latest_version_num(TERRIER_PKG, "terrier-assemblies")
    else:
        terrier_version = str(configure['terrier_version']) # just in case its a float
    configure['terrier_version'] = terrier_version # save this specific version

    # obtain the fat jar from Maven
    # "snapshot" means use Jitpack.io to get a build of the current
    # 5.x branch from Github - see https://jitpack.io/#terrier-org/terrier-core/5.x-SNAPSHOT
    if terrier_version == "snapshot":
        trJar = pt.java.mavenresolver.get_package_jar("com.github.terrier-org.terrier-core", "terrier-assemblies", "5.x-SNAPSHOT", pt.io.pyterrier_home(), "jar-with-dependencies", force_download=configure['force_download'])
    else:
        trJar = pt.java.mavenresolver.get_package_jar(TERRIER_PKG, "terrier-assemblies", terrier_version, pt.io.pyterrier_home(), "jar-with-dependencies")
    pt.java.add_jar(trJar)

    # now the helper classes
    if configure['helper_version'] is None or configure['helper_version'] == 'snapshot':
        helper_version = pt.java.mavenresolver.latest_version_num(TERRIER_PKG, "terrier-python-helper")
    else:
        helper_version = str(configure['helper_version']) # just in case its a float
    configure['helper_version'] = helper_version # save this specific version
    pt.java.add_package(TERRIER_PKG, "terrier-python-helper", helper_version)

    if configure['prf_version'] is not None:
        pt.java.add_package('com.github.terrierteam', 'terrier-prf', configure['prf_version'])


@pt.java.required_raise
def _post_init(jnius):
    global _properties

    jnius.protocol_map["org.terrier.structures.postings.IterablePosting"] = {
        '__iter__': lambda self: self,
        '__next__': lambda self: _iterableposting_next(self),
        '__str__': lambda self: self.toString()
    }

    jnius.protocol_map["org.terrier.structures.CollectionStatistics"] = {
        '__str__': lambda self: self.toString()
    }

    jnius.protocol_map["org.terrier.structures.LexiconEntry"] = {
        '__str__': lambda self: self.toString()
    }

    jnius.protocol_map["org.terrier.structures.Lexicon"] = {
        '__getitem__': _lexicon_getitem,
        '__contains__': lambda self, term: self.getLexiconEntry(term) is not None,
        '__len__': lambda self: self.numberOfEntries()
    }

    jnius.protocol_map["org.terrier.querying.IndexRef"] = {
        '__reduce__' : _index_ref_reduce,
        '__getstate__' : lambda self : None,
    }

    jnius.protocol_map["org.terrier.matching.models.WeightingModel"] = {
        '__reduce__' : _wmodel_reduce,
        '__getstate__' : lambda self : None,
    }

    jnius.protocol_map["org.terrier.python.CallableWeightingModel"] = {
        '__reduce__' : _callable_wmodel_reduce,
        '__getstate__' : lambda self : None,
    }

    jnius.protocol_map["org.terrier.structures.Index"] = {
        # this means that len(index) returns the number of documents in the index
        '__len__': lambda self: self.getCollectionStatistics().getNumberOfDocuments(),

        # document-wise composition of indices: adding more documents to an index, by merging two indices with 
        # different numbers of documents. This implemented by the overloading the `+` Python operator
        '__add__': _index_add,

        # get_corpus_iter returns a yield generator that return {"docno": "d1", "toks" : {'a' : 1}}
        'get_corpus_iter' : _index_corpusiter
    }

    pt.IndexRef = J.IndexRef
    _properties = pt.java.J.Properties()
    pt.ApplicationSetup = J.ApplicationSetup
    J.ApplicationSetup.bootstrapInitialisation(_properties)

    version_string = J.Version.VERSION
    if "BUILD_DATE" in dir(J.Version):
        version_string += f" (built by {J.Version.BUILD_USER} on {J.Version.BUILD_DATE})"

    return f"version={version_string}, helper_version={configure['helper_version']}"


def _new_indexref(s):
    return pt.IndexRef.of(s)


@pt.java.required
def _new_wmodel(b):
    return J.Serialization.deserialize(b, J.ApplicationSetup.getClass("org.terrier.matching.models.WeightingModel"))


def _new_callable_wmodel(byterep):
    import dill as pickle
    from dill import extend
    #see https://github.com/SeldonIO/alibi/issues/447#issuecomment-881552005
    extend(use_dill=False)            
    fn = pickle.loads(byterep)
    #we need to prevent these functions from being GCd.
    global _SAVED_FNS
    _SAVED_FNS.append(fn)
    callback, wmodel = pt.terrier.retriever._function2wmodel(fn)
    _SAVED_FNS.append(callback)
    #print("Stored lambda fn  %s and callback in SAVED_FNS, now %d stored" % (str(fn), len(SAVED_FNS)))
    return wmodel


def _iterableposting_next(self):
    ''' dunder method for iterating IterablePosting '''
    nextid = self.next()
    # 2147483647 is IP.EOL. fix this once static fields can be read from instances.
    if 2147483647 == nextid:
        raise StopIteration()
    return self


def _lexicon_getitem(self, term):
    ''' dunder method for accessing Lexicon '''
    rtr = self.getLexiconEntry(term)
    if rtr is None:
        raise KeyError()
    return rtr


def _index_ref_reduce(self):
    return (
        _new_indexref,
        (str(self.toString()),),
        None
    )


# handles the pickling of WeightingModel classes, which are themselves usually Serializable in Java
@pt.java.required
def _wmodel_reduce(self):
    serialized = bytes(J.Serialization.serialize(self))
    return (
        _new_wmodel,
        (serialized, ),
        None
    )


def _callable_wmodel_reduce(self):
    # get bytebuffer representation of lambda
    # convert bytebyffer to python bytearray
    serlzd = self.scoringClass.serializeFn()
    bytesrep = pt.java.bytebuffer_to_array(serlzd)
    del(serlzd)
    return (
        _new_callable_wmodel,
        (bytesrep, ),
        None
    )


@pt.java.required
def _index_add(self, other):
    fields_1 = self.getCollectionStatistics().getNumberOfFields()
    fields_2 = self.getCollectionStatistics().getNumberOfFields()
    if fields_1 != fields_2:
        raise ValueError("Cannot document-wise merge indices with different numbers of fields (%d vs %d)" % (fields_1, fields_2))
    blocks_1 = self.getCollectionStatistics().hasPositions()
    blocks_2 = other.getCollectionStatistics().hasPositions()
    if blocks_1 != blocks_2:
        raise ValueError("Cannot document-wise merge indices with and without positions (%r vs %r)" % (blocks_1, blocks_2))
    return J.MultiIndex([self, other], blocks_1, fields_1 > 0)


def _index_corpusiter(self, return_toks=True):
    def _index_corpusiter_meta(self):
        meta_inputstream = self.getIndexStructureInputStream("meta")
        keys = self.getMetaIndex().getKeys()
        keys_offset = { k: offset for offset, k in enumerate(keys) }
        while meta_inputstream.hasNext():
            item = meta_inputstream.next()
            yield {k : item[keys_offset[k]] for k in keys_offset}

    def _index_corpusiter_direct_pretok(self):
        MIN_PYTHON = (3, 8)
        if sys.version_info < MIN_PYTHON:
            raise NotImplementedError("Sorry, Python 3.8+ is required for this functionality")

        meta_inputstream = self.getIndexStructureInputStream("meta")
        keys = self.getMetaIndex().getKeys()
        keys_offset = { k: offset for offset, k in enumerate(keys) }
        keys_offset = {'docno' : keys_offset['docno'] }
        direct_inputstream = self.getIndexStructureInputStream("direct")
        lex = self.getLexicon()

        ip = None
        while (ip := direct_inputstream.getNextPostings()) is not None: # this is the next() method

            # yield empty toks dicts for empty documents
            for skipped in range(0, direct_inputstream.getEntriesSkipped()):
                meta = meta_inputstream.next()
                rtr = {k : meta[keys_offset[k]] for k in keys_offset}   
                rtr['toks'] = {}
                yield rtr

            toks = {}
            while ip.next() != ip.EOL:
                t, _ = lex[ip.getId()]
                toks[t] = ip.getFrequency()
            meta = meta_inputstream.next()
            rtr = {'toks' : toks}
            rtr.update({k : meta[keys_offset[k]] for k in keys_offset})
            yield rtr

        # yield for trailing empty documents
        for skipped in range(0, direct_inputstream.getEntriesSkipped()):
            meta = meta_inputstream.next()
            rtr = {k : meta[keys_offset[k]] for k in keys_offset}   
            rtr['toks'] = {}
            yield rtr
    
    if return_toks:
        if not self.hasIndexStructureInputStream("direct"):
            raise ValueError("No direct index input stream available, cannot use return_toks=True")
        return _index_corpusiter_direct_pretok(self)
    return _index_corpusiter_meta(self)



@pt.java.required
def extend_classpath(packages: Union[str, List[str]]):
    """
        Allows to add packages to Terrier's classpath after the JVM has started.
    """
    if isinstance(packages, str):
        packages = [packages]
    assert check_version(5.3), "Terrier 5.3 required for this functionality"
    package_list = pt.java.J.ArrayList()
    for package in packages:
        package_list.add(package)
    mvnr = J.ApplicationSetup.getPlugin("MavenResolver")
    assert mvnr is not None
    mvnr = pt.java.cast("org.terrier.utility.MavenResolver", mvnr)
    mvnr.addDependencies(package_list)


@pt.java.required
def set_property(k, v):
    """
        Allows to set a property in Terrier's global properties configuration. Example::

            pt.set_property("termpipelines", "")

        While Terrier has a variety of properties -- as discussed in its 
        `indexing <https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_indexing.md>`_ 
        and `retrieval <https://github.com/terrier-org/terrier-core/blob/5.x/doc/configure_retrieval.md>`_ 
        configuration guides -- in PyTerrier, we aim to expose Terrier configuration through appropriate 
        methods or arguments. So this method should be seen as a safety-valve - a way to override the 
        Terrier configuration not explicitly supported by PyTerrier.
    """
    _properties[str(k)] = str(v)
    J.ApplicationSetup.bootstrapInitialisation(_properties)


@pt.java.required
def set_properties(kwargs):
    """
        Allows to set many properties in Terrier's global properties configuration
    """
    for key, value in kwargs.items():
        _properties[str(key)] = str(value)
    J.ApplicationSetup.bootstrapInitialisation(_properties)


@pt.java.required
def run(cmd, args=[]):
    """
        Allows to run a Terrier executable class, i.e. one that can be access from the `bin/terrier` commandline programme.
    """
    J.CLITool.main([cmd] + args)


@pt.java.required
def version():
    """
        Returns the version string from the underlying Terrier platform.
    """
    return J.Version.VERSION


def check_version(min):
    """
        Returns True iff the underlying Terrier version is no older than the requested version.
    """
    current_ver = version()
    assert current_ver is not None, "Could not obtain Terrier version"
    current_ver = Version(current_ver.replace("-SNAPSHOT", ""))

    min = Version(str(min))
    return current_ver >= min


def check_helper_version(min):
    """
        Returns True iff the underlying Terrier helper version is no older than the requested version.
    """
    current_ver = configure['helper_version']
    assert current_ver is not None, "Could not obtain Terrier helper version" 
    current_ver = Version(current_ver.replace("-SNAPSHOT", ""))

    min = Version(str(min))
    return current_ver >= min



# Terrier-specific classes
J = pt.java.JavaClasses({
    'ApplicationSetup': 'org.terrier.utility.ApplicationSetup',
    'IndexRef': 'org.terrier.querying.IndexRef',
    'Version': 'org.terrier.Version',
    'Tokenizer': 'org.terrier.indexing.tokenisation.Tokeniser',
    'Serialization': 'org.terrier.python.Serialization',
    'IndexOnDisk': 'org.terrier.structures.IndexOnDisk',
    'IndexFactory': 'org.terrier.structures.IndexFactory',
    'MultiIndex': 'org.terrier.realtime.multi.MultiIndex',
    'CLITool': 'org.terrier.applications.CLITool',
    'ApplyTermPipeline': 'org.terrier.querying.ApplyTermPipeline',
    'ManagerFactory': 'org.terrier.querying.ManagerFactory',
    'Request': 'org.terrier.querying.Request',

    # Indexing
    'TaggedDocument': 'org.terrier.indexing.TaggedDocument',
    'FlatJSONDocument': 'org.terrier.indexing.FlatJSONDocument',
    'Tokeniser': 'org.terrier.indexing.tokenisation.Tokeniser',
    'TRECCollection': 'org.terrier.indexing.TRECCollection',
    'SimpleFileCollection': 'org.terrier.indexing.SimpleFileCollection',
    'BasicIndexer': 'org.terrier.structures.indexing.classical.BasicIndexer',
    'BlockIndexer': 'org.terrier.structures.indexing.classical.BlockIndexer',
    'BasicSinglePassIndexer': 'org.terrier.structures.indexing.singlepass.BasicSinglePassIndexer',
    'BlockSinglePassIndexer': 'org.terrier.structures.indexing.singlepass.BlockSinglePassIndexer',
    'BasicMemoryIndexer': lambda: 'org.terrier.realtime.memory.MemoryIndexer' if check_version("5.7") else 'org.terrier.python.MemoryIndexer',
    'Collection': 'org.terrier.indexing.Collection',
    'StructureMerger': 'org.terrier.structures.merging.StructureMerger',
    'BlockStructureMerger': 'org.terrier.structures.merging.BlockStructureMerger',
    'DocumentPostingList': 'org.terrier.structures.indexing.DocumentPostingList',
    'MapEntry': 'org.terrier.structures.collections.MapEntry',
    'MultiDocumentFileCollection': 'org.terrier.indexing.MultiDocumentFileCollection',
    'CollectionFactory': 'org.terrier.indexing.CollectionFactory',
    'TagSet': 'org.terrier.utility.TagSet',
    'CollectionFromDocumentIterator': 'org.terrier.python.CollectionFromDocumentIterator',
    'PTUtils': 'org.terrier.python.PTUtils',
    'Index': 'org.terrier.structures.Index',
    'JsonlDocumentIterator': 'org.terrier.python.JsonlDocumentIterator',
    'JsonlPretokenisedIterator': 'org.terrier.python.JsonlPretokenisedIterator',
    'ParallelIndexer': 'org.terrier.python.ParallelIndexer',

    # PRF
    'TerrierQLParser': 'org.terrier.querying.TerrierQLParser',
    'TerrierQLToMatchingQueryTerms': 'org.terrier.querying.TerrierQLToMatchingQueryTerms',
    'QueryResultSet': 'org.terrier.matching.QueryResultSet',
    'DependenceModelPreProcess': 'org.terrier.querying.DependenceModelPreProcess',
    'RM3': 'org.terrier.querying.RM3',
    'AxiomaticQE': 'org.terrier.querying.AxiomaticQE',
})
