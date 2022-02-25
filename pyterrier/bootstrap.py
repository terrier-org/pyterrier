import deprecation

from . import mavenresolver

stdout_ref = None
stderr_ref = None
TERRIER_PKG = "org.terrier"
SAVED_FNS=[]

def logging(level):
    from jnius import autoclass
    autoclass("org.terrier.python.PTUtils").setLogLevel(level, None)
# make an alias
_logging = logging

def new_indexref(s):
    from . import IndexRef
    return IndexRef.of(s)

def new_wmodel(bytes):
    from . import autoclass
    serUtils = autoclass("org.terrier.python.Serialization")
    return serUtils.deserialize(bytes, autoclass("org.terrier.utility.ApplicationSetup").getClass("org.terrier.matching.models.WeightingModel") )

def new_callable_wmodel(byterep):
    import dill as pickle
    from dill import extend
    #see https://github.com/SeldonIO/alibi/issues/447#issuecomment-881552005
    extend(use_dill=False)            
    fn = pickle.loads(byterep)
    #we need to prevent these functions from being GCd.
    global SAVED_FNS
    SAVED_FNS.append(fn)
    from .batchretrieve import _function2wmodel
    callback, wmodel = _function2wmodel(fn)
    SAVED_FNS.append(callback)
    #print("Stored lambda fn  %s and callback in SAVED_FNS, now %d stored" % (str(fn), len(SAVED_FNS)))
    return wmodel

def javabytebuffer2array(buffer):
    assert buffer is not None
    def unsign(signed):
        return signed + 256 if signed < 0 else signed
    return bytearray([ unsign(buffer.get(offset)) for offset in range(buffer.capacity()) ])

def setup_jnius():
    from jnius import protocol_map # , autoclass

    def _iterableposting_next(self):
        ''' dunder method for iterating IterablePosting '''
        nextid = self.next()
        # 2147483647 is IP.EOL. fix this once static fields can be read from instances.
        if 2147483647 == nextid:
            raise StopIteration()
        return self

    protocol_map["org.terrier.structures.postings.IterablePosting"] = {
        '__iter__': lambda self: self,
        '__next__': lambda self: _iterableposting_next(self),
        '__str__': lambda self: self.toString()
    }

    protocol_map["org.terrier.structures.CollectionStatistics"] = {
        '__str__': lambda self: self.toString()
    }

    protocol_map["org.terrier.structures.LexiconEntry"] = {
        '__str__': lambda self: self.toString()
    }

    def _lexicon_getitem(self, term):
        ''' dunder method for accessing Lexicon '''
        rtr = self.getLexiconEntry(term)
        if rtr is None:
            raise KeyError()
        return rtr

    protocol_map["org.terrier.structures.Lexicon"] = {
        '__getitem__': _lexicon_getitem,
        '__contains__': lambda self, term: self.getLexiconEntry(term) is not None,
        '__len__': lambda self: self.numberOfEntries()
    }

    def index_ref_reduce(self):
        return (
            new_indexref,
            (str(self.toString()),),
            None
        )

    protocol_map["org.terrier.querying.IndexRef"] = {
        '__reduce__' : index_ref_reduce,
        '__getstate__' : lambda self : None,
    }


    # handles the pickling of WeightingModel classes, which are themselves usually Serializable in Java
    def wmodel_reduce(self):
        from . import autoclass
        serUtils = autoclass("org.terrier.python.Serialization")
        serialized = bytes(serUtils.serialize(self))
        return (
            new_wmodel,
            (serialized, ),
            None
        )

    protocol_map["org.terrier.matching.models.WeightingModel"] = {
        '__reduce__' : wmodel_reduce,
        '__getstate__' : lambda self : None,
    }

    def callable_wmodel_reduce(self):
        from . import autoclass
        # get bytebuffer representation of lambda
        # convert bytebyffer to python bytearray
        bytesrep = javabytebuffer2array(self.scoringClass.serializeFn())
        return (
            new_callable_wmodel,
            (bytesrep, ),
            None
        )

    protocol_map["org.terrier.python.CallableWeightingModel"] = {
        '__reduce__' : callable_wmodel_reduce,
        '__getstate__' : lambda self : None,
    }

    def _index_add(self, other):
        from . import autoclass
        fields_1 = self.getCollectionStatistics().getNumberOfFields()
        fields_2 = self.getCollectionStatistics().getNumberOfFields()
        if fields_1 != fields_2:
            raise ValueError("Cannot document-wise merge indices with different numbers of fields (%d vs %d)" % (fields_1, fields_2))
        blocks_1 = self.getCollectionStatistics().hasPositions()
        blocks_2 = other.getCollectionStatistics().hasPositions()
        if blocks_1 != blocks_2:
            raise ValueError("Cannot document-wise merge indices with and without positions (%r vs %r)" % (blocks_1, blocks_2))
        multiindex_cls = autoclass("org.terrier.realtime.multi.MultiIndex")
        return multiindex_cls([self, other], blocks_1, fields_1 > 0)

    protocol_map["org.terrier.structures.Index"] = {
        # this means that len(index) returns the number of documents in the index
        '__len__': lambda self: self.getCollectionStatistics().getNumberOfDocuments(),

        # document-wise composition of indices: adding more documents to an index, by merging two indices with 
        # different numbers of documents. This implemented by the overloading the `+` Python operator
        '__add__': _index_add
    }

def setup_terrier(file_path, terrier_version=None, helper_version=None, boot_packages=[], force_download=True):
    """
    Download Terrier's jar file for the given version at the given file_path
    Called by pt.init()

    Args:
        file_path(str): Where to download
        terrier_version(str): Which version of Terrier - None is latest release; "snapshot" uses Jitpack to download a build of the current Github 5.x branch.
        helper_version(str): Which version of the helper - None is latest
    """
    # If version is not specified, find newest and download it
    if terrier_version is None:
        terrier_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-assemblies")
    else:
        terrier_version = str(terrier_version) # just in case its a float
    # obtain the fat jar from Maven
    # "snapshot" means use Jitpack.io to get a build of the current
    # 5.x branch from Github - see https://jitpack.io/#terrier-org/terrier-core/5.x-SNAPSHOT
    if terrier_version == "snapshot":
        trJar = mavenresolver.downloadfile("com.github.terrier-org.terrier-core", "terrier-assemblies", "5.x-SNAPSHOT", file_path, "jar-with-dependencies", force_download=force_download)
    else:
        trJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-assemblies", terrier_version, file_path, "jar-with-dependencies")

    # now the helper classes
    if helper_version is None:
        helper_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-python-helper")
    else:
        helper_version = str(helper_version) # just in case its a float
    helperJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-python-helper", helper_version, file_path, "jar")

    classpath=[trJar, helperJar]
    for b in boot_packages:
        parts = b.split(":")
        if len(parts)  < 2 or len(parts) > 4:
            raise ValueError("Invalid format for package %s" % b)
        group = parts[0]
        pkg = parts[1]
        filetype = "jar"
        version = None
        if len(parts) > 2:
            version = parts[2]
            if len(parts) > 3:
                filetype = parts[3]
        #print((group, pkg, filetype, version))
        filename = mavenresolver.downloadfile(group, pkg, version, file_path, filetype)
        classpath.append(filename)

    return classpath

def is_windows() -> bool:
    import platform
    return platform.system() == 'Windows'

def is_binary(f):
    import io
    return isinstance(f, (io.RawIOBase, io.BufferedIOBase))

def redirect_stdouterr():
    from jnius import autoclass, PythonJavaClass, java_method

    # TODO: encodings may be a probem here
    class MyOut(PythonJavaClass):
        __javainterfaces__ = ['org.terrier.python.OutputStreamable']

        def __init__(self, pystream):
            super(MyOut, self).__init__()
            self.pystream = pystream
            self.binary = is_binary(pystream)

        @java_method('()V')
        def close(self):
            self.pystream.close()

        @java_method('()V')
        def flush(self):
            self.pystream.flush()

        @java_method('([B)V', name='write')
        def writeByteArray(self, byteArray):
            # TODO probably this could be faster.
            for c in byteArray:
                self.writeChar(c)

        @java_method('([BII)V', name='write')
        def writeByteArrayIntInt(self, byteArray, offset, length):
            # TODO probably this could be faster.
            for i in range(offset, offset + length):
                self.writeChar(byteArray[i])

        @java_method('(I)V', name='write')
        def writeChar(self, chara):
            if self.binary:
                return self.pystream.write(bytes([chara]))
            return self.pystream.write(chr(chara))

    # we need to hold lifetime references to stdout_ref/stderr_ref, to ensure
    # they arent GCd. This prevents a crash when Java callsback to  GCd py obj

    global stdout_ref
    global stderr_ref
    import sys
    stdout_ref = MyOut(sys.stdout)
    stderr_ref = MyOut(sys.stderr)
    jls = autoclass("java.lang.System")
    jls.setOut(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')(stdout_ref),
            signature="(Ljava/io/OutputStream;)V"))
    jls.setErr(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')(stderr_ref),
            signature="(Ljava/io/OutputStream;)V"))
