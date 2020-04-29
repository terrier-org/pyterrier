import deprecation

from . import mavenresolver

stdout_ref = None
stderr_ref = None
TERRIER_PKG = "org.terrier"

@deprecation.deprecated(deprecated_in="0.1.3",
                        # remove_id="",
                        details="Use the logging(level) function instead")
def setup_logging(level):
    logging(level)

def logging(level):
    from jnius import autoclass
    autoclass("org.terrier.python.PTUtils").setLogLevel(level, None)

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
        '__next__': lambda self: _iterableposting_next(self)
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

def setup_terrier(file_path, terrier_version=None, helper_version=None, boot_packages=[]):
    """
    Download Terrier's jar file for the given version at the given file_path
    Called by pt.init()

    Args:
        file_path(str): Where to download
        terrier_version(str): Which version of Terrier - None is latest
        helper_version(str): Which version of the helper - None is latest
    """
    # If version is not specified, find newest and download it
    if terrier_version is None:
        terrier_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-assemblies")
    else:
        terrier_version = str(terrier_version) # just in case its a float
    # obtain the fat jar from Maven
    trJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-assemblies", terrier_version, file_path, "jar-with-dependencies")

    # now the helper classes
    if helper_version is None:
        helper_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-python-helper")
    else:
        helper_version = str(helper_version) # just in case its a float
    helperJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-python-helper", helper_version, file_path, "jar")

    classpath=[trJar, helperJar]
    for b in boot_packages:
        group, pkg, filetype, version = b.split(":")
        if version is None:
            version = filetype
            filetype = "jar"
        print((group, pkg, filetype, version))
        filename = mavenresolver.downloadfile(group, pkg, version, file_path, filetype)
        classpath.append(filename)

    return classpath

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
