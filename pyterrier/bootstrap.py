

import os
from . import mavenresolver


TERRIER_PKG = "org.terrier"

def setup_logging(level):
    from jnius import autoclass
    autoclass("org.terrier.python.PTUtils").setLogLevel(level, None)

def setup_terrier(file_path, terrier_version=None, helper_version=None):
    """
    Download Terrier's jar file for the given version at the given file_path
    Called by pt.init()

    Args:
        file_path(str): Where to download
        terrier_version(str): Which version of Terrier - None is latest
        helper_version(str): Which version of the helper - None is latest
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    # If version is not specified, find newest and download it
    if terrier_version is None:
        terrier_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-assemblies")
    else: 
        terrier_version = str(terrier_version) #just in case its a float
    #obtain the fat jar from Maven
    trJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-assemblies", terrier_version, file_path, "jar-with-dependencies")
    
    #now the helper classes
    if helper_version is None:
        helper_version = mavenresolver.latest_version_num(TERRIER_PKG, "terrier-python-helper")
    else: 
        helper_version = str(helper_version) #just in case its a float
    helperJar = mavenresolver.downloadfile(TERRIER_PKG, "terrier-python-helper", helper_version, file_path, "jar")
    return [trJar, helperJar]


def is_binary(f):
        import io
        return isinstance(f, (io.RawIOBase, io.BufferedIOBase))

def redirect_stdouterr():
    from jnius import autoclass, PythonJavaClass, java_method

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
            #TODO probably this could be faster.
            for c in byteArray:
                self.writeChar(c)
        
        @java_method('([BII)V', name='write')
        def writeByteArrayIntInt(self, byteArray, offset, length):
            #TODO probably this could be faster.
            for i in range(offset, offset+length):
                self.writeChar(byteArray[i])
        
        @java_method('(I)V', name='write')
        def writeChar(self, chara):
            if self.binary:
                return self.pystream.write(bytes([chara]))
            return self.pystream.write(chr(chara))

    #from utils import MyOut
    import sys
    jls = autoclass("java.lang.System")
    jls.setOut(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')( MyOut(sys.stdout)), 
            signature="(Ljava/io/OutputStream;)V"))
    jls.setErr(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')( MyOut(sys.stderr)), 
            signature="(Ljava/io/OutputStream;)V"))