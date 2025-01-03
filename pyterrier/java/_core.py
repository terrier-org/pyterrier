# type: ignore
import os
from pyterrier.java import required_raise, required, before_init, started, mavenresolver, JavaClasses, JavaInitializer, register_config
from typing import Optional
import pyterrier as pt


_stdout_ref = None
_stderr_ref = None


# ----------------------------------------------------------
# Java Initialization
# ----------------------------------------------------------

class CoreJavaInit(JavaInitializer):
    def priority(self) -> int:
        return -100 # run this initializer before anything else

    def pre_init(self, jnius_config):
        if configure['java_home']:
            os.environ['JAVA_HOME'] = configure['java_home']

        if pt.utils.is_windows():
            if "JAVA_HOME" in os.environ:
                java_home =  os.environ["JAVA_HOME"]
                fix = f'{java_home}\\jre\\bin\\server\\;{java_home}\\jre\\bin\\client\\;{java_home}\\bin\\server\\'
                os.environ["PATH"] = os.environ["PATH"] + ";" + fix

        if pt.java.configure['mem'] is not None:
            jnius_config.add_options('-Xmx' + str(pt.java.configure['mem']) + 'm')

        for opt in pt.java.configure['options']:
            jnius_config.add_options(opt)

        for jar in pt.java.configure['jars']:
            jnius_config.add_classpath(jar)

    @required_raise
    def post_init(self, jnius):
        pt.java.set_log_level(pt.java.configure['log_level'])

        if pt.java.configure['redirect_io']:
            pt.java.redirect_stdouterr()

        java_version = pt.java.J.System.getProperty("java.version")
        if java_version.startswith("1.") or java_version.startswith("9."):
            raise RuntimeError(f"Pyterrier requires Java 11 or newer, we only found Java version {java_version};"
                + " install a more recent Java, or change os.environ['JAVA_HOME'] to point to the proper Java installation")

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


def _is_binary(f):
    import io
    return isinstance(f, (io.RawIOBase, io.BufferedIOBase))


@required
def redirect_stdouterr():
    from jnius import autoclass, PythonJavaClass, java_method

    # TODO: encodings may be a probem here
    class MyOut(PythonJavaClass):
        __javainterfaces__ = ['org.terrier.python.OutputStreamable']

        def __init__(self, pystream):
            super(MyOut, self).__init__()
            self.pystream = pystream
            self.binary = _is_binary(pystream)

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

    # we need to hold lifetime references to _stdout_ref/_stderr_ref, to ensure
    # they arent GCd. This prevents a crash when Java callsback to  GCd py obj

    global _stdout_ref
    global _stderr_ref
    import sys
    _stdout_ref = MyOut(sys.stdout)
    _stderr_ref = MyOut(sys.stderr)
    jls = autoclass("java.lang.System")
    jls.setOut(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')(_stdout_ref),
            signature="(Ljava/io/OutputStream;)V"))
    jls.setErr(
        autoclass('java.io.PrintStream')(
            autoclass('org.terrier.python.ProxyableOutputStream')(_stderr_ref),
            signature="(Ljava/io/OutputStream;)V"))


def bytebuffer_to_array(buffer):
    assert buffer is not None
    def unsign(signed):
        return signed + 256 if signed < 0 else signed
    return bytearray([ unsign(buffer.get(offset)) for offset in range(buffer.capacity()) ])


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

configure = register_config('pyterrier.java', {
    'jars': [],
    'options': [],
    'mem': None,
    'log_level': 'WARN',
    'redirect_io': True,
    'java_home': None,
})


@before_init
def add_jar(jar_path):
    configure.append('jars', jar_path)


@before_init
def add_package(org_name : str, package_name : str, version : Optional[str] = None, file_type : str = 'jar'):
    if version is None or version == 'snapshot':
        version = mavenresolver.latest_version_num(org_name, package_name)
    file_name = mavenresolver.get_package_jar(org_name, package_name, version, artifact=file_type)
    add_jar(file_name)


@before_init
def set_memory_limit(mem: Optional[float]):
    configure['mem'] = mem


@before_init
def add_option(option: str):
    configure.append('options', option)


@before_init
def set_redirect_io(redirect_io: bool):
    configure['redirect_io'] = redirect_io


@before_init
def set_java_home(java_home: str):
    """
    Sets the directory to search when loading Java.

    Note that you can achieve the same outcome by setting the `JAVA_HOME` environment variable.
    """
    configure['java_home'] = java_home

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
        configure['log_level'] = level
    else:
        J.PTUtils.setLogLevel(level, None) # noqa: PT100 handled by started() check above


# ----------------------------------------------------------
# Common classes (accessible via pt.java.J.[ClassName])
# ----------------------------------------------------------

J = JavaClasses(
    ArrayList = 'java.util.ArrayList',
    Properties = 'java.util.Properties',
    PTUtils = 'org.terrier.python.PTUtils',
    System = 'java.lang.System',
    StringReader = 'java.io.StringReader',
    HashMap = 'java.util.HashMap',
    Arrays = 'java.util.Arrays',
    Array = 'java.lang.reflect.Array',
    String = 'java.lang.String',
    List = 'java.util.List',
)
