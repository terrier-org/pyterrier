import pyterrier as pt
from pyterrier.java import required_raise, before_init, started, configure, mavenresolver
from typing import Dict, Optional


stdout_ref = None
stderr_ref = None


def _is_binary(f):
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


def bytebuffer_to_array(buffer):
    assert buffer is not None
    def unsign(signed):
        return signed + 256 if signed < 0 else signed
    return bytearray([ unsign(buffer.get(offset)) for offset in range(buffer.capacity()) ])


@before_init
def add_jar(jar_path):
    configure.append('jars', jar_path)


@before_init
def add_package(org_name: str = None, package_name: str = None, version: str = None, file_type='jar'):
    if version is None or version == 'snapshot':
        version = mavenresolver.latest_version_num(org_name, package_name)
    file_name = mavenresolver.get_package_jar(org_name, package_name, version, pt.io.pyterrier_home(), file_type)
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
        J.PTUtils.setLogLevel(level, None)


class JavaClasses:
    def __init__(self, mapping: Dict[str, str]):
        self._mapping = mapping
        self._cache = {}

    def __dir__(self):
        return list(self._mapping.keys())

    @required_raise
    def __getattr__(self, key):
        if key not in self._mapping:
            return AttributeError(f'{self} has no attribute {key!r}')
        if key not in self._cache:
            clz = self._mapping[key]
            if callable(clz):
                clz = clz()
            self._cache[key] = pt.java.autoclass(clz)
        return self._cache[key]


J = JavaClasses({
    'ArrayList': 'java.util.ArrayList',
    'Properties': 'java.util.Properties',
    'PTUtils': 'org.terrier.python.PTUtils',
    'System': 'java.lang.System',
    'StringReader': 'java.io.StringReader',
    'HashMap': 'java.util.HashMap',
    'Arrays': 'java.util.Arrays',
    'Array': 'java.lang.reflect.Array',
})
