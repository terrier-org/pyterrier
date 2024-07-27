import sys
from typing import Optional
import pyterrier as pt

TERRIER_PKG = "org.terrier"

_resolved_helper_version = None

configure = pt.java.config.register('pt.terrier.java', {
    'terrier_version': None,
    'helper_version': None,
    'boot_packages': [],
    'force_download': True,
})


@pt.java.before_init()
def set_terrier_version(version: Optional[str] = None):
    configure(terrier_version=version)


@pt.java.before_init()
def set_helper_version(version: Optional[str] = None):
    configure(helper_version=version)


@pt.java.before_init()
def enable_prf(version: Optional[str] = None):
    pt.java.add_package('com.github.terrierteam', 'terrier-prf', version)



def _pre_init(jnius_config):
    global _resolved_helper_version
    """
    Download Terrier's jar file for the given version at the given file_path
    Called by pt.init()

    Args:
        file_path(str): Where to download
        terrier_version(str): Which version of Terrier - None is latest release; "snapshot" uses Jitpack to download a build of the current Github 5.x branch.
        helper_version(str): Which version of the helper - None is latest
    """
    # If version is not specified, find newest and download it
    cfg = configure()
    if cfg['terrier_version'] is None:
        terrier_version = pt.java.mavenresolver.latest_version_num(TERRIER_PKG, "terrier-assemblies")
    else:
        terrier_version = str(cfg['terrier_version']) # just in case its a float

    # obtain the fat jar from Maven
    # "snapshot" means use Jitpack.io to get a build of the current
    # 5.x branch from Github - see https://jitpack.io/#terrier-org/terrier-core/5.x-SNAPSHOT
    if terrier_version == "snapshot":
        trJar = pt.java.mavenresolver.downloadfile("com.github.terrier-org.terrier-core", "terrier-assemblies", "5.x-SNAPSHOT", pt.io.pyterrier_home(), "jar-with-dependencies", force_download=cfg['force_download'])
    else:
        trJar = pt.java.mavenresolver.downloadfile(TERRIER_PKG, "terrier-assemblies", terrier_version, pt.io.pyterrier_home(), "jar-with-dependencies")
    pt.java.add_jar(trJar)

    # now the helper classes
    if cfg['helper_version'] is None or cfg['helper_version'] == 'snapshot':
        helper_version = pt.java.mavenresolver.latest_version_num(TERRIER_PKG, "terrier-python-helper")
    else:
        helper_version = str(cfg['helper_version']) # just in case its a float
    _resolved_helper_version = helper_version
    pt.java.add_package(TERRIER_PKG, "terrier-python-helper", helper_version)


def _post_init(jnius):
    version_string = J.Version.VERSION
    if "BUILD_DATE" in dir(J.Version):
        version_string += f" (built by {J.Version.BUILD_USER} on {J.Version.BUILD_DATE})"

    print(f"PyTerrier {pt.__version__} has loaded Terrier {version_string} and "
          f"terrier-helper {_resolved_helper_version}\n", file=sys.stderr)

    pt.IndexRef = J.IndexRef


@pt.java.required()
def extend_package(package):
    """
        Allows to add packages to Terrier's classpath after the JVM has started.
    """
    assert pt.check_version(5.3), "Terrier 5.3 required for this functionality"
    package_list = pt.java.J.ArrayList()
    package_list.add(package)
    mvnr = pt.ApplicationSetup.getPlugin("MavenResolver")
    assert mvnr is not None
    mvnr = pt.java.cast("org.terrier.utility.MavenResolver", mvnr)
    mvnr.addDependencies(package_list)


# Terrier-specific classes
J = pt.java.JavaClasses({
    'IndexRef': 'org.terrier.querying.IndexRef',
    'Version': 'org.terrier.Version',
    'Tokenizer': 'org.terrier.indexing.tokenisation.Tokeniser',
})
