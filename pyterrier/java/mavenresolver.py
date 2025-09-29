from xml.dom import minidom
import urllib.request
from packaging.version import Version
from pathlib import Path
from enum import Enum
from warnings import warn
import requests
import os
import re
import pyterrier as pt

MAVEN_BASE_URL = "https://repo1.maven.org/maven2/"
JITPACK_BASE_URL = "https://jitpack.io/"
USER_AGENT = "curl/7.81.0"

class OnlineMode(Enum):
    ONLINE = 'ONLINE'
    OFFLINE = 'OFFLINE'
    UNSET = 'UNSET'


_online_mode = OnlineMode.UNSET


def online_mode() -> OnlineMode:
    """
    Returns whether mavenresolver is operating in "online" mode, "offline" mode, or if the mode has not been set yet.

    In "online" mode, packages will always be requested from maven. Network failures will raise errors in this mode.

    In "offline" mode, maven will not be contacted, and the most recent versions of packages that are currently cached
    on the system will be used when requesting package versions, etc. If no versions are available, an error will be
    raised.

    If not set, mavenresolver will behave in "online" mode until it receives a network error, at which point it will
    move automatically to "offline" mode.
    """
    return _online_mode


def offline(is_offline: bool = True):
    """
    Enables offline mode. 
    """
    global _online_mode
    _online_mode = {
        True: OnlineMode.OFFLINE,
        False: OnlineMode.ONLINE
    }[is_offline]


def online(is_online: bool = True):
    global _online_mode
    _online_mode = {
        True: OnlineMode.ONLINE,
        False: OnlineMode.OFFLINE
    }[is_online]


# obtain a file from maven
def get_package_jar(orgName, packageName, version, file_path=None, artifact="jar", force_download=False):
    if file_path is None:
        file_path = pt.io.pyterrier_home()
    orgName = orgName.replace(".", "/")
    suffix = ""
    ext = "jar"
    if artifact == "jar-with-dependencies" or artifact == "fatjar":
        suffix = "-" + artifact
        ext = "jar"
    if artifact == "pom":
        ext = "pom"
    filename = packageName + "-" + version + suffix + "." + ext

    filelocation = orgName + "/" + packageName + "/" + version + "/" + filename

    mode = online_mode()

    if force_download and mode == OnlineMode.OFFLINE:
        force_download = False # OFFLINE overrides force_download
    
    target_file = os.path.join(file_path, filename)
    file_exists = os.path.isfile(target_file)
    if file_exists:
        if not force_download:
            return target_file
        else:
            # ensure that we put the file in a different name
            os.rename(target_file, f'{target_file}.bak')

    # check local Maven repo, and use that if it exists
    from os.path import expanduser
    userhome = expanduser("~")
    mavenRepoHome = os.path.join(userhome, ".m2", "repository")
    mvnLocalLocation = os.path.join(mavenRepoHome, filelocation)
    if os.path.isfile(mvnLocalLocation):
        return mvnLocalLocation

    if mode == OnlineMode.OFFLINE:
        raise ValueError(f'Offline mode, and {filename} not found')

    if force_download:
        print("Downloading "+ packageName + " " + version + " " + artifact  + " to " + file_path + "...")
    else:
        print(packageName + " " + version + " " + artifact  + " not found, downloading to " + file_path + "...")
    
    if "com/github" in orgName:
        mvnUrl = JITPACK_BASE_URL + filelocation
    else:
        mvnUrl = MAVEN_BASE_URL + filelocation

    try:
        pt.io.download(mvnUrl, target_file, verbose=True, headers={"User-Agent": USER_AGENT})
    except requests.exceptions.ConnectionError as he:
        if mode == OnlineMode.UNSET:
            offline() # now we're in offline mode
        if file_exists:
            os.rename(f'{target_file}.bak', target_file) # move back
            if mode == OnlineMode.UNSET:
                warn(f'Attempted to download {mvnUrl}, but was offline. Using cached version: {target_file}')
                return target_file
        raise ValueError("Could not fetch " + mvnUrl) from he

    if file_exists:
        os.remove(f'{target_file}.bak') # clean up temp file
    print("Done")

    return os.path.join(file_path, filename)

# returns the latest version
def latest_version_num(orgName, packageName):
    orgName = orgName.replace(".", "/")
    if "com/github" in orgName:
        # its jitpack
        return "-SNAPSHOT"

    mode = online_mode()

    if mode == OnlineMode.OFFLINE:
        version = latest_local_version_num(packageName)
        if version is None:
            raise ValueError(f'Could not find latest version of {packageName}')
        return version

    url_str = MAVEN_BASE_URL + orgName + "/" + packageName + "/maven-metadata.xml"
    try:
        with urllib.request.urlopen(urllib.request.Request( url_str, headers={"User-Agent": USER_AGENT})) as url:
            xml_str = url.read()
    except urllib.error.URLError as ue:
        if mode == OnlineMode.UNSET:
            offline()
            # no internet connection, use the latest version found locally.
            version = latest_local_version_num(packageName)
            if version is None:
                raise # version not found, re-raise the URLError error
            else:
                warn(f'Attempted to get latest version of {packageName} from maven, but was offline ({ue.url} {ue.code} {ue.reason}). Using latest cached version: {version}')
                return version
        else: # mode == OnlineMode.ONLINE
            raise
    xmldoc = minidom.parseString(xml_str)
    obs_values = xmldoc.getElementsByTagName("latest")
    version = obs_values[0].firstChild.nodeValue
    return version


def latest_local_version_num(packageName):
    versions = []
    for jar_path in Path(pt.io.pyterrier_home()).glob(f'{packageName}-*.jar'):
        match = re.search(rf'{packageName}-([0-9]+(\.[0-9]+)+).*.jar', jar_path.name)
        if match:
            versions.append(Version(match.group(1)))
    if len(versions) > 0:
        return str(max(versions))
    return None # no local version found
