from xml.dom import minidom
import urllib.request
import wget
import os

MAVEN_BASE_URL = "https://repo1.maven.org/maven2/"
JITPACK_BASE_URL = "https://jitpack.io/"

# obtain a file from maven
def downloadfile(orgName, packageName, version, file_path, artifact="jar", force_download=False):
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
    
    target_file = os.path.join(file_path, filename)
    file_exists = os.path.isfile(target_file)
    if file_exists:
        if not force_download:
            return target_file
        else:
            # ensure that wget doesnt put the file in a different name
            os.remove(target_file)

    # check local Maven repo, and use that if it exists
    from os.path import expanduser
    userhome = expanduser("~")
    mavenRepoHome = os.path.join(userhome, ".m2", "repository")
    mvnLocalLocation = os.path.join(mavenRepoHome, filelocation)
    if os.path.isfile(mvnLocalLocation):
        return mvnLocalLocation

    if force_download:
        print("Downloading "+ packageName + " " + version + " " + artifact  + " to " + file_path + "...")
    else:
        print(packageName + " " + version + " " + artifact  + " not found, downloading to " + file_path + "...")
    
    if "com/github" in orgName:
        mvnUrl = JITPACK_BASE_URL + filelocation
    else:
        mvnUrl = MAVEN_BASE_URL + filelocation

    try:
        wget.download(mvnUrl, file_path)
    except urllib.error.HTTPError as he:
        raise ValueError("Could not fetch " + mvnUrl) from he
    print("Done")

    return (os.path.join(file_path, filename))

# returns the latest version
def latest_version_num(orgName, packageName):
    orgName = orgName.replace(".", "/")
    if "com/github" in orgName:
        # its jitpack
        return("-SNAPSHOT")

    url_str = MAVEN_BASE_URL + orgName + "/" + packageName + "/maven-metadata.xml"
    with urllib.request.urlopen(url_str) as url:
        xml_str = url.read()
    xmldoc = minidom.parseString(xml_str)
    obs_values = xmldoc.getElementsByTagName("latest")
    version = obs_values[0].firstChild.nodeValue
    return(version)
