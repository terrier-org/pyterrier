# import jnius_config
# jnius_config.add_classpath("../terrier-project-5.1-jar-with-dependencies.jar")
# from jnius import autoclass, cast

# import pytrec_eval
import os, json, wget
import numpy as np
import pandas as pd
from xml.dom import minidom
import urllib.request

def setup_terrier(file_path, version):
    file_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(os.path.join(file_path,"terrier-assemblies-"+version+"-jar-with-dependencies.jar")):
        print("Terrier "+ version +" not found, downloading")
        # url = 'https://repo.maven.apache.org/maven2/org/terrier/terrier-assemblies/5.1/terrier-assemblies-5.1-jar-with-dependencies.jar'
        url = "https://repo.maven.apache.org/maven2/org/terrier/terrier-assemblies/"+version+"/terrier-assemblies-"+version+"-jar-with-dependencies.jar"
        wget.download(url, file_path)

file_path = os.path.dirname(os.path.abspath(__file__))
firstInit = False
# print(file_path)
# setup_terrier(file_path)
# import jnius_config
# jnius_config.add_classpath(os.path.join(file_path,"terrier-assemblies-5.1-jar-with-dependencies.jar"))
# from jnius import autoclass, cast
# from utils import *
# from batchretrieve import *
# from index import *

def init(version=None, mem="4096", packages=[]):
    # If version is not specified, find newest and download it
    if version is None:
        url_str = "https://repo1.maven.org/maven2/org/terrier/terrier-assemblies/maven-metadata.xml"
        with urllib.request.urlopen(url_str) as url:
            xml_str = url.read()
        xmldoc = minidom.parseString(xml_str)
        obs_values = xmldoc.getElementsByTagName("latest")
        version = obs_values[0].firstChild.nodeValue
    else: version = str(version)
    url = "https://repo.maven.apache.org/maven2/org/terrier/terrier-assemblies/"+version+"/terrier-assemblies-"+version+"-jar-with-dependencies.jar"
    setup_terrier(file_path, version)

    # Import pyjnius and other classes
    import jnius_config
    jnius_config.set_classpath(os.path.join(file_path,"terrier-assemblies-"+version+"-jar-with-dependencies.jar"))
    jnius_config.add_options('-Xrs', '-Xmx'+str(mem)+'m')
    from jnius import autoclass, cast
    from utils import Utils
    from batchretrieve import BatchRetrieve, FeaturesBatchRetrieve
    from index import BasicIndex, createFilesIndex, createTRECIndex
    # Make imports global
    globals()["Utils"]=Utils
    globals()["autoclass"] = autoclass
    globals()["cast"] = cast
    globals()["BatchRetrieve"] = BatchRetrieve
    globals()["FeaturesBatchRetrieve"] = FeaturesBatchRetrieve
    globals()["BasicIndex"] = BasicIndex
    globals()["createFilesIndex"] = createFilesIndex
    globals()["createTRECIndex"] = createTRECIndex

    # Import other java packages
    if packages != []:
        ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')
        properties = autoclass('java.util.Properties')()
        pkgs_string = ",".join(packages)
        properties.put("terrier.mvn.coords",pkgs_string)
        ApplicationSetup.bootstrapInitialisation(properties)
        # sqlClass = ApplicationSetup.getClass("com.harium.database.sqlite.module.SQLiteDatabaseModule"))
        print(ApplicationSetup.getProperty("terrier.mvn.coords",None))

if __name__ == "__main__":
    # init(packages=["com.harium.database:sqlite:1.0.5"])
    init()
    # JIR = autoclass('org.terrier.querying.IndexRef')
    # indexref = JIR.of("../index/data.properties")
    index_path = "../index/data.properties"

    topics = Utils.parse_trec_topics_file("../vaswani_npl/query-text.trec")
    topics_light = Utils.parse_trec_topics_file("../vaswani_npl/query_light.trec")

    # feat_retrieve = FeaturesBatchRetrieve(index_path, ["WMODEL:BM25","WMODEL:PL2"])
    # feat_res = feat_retrieve.transform(topics_light)
    # print(feat_res)

    # retr = BatchRetrieve(index_path)
    # batch_retrieve_results=retr.transform(topics)
    # print(batch_retrieve_results)

    # retr.saveLastResult("dph.res")
    # retr.saveResult(batch_retrieve_results,"/home/alex/Documents/Pyterrier/result.res")

    # qrels = Utils.parse_qrels("../vaswani_npl/qrels")
    # eval = Utils.evaluate(batch_retrieve_results,qrels)
    # print(eval)
    # lst = ["Doc1", "Doc2", "Doc3"]

    dct = {"text": [
    "He ran out of money, so he had to stop playing poker.",
    "The waves were crashing on the shore; it was a lovely sight.",
    "The body may perhaps compensates for the loss of a true metaphysics."],
    "docno": ["1","2","3"]}
    df = pd.DataFrame(dct)

    index_path2 = "/home/alex/Documents/index"

    path = "/home/alex/Documents/pyterrier/vaswani_npl/corpus/"
    path2 = "/home/alex/Downloads/books"
    path3 = "/home/alex/Downloads/books/doc-text.trec"

# TREC INDEX
    # basicIndex = BasicIndex(path3, index_path2)
    # retr = BatchRetrieve(index_path2+"/data.properties")
    # batch_retrieve_results=retr.transform("file")
    # print(batch_retrieve_results)

    index_path = createTRECIndex(path3, index_path2)
    retr = BatchRetrieve(index_path)
    batch_retrieve_results=retr.transform("file")
    print(batch_retrieve_results)

#  DATAFRAME INDEX
    # basicIndex = BasicIndex(df, index_path2)
    # retr = BatchRetrieve(index_path2+"/data.properties")
    # batch_retrieve_results=retr.transform("sight")
    # print(batch_retrieve_results)

# TXT INDEX
    # basicIndex = BasicIndex(path2, index_path2)
    # retr = BatchRetrieve(index_path2+"/data.properties")
    # batch_retrieve_results=retr.transform("file")
    # print(batch_retrieve_results)
