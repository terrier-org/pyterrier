import urllib.request
import wget
import os

STANDARD_TERRIER_INDEX_FILES=[
    "data.direct.bf",
    "data.document.fsarrayfile",
    "data.inverted.bf",
    "data.lexicon.fsomapfile",
    "data.lexicon.fsomaphash",
    "data.lexicon.fsomapid",
    "data.meta.idx",
    "data.meta.zdata",
    "data.properties"
]


class Dataset():

    def get_corpus_location(self):
        pass

    def get_index(self):
        pass

    def get_topics(self):
        pass

    def get_qrels(self):
        pass

class RemoteDataset(Dataset):

    def __init__(self, name, locations):
        from os.path import expanduser
        userhome = expanduser("~")
        pt_home = os.path.join(userhome, ".pyterrier")
        self.corpus_home = os.path.join(pt_home, "corpora", name) 
        self.locations = locations
        self.name = name

    def _get_one_file(self, component):
        (local, URL) = self.locations[component][0]
        if not os.path.exists(self.corpus_home):
            os.makedirs(self.corpus_home)
        local = os.path.join(self.corpus_home, local)
        if not os.path.exists(local):
            try:
                print("Downloading %s %s to %s" % (self.name, component, local))
                wget.download(URL, local)
            except urllib.error.HTTPError as he:
                raise ValueError("Could not fetch " + URL) from he
        return local

    def _get_all_files(self, component):
        localDir = os.path.join(self.corpus_home, component)
        if not os.path.exists(localDir):
            os.makedirs(localDir)
            print("Downloading %s %s to %s" % (self.name, component, localDir))
        for (local, URL) in self.locations[component]:
            local = os.path.join(localDir, local)
            if not os.path.exists(local):
                try:
                    wget.download(URL, local)
                except urllib.error.HTTPError as he:
                    raise ValueError("Could not fetch " + URL) from he
        return localDir

    def get_qrels(self):
        import pyterrier as pt    
        return pt.Utils.parse_qrels(self._get_one_file("qrels"))

    def get_topics(self):
        import pyterrier as pt    
        return pt.Utils.parse_trec_topics_file(self._get_one_file("topics"))

    def get_index(self):
        import pyterrier as pt  
        thedir = self._get_all_files("index")
        return pt.autoclass("org.terrier.querying.IndexRef").of(os.path.join(thedir, "data.properties"))


VASWANI_CORPUS_BASE = "https://raw.githubusercontent.com/terrier-org/pyterrier/master/tests/fixtures/vaswani_npl/"
VASWANI_INDEX_BASE = "https://raw.githubusercontent.com/terrier-org/pyterrier/master/tests/fixtures/index/"
VASWANI_FILES= {
    "corpus": 
        [  ("doc-text.trec" , VASWANI_CORPUS_BASE + "corpus/doc-text.trec" ) ],
    "topics" : 
        [ ( "query-text.trec" , VASWANI_CORPUS_BASE + "query-text.trec") ],
    "qrels" : 
        [  ( "qrels" , VASWANI_CORPUS_BASE + "qrels") ],
    "index" :
        [  (filename, VASWANI_INDEX_BASE + filename) for filename in STANDARD_TERRIER_INDEX_FILES]
}

DATASET_MAP = {
    "vaswani" : RemoteDataset("vaswani", VASWANI_FILES)
}

def get_dataset(name):
    return DATASET_MAP[name]

