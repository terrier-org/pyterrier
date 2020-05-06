import urllib.request
import wget
import os

STANDARD_TERRIER_INDEX_FILES = [
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

    def _get_one_file(self, component, variant=None):
        filetype=None
        location = self.locations[component][0] if variant is None else self.locations[component][variant]
        local = location[0]
        URL = location [1]
        if len(location) > 2:
            filetype = location[2]

        if not os.path.exists(self.corpus_home):
            os.makedirs(self.corpus_home)
        local = os.path.join(self.corpus_home, local)
        if not os.path.exists(local):
            try:
                print("Downloading %s %s to %s" % (self.name, component, local))
                wget.download(URL, local)
            except urllib.error.HTTPError as he:
                raise ValueError("Could not fetch " + URL) from he
        return (local, filetype)

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

    def get_corpus(self):
        import pyterrier as pt
        return pt.Utils.get_files_in_dir(self._get_all_files("corpus"))

    def get_qrels(self, variant=None):
        import pyterrier as pt
        return pt.Utils.parse_qrels(self._get_one_file("qrels", variant)[0])

    def get_topics(self, variant=None):
        import pyterrier as pt
        file, filetype = self._get_one_file("topics", variant)
        if filetype is None and filetype == "trec":
            return pt.Utils.parse_trec_topics_file(file)
        elif filetype == "singleline":
            return pt.Utils.parse_singleline_topics_file(file)

    def get_index(self):
        import pyterrier as pt
        thedir = self._get_all_files("index")
        return pt.autoclass("org.terrier.querying.IndexRef").of(os.path.join(thedir, "data.properties"))


TREC_DEEPLEARNING_MSMARCO_FILES = {
    "corpus" : 
        [("msmarco-docs.trec.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz")],
    "topics" : 
        { 
            "train" : ("msmarco-doctrain-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz", "singleline"),
            "dev" : ("msmarco-docdev-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz", "singleline"),
            "test" : ("msmarco-test2019-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz", "singleline")
        },
    "qrels" : 
        { 
            "train" : ("msmarco-doctrain-qrels.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz"),
            "dev" : ("msmarco-docdev-qrels.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz"),
            "test" : ("2019qrels-docs.txt", "https://trec.nist.gov/data/deep/2019qrels-docs.txt")
        }
}

TREC_ROBUST_04_FILES = {
    "qrels" : [ ("qrels.robust2004.txt", "https://trec.nist.gov/data/robust/qrels.robust2004.txt") ],
    "topics" : [ (  "04.testset.gz", "https://trec.nist.gov/data/robust/04.testset.gz" ) ]
}
TREC_ROBUST_05_FILES = {
    "qrels" : [ ("TREC2005.qrels.txt", "https://trec.nist.gov/data/robust/05/TREC2005.qrels.txt") ],
    "topics" : [ (  "05.50.topics.txt", "https://trec.nist.gov/data/robust/05/05.50.topics.txt" ) ]
}

VASWANI_CORPUS_BASE = "https://raw.githubusercontent.com/terrier-org/pyterrier/master/tests/fixtures/vaswani_npl/"
VASWANI_INDEX_BASE = "https://raw.githubusercontent.com/terrier-org/pyterrier/master/tests/fixtures/index/"
VASWANI_FILES = {
    "corpus":
        [("doc-text.trec", VASWANI_CORPUS_BASE + "corpus/doc-text.trec")],
    "topics":
        [("query-text.trec", VASWANI_CORPUS_BASE + "query-text.trec")],
    "qrels":
        [("qrels", VASWANI_CORPUS_BASE + "qrels")],
    "index":
        [(filename, VASWANI_INDEX_BASE + filename) for filename in STANDARD_TERRIER_INDEX_FILES]
}

DATASET_MAP = {
    "vaswani": RemoteDataset("vaswani", VASWANI_FILES),
    "trec-deep-learning-docs" : RemoteDataset("trec-deep-learning-docs", TREC_DEEPLEARNING_MSMARCO_FILES),
    "trec-robust-2004" : RemoteDataset("trec-robust-2004", TREC_ROBUST_04_FILES),
    "trec-robust-2005" : RemoteDataset("trec-robust-2005", TREC_ROBUST_05_FILES),
}

def get_dataset(name):
    return DATASET_MAP[name]
