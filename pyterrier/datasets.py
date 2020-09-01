import urllib.request
import wget
import os
import pandas as pd
from .transformer import is_lambda
import types
import requests
from tqdm import tqdm

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

    def _configure(self, **kwargs):
        pass

    def get_corpus_location(self):
        ''' Returns the location of the files to allow indexing the corpus '''
        pass

    def get_index(self):
        ''' Returns the IndexRef of the index to allow retrieval '''
        pass

    def get_topics(self):
        ''' Returns the topics, as a dataframe, ready for retrieval '''
        pass

    def get_qrels(self):
        ''' Returns the qrels, as a dataframe, ready for evaluation '''
        pass

class RemoteDataset(Dataset):

    def __init__(self, name, locations):
        from os.path import expanduser
        userhome = expanduser("~")
        pt_home = os.path.join(userhome, ".pyterrier")
        self.corpus_home = os.path.join(pt_home, "corpora", name)
        self.locations = locations
        self.name = name
        self.user = None
        self.password = None

    def _configure(self, **kwargs):
        if 'user' in kwargs:
            self.user = kwargs['user']
            self.password = kwargs['password']

    @staticmethod
    def download(URL, filename, **kwargs):
        basename = os.path.basename(filename)
        r = requests.get(URL, allow_redirects=True, stream=True, **kwargs)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(filename, 'wb') as file, tqdm(
                desc=basename,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in r.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    def _check_variant(self, component, variant=None):
        name=self.name
        if not component in self.locations:
            raise ValueError("No %s in dataset %s" % (component, name))
        if variant is None:
            if not isinstance(self.locations[component], list):
                raise ValueError("For %s in dataset %s, you must specify a variant=. Available are: %s" % (component, name, str(self.locations[component].keys())))
            location = self.locations[component][0]
        else:
            if isinstance(self.locations[component], list):
                raise ValueError("For %s in dataset %s, there are no variants, but you specified %s" % (component, name, variant))
            if not variant in self.locations[component]:
                raise ValueError("For %s in dataset %s, there is no variant %s. Available are: %s" % (component, name, variant, str(self.locations[component].keys())))

    def _get_one_file(self, component, variant=None):
        filetype=None        
        self._check_variant(component, variant)
        location = self.locations[component][0] if variant is None else self.locations[component][variant]

        if is_lambda(location) or isinstance(location, types.FunctionType):
            argcount = location.__code__.co_argcount
            if argcount == 0:
                return location()
            elif argcount == 3:
                return location(self, component, variant)
            else:
                raise TypeError("Expected function with 0 or 3 arguments for  %s %s %s" % (component, name, variant))

        local = location[0]
        URL = location[1]
        if len(location) > 2:
            filetype = location[2]

        if not os.path.exists(self.corpus_home):
            os.makedirs(self.corpus_home)
        local = os.path.join(self.corpus_home, local)
        if not os.path.exists(local):
            try:
                print("Downloading %s %s to %s" % (self.name, component, local))
                kwargs = {}
                if self.user is not None:
                    kwargs["auth"]=(self.user, self.password)
                RemoteDataset.download(URL, local, **kwargs)
            except urllib.error.HTTPError as he:
                raise ValueError("Could not fetch " + URL) from he
        return (local, filetype)

    def _get_all_files(self, component, **kwargs):
        localDir = os.path.join(self.corpus_home, component)
        if not os.path.exists(localDir):
            os.makedirs(localDir)
            print("Downloading %s %s to %s" % (self.name, component, localDir))
        kwargs = {}
        if self.user is not None:
            kwargs["auth"]=(self.user, self.password)
        for (local, URL) in self.locations[component]:
            local = os.path.join(localDir, local)
            if not os.path.exists(local):
                try:
                    RemoteDataset.download(URL, local, **kwargs)
                except urllib.error.HTTPError as he:
                    raise ValueError("Could not fetch " + URL) from he
        return localDir

    def _describe_component(self, component):
        if component not in self.locations:
            return None
        if type(self.locations[component]) == type([]):
            return True
        return self.locations[component].keys()

    def get_corpus(self, **kwargs):
        import pyterrier as pt
        return pt.Utils.get_files_in_dir(self._get_all_files("corpus", **kwargs))

    def get_qrels(self, variant=None):
        import pyterrier as pt
        filename, type = self._get_one_file("qrels", variant)
        if type == "direct":
            return filename 
        return pt.io.read_qrels(filename)

    def get_topics(self, variant=None, **kwargs):
        import pyterrier as pt
        file, filetype = self._get_one_file("topics", variant)
        if filetype is None or filetype in pt.io.SUPPORTED_TOPICS_FORMATS:
            return pt.io.read_topics(file, format=filetype, **kwargs)
        elif filetype == "direct":
            return file
        raise ValueError("Unknown filetype %s for %s topics %s"  % (filetype, self.name, variant))

    def get_index(self):
        import pyterrier as pt
        thedir = self._get_all_files("index")
        return pt.autoclass("org.terrier.querying.IndexRef").of(os.path.join(thedir, "data.properties"))

    def __repr__(self):
        return "RemoteDataset for %s, with %s" % (self.name, str(list(self.locations.keys())))

TREC_COVID_FILES = {
    "topics" : {
        "round1" : ("topics-rnd1.xml", "https://ir.nist.gov/covidSubmit/data/topics-rnd1.xml", "trecxml"),
        "round2" : ("topics-rnd2.xml", "https://ir.nist.gov/covidSubmit/data/topics-rnd2.xml", "trecxml"),
        "round3" : ("topics-rnd3.xml", "https://ir.nist.gov/covidSubmit/data/topics-rnd3.xml", "trecxml"),
        "round4" : ("topics-rnd4.xml", "https://ir.nist.gov/covidSubmit/data/topics-rnd4.xml", "trecxml"),
        "round5" : ("topics-rnd5.xml", "https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml", "trecxml"),
    },
    "qrels" : {
        "round1" : ("qrels-rnd1.txt", "https://ir.nist.gov/covidSubmit/data/qrels-rnd1.txt"),
        "round2" : ("qrels-rnd2.txt", "https://ir.nist.gov/covidSubmit/data/qrels-rnd2.txt"),
        "round3" : ("qrels-rnd3.txt", "https://ir.nist.gov/covidSubmit/data/qrels-covid_d3_j2.5-3.txt"),
        "round3-cumulative" : ("qrels-rnd3-cumulative.txt", "https://ir.nist.gov/covidSubmit/data/qrels-covid_d3_j0.5-3.txt"),
        "round4" : ("qrels-rnd4.txt", "https://ir.nist.gov/covidSubmit/data/qrels-covid_d4_j3.5-4.txt"),
        "round4-cumulative" : ("qrels-rnd4-cumulative.txt", "https://ir.nist.gov/covidSubmit/data/qrels-covid_d4_j0.5-4.txt"),
        "round5" : ("qrels-covid_d5_j4.5-5.txt", "https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j4.5-5.txt"),
    },
    "corpus" : {
        "round4": ("round4.tar.gz", "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-06-19.tar.gz"),
        "round5": ("round5.tar.gz", "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-07-16.tar.gz"), 
    },
    "docids" : { 
        "docids-rnd3" : ("docids-rnd3.txt", "https://ir.nist.gov/covidSubmit/data/docids-rnd3.txt"),
        "docids-rnd4" : ("docids-rnd4.txt", "https://ir.nist.gov/covidSubmit/data/docids-rnd4.txt"),
        "docids-rnd5" : ("docids-rnd5.txt", "https://ir.nist.gov/covidSubmit/data/docids-rnd5.txt")
    },
}

TREC_DEEPLEARNING_MSMARCO_FILES = {
    "corpus" : 
        [("msmarco-docs.trec.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz")],
    "corpus-tsv":
        [("msmarco-docs.tsv.gz",  "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz")],
    "topics" : 
        { 
            "train" : ("msmarco-doctrain-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz", "singleline"),
            "dev" : ("msmarco-docdev-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz", "singleline"),
            "test" : ("msmarco-test2019-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz", "singleline"),
            "test-2020" : ("msmarco-test2020-queries.tsv.gz" , "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz", "singleline"),
            'leaderboard-2020' : ("docleaderboard-queries.tsv.gz" , "https://msmarco.blob.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz", "singleline")
        },
    "qrels" : 
        { 
            "train" : ("msmarco-doctrain-qrels.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz"),
            "dev" : ("msmarco-docdev-qrels.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz"),
            "test" : ("2019qrels-docs.txt", "https://trec.nist.gov/data/deep/2019qrels-docs.txt")
        }
}

# remove WT- prefix from topics
def remove_prefix(self, component, variant):
    import pyterrier as pt
    topics_file, type = self._get_one_file("topics_prefixed", variant)
    if type in pt.io.SUPPORTED_TOPICS_FORMATS:
        topics = pt.io.read_topics(topics_file, type)
    else:
        raise ValueError("Unknown topic type %s" % type)
    topics["qid"] = topics.apply(lambda row: row["qid"].split("-")[1], axis=1)
    return (topics, "direct")


# a function to fix the namedpage TREC Web track 2002
def parse_desc_only(self, component, variant):
    import pyterrier as pt
    file, type = self._get_one_file("topics_special_np")
    topics = pt.io.read_topics(file, format="trec", whitelist=["DESC"], blacklist=None)
    topics["qid"] = topics.apply(lambda row: row["qid"].replace("NP", ""), axis=1)
    return (topics, "direct")

TREC_WT_2002_FILES = {
    "topics" : 
        { 
            "td" : ("webtopics_551-600.txt.gz", "https://trec.nist.gov/data/topics_eng/webtopics_551-600.txt.gz", "trec"),
            "np" : parse_desc_only
        },
    "topics_special_np" : [
        ("webnamed_page_topics.1-150.txt.gz", "https://trec.nist.gov/data/topics_eng/webnamed_page_topics.1-150.txt.gz", "trec")
    ],
    "qrels" : 
        { 
            "np" : ("qrels.named-page.txt.gz", "https://trec.nist.gov/data/qrels_eng/qrels.named-page.txt.gz"),
            "td" : ("qrels.distillation.txt.gz", "https://trec.nist.gov/data/qrels_eng/qrels.distillation.txt.gz")
        }
}

TREC_WT_2003_FILES = {
    "topics" : 
        { 
            "np" : ("webtopics_551-600.txt.gz", "https://trec.nist.gov/data/topics_eng/webtopics_551-600.txt.gz", "trec"),
            "td" : ("2003.distillation_topics.1-50.txt", "https://trec.nist.gov/data/topics_eng/2003.distillation_topics.1-50.txt", "trec"),
        },
    "qrels" : 
        { 
            "np" : ("qrels.named-page.txt.gz", "https://trec.nist.gov/data/qrels_eng/qrels.named-page.txt.gz"),
            "td" : ("qrels.distillation.2003.txt", "https://trec.nist.gov/data/qrels_eng/qrels.distillation.2003.txt")
        }
}

def filter_on_qid_type(self, component, variant):
    if component == "topics":
        data = self.get_topics("all")
    elif component == "qrels":
        data = self.get_qrels("all")
    qid2type = pd.read_csv("https://trec.nist.gov/data/web/04.topic-map.official.txt", names=["qid", "type"], sep=" ")
    qid2type["qid"] = qid2type.apply(lambda row: row["qid"].split("-")[1], axis=1)
    rtr = data.merge(qid2type[qid2type["type"] == variant], on=["qid"])
    if len(rtr) == 0:
        raise ValueError("No such topic type '%s'" % variant)
    rtr.drop(columns=['type'], inplace=True)
    return (rtr, "direct")

TREC_WT_2004_FILES = {
    "topics" : 
        { 
            "all" : remove_prefix,
            "np": filter_on_qid_type,
            "hp": filter_on_qid_type,
            "td": filter_on_qid_type,
        },
    "topics_prefixed" : 
        { 
            "all" : ("Web2004.query.stream.trecformat.txt", "https://trec.nist.gov/data/web/Web2004.query.stream.trecformat.txt", "trec")
        },
    "qrels" : 
        {
            "hp" : filter_on_qid_type,
            "td" : filter_on_qid_type,
            "np" : filter_on_qid_type,
            "all" : ("04.qrels.web.mixed.txt", "https://trec.nist.gov/data/web/04.qrels.web.mixed.txt")
        }
}

FIFTY_PCT_INDEX_BASE = "http://www.dcs.gla.ac.uk/~craigm/IR_HM/index/"
FIFTY_PCT_FILES = {
    "index":
        [(filename, FIFTY_PCT_INDEX_BASE + filename) for filename in ["data.meta-0.fsomapfile"] + STANDARD_TERRIER_INDEX_FILES]
}



# a function for the TREC Web track 2009 qrels, to make prels into qrels
def prel2qrel(self, component, variant): 
    prel_file, _ = self._get_one_file("prels", variant)
    df = pd.read_csv(prel_file, sep=" ", names=["qid", "docno", "label", "oth1", "oth2"])[["qid", "docno", "label"]]
    df["qid"] = df["qid"].astype(str)
    df["docno"] = df["docno"].astype(str)
    return (df, "direct")

TREC_WT_2009_FILES = {
    "topics" : [  
            remove_prefix
        ],

    "topics_prefixed" : [  
            ("wt09.topics.queries-only", "https://trec.nist.gov/data/web/09/wt09.topics.queries-only", "singleline")
        ],
    "qrels" :  {
        "adhoc" : prel2qrel, 
        "adhoc.catA" : prel2qrel,
        "adhoc.catB" : prel2qrel,
    },
    "prels" : {
        "adhoc" : ("prels.1-50.gz", "https://trec.nist.gov/data/web/09/prels.1-50.gz"),
        "adhoc.catA" : ("prels.catA.1-50.gz", "https://trec.nist.gov/data/web/09/prels.catA.1-50.gz"),
        "adhoc.catB" : ("prels.catB.1-50.gz", "https://trec.nist.gov/data/web/09/prels.catB.1-50.gz")
    }
}

TREC_WT_2010_FILES = {
     "topics" : [  
            ("wt2010-topics.queries-only", "https://trec.nist.gov/data/web/10/wt2010-topics.queries-only", "singleline")
        ],
    "qrels" : 
        { 
            "adhoc" : ("qrels.adhoc", "https://trec.nist.gov/data/web/10/10.adhoc-qrels.final")
        }
}

TREC_WT_2011_FILES = {
     "topics" : [  
            ("queries.101-150.txt", "https://trec.nist.gov/data/web/11/queries.101-150.txt", "singleline")
        ],
    "qrels" : 
        { 
            "adhoc" : ("qrels.adhoc", "https://trec.nist.gov/data/web/11/qrels.adhoc")
        }
}

TREC_WT_2012_FILES = {
     "topics" : [  
            ("queries.151-200.txt", "https://trec.nist.gov/data/web/12/queries.151-200.txt", "singleline")
        ],
    "qrels" : 
        { 
            "adhoc" : ("qrels.adhoc", "https://trec.nist.gov/data/web/12/qrels.adhoc")
        }
}

TREC_WT2G_FILES = {
    "qrels" : [ ("qrels.trec8.small_web.gz", "https://trec.nist.gov/data/qrels_eng/qrels.trec8.small_web.gz") ],
    "topics" : [ (  "topics.401-450.gz", "https://trec.nist.gov/data/topics_eng/topics.401-450.gz" ) ]
}

TREC_ROBUST_04_FILES = {
    "qrels" : [ ("qrels.robust2004.txt", "https://trec.nist.gov/data/robust/qrels.robust2004.txt") ],
    "topics" : [ (  "04.testset.gz", "https://trec.nist.gov/data/robust/04.testset.gz" ) ]
}
TREC_ROBUST_05_FILES = {
    "qrels" : [ ("TREC2005.qrels.txt", "https://trec.nist.gov/data/robust/05/TREC2005.qrels.txt") ],
    "topics" : [ (  "05.50.topics.txt", "https://trec.nist.gov/data/robust/05/05.50.topics.txt" ) ]
}

TREC_PRECISION_MEDICINE_FILES = {
    "topics" : {
        "2017" : ("topics2017.xml", "http://www.trec-cds.org/topics2017.xml", "trecxml"),
        "2018" : ("topics2018.xml", "http://www.trec-cds.org/topics2018.xml", "trecxml"),
        "2019" : ("topics2019.xml", "http://www.trec-cds.org/topics2019.xml", "trecxml"),
        "2020" : ("topics2020.xml", "http://www.trec-cds.org/topics2020.xml", "trecxml")
    },
    "qrels" : {
        "qrels-2017-abstracts" : ("qrels-2017-abstracts.txt", "https://trec.nist.gov/data/precmed/qrels-final-abstracts.txt"),  #TODO keep original names?
        "qrels-2017-abstracts-sample" : ("qrels-2017-abstracts-sample.txt", "https://trec.nist.gov/data/precmed/sample-qrels-final-abstracts.txt"),
        "qrels-2017-trials" : ("qrels-2017-trials.txt", "https://trec.nist.gov/data/precmed/qrels-final-trials.txt"),
        "qrels-2018-abstracts" : ("qrels-2018-abstracts.txt", "https://trec.nist.gov/data/precmed/qrels-treceval-abstracts-2018-v2.txt"),
        "qrels-2018-abstracts-sample" : ("qrels-2018-abstracts-sample.txt", "https://trec.nist.gov/data/precmed/qrels-sample-abstracts-v2.txt"),
        "qrels-2018-trials" : ("qrels-2018-trials.txt", "https://trec.nist.gov/data/precmed/qrels-treceval-clinical_trials-2018-v2.txt"),
        "qrels-2018-trials-sample" : ("qrels-2018-trials-sample.txt", "https://trec.nist.gov/data/precmed/qrels-sample-trials-v2.txt"),
        "qrels-2019-abstracts" : ("qrels-2019-abstracts.txt", "https://trec.nist.gov/data/precmed/qrels-treceval-abstracts.2019.txt"),
        "qrels-2019-trials" : ("qrels-2019-trials.txt", "https://trec.nist.gov/data/precmed/qrels-treceval-trials.38.txt"),
        "qrels-2019-abstracts-sample" : ("qrels-2019-abstracts-sample.txt", "https://trec.nist.gov/data/precmed/qrels-sampleval-abstracts.2019.txt"),
        "qrels-2019-trials-sample" : ("qrels-2019-trials-sample.txt", "https://trec.nist.gov/data/precmed/qrels-sampleval-trials.38.txt")
    }
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
    "50pct" : RemoteDataset("50pct", FIFTY_PCT_FILES),
    "vaswani": RemoteDataset("vaswani", VASWANI_FILES),
    "trec-deep-learning-docs" : RemoteDataset("trec-deep-learning-docs", TREC_DEEPLEARNING_MSMARCO_FILES),
    "trec-robust-2004" : RemoteDataset("trec-robust-2004", TREC_ROBUST_04_FILES),
    "trec-robust-2005" : RemoteDataset("trec-robust-2005", TREC_ROBUST_05_FILES),
    #medical-like tracks
    "trec-precision-medicine" : RemoteDataset("trec-precicion-medicine", TREC_PRECISION_MEDICINE_FILES),
    "trec-covid" : RemoteDataset("trec-covid", TREC_COVID_FILES),
    #wt2g
    "trec-wt2g" : RemoteDataset("trec-wt2g", TREC_WT2G_FILES),
    #.gov
    "trec-wt-2002" : RemoteDataset("trec-wt-2002", TREC_WT_2002_FILES),
    "trec-wt-2003" : RemoteDataset("trec-wt-2003", TREC_WT_2002_FILES),
    "trec-wt-2004" : RemoteDataset("trec-wt-2004", TREC_WT_2004_FILES),
    #.clueweb09
    "trec-wt-2009" : RemoteDataset("trec-wt-2009", TREC_WT_2009_FILES),
    "trec-wt-2010" : RemoteDataset("trec-wt-2010", TREC_WT_2010_FILES),
    "trec-wt-2011" : RemoteDataset("trec-wt-2011", TREC_WT_2011_FILES),
    "trec-wt-2012" : RemoteDataset("trec-wt-2012", TREC_WT_2012_FILES),
}

def get_dataset(name, **kwargs):
    '''
        Get a dataset by name
    '''
    rtr = DATASET_MAP[name]
    rtr._configure(**kwargs)
    return rtr

def datasets():
    '''
        Lists all the names of the datasets
    '''
    return DATASET_MAP.keys()

def list_datasets():
    '''
        Returns a dataframe of all datasets, listing which topics, qrels, corpus files or indices are available
    '''
    import pandas as pd
    rows=[]
    for k in datasets():
        dataset = get_dataset(k)
        rows.append([
            k, 
            dataset._describe_component("topics"), 
            dataset._describe_component("qrels"), 
            dataset._describe_component("corpus"), 
            dataset._describe_component("index") ])
    return pd.DataFrame(rows, columns=["dataset", "topics", "qrels", "corpus", "index"])