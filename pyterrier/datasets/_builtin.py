import pandas as pd
from typing import Iterable, Dict
import requests
import pyterrier as pt


TERRIER_DATA_BASE="http://data.terrier.org/indices/"
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


def passage_generate(dataset):
    for filename in dataset.get_corpus():
        with pt.io.autoopen(filename, 'rt') as corpusfile:
            for line in corpusfile: #for each line
                docno, passage = line.split("\t")
                yield {'docno' : docno, 'text' : passage}

def _datarepo_index(self, component, variant=None, version='latest', **kwargs):
    if variant is None:
        raise ValueError(f"Must specify index variant for {self.name}. See http://data.terrier.org/{self.name}.dataset.html")
    urlprefix= f"http://data.terrier.org/indices/{self.name}/{variant}/{version}/"
    url = urlprefix + "files"
    try:
        r = requests.get(url, **kwargs)
        r.raise_for_status()
        file = r.text.splitlines()
    except Exception as e:
        raise ValueError(f"Could not find index variant {variant} for dataset {self.name} at {url}. See available variants at http://data.terrier.org/{self.name}.dataset.html") from e
    rtr = []
    import re
    for linenum, line in enumerate(file):
        # skip comments
        if line.startswith("#"):
            continue
        try:
            (length, filename) = re.split(r"\s+", line.strip(), maxsplit=2)
            rtr.append((filename, urlprefix+filename, int(length)))
        except Exception as e:
            raise ValueError(f"Could not parse {url} line {linenum} '{line}'") from e
    return rtr
    
def _datarepo_index_default_none(self, component, variant=None, version='latest', **kwargs):
    """
    For backward compatability with vaswani - use default for variant 
    """
    if variant is None:
        variant = 'terrier_stemmed'
    return _datarepo_index(self, component, variant=variant, version=version, **kwargs)

ANTIQUE_FILES = {
    "topics" : {
        "train" : ("antique-train-queries.txt", "http://ciir.cs.umass.edu/downloads/Antique/antique-train-queries.txt", "singleline"),
        "test" : ("antique-test-queries.txt", "http://ciir.cs.umass.edu/downloads/Antique/antique-test-queries.txt", "singleline"),
    },
    "qrels" : {
        "train" : ("antique-train.qrel", "http://ciir.cs.umass.edu/downloads/Antique/antique-train.qrel", "singleline"),
        "test" : ("antique-test.qrel", "http://ciir.cs.umass.edu/downloads/Antique/antique-test.qrel", "singleline"),
    },
    "corpus" : 
        [("antique-collection.txt", "http://ciir.cs.umass.edu/downloads/Antique/antique-collection.txt")],
    "info_url" : "https://ciir.cs.umass.edu/downloads/Antique/readme.txt",
    "corpus_iter" : passage_generate
}

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
    "info_url"  : "https://ir.nist.gov/covidSubmit/",
    "index": _datarepo_index
}

def msmarco_document_generate(dataset):
    for filename in dataset.get_corpus(variant="corpus-tsv"):
        with pt.io.autoopen(filename, 'rt') as corpusfile:
            for line in corpusfile: #for each line
                docno, url, title, passage = line.split("\t")
                yield {'docno' : docno, 'url' : url, 'title' : title, 'text' : passage}

MSMARCO_DOC_FILES = {
    "corpus" : 
        [("msmarco-docs.trec.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.trec.gz")],
    "corpus-tsv":
        [("msmarco-docs.tsv.gz",  "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz")],
    "topics" : 
        { 
            "train" : ("msmarco-doctrain-queries.tsv.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz", "singleline"),
            "dev" : ("msmarco-docdev-queries.tsv.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz", "singleline"),
            "test" : ("msmarco-test2019-queries.tsv.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz", "singleline"),
            "test-2020" : ("msmarco-test2020-queries.tsv.gz" , "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz", "singleline"),
            'leaderboard-2020' : ("docleaderboard-queries.tsv.gz" , "https://msmarco.z22.web.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz", "singleline")
        },
    "qrels" : 
        { 
            "train" : ("msmarco-doctrain-qrels.tsv.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz"),
            "dev" : ("msmarco-docdev-qrels.tsv.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz"),
            "test" : ("2019qrels-docs.txt", "https://trec.nist.gov/data/deep/2019qrels-docs.txt"),
            "test-2020" : ("2020qrels-docs.txt", "https://trec.nist.gov/data/deep/2020qrels-docs.txt")
        },
    "info_url" : "https://microsoft.github.io/msmarco/",
    "corpus_iter" : msmarco_document_generate,
    "index" : _datarepo_index
}

MSMARCO_PASSAGE_FILES = {
    "corpus" : 
        [("collection.tsv", "collection.tar.gz#collection.tsv")],
    "topics" :
        { 
            "train" : ("queries.train.tsv", "queries.tar.gz#queries.train.tsv", "singleline"),
            "dev" : ("queries.dev.tsv", "queries.tar.gz#queries.dev.tsv", "singleline"),
            "dev.small" : ("queries.dev.small.tsv", "collectionandqueries.tar.gz#queries.dev.small.tsv", "singleline"),
            "eval" : ("queries.eval.tsv", "queries.tar.gz#queries.eval.tsv", "singleline"),
            "eval.small" : ("queries.eval.small.tsv", "collectionandqueries.tar.gz#queries.eval.small.tsv", "singleline"),
            "test-2019" : ("msmarco-test2019-queries.tsv.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz", "singleline"),
            "test-2020" : ("msmarco-test2020-queries.tsv.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz", "singleline")
        },        
    "tars" : {
        "queries.tar.gz" : ("queries.tar.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"),
        "collection.tar.gz" : ("collection.tar.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"),
        "collectionandqueries.tar.gz" : ("collectionandqueries.tar.gz", "https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz")
    },
    "qrels" : 
        { 
            "train" : ("qrels.train.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv"),
            "dev" : ("qrels.dev.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv"),
            "test-2019" : ("2019qrels-docs.txt", "https://trec.nist.gov/data/deep/2019qrels-pass.txt"),
            "test-2020" : ("2020qrels-docs.txt", "https://trec.nist.gov/data/deep/2020qrels-pass.txt"),
            "dev.small" : ("qrels.dev.small.tsv", "collectionandqueries.tar.gz#qrels.dev.small.tsv"),
        },
    "info_url" : "https://microsoft.github.io/MSMARCO-Passage-Ranking/",
    "corpus_iter" : passage_generate,
    "index" : _datarepo_index
}

MSMARCOv2_DOC_FILES = {
    "info_url" : "https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
    "topics" : {
        "train" : ("docv2_train_queries.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_queries.tsv", "singleline"),
        "dev1"  :("docv2_dev_queries.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_queries.tsv", "singleline"),
        "dev2"  :("docv2_dev2_queries.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_queries.tsv", "singleline"),
        "valid1" : ("msmarco-test2019-queries.tsv.gz" , "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz", "singleline"),
        "valid2" : ("msmarco-test2020-queries.tsv.gz" , "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz", "singleline"),
        "trec_2021" : ("2021_queries.tsv" , "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv", "singleline"),
    },
    "qrels" : {
        "train" : ("docv2_train_qrels.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_qrels.tsv"),
        "dev1"  :("docv2_dev_qrels.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_qrels.tsv"),
        "dev2"  :("docv2_dev2_qrels.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_qrels.tsv"),
        "valid1" : ("docv2_trec2019_qrels.txt.gz" , "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_trec2019_qrels.txt.gz"),
        "valid2" : ("docv2_trec2020_qrels.txt.gz" , "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_trec2020_qrels.txt.gz")
    },
    "index" : _datarepo_index,
}

MSMARCOv2_PASSAGE_FILES = {
    "info_url" : "https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
    "topics" : {
        "train" : ("passv2_train_queries.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_queries.tsv", "singleline"),
        "dev1"  : ("passv2_dev_queries.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_queries.tsv", "singleline"),
        "dev2"  : ("passv2_dev2_queries.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_queries.tsv", "singleline"),
        "trec_2021" : ("2021_queries.tsv" , "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv", "singleline"),
    },
    "qrels" : {
        "train" : ("passv2_train_qrels.tsv" "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_qrels.tsv"),
        "dev1"  : ("passv2_dev_qrels.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_qrels.tsv"),
        "dev2"  : ("passv2_dev2_qrels.tsv", "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_qrels.tsv"),
    },
    "index" : _datarepo_index,
}

# remove WT- prefix from topics
def remove_prefix(self, component, variant):
    topics_file, type = self._get_one_file("topics_prefixed", variant)
    if type in pt.io.SUPPORTED_TOPICS_FORMATS:
        topics = pt.io.read_topics(topics_file, type)
    else:
        raise ValueError("Unknown topic type %s" % type)
    topics["qid"] = topics.apply(lambda row: row["qid"].split("-")[1], axis=1)
    return (topics, "direct")


# a function to fix the namedpage TREC Web tracks 2001 and 2002
def parse_desc_only(self, component, variant):
    file, type = self._get_one_file("topics_desc_only", variant=variant)
    topics = pt.io.read_topics(file, format="trec", whitelist=["DESC"], blacklist=None)
    topics["qid"] = topics.apply(lambda row: row["qid"].replace("NP", ""), axis=1)
    topics["qid"] = topics.apply(lambda row: row["qid"].replace("EP", ""), axis=1)
    return (topics, "direct")

TREC_WT_2002_FILES = {
    "topics" : 
        { 
            "td" : ("webtopics_551-600.txt.gz", "https://trec.nist.gov/data/topics_eng/webtopics_551-600.txt.gz", "trec"),
            "np" : parse_desc_only
        },
    "topics_desc_only" : {
        "np" : ("webnamed_page_topics.1-150.txt.gz", "https://trec.nist.gov/data/topics_eng/webnamed_page_topics.1-150.txt.gz", "trec")
    },
    "qrels" : 
        { 
            "np" : ("qrels.named-page.txt.gz", "https://trec.nist.gov/data/qrels_eng/qrels.named-page.txt.gz"),
            "td" : ("qrels.distillation.txt.gz", "https://trec.nist.gov/data/qrels_eng/qrels.distillation.txt.gz")
        },
    "info_url" : "https://trec.nist.gov/data/t11.web.html",
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
        },
    "info_url" : "https://trec.nist.gov/data/t12.web.html",
}

def irds_mirror(md5):
    return f'http://mirror.ir-datasets.com/{md5}'

def filter_on_qid_type(self, component, variant):
    if component == "topics":
        data = self.get_topics("all")
    elif component == "qrels":
        data = self.get_qrels("all")
    qid2type_file = self._get_one_file("topics_map")[0]
    qid2type = pd.read_csv(qid2type_file, names=["qid", "type"], sep=" ")
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
    "topics_map" : [("04.topic-map.official.txt", [
        "https://trec.nist.gov/data/web/04.topic-map.official.txt",
        irds_mirror("79737768b3be1aa07b14691aa54802c5"),
        "https://www.dcs.gla.ac.uk/~craigm/04.topic-map.official.txt"
        ] )],
    "topics_prefixed" : { 
        "all" : ("Web2004.query.stream.trecformat.txt", [
                "https://trec.nist.gov/data/web/Web2004.query.stream.trecformat.txt",
                irds_mirror("10821f7a000b8bec058097ede39570be"),
                "https://www.dcs.gla.ac.uk/~craigm/Web2004.query.stream.trecformat.txt"], 
            "trec")
    },
    "qrels" : 
        {
            "hp" : filter_on_qid_type,
            "td" : filter_on_qid_type,
            "np" : filter_on_qid_type,
            "all" : ("04.qrels.web.mixed.txt", [
                "https://trec.nist.gov/data/web/04.qrels.web.mixed.txt",
                irds_mirror("93daa0e4b4190c84e30d2cce78a0f674"),
                "https://www.dcs.gla.ac.uk/~craigm/04.qrels.web.mixed.txt"])
        },
    "info_url" : "https://trec.nist.gov/data/t13.web.html",
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
    },
    "info_url" : "https://trec.nist.gov/data/web09.html",
}

TREC_WT_2010_FILES = {
     "topics" : [  
            ("wt2010-topics.queries-only", "https://trec.nist.gov/data/web/10/wt2010-topics.queries-only", "singleline")
        ],
    "qrels" : 
        { 
            "adhoc" : ("qrels.adhoc", "https://trec.nist.gov/data/web/10/10.adhoc-qrels.final")
        },
    "info_url" : "https://trec.nist.gov/data/web10.html",
}

TREC_WT_2011_FILES = {
     "topics" : [  
            ("queries.101-150.txt", "https://trec.nist.gov/data/web/11/queries.101-150.txt", "singleline")
        ],
    "qrels" : 
        { 
            "adhoc" : ("qrels.adhoc", "https://trec.nist.gov/data/web/11/qrels.adhoc")
        },
    "info_url" : "https://trec.nist.gov/data/web2011.html",
}

TREC_WT_2012_FILES = {
     "topics" : [  
            ("queries.151-200.txt", "https://trec.nist.gov/data/web/12/queries.151-200.txt", "singleline")
        ],
    "qrels" : 
        { 
            "adhoc" : ("qrels.adhoc", "https://trec.nist.gov/data/web/12/qrels.adhoc")
        },
    "info_url" : "https://trec.nist.gov/data/web2012.html",
}

TREC_WT2G_FILES = {
    "qrels" : [ ("qrels.trec8.small_web.gz", "https://trec.nist.gov/data/qrels_eng/qrels.trec8.small_web.gz") ],
    "topics" : [ (  "topics.401-450.gz", "https://trec.nist.gov/data/topics_eng/topics.401-450.gz" ) ],
    "info_url" : "https://trec.nist.gov/data/t8.web.html",
}

TREC_WT10G_FILES = {
    "qrels" : {
        "trec9" : ("qrels.trec9.main_web.gz", "https://trec.nist.gov/data/qrels_eng/qrels.trec9.main_web.gz"),
        "trec10-adhoc" : ("qrels.trec10.main_web.gz", "https://trec.nist.gov/data/qrels_eng/qrels.trec10.main_web.gz"),
        "trec10-hp" : ("qrels.trec10.entrypage.gz", "https://trec.nist.gov/data/qrels_eng/qrels.trec10.entrypage.gz"),
    },
    "topics" : {
        "trec9" : (  "topics.451-500.gz", "https://trec.nist.gov/data/topics_eng/topics.451-500.gz" ),
        "trec10-adhoc" : (  "topics.501-550.txt", "https://trec.nist.gov/data/topics_eng/topics.501-550.txt" ),
         "trec10-hp" : parse_desc_only
    },
    "topics_desc_only" : {
         "trec10-hp" : (  "entry_page_topics.1-145.txt", "https://trec.nist.gov/data/topics_eng/entry_page_topics.1-145.txt" ),
    },
    "info_url" : "https://trec.nist.gov/data/t9.web.html",
}

def _merge_years(self, component, variant):
    MAP_METHOD = { 
        "topics" : pt.datasets.RemoteDataset.get_topics,
        "qrels" : pt.datasets.RemoteDataset.get_qrels,  
    }
    dfs = []
    low, hi = variant.split("-")
    for y in range(int(low), int(hi)+1):
        df = MAP_METHOD[component](self, variant=str(y))
        dfs.append(df)
    return (pd.concat(dfs), "direct")

TREC_TB_FILES = {
    "topics" : {
        "2004" : ( "04topics.701-750.txt", "https://trec.nist.gov/data/terabyte/04/04topics.701-750.txt" ),
        "2005" : ( "04topics.701-750.txt", "https://trec.nist.gov/data/terabyte/05/05.topics.751-800.txt" ),
        "2006" : ( "06.topics.801-850.txt", "https://trec.nist.gov/data/terabyte/06/06.topics.801-850.txt" ),
        "2004-2006" : ("06.topics.701-850.txt", "https://trec.nist.gov/data/terabyte/06/06.topics.701-850.txt"),

        "2006-np" : ( "06.np_topics.901-1081.txt", "https://trec.nist.gov/data/terabyte/06/06.np_topics.901-1081.txt" ),
        "2005-np" : ( "05.np_topics.601-872.final.txt", "https://trec.nist.gov/data/terabyte/05/05.np_topics.601-872.final.txt")
    },

    "qrels" : {
        "2004" : ( "04.qrels.12-Nov-04", "https://trec.nist.gov/data/terabyte/04/04.qrels.12-Nov-04"),
        "2005" : ( "05.adhoc_qrels", "https://trec.nist.gov/data/terabyte/05/05.adhoc_qrels"),
        "2006" : ( "qrels.tb06.top50", "https://trec.nist.gov/data/terabyte/06/qrels.tb06.top50"),
        "2004-2006" : _merge_years,

        "2005-np" : ( "05.np_qrels", "https://trec.nist.gov/data/terabyte/05/05.np_qrels"),
        "2006-np" : ( "qrels.tb06.np", "https://trec.nist.gov/data/terabyte/06/qrels.tb06.np"),
    },
    "info_url" : "https://trec.nist.gov/data/terabyte.html"
}

TREC_ROBUST_04_FILES = {
    "qrels" : [ ("qrels.robust2004.txt", "https://trec.nist.gov/data/robust/qrels.robust2004.txt") ],
    "topics" : [ (  "04.testset.gz", "https://trec.nist.gov/data/robust/04.testset.gz" ) ],
    "info_url" : "https://trec.nist.gov/data/t13_robust.html",
}
TREC_ROBUST_05_FILES = {
    "qrels" : [ ("TREC2005.qrels.txt", "https://trec.nist.gov/data/robust/05/TREC2005.qrels.txt") ],
    "topics" : [ (  "05.50.topics.txt", "https://trec.nist.gov/data/robust/05/05.50.topics.txt" ) ],
    "info_url" : "https://trec.nist.gov/data/t14_robust.html",
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
    },
    "info_url" : "https://trec.nist.gov/data/precmed.html",
}



VASWANI_CORPUS_BASE = "https://raw.githubusercontent.com/terrier-org/pyterrier/master/tests/fixtures/vaswani_npl/"
VASWANI_INDEX_BASE = "https://raw.githubusercontent.com/terrier-org/pyterrier/master/tests/fixtures/index/"
VASWANI_FILES = {
    "corpus": [("doc-text.trec", [
        VASWANI_CORPUS_BASE + "corpus/doc-text.trec",
        irds_mirror("a059e713c50350e39999467c8c73b7c5")])],
    "topics": [("query-text.trec", [
        VASWANI_CORPUS_BASE + "query-text.trec",
        irds_mirror("3a624be2b0ef7c9534cf848891679bec")])],
    "qrels": [("qrels", [
        VASWANI_CORPUS_BASE + "qrels",
        irds_mirror("6acb6db9969da8b8c6c23c09551af8d9")])],
    "index": _datarepo_index_default_none,
    #"index":
    #    [(filename, VASWANI_INDEX_BASE + filename) for filename in STANDARD_TERRIER_INDEX_FILES + ["data.meta-0.fsomapfile"]],
    "info_url" : "http://ir.dcs.gla.ac.uk/resources/test_collections/npl/",
    "corpus_iter" : lambda dataset, **kwargs : pt.index.treccollection2textgen(dataset.get_corpus(), num_docs=11429, verbose=kwargs.get("verbose", False))
}

DATASET_MAP : Dict[str, pt.datasets.Dataset] = {
    # umass antique corpus - see http://ciir.cs.umass.edu/downloads/Antique/ 
    "antique" : pt.datasets.RemoteDataset("antique", ANTIQUE_FILES),
    # generated from http://ir.dcs.gla.ac.uk/resources/test_collections/npl/
    "vaswani": pt.datasets.RemoteDataset("vaswani", VASWANI_FILES),
    "msmarco_document" : pt.datasets.RemoteDataset("msmarco_document", MSMARCO_DOC_FILES),
    "msmarcov2_document" : pt.datasets.RemoteDataset("msmarcov2_document", MSMARCOv2_DOC_FILES),
    "msmarco_passage" : pt.datasets.RemoteDataset("msmarco_passage", MSMARCO_PASSAGE_FILES),
    "msmarcov2_passage" : pt.datasets.RemoteDataset("msmarcov2_passage", MSMARCOv2_PASSAGE_FILES),
    "trec-robust-2004" : pt.datasets.RemoteDataset("trec-robust-2004", TREC_ROBUST_04_FILES),
    "trec-robust-2005" : pt.datasets.RemoteDataset("trec-robust-2005", TREC_ROBUST_05_FILES),
    "trec-terabyte" : pt.datasets.RemoteDataset("trec-terabyte", TREC_TB_FILES),
    #medical-like tracks
    "trec-precision-medicine" : pt.datasets.RemoteDataset("trec-precicion-medicine", TREC_PRECISION_MEDICINE_FILES),
    "trec-covid" : pt.datasets.RemoteDataset("trec-covid", TREC_COVID_FILES),
    #wt2g
    "trec-wt2g" : pt.datasets.RemoteDataset("trec-wt2g", TREC_WT2G_FILES),
    #wt10g
    "trec-wt10g" : pt.datasets.RemoteDataset("trec-wt10g", TREC_WT10G_FILES),
    #.gov
    "trec-wt-2002" : pt.datasets.RemoteDataset("trec-wt-2002", TREC_WT_2002_FILES),
    "trec-wt-2003" : pt.datasets.RemoteDataset("trec-wt-2003", TREC_WT_2002_FILES),
    "trec-wt-2004" : pt.datasets.RemoteDataset("trec-wt-2004", TREC_WT_2004_FILES),
    #clueweb09
    "trec-wt-2009" : pt.datasets.RemoteDataset("trec-wt-2009", TREC_WT_2009_FILES),
    "trec-wt-2010" : pt.datasets.RemoteDataset("trec-wt-2010", TREC_WT_2010_FILES),
    "trec-wt-2011" : pt.datasets.RemoteDataset("trec-wt-2011", TREC_WT_2011_FILES),
    "trec-wt-2012" : pt.datasets.RemoteDataset("trec-wt-2012", TREC_WT_2012_FILES),
}

# "trec-deep-learning-docs"
#DATASET_MAP['msmarco_document'] = DATASET_MAP["trec-deep-learning-docs"]
#DATASET_MAP['msmarco_passage'] = DATASET_MAP["trec-deep-learning-passages"]
DATASET_MAP["trec-deep-learning-docs"] = DATASET_MAP['msmarco_document']
DATASET_MAP["trec-deep-learning-passages"] = DATASET_MAP['msmarco_passage']


class BuiltinDatasetProvider(pt.datasets.DatasetProvider):
    def get_dataset(self, name: str) -> pt.datasets.Dataset:
        return DATASET_MAP[name]

    def list_dataset_names(self) -> Iterable[str]:
        return list(DATASET_MAP.keys())