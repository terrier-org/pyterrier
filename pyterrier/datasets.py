import urllib.request
import wget
import os
import pandas as pd
from .transformer import is_lambda
import types
import requests
from .io import autoopen
from . import tqdm, HOME_DIR
import tarfile

import pyterrier

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

class GeneratorLen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self): 
        return self.length

    def __iter__(self):
        return self.gen

class Dataset():
    """
        Represents a dataset (test collection) for indexing or retrieval. A common use-case is to use the Dataset within an Experiment::

            dataset = pt.get_dataset("trec-robust-2004")
            pt.Experiment([br1, br2], dataset.get_topics(), dataset.get_qrels(), eval_metrics=["map", "recip_rank"])

    """

    def _configure(self, **kwargs):
        pass

    def get_corpus(self):
        """ 
            Returns the location of the files to allow indexing the corpus, i.e. it returns a list of filenames.
        """
        pass

    def get_corpus_iter(self, verbose=True):
        """
            Returns an iter of dicts for this collection. If verbose=True, a tqdm pbar shows the progress over this iterator.
        """
        pass

    def get_corpus_lang(self):
        """
            Returns the ISO 639-1 language code for the corpus, or None for multiple/other/unknown
        """
        return None

    def get_index(self, variant=None):
        """ 
            Returns the IndexRef of the index to allow retrieval. Only a few datasets provide indices ready made.
        """
        pass

    def get_topics(self, variant=None):
        """
            Returns the topics, as a dataframe, ready for retrieval. 
        """
        pass

    def get_topics_lang(self):
        """
            Returns the ISO 639-1 language code for the topics, or None for multiple/other/unknown
        """
        return None

    def get_qrels(self, variant=None):
        """ 
            Returns the qrels, as a dataframe, ready for evaluation.
        """
        pass

    def info_url(self):
        """
            Returns a url that provides more information about this dataset.
        """
        return None

class RemoteDataset(Dataset):

    def __init__(self, name, locations):
        self.locations = locations
        self.name = name
        self.user = None
        self.password = None

    def _configure(self, **kwargs):
        from os.path import expanduser
        pt_home = HOME_DIR
        if pt_home is None:
            from os.path import expanduser
            userhome = expanduser("~")
            pt_home = os.path.join(userhome, ".pyterrier")
        self.corpus_home = os.path.join(pt_home, "corpora", self.name)
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
        if "#" in URL:
            tarname, intarfile = URL.split("#")
            assert not "/" in intarfile
            assert ".tar" in tarname or ".tgz" in tarname
            localtarfile, _ = self._get_one_file("tars", tarname)
            tarobj = tarfile.open(localtarfile, "r")
            tarobj.extract(intarfile, path=self.corpus_home)
            local = os.path.join(self.corpus_home, local)
            os.rename(os.path.join(self.corpus_home, intarfile), local)
            return (local, filetype)

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

    def _get_all_files(self, component, variant=None, **kwargs):
        localDir = os.path.join(self.corpus_home, component)
        if not os.path.exists(localDir):
            os.makedirs(localDir)
            print("Downloading %s %s to %s" % (self.name, component, localDir))
        kwargs = {}
        if self.user is not None:
            kwargs["auth"]=(self.user, self.password)
        file_list = self.locations[component] if variant is None else self.locations[component][variant]
        for (local, URL) in file_list:
            local = os.path.join(localDir, local)
            if not os.path.exists(local):
                if "#" in URL:
                    tarname, intarfile = URL.split("#")
                    assert not "/" in intarfile
                    assert ".tar" in tarname or ".tgz" in tarname
                    localtarfile, _ = self._get_one_file("tars", tarname)
                    tarobj = tarfile.open(localtarfile, "r")
                    tarobj.extract(intarfile, path=self.corpus_home)
                    local = os.path.join(self.corpus_home, local)
                    #TODO, files could be recompressed here to save space
                    os.rename(os.path.join(self.corpus_home, intarfile), local)
                else:
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
        return list(self.locations[component].keys())

    def get_corpus(self, **kwargs):
        import pyterrier as pt
        return pt.io.find_files(self._get_all_files("corpus", **kwargs))

    def get_corpus_iter(self, **kwargs):
        if not "corpus_iter" in self.locations:
            raise ValueError("Cannot supply a corpus iterator on dataset %s" % self.name)
        return self.locations["corpus_iter"](self, **kwargs)
        
    def get_corpus_lang(self):
        if 'corpus' in self.locations:
            return 'en' # all are english
        return None

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
    
    def get_topics_lang(self):
        if 'topics' in self.locations:
            return 'en' # all are english
        return None

    def get_index(self, variant=None):
        import pyterrier as pt
        if self.name == "50pct" and variant is None:
            variant="ex1"
        thedir = self._get_all_files("index", variant=variant)
        return pt.autoclass("org.terrier.querying.IndexRef").of(os.path.join(thedir, "data.properties"))

    def __repr__(self):
        return "RemoteDataset for %s, with %s" % (self.name, str(list(self.locations.keys())))

    def info_url(self):
        return self.locations['info_url'] if "info_url" in self.locations else None


class IRDSDataset(Dataset):
    def __init__(self, irds_id):
        self._irds_id = irds_id
        self._irds_ref = None

    def irds_ref(self):
        if self._irds_ref is None:
            self._irds_ref = ir_datasets.load(self._irds_id)
        return self._irds_ref

    def get_corpus(self):
        raise NotImplementedError("IRDSDataset doesn't support get_corpus; use get_corpus_iter instead")

    def get_corpus_iter(self, verbose=True):
        ds = self.irds_ref()
        assert ds.has_docs(), f"{self._irds_id} doesn't support get_corpus_iter"
        it = ds.docs_iter()
        if verbose:
            it = tqdm(it, desc=f'{self._irds_id} documents', total=ds.docs_count())
        def gen():
            for doc in it:
                doc = doc._asdict()
                # pyterrier uses "docno"
                doc['docno'] = doc.pop('doc_id')
                yield doc
        return GeneratorLen(gen(), ds.docs_count())

    def get_corpus_lang(self):
        ds = self.irds_ref()
        if ds.has_docs():
            return ds.docs_lang()
        return None

    def get_index(self, variant=None):
        # this is only for indices where Terrier provides an index already
        raise NotImplementedError("IRDSDataset doesn't support get_index")

    def get_topics(self, variant=None, tokenise_query=True):
        """
            Returns the topics, as a dataframe, ready for retrieval. 
        """
        ds = self.irds_ref()
        assert ds.has_queries(), f"{self._irds_id} doesn't support get_topics"
        qcls = ds.queries_cls()
        assert variant is None or variant in qcls._fields[1:], f"{self._irds_id} only supports the following topic variants {qcls._fields[1:]}"
        df = pd.DataFrame(ds.queries_iter())

        df.rename(columns={"query_id": "qid"}, inplace=True) # pyterrier uses "qid"

        if variant is not None:
            df.rename(columns={variant: "query"}, inplace=True) # user specified which version of the query they want
            df.drop(df.columns.difference(['qid','query']), 1, inplace=True)
        elif len(qcls._fields) == 2:
            # auto-rename single query field to "query" if there's only query_id and that field
            df.rename(columns={qcls._fields[1]: "query"}, inplace=True)
        else:
            print(f'There are multiple query fields available: {qcls._fields[1:]}. To use with pyterrier, provide variant or modify dataframe to add query column.')

        # apply pyterrier tokenisation (otherwise the queries may not play well with batchretrieve)
        if tokenise_query and 'query' in df:
            import pyterrier as pt
            tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
            def pt_tokenise(text):
                return ' '.join(tokeniser.getTokens(text))
            df['query'] = df['query'].apply(pt_tokenise)

        return df

    def get_topics_lang(self):
        ds = self.irds_ref()
        if ds.has_queries():
            return ds.queries_lang()
        return None

    def get_qrels(self, variant=None):
        """ 
            Returns the qrels, as a dataframe, ready for evaluation.
        """
        ds = self.irds_ref()
        assert ds.has_qrels(), f"{self._irds_id} doesn't support get_qrels"
        qrelcls = ds.qrels_cls()
        qrel_fields = [f for f in qrelcls._fields if f not in ('query_id', 'doc_id', 'iteration')]
        assert variant is None or variant in qrel_fields, f"{self._irds_id} only supports the following qrel variants {qrel_fields}"
        df = pd.DataFrame(ds.qrels_iter())

        # pyterrier uses "qid" and "docno"
        df.rename(columns={
            "query_id": "qid",
            "doc_id": "docno"}, inplace=True)

        # pyterrier uses "label"
        if variant is not None:
            df.rename(columns={variant: "label"}, inplace=True)
        if len(qrel_fields) == 1:
            # usually "relevance"
            df.rename(columns={qrel_fields[0]: "label"}, inplace=True)
        elif 'relevance' in qrel_fields:
            print(f'There are multiple qrel fields available: {qrel_fields}. Defaulting to "relevance", but to use a different one, supply variant')
            df.rename(columns={'relevance': "label"}, inplace=True)
        else:
            print(f'There are multiple qrel fields available: {qrel_fields}. To use with pyterrier, provide variant or modify dataframe to add query column.')

        return df

    def _describe_component(self, component):
        ds = self.irds_ref()
        if component == "topics":
            if ds.has_queries():
                fields = ds.queries_cls()._fields[1:]
                if len(fields) > 1:
                    return list(fields)
                return True
            return None
        if component == "qrels":
            if ds.has_qrels():
                fields = [f for f in ds.qrels_cls()._fields if f not in ('query_id', 'doc_id', 'iteration')]
                if len(fields) > 1:
                    return list(fields)
                return True
            return None
        if component == "corpus":
            return ds.has_docs() or None
        return None

    def info_url(self):
        top_id = self._irds_id.split('/', 1)[0]
        suffix = f'#{self._irds_id}' if top_id != self._irds_id else ''
        return f'https://ir-datasets.com/{top_id}.html{suffix}'

    def __repr__(self):
        return f"IRDSDataset({repr(self._irds_id)})"


def passage_generate(dataset):
    for filename in dataset.get_corpus():
        with autoopen(filename, 'rt') as corpusfile:
            for l in corpusfile: #for each line
                docno, passage = l.split("\t")
                yield {'docno' : docno, 'text' : passage}


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
    "info_url"  : "https://ir.nist.gov/covidSubmit/"
}

def msmarco_document_generate(dataset):
    for filename in dataset.get_corpus(variant="corpus-tsv"):
        with autoopen(filename, 'rt') as corpusfile:
            for l in corpusfile: #for each line
                docno, url, title, passage = l.split("\t")
                yield {'docno' : docno, 'url' : url, 'title' : title, 'text' : passage}

TREC_DEEPLEARNING_DOCS_MSMARCO_FILES = {
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
            "test" : ("2019qrels-docs.txt", "https://trec.nist.gov/data/deep/2019qrels-docs.txt"),
            "test-2020" : ("2020qrels-docs.txt", "https://trec.nist.gov/data/deep/2020qrels-docs.txt")
        },
    "info_url" : "https://microsoft.github.io/msmarco/",
    "corpus_iter" : msmarco_document_generate
}

TREC_DEEPLEARNING_PASSAGE_MSMARCO_FILES = {
    "corpus" : 
        [("collection.tsv", "collection.tar.gz#collection.tsv")],
    "topics" :
        { 
            "train" : ("queries.train.tsv", "queries.tar.gz#queries.train.tsv", "singleline"),
            "dev" : ("queries.dev.tsv", "queries.tar.gz#queries.dev.tsv", "singleline"),
            "eval" : ("queries.eval.tsv", "queries.tar.gz#queries.eval.tsv", "singleline"),
            "test-2019" : ("msmarco-test2019-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz", "singleline"),
            "test-2020" : ("msmarco-test2020-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz", "singleline")
        },        
    "tars" : {
        "queries.tar.gz" : ("queries.tar.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz"),
        "collection.tar.gz" : ("collection.tar.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz")
    },
    "qrels" : 
        { 
            "train" : ("qrels.train.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv"),
            "dev" : ("qrels.dev.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv"),
            "test-2019" : ("2019qrels-docs.txt", "https://trec.nist.gov/data/deep/2019qrels-pass.txt"),
            "test-2020" : ("2020qrels-docs.txt", "https://trec.nist.gov/data/deep/2020qrels-pass.txt")
        },
    "info_url" : "https://microsoft.github.io/MSMARCO-Passage-Ranking/",
    "corpus_iter" : passage_generate
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


# a function to fix the namedpage TREC Web tracks 2001 and 2002
def parse_desc_only(self, component, variant):
    import pyterrier as pt
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
        },
    "info_url" : "https://trec.nist.gov/data/t13.web.html",
}

FIFTY_PCT_INDEX_BASE = "http://www.dcs.gla.ac.uk/~craigm/IR_HM/"
FIFTY_PCT_FILES = {
    "index": {
        "ex1" : [(filename, FIFTY_PCT_INDEX_BASE + "index/" + filename) for filename in ["data.meta-0.fsomapfile"] + STANDARD_TERRIER_INDEX_FILES],
        "ex2" : [(filename, FIFTY_PCT_INDEX_BASE + "index_block_fields_2021_content/" + filename) for filename in ["data.meta-0.fsomapfile", "data-pagerank.oos"] + STANDARD_TERRIER_INDEX_FILES],   
    },
    "topics": { 
            "training" : ("training.topics", FIFTY_PCT_INDEX_BASE + "topics/" + "training.topics", "trec"),
            "validation" : ("validation.topics", FIFTY_PCT_INDEX_BASE + "topics/" + "validation.topics", "trec"),
    },
    "qrels": { 
            "training" : ("training.qrels", FIFTY_PCT_INDEX_BASE + "topics/" + "training.qrels", "trec"),
            "validation" : ("validation.qrels", FIFTY_PCT_INDEX_BASE + "topics/" + "validation.qrels", "trec"),
    }    
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
        "topics" : RemoteDataset.get_topics,
        "qrels" : RemoteDataset.get_qrels,  
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
    "corpus":
        [("doc-text.trec", VASWANI_CORPUS_BASE + "corpus/doc-text.trec")],
    "topics":
        [("query-text.trec", VASWANI_CORPUS_BASE + "query-text.trec")],
    "qrels":
        [("qrels", VASWANI_CORPUS_BASE + "qrels")],
    "index":
        [(filename, VASWANI_INDEX_BASE + filename) for filename in STANDARD_TERRIER_INDEX_FILES + ["data.meta-0.fsomapfile"]],
    "info_url" : "http://ir.dcs.gla.ac.uk/resources/test_collections/npl/",
    "corpus_iter" : lambda dataset, **kwargs : pyterrier.index.treccollection2textgen(dataset.get_corpus(), num_docs=11429, verbose=kwargs.get("verbose", False))
}

DATASET_MAP = {
    # used for UGlasgow teaching
    "50pct" : RemoteDataset("50pct", FIFTY_PCT_FILES),
    # umass antique corpus - see http://ciir.cs.umass.edu/downloads/Antique/ 
    "antique" : RemoteDataset("antique", ANTIQUE_FILES),
    # generated from http://ir.dcs.gla.ac.uk/resources/test_collections/npl/
    "vaswani": RemoteDataset("vaswani", VASWANI_FILES),
    "trec-deep-learning-docs" : RemoteDataset("trec-deep-learning-docs", TREC_DEEPLEARNING_DOCS_MSMARCO_FILES),
    "trec-deep-learning-passages" : RemoteDataset("trec-deep-learning-passages", TREC_DEEPLEARNING_PASSAGE_MSMARCO_FILES),
    "trec-robust-2004" : RemoteDataset("trec-robust-2004", TREC_ROBUST_04_FILES),
    "trec-robust-2005" : RemoteDataset("trec-robust-2005", TREC_ROBUST_05_FILES),
    "trec-terabyte" : RemoteDataset("trec-terabyte", TREC_TB_FILES),
    #medical-like tracks
    "trec-precision-medicine" : RemoteDataset("trec-precicion-medicine", TREC_PRECISION_MEDICINE_FILES),
    "trec-covid" : RemoteDataset("trec-covid", TREC_COVID_FILES),
    #wt2g
    "trec-wt2g" : RemoteDataset("trec-wt2g", TREC_WT2G_FILES),
    #wt10g
    "trec-wt10g" : RemoteDataset("trec-wt10g", TREC_WT10G_FILES),
    #.gov
    "trec-wt-2002" : RemoteDataset("trec-wt-2002", TREC_WT_2002_FILES),
    "trec-wt-2003" : RemoteDataset("trec-wt-2003", TREC_WT_2002_FILES),
    "trec-wt-2004" : RemoteDataset("trec-wt-2004", TREC_WT_2004_FILES),
    #clueweb09
    "trec-wt-2009" : RemoteDataset("trec-wt-2009", TREC_WT_2009_FILES),
    "trec-wt-2010" : RemoteDataset("trec-wt-2010", TREC_WT_2010_FILES),
    "trec-wt-2011" : RemoteDataset("trec-wt-2011", TREC_WT_2011_FILES),
    "trec-wt-2012" : RemoteDataset("trec-wt-2012", TREC_WT_2012_FILES),
}


# Include all datasets from ir_datasets with "irds:" prefix so they don't conflict with pt dataset names
# Results in records like:
# irds:antique
# irds:antique/test
# irds:antique/test/non-offensive
# irds:antique/train
# ...
import ir_datasets
for ds_id in ir_datasets.registry:
    DATASET_MAP[f'irds:{ds_id}'] = IRDSDataset(ds_id)


def get_dataset(name, **kwargs):
    """
        Get a dataset by name
    """
    rtr = DATASET_MAP[name]
    rtr._configure(**kwargs)
    return rtr

def datasets():
    """
        Lists all the names of the datasets
    """
    return DATASET_MAP.keys()

def list_datasets(en_only=True):
    """
        Returns a dataframe of all datasets, listing which topics, qrels, corpus files or indices are available.
        By default, filters to only datasets with both a corpus and topics in English.
    """
    import pandas as pd
    rows=[]
    for k in datasets():
        dataset = get_dataset(k)
        rows.append([
            k, 
            dataset._describe_component("topics"), 
            dataset.get_topics_lang(), 
            dataset._describe_component("qrels"), 
            dataset._describe_component("corpus"), 
            dataset.get_corpus_lang(), 
            dataset._describe_component("index"), 
            dataset.info_url() ])
    result = pd.DataFrame(rows, columns=["dataset", "topics", "topics_lang", "qrels", "corpus", "corpus_lang", "index", "info_url"])
    if en_only:
        topics_filter = (result['topics'].isnull()) | (result['topics_lang'] == 'en')
        corpus_filter = (result['corpus'].isnull()) | (result['corpus_lang'] == 'en')
        result = result[topics_filter & corpus_filter]
    return result
