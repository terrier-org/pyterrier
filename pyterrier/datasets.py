import urllib.request
import wget
import os
import pandas as pd
from .transformer import is_lambda
import types
from typing import Union, Tuple, Iterator, Dict, Any, List
from warnings import warn
import requests
from .io import autoopen, touch
from . import tqdm, HOME_DIR
import tarfile
from warnings import warn

import pyterrier

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

    def get_corpus_iter(self, verbose=True) -> Iterator[Dict[str,Any]]:
        """
            Returns an iter of dicts for this collection. If verbose=True, a tqdm pbar shows the progress over this iterator.
        """
        pass

    def get_corpus_lang(self) -> Union[str,None]:
        """
            Returns the ISO 639-1 language code for the corpus, or None for multiple/other/unknown
        """
        return None

    def get_index(self, variant=None, **kwargs):
        """ 
            Returns the IndexRef of the index to allow retrieval. Only a few datasets provide indices ready made.
        """
        pass

    def get_topics(self, variant=None) -> pd.DataFrame:
        """
            Returns the topics, as a dataframe, ready for retrieval. 
        """
        pass

    def get_topics_lang(self) -> Union[str,None]:
        """
            Returns the ISO 639-1 language code for the topics, or None for multiple/other/unknown
        """
        return None

    def get_qrels(self, variant=None) -> pd.DataFrame:
        """ 
            Returns the qrels, as a dataframe, ready for evaluation.
        """
        pass

    def get_topicsqrels(self, variant=None) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
            Returns both the topics and qrels in a tuple. This is useful for pt.Experiment().
        """
        return (
            self.get_topics(variant=variant),
            self.get_qrels(variant=variant)
        )

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
    def download(URLs : Union[str,List[str]], filename : str, **kwargs):
        import pyterrier as pt
        basename = os.path.basename(filename)

        if isinstance(URLs, str):
            URLs = [URLs]
        
        finalattempt=len(URLs)-1
        error = None
        for i, url in enumerate(URLs):            
            try:
                r = requests.get(url, allow_redirects=True, stream=True, **kwargs)
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with pt.io.finalized_open(filename, 'b') as file, tqdm(
                        desc=basename,
                        total=total,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in r.iter_content(chunk_size=1024):
                        size = file.write(data)
                        bar.update(size)
                    break
            except Exception as e:
                if error is not None:
                    e.__cause__ = error # chain errors to show all if fails
                error = e
                if i == finalattempt:
                    raise error
                else:
                    warn("Problem fetching %s, resorting to next mirror" % url)
            

    def _check_variant(self, component, variant=None):
        name=self.name
        if not component in self.locations:
            raise ValueError("No %s in dataset %s" % (component, name))
        if variant is None:
            if not isinstance(self.locations[component], list):
                raise ValueError("For %s in dataset %s, you must specify a variant. Available are: %s" % (component, name, str(list(self.locations[component].keys()))))
            location = self.locations[component][0]
        else:
            if isinstance(self.locations[component], list):
                raise ValueError("For %s in dataset %s, there are no variants, but you specified %s" % (component, name, variant))
            if not variant in self.locations[component]:
                raise ValueError("For %s in dataset %s, there is no variant %s. Available are: %s" % (component, name, variant, str(list(self.locations[component].keys()))))

    def _get_one_file(self, component, variant=None):
        filetype=None
        name=self.name
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
        actualURL = URL if isinstance(URL, str) else URL[0]
        if "#" in actualURL and not os.path.exists(local):
            tarname, intarfile = actualURL.split("#")
            assert not "/" in intarfile
            assert ".tar" in tarname or ".tgz" in tarname
            localtarfile, _ = self._get_one_file("tars", tarname)
            tarobj = tarfile.open(localtarfile, "r")
            tarobj.extract(intarfile, path=self.corpus_home)
            os.rename(os.path.join(self.corpus_home, intarfile), local)
            return (local, filetype)        
        
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
        if variant is None:
            localDir = os.path.join(self.corpus_home, component)
        else:
            localDir = os.path.join(self.corpus_home, component, variant)

        kwargs = {}
        if self.user is not None:
            kwargs["auth"]=(self.user, self.password)

        direxists = os.path.exists(localDir)
        
        location = self.locations[component]
        if is_lambda(location) or isinstance(location, types.FunctionType):
            # functions are expensive to call, normally another HTTP is needed.
            # just assume we have everthing we need if we have the local directory already
            # and it contains a .complete file.
            if direxists and os.path.exists(os.path.join(localDir, ".complete")):
                return localDir

            # call the function, and get the file list
            file_list = location(self, component, variant, **kwargs)
        else:
            file_list = self.locations[component] if variant is None else self.locations[component][variant]

        if not direxists:
            os.makedirs(localDir)
            print("Downloading %s %s to %s" % (self.name, component, localDir))
        

        # check for how much space is required and available space
        def _totalsize(file_list):
            total = -1
            for f in file_list:
                if len(f) > 2:
                    total += f[2]
            if total != -1:
                total += 1
            return total

        totalsize = _totalsize(file_list)
        if totalsize > 0:
            import shutil
            total, used, free = shutil.disk_usage(localDir)
            if free < totalsize:
                raise ValueError("Insufficient freedisk space at %s to download index" % localDir)
            if totalsize > 2 * 2**30:
                warn("Downloading index of > 2GB.")

        for fileentry in file_list:
            local = fileentry[0]
            URL = fileentry[1]
            expectedlength = -1
            if len(fileentry) == 3:
                expectedlength = fileentry[2]
            local = os.path.join(localDir, local)
            
            # if file exists and we know length, check if dowload is complete
            fileexists = os.path.exists(local)
            if fileexists and expectedlength >= 0:
                length = os.stat(local).st_size
                if expectedlength != length:
                    warn("Removing partial download of %s (expected %d bytes, found %d)" % (local, expectedlength, length ))
                    os.remove(local)
                    fileexists = False

            if not fileexists:
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

                    # verify file if exists
                    if expectedlength >= 0:
                        length = os.stat(local).st_size
                        if expectedlength != length:
                            raise ValueError("Failed download of %s to %s (expected %d bytes, found %d)" % (URL, local, expectedlength, length ))

        # finally, touch a file signifying that download has been completed
        touch(os.path.join(localDir, ".complete"))
        return localDir

    def _describe_component(self, component):
        if component not in self.locations:
            return None
        if type(self.locations[component]) == type([]):
            return True
        if isinstance(self.locations[component], dict):
            return list(self.locations[component].keys())
        return True

    def get_corpus(self, **kwargs):
        import pyterrier as pt
        return list(filter(lambda f : not f.endswith(".complete"), pt.io.find_files(self._get_all_files("corpus", **kwargs))))

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

    def get_index(self, variant=None, **kwargs):
        import pyterrier as pt
        if self.name == "50pct" and variant is None:
            variant="ex1"
        thedir = self._get_all_files("index", variant=variant, **kwargs)
        return thedir
        #return pt.autoclass("org.terrier.querying.IndexRef").of(os.path.join(thedir, "data.properties"))

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
        raise NotImplementedError("IRDSDataset doesn't support get_corpus; use get_corpus_iter instead. If you "
                                  "are indexing, get_corpus_iter should be used in conjunction with IterDictIndexer.")

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
            (length, filename) = re.split(r"\s+", line.strip(), 2)
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
        with autoopen(filename, 'rt') as corpusfile:
            for l in corpusfile: #for each line
                docno, url, title, passage = l.split("\t")
                yield {'docno' : docno, 'url' : url, 'title' : title, 'text' : passage}

MSMARCO_DOC_FILES = {
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
    "corpus_iter" : msmarco_document_generate,
    "index" : _datarepo_index
}

MSMARCO_PASSAGE_FILES = {
    "corpus" : 
        [("collection.tsv", "collection.tar.gz#collection.tsv")],
    "index": {
        "terrier_stemmed" : [(filename, TERRIER_DATA_BASE + "/msmarco_passage/terrier_stemmed/latest/" + filename) for filename in STANDARD_TERRIER_INDEX_FILES],
        "terrier_unstemmed" : [(filename, TERRIER_DATA_BASE + "/msmarco_passage/terrier_unstemmed/latest/" + filename) for filename in STANDARD_TERRIER_INDEX_FILES],
        "terrier_stemmed_text" : [(filename, TERRIER_DATA_BASE + "/msmarco_passage/terrier_stemmed_text/latest/" + filename) for filename in STANDARD_TERRIER_INDEX_FILES],
        "terrier_unstemmed_text" : [(filename, TERRIER_DATA_BASE + "/msmarco_passage/terrier_unstemmed_text/latest/" + filename) for filename in STANDARD_TERRIER_INDEX_FILES],
        "terrier_stemmed_deepct" : [(filename, TERRIER_DATA_BASE + "/msmarco_passage/terrier_stemmed_deepct/latest/" + filename) for filename in STANDARD_TERRIER_INDEX_FILES],
        "terrier_stemmed_docT5query" : [(filename, TERRIER_DATA_BASE + "/msmarco_passage/terrier_stemmed_docT5query/latest/" + filename) for filename in STANDARD_TERRIER_INDEX_FILES],
    },
    "topics" :
        { 
            "train" : ("queries.train.tsv", "queries.tar.gz#queries.train.tsv", "singleline"),
            "dev" : ("queries.dev.tsv", "queries.tar.gz#queries.dev.tsv", "singleline"),
            "dev.small" : ("queries.dev.small.tsv", "collectionandqueries.tar.gz#queries.dev.small.tsv", "singleline"),
            "eval" : ("queries.eval.tsv", "queries.tar.gz#queries.eval.tsv", "singleline"),
            "eval.small" : ("queries.eval.small.tsv", "collectionandqueries.tar.gz#queries.eval.small.tsv", "singleline"),
            "test-2019" : ("msmarco-test2019-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz", "singleline"),
            "test-2020" : ("msmarco-test2020-queries.tsv.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz", "singleline")
        },        
    "tars" : {
        "queries.tar.gz" : ("queries.tar.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz"),
        "collection.tar.gz" : ("collection.tar.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz"),
        "collectionandqueries.tar.gz" : ("collectionandqueries.tar.gz", "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz")
    },
    "qrels" : 
        { 
            "train" : ("qrels.train.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv"),
            "dev" : ("qrels.dev.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv"),
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
        "train" : ("docv2_train_queries.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_train_queries.tsv", "singleline"),
        "dev1"  :("docv2_dev_queries.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_dev_queries.tsv", "singleline"),
        "dev2"  :("docv2_dev2_queries.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_dev2_queries.tsv", "singleline"),
        "valid1" : ("msmarco-test2019-queries.tsv.gz" , "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz", "singleline"),
        "valid2" : ("msmarco-test2020-queries.tsv.gz" , "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz", "singleline"),
        "trec_2021" : ("2021_queries.tsv" , "https://msmarco.blob.core.windows.net/msmarcoranking/2021_queries.tsv", "singleline"),
    },
    "qrels" : {
        "train" : ("docv2_train_qrels.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_train_qrels.tsv"),
        "dev1"  :("docv2_dev_qrels.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_dev_qrels.tsv"),
        "dev2"  :("docv2_dev2_qrels.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_dev2_qrels.tsv"),
        "valid1" : ("docv2_trec2019_qrels.txt.gz" , "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_trec2019_qrels.txt.gz"),
        "valid2" : ("docv2_trec2020_qrels.txt.gz" , "https://msmarco.blob.core.windows.net/msmarcoranking/docv2_trec2020_qrels.txt.gz")
    },
    "index" : _datarepo_index,
}

MSMARCOv2_PASSAGE_FILES = {
    "info_url" : "https://microsoft.github.io/msmarco/TREC-Deep-Learning.html",
    "topics" : {
        "train" : ("passv2_train_queries.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/passv2_train_queries.tsv", "singleline"),
        "dev1"  : ("passv2_dev_queries.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/passv2_dev_queries.tsv", "singleline"),
        "dev2"  : ("passv2_dev2_queries.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/passv2_dev2_queries.tsv", "singleline"),
        "trec_2021" : ("2021_queries.tsv" , "https://msmarco.blob.core.windows.net/msmarcoranking/2021_queries.tsv", "singleline"),
    },
    "qrels" : {
        "train" : ("passv2_train_qrels.tsv" "https://msmarco.blob.core.windows.net/msmarcoranking/passv2_train_qrels.tsv"),
        "dev1"  : ("passv2_dev_qrels.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/passv2_dev_qrels.tsv"),
        "dev2"  : ("passv2_dev2_qrels.tsv", "https://msmarco.blob.core.windows.net/msmarcoranking/passv2_dev2_qrels.tsv"),
    },
    "index" : _datarepo_index,
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
        "http://mirror.ir-datasets.com/79737768b3be1aa07b14691aa54802c5",
        "https://www.dcs.gla.ac.uk/~craigm/04.topic-map.official.txt"
        ] )],
    "topics_prefixed" : { 
        "all" : ("Web2004.query.stream.trecformat.txt", [
                "https://trec.nist.gov/data/web/Web2004.query.stream.trecformat.txt",
                "https://mirror.ir-datasets.com/10821f7a000b8bec058097ede39570be",
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
                "https://mirror.ir-datasets.com/93daa0e4b4190c84e30d2cce78a0f674",
                "https://www.dcs.gla.ac.uk/~craigm/04.qrels.web.mixed.txt"])
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
    "index": _datarepo_index_default_none,
    #"index":
    #    [(filename, VASWANI_INDEX_BASE + filename) for filename in STANDARD_TERRIER_INDEX_FILES + ["data.meta-0.fsomapfile"]],
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
    "msmarco_document" : RemoteDataset("msmarco_document", MSMARCO_DOC_FILES),
    "msmarcov2_document" : RemoteDataset("msmarcov2_document", MSMARCOv2_DOC_FILES),
    "msmarco_passage" : RemoteDataset("msmarco_passage", MSMARCO_PASSAGE_FILES),
    "msmarcov2_passage" : RemoteDataset("msmarcov2_passage", MSMARCOv2_PASSAGE_FILES),
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

# "trec-deep-learning-docs"
#DATASET_MAP['msmarco_document'] = DATASET_MAP["trec-deep-learning-docs"]
#DATASET_MAP['msmarco_passage'] = DATASET_MAP["trec-deep-learning-passages"]
DATASET_MAP["trec-deep-learning-docs"] = DATASET_MAP['msmarco_document']
DATASET_MAP["trec-deep-learning-passages"] = DATASET_MAP['msmarco_passage']


def get_dataset(name, **kwargs):
    """
        Get a dataset by name
    """
    # Some datasets in ir_datasets are built on-the-fly (e.g., clirmatrix).
    # Handle this by allocating it on demand here.
    if name not in DATASET_MAP and name.startswith('irds:'):
        # remove irds: prefix
        ds_id = name[len('irds:'):]
        DATASET_MAP[name] = IRDSDataset(ds_id)
    rtr = DATASET_MAP[name]
    rtr._configure(**kwargs)
    return rtr

def datasets():
    """
        Lists all the names of the datasets
    """
    return DATASET_MAP.keys()

def find_datasets(query, en_only=True):
    """
    A grep-like method to help identify datasets. Filters the output of list_datasets() based on the name containing the query
    """
    datasets = list_datasets(en_only=en_only)
    return datasets[datasets['dataset'].str.contains(query)]

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
