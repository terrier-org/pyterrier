from .transformer import TransformerBase
import hashlib
from . import HOME_DIR 
import os
from os import path
CACHE_DIR = None
import pandas as pd
import pickle
from functools import partial
import datetime
from warnings import warn

DEFINITION_FILE = ".transformer"

#https://stackoverflow.com/a/10171475
from math import log
unit_list = list( zip(['bytes', 'kB', 'MB', 'GB', 'TB', 'PB'], [0, 0, 1, 2, 2, 2]) )
def sizeof_fmt(num):
    """Human friendly file size"""
    if num > 1:
        exponent = min(int(log(num, 1024)), len(unit_list) - 1)
        quotient = float(num) / 1024**exponent
        unit, num_decimals = unit_list[exponent]
        format_string = '{:.%sf} {}' % (num_decimals)
        return format_string.format(quotient, unit)
    if num == 0:
        return '0 bytes'
    if num == 1:
        return '1 byte'

def init():
    global CACHE_DIR
    CACHE_DIR = path.join(HOME_DIR,"transformer_cache") 

def list_cache():
    if CACHE_DIR is None:
        init()
    rtr={}
    for dirname in os.listdir(CACHE_DIR):
        elem={}
        dir = path.join(CACHE_DIR, dirname)
        if not path.isdir(dir):
            continue
        def_file = path.join(dir, DEFINITION_FILE)
        if path.exists(def_file):
            with open(def_file, "r") as f:
                elem["transformer"] = f.readline()
        elem["size"] = sum(d.stat().st_size for d in os.scandir(dir) if d.is_file())
        elem["size_str"] = sizeof_fmt(elem["size"])
        elem["queries"] = len(os.listdir(dir)) -2 #subtract .keys and DEFINITION_FILE
        elem["lastmodified"] = path.getmtime(dir)
        elem["lastmodified_str"] = datetime.datetime.fromtimestamp(elem["lastmodified"]).strftime('%Y-%m-%dT%H:%M:%S')
        rtr[dirname] = elem
    return rtr


def clear_cache():
    if CACHE_DIR is None:
        init()
    import shutil
    shutil.rmtree(CACHE_DIR)

class ChestCacheTransformer(TransformerBase):
    """
        A transformer that cache the results of the consituent (inner) transformer. 
        This is instantiated using the `~` operator on any transformer.

        Caching is unqiue based on the configuration of the pipeline, as read by executing
        retr() on the pipeline. Caching lookup is based on the qid, so any change in query
        _formulation_ will not be reflected in a cache's results.

        Example Usage::

            dataset = pt.get_dataset("trec-robust-2004")
            # use for first pass and 2nd pass
            BM25 = pt.BatchRetrieve(index, wmodel="BM25")

            # used for query expansion
            RM3 = pt.rewrite.RM3(index)
            pt.Experiment([
                    ~BM25,
                    (~BM25) >> RM3 >> BM25
                ],
                dataset.get_topics(),
                dataset.get_qrels(),
                eval_metrics=["map"]
            )

        In the above example, we use the `~` operator on the first pass retrieval using BM25, but not on the 2nd pass retrieval, 
        as the query formulation will differ during the second pass.

        Caching is not supported for re-ranking transformers.        
    """

    def __init__(self, inner, **kwargs):
        super().__init__(**kwargs)
        on="qid"
        self.inner = inner
        self.disable = False
        if CACHE_DIR is None:
            init()

        # we take the md5 of the __repr__ of the pipeline to make a unique identifier for the pipeline
        # all different pipelines should return unique __repr_() values, as these are intended to be
        # unambiguous
        trepr = repr(self.inner)
        if "object at 0x" in trepr:
            warn("Cannot cache pipeline %s has a component has not overridden __repr__" % trepr)
            self.disable = True
            
        uid = hashlib.md5( bytes(trepr, "utf-8") ).hexdigest()
        destdir = path.join(CACHE_DIR, uid)
        os.makedirs(destdir, exist_ok=True)
        definition_file=path.join(destdir, DEFINITION_FILE)
        if not path.exists(definition_file):
            with open(definition_file, "w") as f:
                f.write(trepr)
        from chest import Chest
        self.chest = Chest(path=destdir, 
            dump=lambda data, filename: pd.DataFrame.to_pickle(data, filename) if isinstance(data, pd.DataFrame) else pickle.dump(data, filename, protocol=1),
            load=lambda filehandle: pickle.load(filehandle) if ".keys" in filehandle.name else pd.read_pickle(filehandle)
        )
        self.hits = 0
        self.requests = 0

    def stats(self):
        return self.hits / self.requests if self.requests > 0 else 0

    # dont double cache - we cannot cache ourselves
    def __invert__(self):
        return self

    def __repr__(self):
        return "Cache("+self.inner.__repr__()+")"

    def __str__(self):
        return "Cache("+str(self.inner)+")"

    @property
    def NOCACHE(self):
        return self.inner

    def transform(self, input_res):
        if self.disable:
            return self.inner.transform(input_res)
        if "docid" in input_res.columns or "docno" in input_res.columns:
            raise ValueError("Caching of %s for re-ranking is not supported. Caching currently only supports input dataframes with queries as inputs and cannot be used for re-rankers." % self.inner.__repr__())
        return self._transform_qid(input_res)

    def _transform_qid(self, input_res):
        rtr = []
        todo=[]
        
        # We cannot remove this iterrows() without knowing how to take named tuples into a dataframe
        for index, row in input_res.iterrows():
            qid = str(row["qid"])
            self.requests += 1
            try:
                df = self.chest.get(qid, None)
            except:
                # occasionally we have file not founds, 
                # lets remove from the cache and continue
                del self.chest[qid]
                df = None
            if df is None:
                todo.append(row.to_frame().T)
            else:
                self.hits += 1
                rtr.append(df)
        if len(todo) > 0:
            todo_df = pd.concat(todo)
            todo_res = self.inner.transform(todo_df)
            for row in todo_df.itertuples():
                qid = row.qid
                this_query_res = todo_res[todo_res["qid"] == qid]
                self.chest[qid] = this_query_res
                rtr.append(this_query_res)
        self.chest.flush()
        return pd.concat(rtr)
