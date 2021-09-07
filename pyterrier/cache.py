from .transformer import TransformerBase
import hashlib
from chest import Chest
from . import HOME_DIR 
import os
from os import path
CACHE_DIR = None
DEFAULT_CACHE_STORE = "shelve" #or "chest"
import pandas as pd
import pickle
from functools import partial
import datetime
from warnings import warn
from typing import List

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
        shelve_file = path.join(dir, "shelve.db")
        if path.exists(shelve_file):
            elem["size"] = path.getsize(shelve_file)
            elem["lastmodified"] = path.getmtime(shelve_file)            
        else:
            #we assume it is a chest
            elem["size"] = sum(d.stat().st_size for d in os.scandir(dir) if d.is_file())
            elem["queries"] = len(os.listdir(dir)) -2 #subtract .keys and DEFINITION_FILE
            elem["lastmodified"] = path.getmtime(dir)
        
        elem["lastmodified_str"] = datetime.datetime.fromtimestamp(elem["lastmodified"]).strftime('%Y-%m-%dT%H:%M:%S')
        elem["size_str"] = sizeof_fmt(elem["size"])
        rtr[dirname] = elem
    return rtr


def clear_cache():
    if CACHE_DIR is None:
        init()
    import shutil
    shutil.rmtree(CACHE_DIR)


class GenericCacheTransformer(TransformerBase):
    """
        A transformer that cache the results of the consituent (inner) transformer. 
        This is instantiated using the `~` operator on any transformer.

        Caching is based on the configuration of the pipeline, as read by executing
        repr() on the pipeline. Caching lookup is by default based on the qid, so any change in query
        _formulation_ will not be reflected in a cache's results.

        Caching lookup can be changed by altering the `on` attribute in the cache object.

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

        
    """

    def __init__(self, inner, on=["qid"], verbose=False, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.on = on
        self.inner = inner
        self.disable = False
        self.hits = 0
        self.requests = 0
        self.debug = debug
        self.verbose = verbose

        if CACHE_DIR is None:
            init()

        # we take the md5 of the __repr__ of the pipeline to make a unique identifier for the pipeline
        # all different pipelines should return unique __repr_() values, as these are intended to be
        # unambiguous
        self.trepr = repr(self.inner)
        if "object at 0x" in self.trepr:
            warn("Cannot cache pipeline %s across PyTerrier sessions, as it has a transient component, which has not overridden __repr__()" % self.trepr)
            #return
            #self.disable = True
            
        uid = hashlib.md5( bytes(self.trepr, "utf-8") ).hexdigest()
        self.destdir = path.join(CACHE_DIR, uid)
        os.makedirs(self.destdir, exist_ok=True)

        definition_file=path.join(self.destdir, DEFINITION_FILE)
        if not path.exists(definition_file):
            if self.debug:
                print("Creating new cache store at %s for %s" % (self.destdir, self.trepr))
            with open(definition_file, "w") as f:
                f.write(self.trepr)
                
    def stats(self):
        return self.hits / self.requests if self.requests > 0 else 0

    # dont double cache - we cannot cache ourselves
    def __invert__(self):
        return self

    def __repr__(self):
        return "Cache("+self.inner.__repr__()+")"

    def __str__(self):
        return "Cache("+str(self.inner)+")"

    def __del__(self):
        self.close()

    @property
    def NOCACHE(self):
        return self.inner

    def flush(self):
        self.chest.flush()

    def close(self):
        pass

    def transform(self, input_res):
        if self.disable:
            return self.inner.transform(input_res)
        for col in self.on:
            if col not in input_res.columns:
                raise ValueError("Caching on %s, but did not find column %s among input columns %s"
                    % (str(self.on)), col, str(input_res.columns))
        for col in ["docno"]:
            if col in input_res.columns and not col in self.on and len(self.on) == 1:
                warn(("Caching on=%s, but found column %s among input columns %s. You may want " % (str(self.on)), col, str(input_res.columns) ) +
                    "to update the on attribute for the cache transformer")
        return self._transform_qid(input_res)

    def _transform_qid(self, input_res):
        # output dataframes to /return/
        rtr = []
        # input rows to execute on the inner transformer
        todo=[]
        import pyterrier as pt
        iter = input_res.itertuples(index=False)
        for row in pt.tqdm(
                iter,
                desc="%s lookups" % self, 
                unit='row', 
                total=len(input_res)) if self.verbose else iter:
            # we calculate what we will key this cache on
            key = ''.join([getattr(row, k) for k in self.on])
            qid = str(row.qid)
            self.requests += 1
            try:
                df = self.chest.get(key, None)
            except:
                # occasionally we have file not founds, 
                # lets remove from the cache and continue
                del self.chest[key]
                df = None
            if df is None:
                if self.debug:
                    print("%s cache miss for key %s" % (self, key))
                todo.append(row)
            else:
                if self.debug:
                    print("%s cache hit for key %s" % (self, key))
                self.hits += 1
                rtr.append(df)
        if len(todo) > 0:
            todo_df = pd.DataFrame(todo)
            todo_res = self.inner.transform(todo_df)
            for key_vals, group in todo_res.groupby(self.on):
                key = ''.join(key_vals)
                self.chest[key] = group
                if self.debug:
                    print("%s caching %d results for key %s" % (self, len(group), key))
            rtr.append(todo_res)
        self.flush()
        return pd.concat(rtr)


class ChestCacheTransformer(GenericCacheTransformer):
    """
        A cache transformer based on `chest <https://github.com/blaze/chest>`_.
    """
    def __init__(self, inner, **kwargs):
        super().__init__(inner, **kwargs)

        self.chest = Chest(path=self.destdir, 
            dump=lambda data, filename: pd.DataFrame.to_pickle(data, filename) if isinstance(data, pd.DataFrame) else pickle.dump(data, filename, protocol=1),
            load=lambda filehandle: pickle.load(filehandle) if ".keys" in filehandle.name else pd.read_pickle(filehandle)
        )

class ShelveCacheTransformer(GenericCacheTransformer):
    """
        A cache transformer based on Python's `shelve <https://docs.python.org/3/library/shelve.html>`_ library. Compares to the 
        chest-based cache, this transformer MUST be closed before cached instances can be seen by other instances. 
    """
    def __init__(self, inner, **kwargs):
        super().__init__(inner, **kwargs)
        filename = os.path.join(self.destdir, "shelve")
        import shelve
        if os.path.exists(filename) and os.path.getsize(filename) == 0:
            warn("Cache file exists but has 0 size - perhaps a previous transformer cache should have been closed")
        self.chest = shelve.open(filename)  
    
    def flush(self):
        self.chest.sync()

    def close(self):
        self.chest.close()

CACHE_STORES={
    "shelve" : ShelveCacheTransformer,
    "chest" : ChestCacheTransformer    
}

def of(
        inner : TransformerBase, 
        on : List[str] = ["qid"], 
        store : str= DEFAULT_CACHE_STORE, **kwargs
        ) -> GenericCacheTransformer:
    """
    Returns a transformer that caches the inner transformer.
    Arguments:
        inner(TransformerBase): which transformer should be cached
        on(List[str]): which attributes to use as keys when caching
        store(str): name of a cache type, either "shelve" or "chest". Defaults to "shelve".
    """
    if not store in CACHE_STORES:
        raise ValueError("cache store type %s unknown, known types %s" % (store, list(CACHE_STORES.keys())))
    clz = CACHE_STORES[store]
    return clz(inner, on=on, **kwargs)
    
    