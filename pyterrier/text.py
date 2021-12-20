from pyterrier.transformer import TransformerBase
from pyterrier.datasets import IRDSDataset
import more_itertools
from collections import defaultdict
import re
from pyterrier.model import add_ranks
import pandas as pd
import numpy as np
from typing import List, Union
from warnings import warn


def get_text(
        indexlike, 
        metadata : Union[str,List[str]] = "body", 
        by_query : bool = False,
        verbose : bool = False) -> TransformerBase:
    """
    A utility transformer for obtaining the text from the text of documents (or other document metadata) from Terrier's MetaIndex
    or an IRDSDataset docstore.

    Arguments:
        indexlike: a Terrier index or IRDSDataset to retrieve the metadata from
        metadata(list(str) or str): a list of strings of the metadata keys to retrieve from the index. Defaults to ["body"]
        by_query(bool): whether the entire dataframe should be progressed at once, rather than one query at a time. 
            Defaults to false, which means that all document metadata will be fetched at once.
        verbose(bool): whether to print a tqdm progress bar. Defaults to false. Has no effect when by_query=False

    Example::

        pipe = ( pt.BatchRetrieve(index, wmodel="DPH")
            >> pt.text.get_text(index)
            >> pt.text.scorer(wmodel="DPH") )

    """
    import pyterrier as pt
    JIR = pt.autoclass('org.terrier.querying.IndexRef')
    JI = pt.autoclass('org.terrier.structures.Index')

    if isinstance(metadata, str):
        metadata = [metadata]

    if isinstance(indexlike, str) or isinstance(indexlike, JIR):
        index = pt.IndexFactory.of(indexlike)
        add_text_fn = _add_text_terrier_metaindex(index, metadata)
    elif isinstance(indexlike, JI):
        add_text_fn = _add_text_terrier_metaindex(indexlike, metadata)
    elif isinstance(indexlike, IRDSDataset):
        add_text_fn = _add_text_irds_docstore(indexlike, metadata)
    else:
        raise ValueError("indexlike %s of type %s not supported. Pass a string, an IndexRef, an Index, or an IRDSDataset" %
            (str(indexlike), type(indexlike)))

    if by_query:
        return pt.apply.by_query(add_text_fn, verbose=verbose)
    return pt.apply.generic(add_text_fn)


def _add_text_terrier_metaindex(index, metadata):
    metaindex = index.getMetaIndex()
    if metaindex is None:
        raise ValueError("Index %s does not have a metaindex" % str(index))

    for k in metadata:
        if not k in metaindex.getKeys():
            raise ValueError("Index from %s did not have requested metaindex key %s. Keys present in metaindex are %s" % 
            (str(index), k, str( metaindex.getKeys()) ))

    def add_docids(res):
        res = res.copy()
        res["docid"] = res.apply(lambda row: metaindex.getDocument("docno", row.docno), axis=1)
        return res

    def add_text_function_docids(res):
        res = res.copy()
        docids = res.docid.values.tolist()
        # indexed by docid then keys
        allmeta = metaindex.getItems(metadata, docids)
        import numpy as np
        # get transpose to make easier for insertion back into dataframe?
        allmeta = np.array(allmeta).T
        for i, k in enumerate(metadata):
            res[k] = allmeta[i]
        return res

    def add_text_generic(res):
        if not "docid" in res.columns:
            assert "docno" in res.columns, "Neither docid nor docno are in the input dataframe, found %s" % (str(res.columns))
            res = add_docids(res)
        return add_text_function_docids(res)

    return add_text_generic


def _add_text_irds_docstore(irds_dataset, metadata):
    irds = irds_dataset.irds_ref()
    assert irds.has_docs(), f"dataset {irds_dataset} doesn't provide docs"
    docs_cls = irds.docs_cls()

    for k in metadata:
        if not k in docs_cls._fields:
            raise ValueError(f"{irds_dataset} did not have requested field {k}. Keys present are {docs_cls._fields} (from {docs_cls})")
    field_idx = [(f, docs_cls._fields.index(f)) for f in metadata]

    docstore = irds.docs_store()

    def add_text_function_docids(res):
        assert 'docno' in res, "requires docno column"
        res = res.copy()
        docids = res.docno.values.tolist()
        did2idxs = defaultdict(list)
        for i, did in enumerate(docids):
            did2idxs[did].append(i)
        new_columns = {f: [None] * len(docids) for f in metadata}
        for doc in docstore.get_many_iter(docids):
            for didx in did2idxs[doc.doc_id]:
                for f, fidx in field_idx:
                    new_columns[f][didx] = doc[fidx]
        for k, v in new_columns.items():
            res[k] = v
        return res

    return add_text_function_docids


def scorer(*args, **kwargs) -> TransformerBase:
    """
    This allows scoring of the documents with respect to a query, without creating an index first. 
    This is an alias to pt.TextScorer(). Internally, a Terrier memory index is created, before being
    used for scoring.

    Example::
    
        df = pd.DataFrame(
            [
                ["q1", "chemical reactions", "d1", "professor protor poured the chemicals"],
                ["q1", "chemical reactions", "d2", "chemical brothers turned up the beats"],
            ], columns=["qid", "query", "docno", "text"])
        textscorerTf = pt.text.scorer(body_attr="text", wmodel="Tf")
        rtr = textscorerTf.transform(df)
        # rtr will have a score for each document for the query "chemical reactions" based on the provided document contents
        # both attain score 1, as, after stemming, they both contain one occurrence of the query term 'chemical'
        # ["q1", "chemical reactions", "d1", "professor protor poured the chemicals", 0, 1]
        # ["q1", "chemical reactions", "d2", "chemical brothers turned up the beats", 0, 1]

    For calculating the scores of documents using any weighting model with the concept of IDF, it may be useful to make use of
    an existing Terrier index for background statistics::

        textscorerTfIdf = pt.text.scorer(body_attr="text", wmodel="TF_IDF", background_index=index)

    """
    import pyterrier as pt
    return pt.batchretrieve.TextScorer(*args, **kwargs)

def sliding( text_attr='body', length=150, stride=75, join=' ', prepend_attr='title', **kwargs) -> TransformerBase:
    r"""
    A useful transformer for splitting long documents into smaller passages within a pipeline. This applies a *sliding* window over the
    text, where each passage is the give number of tokens long. Passages can overlap, if the stride is set smaller than the length. In
    applying this transformer, docnos are altered by adding '%p' and a passage number. The original scores for each document can be recovered
    by aggregation functions, such as `max_passage()`.

    For the puposes of obtaining passages of a given length, tokenisation is perfomed simply by splitting on one-or-more spaces, i.e. based 
    on the Python regular expression ``re.compile(r'\s+')``.

    Parameters:
        text_attr(str): what is the name of the dataframe attribute containing the main text of the document to be split into passages.
            Default is 'body'.
        length(int): how many tokens in each passage. Default is 150.
        stride(int): how many tokens to advance each passage by. Default is 75.
        prepend_attr(str): whether another document attribute, such as the title of the document, to each passage, following [Dai2019]. Defaults to 'title'. 
        title_attr(str): what is the name of the dataframe attribute containing the title the document to be split into passages.
            Default is 'title'. Only used if prepend_title is set to True.
    
    Example::
    
        pipe = ( pt.BatchRetrieve(index, wmodel="DPH", metadata=["docno", "body"]) 
            >> pt.text.sliding(length=128, stride=64, prepend_attr=None) 
            >> pt.text.scorer(wmodel="DPH") 
            >> pt.text.max_passage() )
        
    """

    # deal with older names for attributes
    if 'passage_length' in kwargs:
        length = kwargs['passage_length']
        del kwargs['passage_length']
        warn("passage_length should be length.", FutureWarning, 2)
    if 'passage_stride' in kwargs:
        stride = kwargs['passage_stride']
        del kwargs['passage_stride']
        warn("passage_stride should be stride.", FutureWarning, 2)
    if 'prepend_title' in kwargs:
        warn("prepend_title and title_attr should be replaced with prepend_attr.", FutureWarning, 2)
        if kwargs['prepend_title']:
            prepend_attr = kwargs['title_attr']
            del kwargs['title_attr']
        else:
            prepend_attr = None
        del kwargs['prepend_title']
        
    return SlidingWindowPassager(
        text_attr=text_attr, 
        passage_length=length, 
        passage_stride=stride,
        prepend_title=prepend_attr is not None,
        title_attr=prepend_attr,
        join=' ',
        **kwargs
    )

def max_passage() -> TransformerBase:
    """
    Scores each document based on the maximum score of any constituent passage. Applied after a sliding window transformation
    has been scored.
    """
    return MaxPassage()

def mean_passage() -> TransformerBase:
    """
    Scores each document based on the mean score of all constituent passages. Applied after a sliding window transformation
    has been scored.
    """
    return MeanPassage()

def first_passage() -> TransformerBase:
    """
    Scores each document based on score of the first passage of that document. Note that this transformer is rarely used in conjunction with
    the sliding window transformer, as all passages would required to be scored, only for the first one to be used.
    """
    return FirstPassage()

def kmaxavg_passage(k : int) -> TransformerBase:
    """
    Scores each document based on the average score of the top scoring k passages. Generalises combination of mean_passage()
    and max_passage(). Proposed in [Chen2020].

    Parameters:
        k(int): The number of top-scored passages for each document to use when scoring
    
    """
    return KMaxAvgPassage(k)

def slidingWindow(sequence : list, winSize : int, step : int) -> list:
    """
    For the specified sequence, break into sliding windows of size winSize, 
    stepping forward by the specified amount each time
    """
    return [x for x in list(more_itertools.windowed(sequence,n=winSize, step=step)) if x[-1] is not None]


class DePassager(TransformerBase):

    def __init__(self, agg="max", **kwargs):
        super().__init__(**kwargs)
        self.agg = agg

    def transform(self, topics_and_res):
        topics_and_res = topics_and_res.copy()
        topics_and_res[["olddocno", "pid"]] = topics_and_res.docno.str.split("%p", expand=True)
        if self.agg == 'max':
            groups = topics_and_res.groupby(['qid', 'olddocno'])
            group_max_idx = groups['score'].idxmax()
            rtr = topics_and_res.loc[group_max_idx, :]
            rtr = rtr.drop(columns=['docno', 'pid']).rename(columns={"olddocno" : "docno"})
        
        if self.agg == 'first':
            #could this be done by just selectin pid = 0?
            topics_and_res.pid = topics_and_res.pid.astype(int)
            rtr = topics_and_res[topics_and_res.pid == 0].rename(columns={"olddocno" : "docno"})
            
            groups = topics_and_res.groupby(['qid', 'olddocno'])
            group_first_idx = groups['pid'].idxmin()
            rtr = topics_and_res.loc[group_first_idx, ]
            rtr = rtr.drop(columns=['docno', 'pid']).rename(columns={"olddocno" : "docno"})

        if self.agg == 'mean':
            rtr = topics_and_res.groupby(['qid', 'olddocno']).mean()['score'].reset_index().rename(columns={'olddocno' : 'docno'})
            from .model import query_columns
            #add query columns back
            rtr = rtr.merge(topics_and_res[query_columns(topics_and_res)].drop_duplicates(), on='qid')

        if self.agg == 'kmaxavg':
            rtr = topics_and_res.groupby(['qid', 'olddocno'])['score'].apply(lambda ser: ser.nlargest(2).mean()).reset_index().rename(columns={'olddocno' : 'docno'})
            from .model import query_columns
            #add query columns back
            rtr = rtr.merge(topics_and_res[query_columns(topics_and_res)].drop_duplicates(), on='qid')

        rtr = add_ranks(rtr)
        return rtr

class KMaxAvgPassage(DePassager):
    """
        See ICIP at TREC-2020 Deep Learning Track, X.Chen et al. Proc. TREC 2020.
        Usage:
            X >> SlidingWindowPassager() >>  Y >>  KMaxAvgPassage(2)
        where X is some kind of model for obtaining the text of documents and Y is a text scorer, such as BERT or ColBERT
    """
    def __init__(self, K, **kwargs):
        kwargs["agg"] = "kmaxavg"
        self.K = K
        super().__init__(**kwargs)

class MaxPassage(DePassager):
    def __init__(self, **kwargs):
        kwargs["agg"] = "max"
        super().__init__(**kwargs)

class FirstPassage(DePassager):
    def __init__(self, **kwargs):
        kwargs["agg"] = "first"
        super().__init__(**kwargs)

class MeanPassage(DePassager):
    def __init__(self, **kwargs):
        kwargs["agg"] = "mean"
        super().__init__(**kwargs)


class SlidingWindowPassager(TransformerBase):

    def __init__(self, text_attr='body', title_attr='title', passage_length=150, passage_stride=75, join=' ', prepend_title=True, **kwargs):
        super().__init__(**kwargs)
        self.text_attr=text_attr
        self.title_attr=title_attr
        self.passage_length = passage_length
        self.passage_stride= passage_stride
        self.join = join
        self.prepend_title = prepend_title

    def _check_columns(self, topics_and_res):
        if not self.text_attr in topics_and_res.columns:
            raise KeyError("%s is a required input column, but not found in input dataframe. Found %s" % (self.text_attr, str(list(topics_and_res.columns))))
        if self.prepend_title and not self.title_attr in topics_and_res.columns:
            raise KeyError("%s is a required input column, but not found in input dataframe. Set prepend_title=False to disable its use. Found %s" % (self.title_attr, str(list(topics_and_res.columns))))
        if not "docno" in topics_and_res.columns:
            raise KeyError("%s is a required input column, but not found in input dataframe. Found %s" % ("docno", str(list(topics_and_res.columns))))

    def transform(self, topics_and_res):
        # validate input columns
        self._check_columns(topics_and_res)
        print("calling sliding on df of %d rows" % len(topics_and_res))

        # now apply the passaging
        if "qid" in topics_and_res.columns: 
            return self.applyPassaging(topics_and_res, labels="label" in topics_and_res.columns)
        return self.applyPassaging_no_qid(topics_and_res)

    def applyPassaging_no_qid(self, df):
        p = re.compile(r"\s+")
        rows=[]
        for row in df.itertuples():
            row = row._asdict()
            toks = p.split(row[self.text_attr])
            if len(toks) < self.passage_length:
                row['docno'] = row['docno'] + "%p0"
                row[self.text_attr] = ' '.join(toks)
                if self.prepend_title:
                    row[self.text_attr] = str(row[self.title_attr]) + self.join + row[self.text_attr]
                    del(row[self.title_attr])
                rows.append(row)
            else:
                passageCount=0
                for i, passage in enumerate( slidingWindow(toks, self.passage_length, self.passage_stride)):
                    newRow = row.copy()
                    newRow['docno'] = row['docno'] + "%p" + str(i)
                    newRow[self.text_attr] = ' '.join(passage)
                    if self.prepend_title:
                        newRow[self.text_attr] = str(row[self.title_attr]) + self.join + newRow[self.text_attr]
                        del(newRow[self.title_attr])
                    rows.append(newRow)
                    passageCount+=1
        return pd.DataFrame(rows)


    def applyPassaging(self, df, labels=True):
        newRows=[]
        labelCount=defaultdict(int)
        p = re.compile(r"\s+")
        currentQid=None
        rank=0
        copy_columns=[]
        for col in ["score", "rank"]:
            if col in df.columns:
                copy_columns.append(col)

        if len(df) == 0:
            return pd.DataFrame(columns=['qid', 'query', 'docno', self.text_attr, 'score', 'rank'])
    
        from pyterrier import tqdm
        with tqdm('passsaging', total=len(df), desc='passaging', leave=False) as pbar:
            for index, row in df.iterrows():
                pbar.update(1)
                qid = row['qid']
                if currentQid is None or currentQid != qid:
                    rank=0
                    currentQid = qid
                rank+=1
                toks = p.split(row[self.text_attr])
                if len(toks) < self.passage_length:
                    newRow = row.copy()
                    newRow['docno'] = row['docno'] + "%p0"
                    newRow[self.text_attr] = ' '.join(toks)
                    if self.prepend_title:
                        newRow.drop(labels=[self.title_attr], inplace=True)
                        newRow[self.text_attr] = str(row[self.title_attr]) + self.join + newRow[self.text_attr]
                    if labels:
                        labelCount[row['label']] += 1
                    for col in copy_columns:
                        newRow[col] = row[col]
                    newRows.append(newRow)
                else:
                    passageCount=0
                    for i, passage in enumerate( slidingWindow(toks, self.passage_length, self.passage_stride)):
                        newRow = row.copy()
                        newRow['docno'] = row['docno'] + "%p" + str(i)
                        newRow[self.text_attr] = ' '.join(passage)
                        if self.prepend_title:
                            newRow.drop(labels=[self.title_attr], inplace=True)
                            newRow[self.text_attr] = str(row[self.title_attr]) + self.join + newRow[self.text_attr]
                        for col in copy_columns:
                            newRow[col] = row[col]
                        if labels:
                            labelCount[row['label']] += 1
                        newRows.append(newRow)
                        passageCount+=1
        newDF = pd.DataFrame(newRows)
        newDF['query'].fillna('',inplace=True)
        newDF[self.text_attr].fillna('',inplace=True)
        newDF['qid'].fillna('',inplace=True)
        newDF.reset_index(inplace=True,drop=True)
        return newDF
