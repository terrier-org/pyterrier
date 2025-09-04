import more_itertools
from collections import defaultdict
import re
import pandas as pd
from typing import Any, List, Union, Literal, Protocol, runtime_checkable
from warnings import warn
import types
import pyterrier as pt

@runtime_checkable
class HasTextLoader(Protocol):
    def text_loader(
        self,
        fields: Union[List[str], str, Literal['*']] = '*',
        *,
        verbose: bool = False,
    ) -> pt.Transformer:
        """
        Returns a transformer that loads and populates text columns for each document in the
        provided input frame.

        :param fields: The names of the fields to load. If a list of strings, all fields are provided.
            If a single string, this single field is provided. If the special value of '*' (default,
            all available fields are provided.
        :param verbose: Show a progress bar.
        """


def get_text(
        indexlike: Union[HasTextLoader, str],
        metadata : Union[str,List[str], Literal['*']] = '*',
        by_query : bool = False,
        verbose : bool = False,
        **kwargs: Any) -> pt.Transformer:
    """
    A utility transformer for obtaining the text from the text of documents (or other document metadata) from Terrier's MetaIndex
    or an IRDSDataset docstore.

    :param indexlike: an object that provides a .text_loader() factory method, such as a Terrier index or IRDSDataset.
        If a ``str`` is provided, it will try to load a Terrier index from the provided path.
    :param metadata: The names of the fields to load. If a list of strings, all fields are provided.
        If a single string, this single field is provided. If the special value of '*' (default), all
        available fields are provided.
    :param by_query: whether the entire dataframe should be progressed at once, rather than one query at a time.
        Defaults to false, which means that all document metadata will be fetched at once.
    :param verbose: whether to print a tqdm progress bar. When by_query=True, prints progress by query. Otherwise,
        the behaviour is defined by the provided ``indexlike``.
    :param kwargs: other arguments to pass through to the text_loader.
    :return: a transformer that loads the text of documents from the provided indexlike.
    :rtype: pt.Transformer
    :raises ValueError: if indexlike does not provide a .text_loader() method.

    Example (Terrier Index)::

        index = pt.IndexFactory.of("./index/")
        pipe = ( pt.terrier.Retriever(index, wmodel="DPH")
            >> pt.text.get_text(index) # load text using a PyTerrier index
            >> pt.text.scorer(wmodel="DPH") )

    Example (IR Datasets)::

        # see https://github.com/terrierteam/pyterrier_t5
        from pyterrier_t5 import MonoT5ReRanker
        bm25 = pt.terrier.Retriever.from_dataset(pt.get_dataset('msmarcov2_passage'), wmodel='BM25')
        # load text using IR Datasets
        loader = pt.text.get_text(pt.get_dataset('irds:msmarco-passage-v2'), ['text'])
        monoT5 = bm25 >> loader >> MonoT5ReRanker()

    """
    if isinstance(indexlike, str):
        # TODO: We'll need to decide how to handle this once terrier is split from core
        # Maybe it should run Artifact.load(indexlike) instead?
        indexlike = pt.IndexFactory.of(indexlike)

    if not isinstance(indexlike, HasTextLoader):
        raise ValueError('indexlike must provide a .text_loader() method.')

    result : pt.Transformer
    result = indexlike.text_loader(metadata, verbose=verbose and not by_query, **kwargs)

    if by_query:
        result = pt.apply.by_query(result, verbose=verbose)

    return result


def scorer(*args, **kwargs) -> pt.Transformer:
    """
    This allows scoring of the documents with respect to a query, without creating an index first. 
    This is an alias to pt.TextScorer(). Internally, a Terrier memory index is created, before being
    used for scoring.

    :pararm body_attr: what dataframe input column contains the text of the document. Default is `"body"`.
    :param wmodel: name of the weighting model to use for scoring.
    :param background_index: An optional background index to use for collection statistics. If a weighting
        model such as BM25 or TF_IDF or PL2 is used without setting the background_index, the background statistics
        will be calculated from the dataframe, which is ususally not the desired behaviour.
    :param args: other arguments to pass through to the TextScorer.
    :param kwargs: other arguments to pass through to the TextScorer.
    :return: a transformer that scores the documents with respect to a query.
    :rtype: pt.Transformer   

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

    For calculating the scores of documents using any weighting model with the concept of IDF, it is strongly advised to make use of
    an existing Terrier index for background statistics. Without a background index, IDF will be calculated based on the supplied
    dataframe (for models such as BM25, this can lead to negative scores)::

        textscorerTfIdf = pt.text.scorer(body_attr="text", wmodel="TF_IDF", background_index=index)

    """
    return pt.terrier.retriever.TextScorer(*args, **kwargs)

def sliding( text_attr='body', length=150, stride=75, join=' ', prepend_attr='title', tokenizer=None, **kwargs) -> pt.Transformer:
    r"""
    A useful transformer for splitting long documents into smaller passages within a pipeline. This applies a *sliding* window over the
    text, where each passage is the give number of tokens long. Passages can overlap, if the stride is set smaller than the length. In
    applying this transformer, docnos are altered by adding '%p' and a passage number. The original scores for each document can be recovered
    by aggregation functions, such as ``max_passage()``.

    For the puposes of obtaining passages of a given length, the tokenisation can be controlled. By default, tokenisation takes place by splitting
    on space, i.e. based on the Python regular expression ``re.compile(r'\s+')``. However, more fine-grained tokenisation can applied by passing 
    an object matching the HuggingFace Transformers `Tokenizer API <https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer>`_ 
    as the `tokenizer` kwarg argument. In short, the `tokenizer` object must have a ``.tokenize(str) -> list[str]`` method and 
    ``.convert_tokens_to_string(list[str]) -> str`` for detokenisation.
    
    :param text_attr: what is the name of the dataframe attribute containing the main text of the document to be split into passages.
        Default is 'body'.
    :param length: how many tokens in each passage. Default is 150.
    :param stride: how many tokens to advance each passage by. Default is 75.
    :param join: how to join the tokens of the passage together. Default is ' '.
    :param prepend_attr: whether another document attribute, such as the title of the document, to each passage, following [Dai2019]. Defaults to 'title'.
    :param tokenizer: which model to use for tokenizing. The object must have a ``.tokenize(str) -> list[str]`` method for tokenization and ``.convert_tokens_to_string(list[str]) -> str`` for detokenization.
            Default is None. Tokenisation is perfomed by splitting on one-or-more spaces, i.e. based on the Python regular expression ``re.compile(r'\s+')``
    :param kwargs: other arguments to pass through to the SlidingWindowPassager.
    :return: a transformer that splits the documents into passages.
    :rtype: pt.Transformer
    :raises KeyError: if the text_attr or title_attr columns are not found in the input dataframe.
    
    Example::
    
        pipe = ( pt.terrier.Retriever(index, wmodel="DPH", metadata=["docno", "body"]) 
            >> pt.text.sliding(length=128, stride=64, prepend_attr=None) 
            >> pt.text.scorer(wmodel="DPH") 
            >> pt.text.max_passage() )

        # tokenizer model 
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        pipe = (pt.terrier.Retriever(index, wmodel="DPH", metadata=["docno", "body"])
            >> pt.text.sliding(length=128, stride=64, prepend_attr=None, tokenizer=tok)
            >> pt.text.scorer(wmodel="DPH")
            >> pt.text.max_passage() )
    """

    # deal with older names for attributes
    if 'passage_length' in kwargs:
        length = kwargs['passage_length']
        del kwargs['passage_length']
        warn(
            "passage_length should be length.", FutureWarning, 2)
    if 'passage_stride' in kwargs:
        stride = kwargs['passage_stride']
        del kwargs['passage_stride']
        warn(
            "passage_stride should be stride.", FutureWarning, 2)
    if 'prepend_title' in kwargs:
        warn(
            "prepend_title and title_attr should be replaced with prepend_attr.", FutureWarning, 2)
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
        tokenizer=tokenizer,
        **kwargs
    )

def max_passage() -> pt.Transformer:
    """
    Scores each document based on the maximum score of any constituent passage. Applied after a sliding window transformation
    has been scored.
    """
    return MaxPassage()

def mean_passage() -> pt.Transformer:
    """
    Scores each document based on the mean score of all constituent passages. Applied after a sliding window transformation
    has been scored.
    """
    return MeanPassage()

def first_passage() -> pt.Transformer:
    """
    Scores each document based on score of the first passage of that document. Note that this transformer is rarely used in conjunction with
    the sliding window transformer, as all passages would required to be scored, only for the first one to be used.
    """
    return FirstPassage()

def kmaxavg_passage(k : int) -> pt.Transformer:
    """
    Scores each document based on the average score of the top scoring k passages. Generalises combination of mean_passage()
    and max_passage(). Proposed in [Chen2020].

    :param k: The number of top-scored passages for each document to use when scoring
    
    """
    return KMaxAvgPassage(k)

def slidingWindow(sequence : list, winSize : int, step : int) -> list:
    """
    For the specified sequence, break into sliding windows of size winSize, 
    stepping forward by the specified amount each time

    :param sequence: the sequence to break into sliding windows
    :param winSize: the size of the sliding window
    :param step: how much to step forward each time
    :return: a list of sliding windows, where each window is a list of the elements in that window
    :rtype: list
    """
    return [x for x in list(more_itertools.windowed(sequence,n=winSize, step=step)) if x[-1] is not None]

def snippets(
        text_scorer_pipe : pt.Transformer, 
        text_attr : str = "text", 
        summary_attr : str = "summary", 
        num_psgs : int = 5, 
        joinstr : str ='...') -> pt.Transformer:
    """
    Applies query-biased summarisation (snippet), by applying the specified text scoring pipeline.
    Takes a return a dataframe with the columns ['qid', 'query', 'docno', text_attr], and returns a dataframe
    with the columns ['qid', 'query', 'docno', text_attr, summary_attr]. The summary_attr column contains the
    query-biased summary for that document, upto num_psgs passages, joined together with the specified joinstr.

    :param text_scorer_pipe: the pipeline for scoring passages in response to the query. Normally this applies passaging.
        The pipeline should take a dataframe with the columns ['qid', 'query', 'docno', text_attr] and return a dataframe
        with the columns ['qid', 'query', 'docno', text_attr, 'score', 'rank'], where these are smaller passages than the input df.
    :param text_attr: what is the name of the attribute that contains the text of the document
    :param summary_attr: what is the name of the attribute that should contain the query-biased summary for that document
    :param num_psgs: how many passages to select for the summary of each document
    :param joinstr: how to join passages for a given document together

    Example::

        # retrieve documents with text
        br = pt.terrier.Retriever(index, metadata=['docno', 'text'])

        # use Tf as a passage scorer on sliding window passages 
        psg_scorer = ( 
            pt.text.sliding(text_attr='text', length=15, prepend_attr=None) 
            >> pt.text.scorer(body_attr="text", wmodel='Tf', takes='docs')
        )
        
        # use psg_scorer for performing query-biased summarisation on docs retrieved by br 
        retr_pipe = br >> pt.text.snippets(psg_scorer)

    """
    tsp = (
        pt.apply.rename({'qid' : 'oldqid'}) 
        >> pt.apply.qid(lambda row: row['oldqid'] + '-' + row['docno']) 
        >> ( text_scorer_pipe % num_psgs )
        >> pt.apply.qid(drop=True)
        >> pt.apply.rename({'oldqid' : 'qid'})
    )

    def _qbsjoin(docres):
        if len(docres) == 0:
            docres[summary_attr] = pd.Series(dtype='str')
            return docres     

        psgres = tsp(docres)
        if len(psgres) == 0:
            print('no passages found in %d documents for query %s' % (len(docres), docres.iloc[0].query))
            docres = docres.copy()
            docres[summary_attr]  = ""
            return docres

        psgres[["olddocno", "pid"]] = psgres.docno.str.split("%p", expand=True)

        newdf = psgres.groupby(['qid', 'olddocno'])[text_attr].agg(joinstr.join).reset_index().rename(columns={text_attr : summary_attr, 'olddocno' : 'docno'})
        
        return docres.merge(newdf, on=['qid', 'docno'], how='left')
    rtr = pt.apply.generic(_qbsjoin, required_columns=['qid', 'query', 'docno', text_attr])
    rtr.subtransformers = types.MethodType(lambda self: {'tsp' : tsp}, rtr) # type: ignore[attr-defined]
    return rtr


class DePassager(pt.Transformer):

    def __init__(self, agg="max", **kwargs):
        super().__init__(**kwargs)
        self.agg = agg

    def transform(self, topics_and_res):
        pt.validate.columns(topics_and_res, includes=['qid', 'docno'] + (['score'] if self.agg != 'first' else []))
        topics_and_res = topics_and_res.copy()
        topics_and_res[["olddocno", "pid"]] = topics_and_res.docno.str.split("%p", expand=True) if len(topics_and_res) > 0 else pd.DataFrame(columns=["olddocno", "pid"])
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
            rtr = topics_and_res.groupby(['qid', 'olddocno'])['score'].mean().reset_index().rename(columns={'olddocno' : 'docno'})
            #add query columns back
            rtr = rtr.merge(topics_and_res[pt.model.query_columns(topics_and_res)].drop_duplicates(), on='qid')

        if self.agg == 'kmaxavg':
            rtr = topics_and_res.groupby(['qid', 'olddocno'])['score'].apply(lambda ser: ser.nlargest(self.K).mean()).reset_index().rename(columns={'olddocno' : 'docno'})
            #add query columns back
            rtr = rtr.merge(topics_and_res[pt.model.query_columns(topics_and_res)].drop_duplicates(), on='qid')

        if "docid" in rtr.columns:
            rtr = rtr.drop(columns=['docid'])
        rtr = pt.model.add_ranks(rtr)
        return rtr

class KMaxAvgPassage(DePassager):
    """
        .. cite.dblp:: conf/trec/ChenHSCH020

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


class SlidingWindowPassager(pt.Transformer):
    schematic = {'label': 'SlidingWindow'}

    def __init__(self, text_attr='body', title_attr='title', passage_length=150, passage_stride=75, join=' ', prepend_title=True, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.text_attr=text_attr
        self.title_attr=title_attr
        self.passage_length = passage_length
        self.passage_stride= passage_stride
        self.join = join
        self.prepend_title = prepend_title
        self.tokenizer = tokenizer

        # check if the tokenizer has the `.tokenize()` and the `.convert_tokens_to_string()` method
        if self.tokenizer is not None:
            self.tokenize = self.tokenizer.tokenize
            self.detokenize = self.tokenizer.convert_tokens_to_string
        else:
            self.tokenize = re.compile(r"\s+").split
            self.detokenize = ' '.join

    def transform(self, topics_and_res):
        with pt.validate.any(topics_and_res) as v:
            cols = [self.text_attr] + ([self.title_attr] if self.prepend_title else [])
            v.result_frame(cols)
            v.document_frame(cols)
        print("calling sliding on df of %d rows" % len(topics_and_res))

        # now apply the passaging
        if "qid" in topics_and_res.columns: 
            return self.applyPassaging(topics_and_res, labels="label" in topics_and_res.columns)
        return self.applyPassaging_no_qid(topics_and_res)

    def applyPassaging_no_qid(self, df):
        result_columns = list(df.columns)
        if self.prepend_title:
            result_columns = [c for c in result_columns if c != self.title_attr]
        rows=[]
        for row in df.itertuples():
            row = row._asdict()
            toks = self.tokenize(row[self.text_attr])
            if len(toks) < self.passage_length:
                row['docno'] = row['docno'] + "%p0"
                row[self.text_attr] = self.detokenize(toks)
                if self.prepend_title:
                    row[self.text_attr] = str(row[self.title_attr]) + self.join + row[self.text_attr]
                    del(row[self.title_attr])
                rows.append(row)
            else:
                passageCount=0
                for i, passage in enumerate( slidingWindow(toks, self.passage_length, self.passage_stride)):
                    newRow = row.copy()
                    newRow['docno'] = row['docno'] + "%p" + str(i)
                    newRow[self.text_attr] = self.detokenize(passage)
                    if self.prepend_title:
                        newRow[self.text_attr] = str(row[self.title_attr]) + self.join + newRow[self.text_attr]
                        del(newRow[self.title_attr])
                    rows.append(newRow)
                    passageCount+=1
        return pd.DataFrame(rows, columns=result_columns)


    def applyPassaging(self, df, labels=True):
        newRows=[]
        labelCount=defaultdict(int)
        currentQid=None
        rank=0
        copy_columns=[]
        for col in ["score", "rank"]:
            if col in df.columns:
                copy_columns.append(col)

        if len(df) == 0:
            return pd.DataFrame(columns=['qid', 'query', 'docno', self.text_attr, 'score', 'rank'])
    
        with pt.tqdm('passsaging', total=len(df), desc='passaging', leave=False) as pbar:
            for index, row in df.iterrows():
                pbar.update(1)
                qid = row['qid']
                if currentQid is None or currentQid != qid:
                    rank=0
                    currentQid = qid
                rank+=1
                toks = self.tokenize(row[self.text_attr])
                if len(toks) < self.passage_length:
                    newRow = row.copy()
                    newRow['docno'] = row['docno'] + "%p0"
                    newRow[self.text_attr] = self.detokenize(toks)
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
                        newRow[self.text_attr] = self.detokenize(passage)
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
        newDF['query'] = newDF['query'].fillna('')
        newDF[self.text_attr] = newDF[self.text_attr].fillna('')
        newDF['qid'] = newDF['qid'].fillna('')
        newDF.reset_index(inplace=True,drop=True)
        return newDF
