"""
This file contains all the indexers.
"""
from enum import Enum
import pandas as pd
import os
import json
import tempfile
import contextlib
import threading
import select
import math
from warnings import warn
from deprecated import deprecated
from collections import deque
from typing import List, Dict, Union
import more_itertools
import pyterrier as pt
from pyterrier.terrier.stemmer import TerrierStemmer
from pyterrier.terrier.tokeniser import TerrierTokeniser
from pyterrier.terrier.stopwords import TerrierStopwords


# These classes are only defined after pt.java.init() in pyterrier.terrier.java._post_init
DocListIterator = None
PythonListIterator = None
FlatJSONDocumentIterator = None
TQDMCollection = None
TQDMSizeCollection = None

# for backward compatibility
class IterDictIndexerBase(pt.Indexer):
    @deprecated(version="0.9", reason="Use pt.Indexer instead of IterDictIndexerBase")
    def __init__(self, *args, **kwargs):
        super(pt.Indexer, self).__init__(*args, **kwargs)

# lastdoc ensures that a Document instance from a Collection is not GCd before Java has used it.
lastdoc = None

type_to_class = {
    'trec' : 'org.terrier.indexing.TRECCollection',
    'trecweb' : 'org.terrier.indexing.TRECWebCollection',
    'warc' : 'org.terrier.indexing.WARC10Collection'
}

@pt.java.required
def createCollection(files_path : List[str], coll_type : str = 'trec', props = {}):
    if coll_type in type_to_class:
        collectionClzName = type_to_class[coll_type]
    else:
        collectionClzName = coll_type
    collectionClzName = collectionClzName.split(",")
    _props = pt.java.J.HashMap()
    for k, v in props.items():
        _props[k] = v
    pt.terrier.J.ApplicationSetup.getProperties().putAll(_props)
    cls_string = pt.java.J.String._class
    cls_list = pt.java.J.List._class
    if len(files_path) == 0:
        raise ValueError("list files_path cannot be empty")
    asList = createAsList(files_path)
    colObj = pt.terrier.J.CollectionFactory.loadCollections(
        collectionClzName,
        [cls_list, cls_string, cls_string, cls_string],
        [asList, pt.terrier.J.TagSet.TREC_DOC_TAGS, "", ""])
    return colObj


def treccollection2textgen(
        files : List[str], 
        meta : List[str] = ["docno"], 
        meta_tags : Dict[str,str] = {"text":"ELSE"}, 
        verbose = False,
        num_docs = None,
        tag_text_length : int = 4096):
    """
    Creates a generator of dictionaries on parsing TREC formatted files. This is useful 
    for parsing TREC-formatted corpora in indexers like IterDictIndexer, or similar 
    indexers in other plugins (e.g. ColBERTIndexer).

    Arguments:
     - files(List[str]): list of files to parse in TREC format.
     - meta(List[str]): list of attributes to expose in the dictionaries as metadata.
     - meta_tags(Dict[str,str]): mapping of TREC tags as metadata.
     - tag_text_length(int): maximium length of metadata. Defaults to 4096.
     - verbose(bool): set to true to show a TQDM progress bar. Defaults to True.
     - num_docs(int): a hint for TQDM to size the progress bar based on document counts rather than file count.

    Example::

        files = pt.io.find_files("/path/to/Disk45")
        gen = pt.index.treccollection2textgen(files)
        index = pt.IterDictIndexer("./index45").index(gen)

    """

    props = {
        "TrecDocTags.doctag": "DOC",
        "TrecDocTags.idtag": "DOCNO",
        "TrecDocTags.skip": "DOCHDR",
        "TrecDocTags.casesensitive": "false",
        # Should the tags from which we create abstracts be case-sensitive?
        'TaggedDocument.abstracts.tags.casesensitive':'false'
    }
    props['TaggedDocument.abstracts'] = ','.join(meta_tags.keys())
    # The tags from which to save the text. ELSE is special tag name, which means anything not consumed by other tags.
    props['TaggedDocument.abstracts.tags'] = ','.join(meta_tags.values())
    # The max lengths of the abstracts. Abstracts will be truncated to this length. Defaults to empty.
    props['TaggedDocument.abstracts.lengths'] = ','.join([str(tag_text_length)] * len(meta_tags) )  

    collection = createCollection(files, props=props)
    if verbose:
        if num_docs is not None:
            collection = TQDMSizeCollection(collection, num_docs)
        else:
            collection = TQDMCollection(collection)
    while collection.nextDocument():
        d = collection.getDocument()
        while not d.endOfDocument():
            d.getNextTerm()
        rtr = {}
        for k in meta:
            rtr[k] = d.getProperty(k)
        for k in meta_tags:
            rtr[k] = d.getProperty(k)
        yield rtr
    

@pt.java.required
def _TaggedDocumentSetup(
        meta : Dict[str,int],  #mapping from meta-key to length
        meta_tags : Dict[str,str] #mapping from meta-key to tag
    ):
    """
    Property setup for TaggedDocument etc to generate abstracts
    """

    abstract_tags=meta_tags.values()
    abstract_names=meta_tags.keys()
    abstract_lengths=[str(meta[name]) for name in abstract_names]

    pt.terrier.J.ApplicationSetup.setProperty("TaggedDocument.abstracts", ",".join(abstract_names))
    # The tags from which to save the text. ELSE is special tag name, which means anything not consumed by other tags.
    pt.terrier.J.ApplicationSetup.setProperty("TaggedDocument.abstracts.tags", ",".join(abstract_tags))
    # The max lengths of the abstracts. Abstracts will be truncated to this length. Defaults to empty.
    pt.terrier.J.ApplicationSetup.setProperty("TaggedDocument.abstracts.lengths", ",".join(abstract_lengths))
    # Should the tags from which we create abstracts be case-sensitive
    pt.terrier.J.ApplicationSetup.setProperty("TaggedDocument.abstracts.tags.casesensitive", "false")


@pt.java.required
def _FileDocumentSetup(   
        meta : Dict[str,int],  #mapping from meta-key to length
        meta_tags : Dict[str,str] #mapping from meta-key to tag
    ):
    """
    Property setup for FileDocument etc to generate abstracts
    """

    meta_name_for_abstract = None
    for k, v in meta_tags.items():
        if v == 'ELSE':
            meta_name_for_abstract = k
    if meta_name_for_abstract is None:
        return
    
    if meta_name_for_abstract not in meta:
        raise ValueError("You need to specify a meta length for " + meta_name_for_abstract)

    abstract_length = meta[meta_name_for_abstract]

    pt.terrier.J.ApplicationSetup.setProperty("FileDocument.abstract", meta_name_for_abstract)
    pt.terrier.J.ApplicationSetup.setProperty("FileDocument.abstract.length", str(abstract_length))



@pt.java.required
def createAsList(files_path : Union[str, List[str]]):
    """
    Helper method to be used by child indexers to add files to Java List
    Returns:
        Created Java List
    """
    if isinstance(files_path, str):
        asList = pt.java.J.Arrays.asList(files_path)
    elif isinstance(files_path, list):
        asList = pt.java.J.Arrays.asList(*files_path)
    else:
        raise ValueError(f"{files_path}: {type(files_path)} must be a List[str] or str")
    return asList

# Using enum class create enumerations
class IndexingType(Enum):
    """
        This enum is used to determine the type of index built by Terrier. The default is CLASSIC. For more information,
        see the relevant Terrier `indexer <https://terrier-core.readthedocs.io/en/latest/indexer_details.html>`_
        and `realtime <https://terrier-core.readthedocs.io/en/latest/realtime_indices.html>`_ documentation.
    """
    CLASSIC = 1 #: A classical indexing regime, which also creates a direct index structure, useful for query expansion
    SINGLEPASS = 2 #: A single-pass indexing regime, which builds an inverted index directly. No direct index structure is created. Typically is faster than classical indexing.
    MEMORY = 3 #: An in-memory index. No persistent index is created.


@pt.java.required
class TerrierIndexer:
    """
    This is the super class for all of the Terrier-based indexers exposed by PyTerrier. It hosts common configuration
    for all index types.
    
    """

    default_properties = {
            "TrecDocTags.doctag": "DOC",
            "TrecDocTags.idtag": "DOCNO",
            "TrecDocTags.skip": "DOCHDR",
            "TrecDocTags.casesensitive": "false",
            "trec.collection.class": "TRECCollection",
    }

    def __init__(self, index_path : str, *args, 
            blocks : bool = False, 
            overwrite: bool = False, 
            verbose : bool = False, 
            meta_reverse : List[str] = ["docno"],
            stemmer : Union[None, str, TerrierStemmer] = TerrierStemmer.porter,
            stopwords : Union[None, TerrierStopwords, List[str]] = TerrierStopwords.terrier,
            tokeniser : Union[str,TerrierTokeniser] = TerrierTokeniser.english,
            type=IndexingType.CLASSIC,
            properties : Dict[str,str] = {}
            ):
        """
        Constructor called by all indexer subclasses. All arguments listed below are available in 
        IterDictIndexer, DFIndexer, TRECCollectionIndexer and FilesIndsexer. 

        Args:
            index_path (str): Directory to store index. Ignored for IndexingType.MEMORY.
            blocks (bool): Create indexer with blocks if true, else without blocks. Default is False.
            overwrite (bool): If index already present at `index_path`, True would overwrite it, False throws an Exception. Default is False.
            verbose (bool): Provide progess bars if possible. Default is False.
            stemmer (TerrierStemmer): the stemmer to apply. Default is ``TerrierStemmer.porter``.
            stopwords (TerrierStopwords): the stopwords list to apply. Default is ``TerrierStemmer.terrier``.
            tokeniser (TerrierTokeniser): the stemmer to apply. Default is ``TerrierTokeniser.english``.
            type (IndexingType): the specific indexing procedure to use. Default is ``IndexingType.CLASSIC``.
            properties (dict): Terrier properties that you wish to overrride.
        """
        if type is IndexingType.MEMORY:
            self.path = None
        else:
            self.path = os.path.join(index_path, "data.properties")
            if not os.path.isdir(index_path):
                os.makedirs(index_path)
        self.index_called = False
        self.index_dir = index_path
        self.blocks = blocks
        self.type = type
        self.stemmer = TerrierStemmer._to_obj(stemmer)
        self.stopwords, self.stopword_list = TerrierStopwords._to_obj(stopwords)
        self.tokeniser = TerrierTokeniser._to_obj(tokeniser)
        self.properties = pt.java.J.Properties()
        self.setProperties(**self.default_properties)
        for k,v in properties.items():
            self.properties[k] = v        
        self.overwrite = overwrite
        self.verbose = verbose
        self.meta_reverse = meta_reverse
        self.cleanup_hooks = []

    def setProperty(self, k, v):
        """
        Set the named property to the specified value.

        Args:
            k(str): name of the Terrier property
            v(str): value of the Terrier property

        Usage::
            indexer.setProperty("termpipelines", "")
        """
        self.properties.put(str(k), str(v))

    def setProperties(self, **kwargs):
        """
        Set the properties to the given ones.

        Args:
            **kwargs: Properties to set to.

        Usage:
            >>> setProperties(**{property1:value1, property2:value2})
        """
        for key, value in kwargs.items():
            self.properties.put(str(key), str(value))

    def checkIndexExists(self):
        """
        Check if index exists at the `path` given when object was created
        """
        if self.path is None:
            return
        if os.path.isfile(self.path):
            if not self.overwrite:
                raise ValueError("Index already exists at " + self.path)
        if self.index_called:
            raise Exception("Index method can be called only once")

    def createIndexer(self):
        """
        Checks `self.type` and
        - if false, check `blocks` and create a BlockIndexer if true, else create BasicIndexer
        - if true, check `blocks` and create a BlockSinglePassIndexer if true, else create BasicSinglePassIndexer
        Returns:
            Created  Java object extending org.terrier.structures.indexing.Indexer
        """
        
        Indexer, _ = self.indexerAndMergerClasses()
        if Indexer is pt.terrier.J.BasicMemoryIndexer:
            index = Indexer()
        else:
            index = Indexer(self.index_dir, "data")
        assert index is not None
        return index

    def indexerAndMergerClasses(self):
        """
        Check `single_pass` and
        - if false, check `blocks` and create a BlockIndexer if true, else create BasicIndexer
        - if true, check `blocks` and create a BlockSinglePassIndexer if true, else create BasicSinglePassIndexer
        Returns:
            type objects for indexer and merger for the given configuration
        """

        # configure the meta index
        self.properties['indexer.meta.forward.keys'] = ','.join(self.meta.keys())
        self.properties['indexer.meta.forward.keylens'] = ','.join([str(le) for le in self.meta.values()])
        self.properties['indexer.meta.reverse.keys'] = ','.join(self.meta_reverse)

        # configure the term pipeline
        if 'termpipelines' in self.properties:
            # use existing configuration if present
            warn(
                "Setting of termpipelines property directly is deprecated", stacklevel=4, category=DeprecationWarning)
        else:
            
            termpipeline = []
            TerrierStopwords._indexing_config(self.stopwords, self.stopword_list, termpipeline, self.properties, self.cleanup_hooks)

            stemmer_clz = TerrierStemmer._to_class(self.stemmer)
            if stemmer_clz is not None:
                termpipeline.append(stemmer_clz)
            
            self.properties['termpipelines'] = ','.join(termpipeline)

        if "tokeniser" in self.properties:
            warn(
                "Setting of tokeniser property directly is deprecated", stacklevel=4, category=DeprecationWarning)
        else:
            self.properties['tokeniser'] = TerrierTokeniser._to_class(self.tokeniser)

        # inform terrier of all properties
        pt.terrier.J.ApplicationSetup.getProperties().putAll(self.properties)

        # now create the indexers 
        if self.type is IndexingType.SINGLEPASS:
            if self.blocks:
                Indexer = pt.terrier.J.BlockSinglePassIndexer
                Merger = pt.terrier.J.BlockStructureMerger
            else:
                Indexer = pt.terrier.J.BasicSinglePassIndexer
                Merger = pt.terrier.J.StructureMerger
        elif self.type is IndexingType.CLASSIC:
            if self.blocks:
                Indexer = pt.terrier.J.BlockIndexer
                Merger = pt.terrier.J.BlockStructureMerger
            else:
                Indexer = pt.terrier.J.BasicIndexer
                Merger = pt.terrier.J.StructureMerger
        elif self.type is IndexingType.MEMORY:
            if self.blocks:
                raise Exception("Memory indexing with positions not yet implemented")
            else:
                Indexer = pt.terrier.J.BasicMemoryIndexer
                Merger = None
        else:
            raise Exception("Unknown indexer type")
        assert Indexer is not None
        return Indexer, Merger

    def getIndexStats(self):
        """
        Prints the index statistics

        Note:
            Does not work with notebooks at the moment
        """
        pt.terrier.J.CLITool.main(["indexstats", "-I" + self.path])

    def getIndexUtil(self, util):
        """
        Utilities for displaying the content of an index

        Note:
            Does not work with notebooks at the moment

        Args:
            util: which util to print

        Possible Utils:
            util(string): possible values:
            printbitentry
            printlex
            printlist
            printlistentry
            printmeta
            printposting
            printpostingfile
            printterm
        """
        if not util.startswith("-"):
            util = "-" + util
        pt.terrier.J.CLITool.main(["indexutil", "-I" + self.path, util])


class DFIndexUtils:

    @staticmethod
    def get_column_lengths(df):
        meta2len = dict([(v, df[v].apply(lambda r: len(str(r)) if r is not None else 0).max())for v in df.columns.values])
        # nan values can arise if df is empty. Here we take a metalength of 1 instead.
        meta2len = {k : 1 if math.isnan(le) else le for k, le in meta2len.items()}
        return meta2len

    @staticmethod
    @pt.java.required
    def create_javaDocIterator(text, *args, **kwargs):
        HashMap = pt.java.J.HashMap
        TaggedDocument = pt.terrier.J.TaggedDocument
        StringReader = pt.java.J.StringReader
        tokeniser = pt.terrier.J.Tokeniser.getTokeniser()

        all_metadata = {}
        for i, arg in enumerate(args):
            if isinstance(arg, pd.Series):
                all_metadata[arg.name] = arg
                assert len(arg) == len(text), "Length of metadata arguments needs to be equal to length of text argument"
            elif isinstance(arg, pd.DataFrame):
                for name, column in arg.items():
                    all_metadata[name] = column
                    assert len(column) == len(text), "Length of metadata arguments needs to be equal to length of text argument"
            else:
                raise ValueError(f"Non-keyword args need to be of type pandas.Series or pandas.DataFrame, argument {i} was {type(arg)}")
        for key, value in kwargs.items():
            if isinstance(value, (pd.Series, list, tuple)):
                all_metadata[key] = value
                assert len(value) == len(value), "Length of metadata arguments needs to be equal to length of text argument"
            elif isinstance(value, pd.DataFrame):
                for name, column in arg.items():
                    all_metadata[name] = column
                    assert len(column) == len(text), "Length of metadata arguments needs to be equal to length of text argument"
            else:
                raise ValueError("Keyword kwargs need to be of type pandas.Series, list or tuple")

        if "docno" not in all_metadata:
            raise ValueError('No docno column specified, while PyTerrier assumes a docno should exist. Found meta columns were %s' % str(list(all_metadata.keys())))
        # this method creates the documents as and when needed.
        def convertDoc(text_row, meta_column):
            if text_row is None:
                text_row = ""
            hashmap = HashMap()
            for column, value in meta_column[1].items():
                if value is None:
                    value = ""
                hashmap.put(column, value)
            return TaggedDocument(StringReader(text_row), hashmap, tokeniser)
            
        df = pd.DataFrame.from_dict(all_metadata, orient="columns")
        lengths = DFIndexUtils.get_column_lengths(df)
        return (
            PythonListIterator(
                text.values,
                df.iterrows(),
                convertDoc,
                len(text.values),
                ),
            lengths)

@deprecated(version='0.11.0', reason="use pt.terrier.IterDictIndexer().index(dataframe.to_dict(orient='records')) instead")
class DFIndexer(TerrierIndexer):
    """
    Use this Indexer if you wish to index a pandas.Dataframe

    """
    @pt.java.required
    def index(self, text, *args, **kwargs):
        """
        Index the specified

        Args:
            text(pd.Series): A pandas.Series(a column) where each row is the body of text for each document
            *args: Either a pandas.Dataframe or pandas.Series.
                If a Dataframe: All columns(including text) will be passed as metadata
                If a Series: The Series name will be the name of the metadata field and the body will be the metadata content
            **kwargs: Either a list, a tuple or a pandas.Series
                The name of the keyword argument will be the name of the metadata field and the keyword argument contents will be the metadata content
        """
        self.checkIndexExists()
        # we need to prevent collectionIterator from being GCd, so assign to a variable that outlives the indexer
        collectionIterator, meta_lengths = DFIndexUtils.create_javaDocIterator(text, *args, **kwargs)
        
        # record the metadata key names and the length of the values
        self.meta = meta_lengths
        
        # make a Collection class for Terrier
        javaDocCollection = pt.terrier.J.CollectionFromDocumentIterator(collectionIterator)
        if self.verbose:
            javaDocCollection = TQDMSizeCollection(javaDocCollection, len(text)) 
        index = self.createIndexer()
        index.index(pt.terrier.J.PTUtils.makeCollection(javaDocCollection))
        global lastdoc
        lastdoc = None
        javaDocCollection.close()
        self.index_called = True
        collectionIterator = None

        for hook in self.cleanup_hooks:
            hook(self, index.getIndex())

        if self.type is IndexingType.MEMORY:
            return index.getIndex().getIndexRef()
        return pt.terrier.J.IndexRef.of(self.index_dir + "/data.properties")


class _BaseIterDictIndexer(TerrierIndexer, pt.Indexer):
    def __init__(self, index_path, *args, meta = {'docno' : 20}, meta_reverse=['docno'], pretokenised=False, threads=1, **kwargs):
        """
        
        Args:
            index_path(str): Directory to store index. Ignored for IndexingType.MEMORY.
            meta(Dict[str,int]): What metadata for each document to record in the index, and what length to reserve. Metadata fields will be truncated to this length. Defaults to `{"docno" : 20}`.
            meta_reverse(List[str]): What metadata should we be able to resolve back to a docid. Defaults to `["docno"]`,      
        """
        pt.Indexer.__init__(self)
        TerrierIndexer.__init__(self, index_path, *args, **kwargs)
        self.threads = threads
        self.meta = meta
        self.meta_reverse = meta_reverse
        self.pretokenised = pretokenised
        if self.pretokenised:
            assert pt.terrier.check_version(5.7), "Terrier too old, this requires 5.7"
            # we disable stemming and stopwords for pretokenised indices
            self.stemmer = None
            self.stopwords = None

    def _setup(self, fields, meta, meta_lengths):
        """
        Index the specified iter of dicts with the (optional) specified fields

        Args:
            it(iter[dict]): an iter of document dict to be indexed
            fields(list[str]): keys to be indexed as fields
            meta(dict[str,int]): keys to be considered as metdata, and their lengths
            meta_lengths(list[int]): deprecated
        """
        self.checkIndexExists()
        if isinstance(meta, dict):
            self.meta = meta
        else: 
            if meta_lengths is None:
                # the ramifications of setting all lengths to a large value is an overhead in memory usage during decompression
                # also increased reverse lookup file if reverse meta lookups are enabled.
                meta_lengths = ['512'] * len(meta)
            self.meta = { k:v for k,v in zip( meta, meta_lengths)}

        if self.pretokenised:
            self.setProperties(**{
                'metaindex.compressed.crop.long' : 'true',
                'FieldTags.process': '',
                'FieldTags.casesensitive': 'true',
            })
        else:
            self.setProperties(**{
                'metaindex.compressed.crop.long' : 'true',
                'FieldTags.process': ','.join(fields),
                'FieldTags.casesensitive': 'true',
            })

    def _filter_iterable(self, it, indexed_fields):
        # Only include necessary fields: those that are indexed, metadata fields, and docno
        # Also, check that the provided iterator is a suitable format

        if self.pretokenised:
            all_fields = {'docno', "toks"} | set(self.meta.keys())
        else:
            all_fields = {'docno'} | set(indexed_fields) | set(self.meta.keys())

        first_docs, it = more_itertools.spy(it) # peek at the first document and validate it
        if len(first_docs) > 0: # handle empty input
            self._validate_doc_dict(first_docs[0])

        # important: return an iterator (i.e. using a generator expression) here, rather than make this 
        # function a generator, to be sure that the validation above happens when  _filter_iterable is 
        # called, rather than on the first invocation of next()
        return ({f: doc[f] for f in all_fields} for doc in it)

    def _is_dict(self, obj):
        return hasattr(obj, '__getitem__') and hasattr(obj, 'items')

    def _validate_doc_dict(self, obj):
        """
        Raise errors/warnings for common indexing mistakes
        """
        if not self._is_dict(obj):
            raise ValueError("Was passed %s while expected dict-like object" % (str(type(obj))))
        if self.meta is not None:
            for k in self.meta:
                if k not in obj:
                    raise ValueError(f"Indexing meta key {k} not found in first document (keys {list(obj.keys())})")
                if len(obj[k]) > int(self.meta[k]):
                    msg = f"Indexing meta key {k} length requested {self.meta[k]} but exceeded in first document (actual length {len(obj[k])}). " + \
                          f"Increase the length in the meta dict for the indexer, e.g., pt.IterDictIndexer(..., meta={ {k: len(obj[k])} })."
                    if k == 'docno':
                        # docnos that are truncated will cause major issues; raise an error
                        raise ValueError(msg)
                    else:
                        # Other fields may not matter as much; just show a warning
                        warn(msg)


class _IterDictIndexer_nofifo(_BaseIterDictIndexer):
    """
    Use this Indexer if you wish to index an iter of dicts (possibly with multiple fields).
    This version is used for Windows -- which doesn't support the faster fifo implementation.
    """
    @pt.java.required
    def index(self, it, fields=('text',), meta=None, meta_lengths=None, threads=None):
        """
        Index the specified iter of dicts with the (optional) specified fields

        Args:
            it(iter[dict]): an iter of document dict to be indexed
            fields(list[str]): keys to be indexed as fields
            meta(list[str]): keys to be considered as metdata. Deprecated
            meta_lengths(list[int]): length of metadata, defaults to 512 characters. Deprecated
        """
        if meta is not None:
            warn(
                'specifying meta and meta_lengths in IterDictIndexer.index() is deprecated, use kwargs in constructor instead', DeprecationWarning, 2)
            self.meta = meta
            if meta_lengths is not None:
                self.meta = {zip(meta, meta_lengths)}

        self._setup(fields, self.meta, None)
        assert self.threads == 1, 'IterDictIndexer does not support multiple threads on Windows'

        indexer = self.createIndexer()
        if self.pretokenised:
            assert not self.blocks, "pretokenised isnt compatible with blocks"

            # we generate DocumentPostingList from a dictionary of pretokenised text, e.g.
            # [
            #     {'docno' : 'd1', 'toks' : {'a' : 1, 'aa' : 2}}
            # ]
            
            iter_docs = DocListIterator(self._filter_iterable(it, fields))
            self.index_called = True
            indexer.indexDocuments(iter_docs)
            iter_docs = None

        else:

            # we need to prevent collectionIterator from being GCd
            collectionIterator = FlatJSONDocumentIterator(self._filter_iterable(it, fields))
            javaDocCollection = pt.terrier.J.CollectionFromDocumentIterator(collectionIterator)
            # remove once 5.7 is now the minimum version
            indexer.index(javaDocCollection if pt.terrier.check_version("5.7") else [javaDocCollection])
            global lastdoc
            lastdoc = None
            self.index_called = True
            collectionIterator = None

        indexref = None
        if self.type is IndexingType.MEMORY:
            index = indexer.getIndex()
            indexref = index.getIndexRef()
        else:
            indexref = pt.terrier.J.IndexRef.of(self.index_dir + "/data.properties")
            if len(self.cleanup_hooks) > 0:
                sindex = pt.terrier.J.Index
                sindex.setIndexLoadingProfileAsRetrieval(False)
                index = pt.terrier.IndexFactory.of(indexref)
                for hook in self.cleanup_hooks:
                    hook(self, index)
                sindex.setIndexLoadingProfileAsRetrieval(True)

        return indexref


class _IterDictIndexer_fifo(_BaseIterDictIndexer):
    """
    Use this Indexer if you wish to index an iter of dicts (possibly with multiple fields).
    This version is optimized by using multiple threads and POSIX fifos to tranfer data,
    which ends up being much faster.
    """
    @pt.java.required
    def index(self, it, fields=('text',), meta=None, meta_lengths=None):
        """
        Index the specified iter of dicts with the (optional) specified fields

        Args:
            it(iter[dict]): an iter of document dict to be indexed
            fields(list[str]): keys to be indexed as fields
            meta(list[str]): keys to be considered as metdata
            meta_lengths(list[int]): length of metadata, defaults to 512 characters
        """
        CollectionFromDocumentIterator = pt.terrier.J.CollectionFromDocumentIterator
        JsonlDocumentIterator = pt.terrier.J.JsonlDocumentIterator
        if self.pretokenised:
            JsonlTokenisedIterator = pt.terrier.J.JsonlPretokenisedIterator
        ParallelIndexer = pt.terrier.J.ParallelIndexer

        if meta is not None:
            warn(
                'specifying meta and meta_lengths in IterDictIndexer.index() is deprecated, use constructor instead', DeprecationWarning, 2)
            self.meta = meta
            if meta_lengths is not None:
                self.meta = {zip(meta, meta_lengths)}

        self._setup(fields, self.meta, None)

        os.makedirs(self.index_dir, exist_ok=True) # ParallelIndexer expects the directory to exist

        Indexer, Merger = self.indexerAndMergerClasses()

        assert self.threads > 0, "threads must be positive"
        if Indexer is pt.terrier.J.BasicMemoryIndexer:
            assert self.threads == 1, 'IterDictIndexer does not support multiple threads for IndexingType.MEMORY'
        if self.threads > 1:
            warn(
                'Using multiple threads results in a non-deterministic ordering of document in the index. For deterministic behavior, use threads=1')

        # Document iterator
        fifos = []
        j_collections = []
        with tempfile.TemporaryDirectory() as d:
            # Make a POSIX FIFO with associated java collection for each thread to use
            for i in range(self.threads):
                fifo = f'{d}/docs-{i}.jsonl'
                os.mkfifo(fifo)
                if (self.pretokenised):
                    j_collections.append(JsonlTokenisedIterator(fifo))
                else:
                    j_collections.append(CollectionFromDocumentIterator(JsonlDocumentIterator(fifo)))
                fifos.append(fifo)

            # Start dishing out the docs to the fifos
            threading.Thread(target=self._write_fifos, args=(self._filter_iterable(it, fields), fifos), daemon=True).start()

            # Different process for memory indexer (still taking advantage of faster fifos)
            if Indexer is pt.terrier.J.BasicMemoryIndexer:
                indexer = Indexer()
                if self.pretokenised:
                    indexer.indexDocuments(j_collections)
                else:
                    indexer.index(j_collections)
                for hook in self.cleanup_hooks:
                    hook(self, indexer.getIndex())
                return indexer.getIndex().getIndexRef()

            # Start the indexing threads
            if self.pretokenised:
                ParallelIndexer.buildParallelTokenised(j_collections, self.index_dir, Indexer, Merger)
            else:
                ParallelIndexer.buildParallel(j_collections, self.index_dir, Indexer, Merger)
            
        indexref = None
        indexref = pt.terrier.J.IndexRef.of(self.index_dir + "/data.properties")
        
        if len(self.cleanup_hooks) > 0:
            sindex = pt.terrier.J.Index
            sindex.setIndexLoadingProfileAsRetrieval(False)
            index = pt.terrier.IndexFactory.of(indexref)
            sindex.setIndexLoadingProfileAsRetrieval(True)

            for hook in self.cleanup_hooks:
                hook(self, index)

        return indexref

    def _write_fifos(self, it, fifos):
        with contextlib.ExitStack() as stack:
            fifos = [stack.enter_context(open(f, 'wt')) for f in fifos]
            ready = None
            for doc in it:
                if not ready: # either first iteration or deque is empty
                    if len(fifos) > 1:
                        # Not all the fifos may be ready yet for the next document. Rather than
                        # witing for the next one to finish up, go ahead and can check wich are ready
                        # with the select syscall. This will block until at least one is ready. This
                        # optimization can actually have a pretty big impact-- on CORD19, indexing
                        # with 8 threads was 30% faster with this.
                        _, ready, _ = select.select([], fifos, [])
                        ready = deque(ready)
                    else:
                        # single threaded mode
                        ready = deque(fifos)
                fifo = ready.popleft()
                if 'toks' in doc:
                    doc['toks'] = {k: int(v) for k, v in doc['toks'].items()} # cast all values as ints
                json.dump(doc, fifo)
                fifo.write('\n')


# Windows doesn't support fifos -- so we have 2 versions.
# Choose which one to expose based on whether os.mkfifo exists.
if hasattr(os, 'mkfifo'):
    IterDictIndexer = _IterDictIndexer_fifo
else:
    IterDictIndexer = _IterDictIndexer_nofifo
IterDictIndexer.__name__ = 'IterDictIndexer' # trick sphinx into not using "alias of"


class TRECCollectionIndexer(TerrierIndexer):


    """
        Use this Indexer if you wish to index a TREC formatted collection
    """

    def __init__(self, 
            index_path : str, 
            collection : str = "trec", 
            verbose : bool = False,
            meta : Dict[str,int] = {"docno" : 20},
            meta_reverse : List[str] = ["docno"],
            meta_tags : Dict[str,str] = {},
            **kwargs
            ):
        """
        Init method

        Args:
            index_path (str): Directory to store index. Ignored for IndexingType.MEMORY.
            blocks (bool): Create indexer with blocks if true, else without blocks. Default is False.
            overwrite (bool): If index already present at `index_path`, True would overwrite it, False throws an Exception. Default is False.
            type (IndexingType): the specific indexing procedure to use. Default is IndexingType.CLASSIC.
            collection (Class name, or Class instance, or one of "trec", "trecweb", "warc"). Default is "trec".
            meta(Dict[str,int]): What metadata for each document to record in the index, and what length to reserve. Metadata fields will be truncated to this length. Defaults to `{"docno" : 20}`.
            meta_reverse(List[str]): What metadata shoudl we be able to resolve back to a docid. Defaults to `["docno"]`.
            meta_tags(Dict[str,str]): For collections formed using tagged data (e.g. HTML), which tags correspond to which metadata. This is useful for recording the text of documents for use in neural rankers - see :ref:`pt.text`.

        """
        super().__init__(index_path, **kwargs)
        if isinstance(collection, str):
            if collection in type_to_class:
                collection = type_to_class[collection]
        self.collection = collection
        self.verbose = verbose
        self.meta = meta
        self.meta_reverse = meta_reverse
        self.meta_tags = meta_tags
    
    @pt.java.required
    def index(self, files_path : Union[str,List[str]]):
        """
        Index the specified TREC formatted files

        Args:
            files_path: can be a String of the path or a list of Strings of the paths for multiple files
        """
        self.checkIndexExists()
        index = self.createIndexer()
        if not isinstance(files_path, list):
            raise ValueError('files_path must be a list')

        _TaggedDocumentSetup(self.meta, self.meta_tags)

        colObj = createCollection(files_path, self.collection)
        if self.verbose and isinstance(colObj, pt.terrier.J.MultiDocumentFileCollection):
            colObj = pt.java.cast("org.terrier.indexing.MultiDocumentFileCollection", colObj)
            colObj = TQDMCollection(colObj)
        # remove once 5.7 is now the minimum version
        if pt.terrier.check_version("5.7"):
            index.index(colObj)
        else:
            index.index(pt.terrier.J.PTUtils.makeCollection(colObj))
        global lastdoc
        lastdoc = None
        colObj.close()
        self.index_called = True
        
        for hook in self.cleanup_hooks:
            hook(self, index.getIndex())

        if self.type is IndexingType.MEMORY:
            return index.getIndex().getIndexRef()
        return pt.terrier.J.IndexRef.of(self.index_dir + "/data.properties")

class FilesIndexer(TerrierIndexer):
    '''
        Use this Indexer if you wish to index a pdf, docx, txt etc files

        Args:
            index_path (str): Directory to store index. Ignored for IndexingType.MEMORY.
            blocks (bool): Create indexer with blocks if true, else without blocks. Default is False.
            type (IndexingType): the specific indexing procedure to use. Default is IndexingType.CLASSIC.
            meta(Dict[str,int]): What metadata for each document to record in the index, and what length to reserve. Metadata fields will be truncated to this length. Defaults to `{"docno" : 20, "filename" : 512}`.
            meta_reverse(List[str]): What metadata shoudl we be able to resolve back to a docid. Defaults to `["docno"]`,
            meta_tags(Dict[str,str]): For collections formed using tagged data (e.g. HTML), which tags correspond to which metadata. Defaults to empty. This is useful for recording the text of documents for use in neural rankers - see :ref:`pt.text`.

    '''

    def __init__(self, index_path, *args, meta={"docno" : 20, "filename" : 512}, meta_reverse=["docno"], meta_tags={}, **kwargs):
        super().__init__(index_path, *args, **kwargs)
        self.meta = meta
        self.meta_reverse = meta_reverse
        self.meta_tags = meta_tags

    @pt.java.required
    def index(self, files_path : Union[str,List[str]]):
        """
        Index the specified files.

        Args:
            files_path: can be a String of the path or a list of Strings of the paths for multiple files
        """
        self.checkIndexExists()
        index = self.createIndexer()
        asList = createAsList(files_path)
        _TaggedDocumentSetup(self.meta, self.meta_tags)
        _FileDocumentSetup(self.meta, self.meta_tags)
        
        simpleColl = pt.terrier.J.SimpleFileCollection(asList, False)
        # remove once 5.7 is now the minimum version
        index.index(simpleColl if pt.terrier.check_version("5.7") else [simpleColl])
        global lastdoc
        lastdoc = None
        self.index_called = True

        for hook in self.cleanup_hooks:
            hook(self, index.getIndex())

        if self.type is IndexingType.MEMORY:
            return index.getIndex().getIndexRef()
        return pt.terrier.J.IndexRef.of(self.index_dir + "/data.properties")
