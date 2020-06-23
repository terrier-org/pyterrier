"""
This file contains all the indexers.
"""

# from jnius import autoclass, cast, PythonJavaClass, java_method
from jnius import autoclass, PythonJavaClass, java_method, cast
# from .utils import *
import pandas as pd
# import numpy as np
import os
import enum
import json

StringReader = None
HashMap = None
TaggedDocument = None
FlatJSONDocument = None
Tokeniser = None
TRECCollection = None
SimpleFileCollection = None
BasicIndexer = None
BlockIndexer = None
Collection = None
BasicSinglePassIndexer = None
BlockSinglePassIndexer = None
Arrays = None
Array = None
ApplicationSetup = None
Properties = None
CLITool = None
IndexRef = None
IndexFactory = None

def run_autoclass():
    global StringReader
    global HashMap
    global TaggedDocument
    global FlatJSONDocument
    global Tokeniser
    global TRECCollection
    global SimpleFileCollection
    global BasicIndexer
    global BlockIndexer
    global BasicSinglePassIndexer
    global BlockSinglePassIndexer
    global Collection
    global Arrays
    global Array
    global ApplicationSetup
    global Properties
    global CLITool
    global IndexRef
    global IndexFactory

    StringReader = autoclass("java.io.StringReader")
    HashMap = autoclass("java.util.HashMap")
    TaggedDocument = autoclass("org.terrier.indexing.TaggedDocument")
    FlatJSONDocument = autoclass("org.terrier.indexing.FlatJSONDocument")
    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    TRECCollection = autoclass("org.terrier.indexing.TRECCollection")
    SimpleFileCollection = autoclass("org.terrier.indexing.SimpleFileCollection")
    BasicIndexer = autoclass("org.terrier.structures.indexing.classical.BasicIndexer")
    BlockIndexer = autoclass("org.terrier.structures.indexing.classical.BlockIndexer")
    BasicSinglePassIndexer = autoclass("org.terrier.structures.indexing.singlepass.BasicSinglePassIndexer")
    BlockSinglePassIndexer = autoclass("org.terrier.structures.indexing.singlepass.BlockSinglePassIndexer")
    Collection = autoclass("org.terrier.indexing.Collection")
    Arrays = autoclass("java.util.Arrays")
    Array = autoclass('java.lang.reflect.Array')
    ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')
    Properties = autoclass('java.util.Properties')
    CLITool = autoclass("org.terrier.applications.CLITool")
    IndexRef = autoclass('org.terrier.querying.IndexRef')
    IndexFactory = autoclass('org.terrier.structures.IndexFactory')


# Using enum class create enumerations
class IndexingType(enum.Enum):
    CLASSIC = 1
    SINGLEPASS = 2
    MEMORY = 3

class Indexer:
    """
    Parent class. It can be used to load an existing index.
    Use one of its children classes if you wish to create a new index.

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + /data.properties
        index_called(bool): True if index() method of child Indexer has been called, false otherwise
        index_dir(str): The index directory
        blocks(bool): If true the index has blocks enabled
        properties: A Terrier Properties object, which is a hashtable with properties and their values
        overwrite(bool): If True the index() method of child Indexer will overwrite any existing index
    """

    default_properties = {
            "TrecDocTags.doctag": "DOC",
            "TrecDocTags.idtag": "DOCNO",
            "TrecDocTags.skip": "DOCHDR",
            "TrecDocTags.casesensitive": "false",
            "trec.collection.class": "TRECCollection",
    }

    def __init__(self, index_path, *args, blocks=False, overwrite=False, type=IndexingType.CLASSIC, **kwargs):
        """
        Init method

        Args:
            index_path (str): Directory to store index
            blocks (bool): Create indexer with blocks if true, else without blocks
            overwrite (bool): If index already present at `index_path`, True would overwrite it, False throws an Exception
            type (IndexingType): the specific indexing procedure to use. Default is IndexingType.CLASSIC.
        """
        if StringReader is None:
            run_autoclass()
        self.path = os.path.join(index_path, "data.properties")
        self.index_called = False
        self.index_dir = index_path
        self.blocks = blocks
        self.type = type
        self.properties = Properties()
        self.setProperties(**self.default_properties)
        self.overwrite = overwrite
        if not os.path.isdir(index_path):
            os.makedirs(index_path)

    def setProperties(self, **kwargs):
        """
        Set the properties to the given ones.

        Args:
            **kwargs: Properties to set to.

        Usage:
            setProperties("property1=value1, property2=value2")
            or
            setProperties("**{property1:value1, property2:value2}")
        """
        for control, value in kwargs.items():
            self.properties.put(control, value)

    def checkIndexExists(self):
        """
        Check if index exists at the `path` given when object was created
        """
        if os.path.isfile(self.path):
            if not self.overwrite:
                raise ValueError("Index already exists at " + self.path)
        if self.index_called:
            raise Exception("Index method can be called only once")

    def createIndexer(self):
        """
        Check `single_pass` and
        - if false, check `blocks` and create a BlockIndexer if true, else create BasicIndexer
        - if true, check `blocks` and create a BlockSinglePassIndexer if true, else create BasicSinglePassIndexer
        Returns:
            Created index object
        """
        ApplicationSetup.getProperties().putAll(self.properties)
        # ApplicationSetup.bootstrapInitialisation(self.properties)
        if self.type is IndexingType.SINGLEPASS:
            if self.blocks:
                index = BlockSinglePassIndexer(self.index_dir, "data")
            else:
                index = BasicSinglePassIndexer(self.index_dir, "data")
        elif self.type is IndexingType.CLASSIC:
            if self.blocks:
                index = BlockIndexer(self.index_dir, "data")
            else:
                index = BasicIndexer(self.index_dir, "data")
        else:
            raise Exception("Memory indexing not yet implemented")

        return index

    def createAsList(self, files_path):
        """
        Helper method to be used by child indexers to add files to Java List
        Returns:
            Created Java List
        """
        if isinstance(files_path, str):
            asList = Arrays.asList(files_path)
        elif isinstance(files_path, list):
            asList = Arrays.asList(*files_path)
        return asList

    def getIndexStats(self):
        """
        Prints the index statistics

        Note:
            Does not work with notebooks at the moment
        """
        CLITool.main(["indexstats", "-I" + self.path])

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
        CLITool.main(["indexutil", "-I" + self.path, util])

class DFIndexUtils:

    @staticmethod
    def get_column_lengths(df):
        return dict([(v, df[v].apply(lambda r: len(str(r)) if r!=None else 0).max())for v in df.columns.values])

    @staticmethod
    def create_javaDocIterator(text, *args, **kwargs):
        if HashMap is None:
            run_autoclass()

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

        # this method creates the documents as and when needed.
        def convertDoc(text_row, meta_column):
            # meta_row = []
            hashmap = HashMap()
            for column, value in meta_column[1].iteritems():
                hashmap.put(column, value)
            return(TaggedDocument(StringReader(text_row), hashmap, Tokeniser.getTokeniser()))
        
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

class DFIndexer(Indexer):
    """
    Use this Indexer if you wish to index a pandas.Dataframe

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + /data.properties
        index_called(bool): True if index() method of child Indexer has been called, false otherwise
        index_dir(str): The index directory
        blocks(bool): If true the index has blocks enabled
        properties: A Terrier Properties object, which is a hashtable with properties and their values
        overwrite(bool): If True the index() method of child Indexer will overwrite any existing index
    """
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
        
        # generate the metadata properties, set their lengths automatically
        mprop1=""
        mprop2=""
        mprop1_def=None
        mprop2_def=None

        # keep track of the previous settings of these indexing properties
        default_props = ApplicationSetup.getProperties()
        if default_props.containsKey("indexer.meta.forward.keys"):
            mprop1_def = default_props.get("indexer.meta.forward.keys")
        if default_props.containsKey("indexer.meta.forward.keylens"):
            mprop2_def = default_props.get("indexer.meta.forward.keylens")

        # update the indexing properties
        for k in meta_lengths:
            mprop1 += k+ ","
            mprop2 += str(meta_lengths[k]) + ","
        ApplicationSetup.setProperty("indexer.meta.forward.keys", mprop1[:-1])
        ApplicationSetup.setProperty("indexer.meta.forward.keylens", mprop2[:-1])

        # make a Collection class for Terrier
        javaDocCollection = autoclass("org.terrier.python.CollectionFromDocumentIterator")(collectionIterator)
        index = self.createIndexer()
        index.index([javaDocCollection])
        self.index_called = True
        collectionIterator = None

        # this block is for restoring the indexing config
        if mprop1_def is not None:
            ApplicationSetup.setProperty("indexer.meta.forward.keys", mprop1_def)
        else:
            default_props.remove("indexer.meta.forward.keys")
        if mprop2_def is not None:
            ApplicationSetup.setProperty("indexer.meta.forward.keylens", mprop2_def)
        else:
            default_props.remove("indexer.meta.forward.keylens")

        return IndexRef.of(self.index_dir + "/data.properties")

class PythonListIterator(PythonJavaClass):
    __javainterfaces__ = ['java/util/Iterator']

    def __init__(self, text, meta, convertFn, len=None, index=0):
        super(PythonListIterator, self).__init__()
        self.text = text
        self.meta = meta
        self.index = index
        self.convertFn = convertFn
        if len is None:
            self.len = len(self.text)
        else:
            self.len = len

    @java_method('()V')
    def remove():
        # 1
        pass

    @java_method('(Ljava/util/function/Consumer;)V')
    def forEachRemaining(action):
        # 1
        pass

    @java_method('()Z')
    def hasNext(self):
        return self.index < self.len

    @java_method('()Ljava/lang/Object;')
    def next(self):
        text = self.text[self.index]
        meta = self.meta.__next__()
        self.index += 1
        if self.convertFn is not None:
            return self.convertFn(text, meta)
        return [text, meta]

class FlatJSONDocumentIterator(PythonJavaClass):
    __javainterfaces__ = ['java/util/Iterator']

    def __init__(self, it):
        super(FlatJSONDocumentIterator, self).__init__()
        if FlatJSONDocument is None:
            run_autoclass()
        self._it = it
        # easiest way to support hasNext is just to start consuming right away, I think
        self._next = next(self._it, StopIteration)

    @java_method('()V')
    def remove():
        # 1
        pass

    @java_method('(Ljava/util/function/Consumer;)V')
    def forEachRemaining(action):
        # 1
        pass

    @java_method('()Z')
    def hasNext(self):
        return self._next is not StopIteration

    @java_method('()Ljava/lang/Object;')
    def next(self):
        result = self._next
        self._next = next(self._it, StopIteration)
        if result is not StopIteration:
            return FlatJSONDocument(json.dumps(result))
        return None

class IterDictIndexer(Indexer):
    """
    Use this Indexer if you wish to index an iter of dicts (possibly with multiple fields)

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + /data.properties
        index_called(bool): True if index() method of child Indexer has been called, false otherwise
        index_dir(str): The index directory
        blocks(bool): If true the index has blocks enabled
        properties: A Terrier Properties object, which is a hashtable with properties and their values
        overwrite(bool): If True the index() method of child Indexer will overwrite any existing index
    """
    def index(self, it, fields=('text',), meta=('docno',)):
        """
        Index the specified iter of dicts with the (optional) specified fields

        Args:
            it(iter[dict]): an iter of document dict to be indexed
            fields(list[str]): fieleds to be indexed (all will be considered metadata)
        """
        self.checkIndexExists()
        self.setProperties(**{
            'FieldTags.process': ','.join(fields),
            'FieldTags.casesensitive': 'true',
            'indexer.meta.forward.keys': ','.join(meta),
            # What are the ramifications of setting all lengths to a large value like this? (storage cost?)
            'indexer.meta.forward.keylens': ','.join(['512'] * len(meta))
        })
        # we need to prevent collectionIterator from being GCd
        collectionIterator = FlatJSONDocumentIterator(it)
        javaDocCollection = autoclass("org.terrier.python.CollectionFromDocumentIterator")(collectionIterator)
        index = self.createIndexer()
        index.index([javaDocCollection])
        self.index_called = True
        collectionIterator = None
        return IndexRef.of(self.index_dir + "/data.properties")

class TRECCollectionIndexer(Indexer):
    type_to_class = {
        'trec' : 'org.terrier.indexing.TRECCollection',
        'trecweb' : 'org.terrier.indexing.TRECWebCollection',
        'warc' : 'org.terrier.indexing.WARC10Collection'
    }
    """
    Use this Indexer if you wish to index a TREC formatted collection

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + /data.properties
        index_called(bool): True if index() method of child Indexer has been called, false otherwise
        index_dir(str): The index directory
        blocks(bool): If true the index has blocks enabled
        properties: A Terrier Properties object, which is a hashtable with properties and their values
        overwrite(bool): If True the index() method of child Indexer will overwrite any existing index
        verbose(bool): If True, will display a progress bar based on files indexed
    """

    def __init__(self, index_path, blocks=False, overwrite=False, type=IndexingType.CLASSIC, collection="trec", verbose=False):
        """
        Init method

        Args:
            index_path (str): Directory to store index
            blocks (bool): Create indexer with blocks if true, else without blocks
            overwrite (bool): If index already present at `index_path`, True would overwrite it, False throws an Exception
            type (IndexingType): the specific indexing procedure to use. Default is IndexingType.CLASSIC.
            collection (Class name, or Class instance, or one of "trec", "trecweb", "warc")
        """
        super().__init__(index_path, blocks=blocks, overwrite=overwrite, type=type)
        if isinstance(collection, str):
            if collection in TRECCollectionIndexer.type_to_class:
                collection = TRECCollectionIndexer.type_to_class[collection]
        self.collection = collection.split(",")
        self.verbose = verbose
    

    def index(self, files_path):
        """
        Index the specified TREC formatted files

        Args:
            files_path: can be a String of the path or a list of Strings of the paths for multiple files
        """
        self.checkIndexExists()
        index = self.createIndexer()
        asList = self.createAsList(files_path)
        cls_string = autoclass("java.lang.String")._class
        cls_list = autoclass("java.util.List")._class
        colObj = autoclass("org.terrier.indexing.CollectionFactory").loadCollections(
            self.collection,
            [cls_list, cls_string, cls_string, cls_string],
            [asList, autoclass("org.terrier.utility.TagSet").TREC_DOC_TAGS, "", ""])
        collsArray = [colObj]
        if self.verbose and isinstance(colObj, autoclass("org.terrier.indexing.MultiDocumentFileCollection")):
            colObj = cast("org.terrier.indexing.MultiDocumentFileCollection", colObj)
            colObj = TQDMCollection(colObj)
            collsArray = autoclass("org.terrier.python.PTUtils").makeCollection(colObj)
        index.index(collsArray)
        colObj.close()
        self.index_called = True
        return IndexRef.of(self.index_dir + "/data.properties")

class FilesIndexer(Indexer):
    '''
    Use this Indexer if you wish to index a pdf, docx, txt etc files

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + /data.properties
        index_called(bool): True if index() method of child Indexer has been called, false otherwise
        index_dir(str): The index directory
        blocks(bool): If true the index has blocks enabled
        properties: A Terrier Properties object, which is a hashtable with properties and their values
        overwrite(bool): If True the index() method of child Indexer will overwrite any existing index
    '''

    def __init__(self, index_path, *args, **kwargs):
        super().__init__(index_path, args, kwargs)
        self.properties["indexer.meta.forward.keys"]="docno,filename"
        self.properties["indexer.meta.forward.keylens"]="20,512"

    def index(self, files_path):
        """
        Index the specified TREC formatted files

        Args:
            files_path: can be a String of the path or a list of Strings of the paths for multiple files
        """
        self.checkIndexExists()
        index = self.createIndexer()
        asList = self.createAsList(files_path)
        simpleColl = SimpleFileCollection(asList, False)
        index.index([simpleColl])
        self.index_called = True
        return IndexRef.of(self.index_dir + "/data.properties")

class TQDMCollection(PythonJavaClass):
    __javainterfaces__ = ['org/terrier/indexing/Collection']

    def __init__(self, collection):
        super(TQDMCollection, self).__init__()
        assert isinstance(collection, autoclass("org.terrier.indexing.MultiDocumentFileCollection"))
        self.collection = collection
        size = self.collection.FilesToProcess.size()
        from tqdm import tqdm
        self.pbar = tqdm(total=size, unit="files")
        self.last = -1
    
    @java_method('()Z')
    def nextDocument(self):
        rtr = self.collection.nextDocument()
        filenum = self.collection.FileNumber
        if filenum > self.last:
            self.pbar.update(filenum - self.last)
            self.last = filenum
        return rtr

    @java_method('()V')
    def reset(self):
        self.pbar.reset()
        self.collection.reset()

    @java_method('()V')
    def close(self):
        self.pbar.close()
        self.collection.close()

    @java_method('()Z')
    def endOfCollection(self):
        return self.collection.endOfCollection()

    @java_method('()Lorg/terrier/indexing/Document;')
    def getDocument(self):
        return self.collection.getDocument()
        
