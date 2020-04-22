"""
This file contains all the indexers.
"""

# from jnius import autoclass, cast, PythonJavaClass, java_method
from jnius import autoclass, PythonJavaClass, java_method
# from .utils import *
import pandas as pd
# import numpy as np
import os

StringReader = None
HashMap = None
TaggedDocument = None
Tokeniser = None
TRECCollection = None
SimpleFileCollection = None
BasicIndexer = None
BlockIndexer = None
Collection = None
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
    global Tokeniser
    global TRECCollection
    global SimpleFileCollection
    global BasicIndexer
    global BlockIndexer
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
    Tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
    TRECCollection = autoclass("org.terrier.indexing.TRECCollection")
    SimpleFileCollection = autoclass("org.terrier.indexing.SimpleFileCollection")
    BasicIndexer = autoclass("org.terrier.structures.indexing.classical.BasicIndexer")
    BlockIndexer = autoclass("org.terrier.structures.indexing.classical.BlockIndexer")
    Collection = autoclass("org.terrier.indexing.Collection")
    Arrays = autoclass("java.util.Arrays")
    Array = autoclass('java.lang.reflect.Array')
    ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')
    Properties = autoclass('java.util.Properties')
    CLITool = autoclass("org.terrier.applications.CLITool")
    IndexRef = autoclass('org.terrier.querying.IndexRef')
    IndexFactory = autoclass('org.terrier.structures.IndexFactory')

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

    def __init__(self, index_path, blocks=False, overwrite=False):
        """
        Init method

        Args:
            index_path (str): Directory to store index
            blocks (bool): Create indexer with blocks if true, else without blocks
            overwrite (bool): If index already present at `index_path`, True would overwrite it, False throws an Exception
        """
        if StringReader is None:
            run_autoclass()
        self.path = os.path.join(index_path, "data.properties")
        self.index_called = False
        self.index_dir = index_path
        self.blocks = blocks
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
        Check `blocks` and create a BlockIndexer if true, else create BasicIndexer
        Returns:
            Created index object
        """
        ApplicationSetup.getProperties().putAll(self.properties)
        # ApplicationSetup.bootstrapInitialisation(self.properties)
        if self.blocks:
            index = BlockIndexer(self.index_dir, "data")
        else:
            index = BasicIndexer(self.index_dir, "data")
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
            s
        """
        if not util.startswith("-"):
            util = "-" + util
        CLITool.main(["indexutil", "-I" + self.path, util])

class DFIndexUtils:

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

        df = pd.DataFrame(all_metadata)
        return PythonListIterator(
                text.values,
                df.iterrows(),
                convertDoc,
                len(text.values)
        )

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
        # we need to prevent collectionIterator from being GCd
        collectionIterator = DFIndexUtils.create_javaDocIterator(text, *args, **kwargs)
        javaDocCollection = autoclass("org.terrier.python.CollectionFromDocumentIterator")(collectionIterator)
        index = self.createIndexer()
        index.index([javaDocCollection])
        self.index_called = True
        collectionIterator = None
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

class TRECCollectionIndexer(Indexer):
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
    """
    def index(self, files_path):
        """
        Index the specified TREC formatted files

        Args:
            files_path: can be a String of the path or a list of Strings of the paths for multiple files
        """
        self.checkIndexExists()
        index = self.createIndexer()
        asList = self.createAsList(files_path)
        trecCol = TRECCollection(asList, "TrecDocTags", "", "")
        index.index([trecCol])
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
