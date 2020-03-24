"""
This file contains all the indexers.
"""

from jnius import autoclass, cast, PythonJavaClass, java_method
from utils import *
import pandas as pd
import numpy as np
import os

CollectionDocumentList = autoclass("org.terrier.indexing.CollectionDocumentList")
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

class Indexer:
    """
    Parent class. It can be used to load an existing index.
    Use one of its children classes if you wish to create a new index.

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + \data.properties
        index_called(bool): True if index() method of child Indexer has been called, false otherwise
        index_dir(str): The index directory
        blocks(bool): If true the index has blocks enabled
        properties: A Terrier Properties object, which is a hashtable with properties and their values
        overwrite(bool): If True the index() method of child Indexer will overwrite any existing index
    """

    default_properties={
            "TrecDocTags.doctag":"DOC",
            "TrecDocTags.idtag":"DOCNO",
            "TrecDocTags.skip":"DOCHDR",
            "TrecDocTags.casesensitive":"false",
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
        Set the properties to the given ones

        Usage:
            setProperties("property1=value1, property2=value2")
            or
            setProperties("**{property1:value1, property2:value2}")

        Args:
            **kwargs: Properties to set to
        """
        for control,value in kwargs.items():
            self.properties.put(control,value)

    def checkIndexExists(self):
        """
        Check if index exists at the `path` given when object was created
        """
        if os.path.isfile(self.path):
            if not self.overwrite :
                raise ValueError("Index already exists at " + self.path)
        if self.index_called:
            raise Exception("Index method can be called only once")

    def createIndexer(self):
        """
        Check `blocks` and create a BlockIndexer if true, else create BasicIndexer
        Returns:
            Created index object
        """
        ApplicationSetup.bootstrapInitialisation(self.properties)
        if self.blocks:
            index = BlockIndexer(self.index_dir,"data")
        else:
            index = BasicIndexer(self.index_dir,"data")
        return index

    def createAsList(self, files_path):
        """
        Helper method to be used by child indexers to add files to Java List
        Returns:
            Created Java List
        """
        if type(files_path) == type(""):
            asList = Arrays.asList(files_path)
        else if type(files_path) == type([]):
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
            util = "-"+util
        CLITool.main(["indexutil", "-I" + self.path, util])

class DFIndexer(Indexer):
    """
    Use this Indexer if you wish to index a pandas.Dataframe

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + \data.properties
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
        all_metadata={}
        for arg in args:
            if isinstance(arg, pd.Series):
                all_metadata[arg.name]=arg
                assert len(arg)==len(text), "Length of metadata arguments needs to be equal to length of text argument"
            elif isinstance(arg, pd.DataFrame):
                for name, column in arg.items():
                    all_metadata[name]=column
                    assert len(column)==len(text), "Length of metadata arguments needs to be equal to length of text argument"
            else:
                raise ValueError("Non-keyword args need to be of type pandas.Series or pandas,DataFrame")
        for key, value in kwargs.items():
            if isinstance(value, (pd.Series, list, tuple)):
                all_metadata[key]=value
                assert len(value)==len(value), "Length of metadata arguments needs to be equal to length of text argument"
            elif isinstance(value, pd.DataFrame):
                for name, column in arg.items():
                    all_metadata[name]=column
                    assert len(column)==len(text), "Length of metadata arguments needs to be equal to length of text argument"
            else:
                raise ValueError("Keyword kwargs need to be of type pandas.Series, list or tuple")

        doc_list=[]
        df=pd.DataFrame(all_metadata)
        for text_row, meta_column in zip(text.values, df.iterrows()):
            meta_row=[]
            hashmap = HashMap()
            for column, value in meta_column[1].iteritems():
                hashmap.put(column,value)
            tagDoc = TaggedDocument(StringReader(text_row), hashmap, Tokeniser.getTokeniser())
            doc_list.append(tagDoc)

        index = self.createIndexer()
        javaDocCollection = CollectionDocumentList(doc_list, "null")
        index.index([javaDocCollection])
        self.index_called=True

class TRECCollectionIndexer(Indexer):
    """
    Use this Indexer if you wish to index a TREC formatted collection

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + \data.properties
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
        trecCol = TRECCollection(asList,"TrecDocTags","","")
        index.index([trecCol])
        self.index_called=True

class FilesIndexer(Indexer):
    '''
    Use this Indexer if you wish to index a pdf, docx, txt etc files

    Attributes:
        default_properties(dict): Contains the default properties
        path(str): The index directory + \data.properties
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
        simpleColl = SimpleFileCollection(asList,False)
        index.index([simpleColl])
        self.index_called=True
