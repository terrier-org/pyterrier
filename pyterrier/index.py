from jnius import autoclass, cast, PythonJavaClass, java_method
from utils import *
import pandas as pd
import numpy as np
import os

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
    '''
    Parent class. Use one of its children classes
    '''

    default_properties={
            "TrecDocTags.doctag":"DOC",
            "TrecDocTags.idtag":"DOCNO",
            "TrecDocTags.skip":"DOCHDR",
            "TrecDocTags.casesensitive":"false",
            "trec.collection.class": "TRECCollection",
    }
    def __init__(self, index_path, blocks=False, overwrite=False):
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
        for control,value in kwargs.items():
            self.properties.put(control,value)

    def checkIndexExists(self):
        if os.path.isfile(self.path):
            if not self.overwrite :
                raise ValueError("Index already exists at " + self.path)
        if self.index_called:
            raise Exception("Index method can be called only once")

    def createIndexer(self):
        #ApplicationSetup.bootstrapInitialisation(self.properties)
        ApplicationSetup.getProperties().putAll(self.properties)
        if self.blocks:
            index = BlockIndexer(self.index_dir,"data")
        else:
            index = BasicIndexer(self.index_dir,"data")
        return index

    def createAsList(self, files_path):
        if type(files_path) == type(""):
            asList = Arrays.asList(files_path)
        if type(files_path) == type([]):
            asList = Arrays.asList(*files_path)
        return asList

    def getIndexStats(self):
        CLITool.main(["indexstats", "-I" + self.path])


    def getIndexUtil(self, util):
        ''' Utilities for displaying the content of an index
        Parameters:
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
        '''
        if not util.startswith("-"):
            util = "-"+util
        CLITool.main(["indexutil", "-I" + self.path, util])

class DFIndexUtils:

    @staticmethod
    def create_javaDocIterator(text, *args, **kwargs):
        all_metadata={}
        for i, arg in enumerate(args):
            if isinstance(arg, pd.Series):
                all_metadata[arg.name]=arg
                assert len(arg)==len(text), "Length of metadata arguments needs to be equal to length of text argument"
            elif isinstance(arg, pd.DataFrame):
                for name, column in arg.items():
                    all_metadata[name]=column
                    assert len(column)==len(text), "Length of metadata arguments needs to be equal to length of text argument"
            else:
                raise ValueError("Non-keyword args need to be of type pandas.Series or pandas.DataFrame, argument %d was %s "% (i, type(arg)))
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

        #this method creates the documents as and when needed.
        def convertDoc(text_row, meta_column):
            meta_row=[]
            hashmap = HashMap()
            for column, value in meta_column[1].iteritems():
                hashmap.put(column,value)
            return(TaggedDocument(StringReader(text_row), hashmap, Tokeniser.getTokeniser()))

        df=pd.DataFrame(all_metadata)
        return PythonListIterator(
                text.values, 
                df.iterrows(),
                convertDoc,
                len(text.values)
            )


class DFIndexer(Indexer):

    '''
    Use for Pandas dataframe
    '''
    def index(self, text, *args, **kwargs):
        self.checkIndexExists()
        
        javaDocCollection = autoclass("org.terrier.python.CollectionFromDocumentIterator")(
            DFIndexUtils.create_javaDocIterator(text, *args, **kwargs)
        )
        index = self.createIndexer()
        index.index([javaDocCollection])
        self.index_called=True
        JIR = autoclass('org.terrier.querying.IndexRef')
        return JIR.of(self.index_dir+ "/data.properties")

from jnius import PythonJavaClass, java_method

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
    def remove(): 1

    @java_method('(Ljava/util/function/Consumer;)V')
    def forEachRemaining(action): 1

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
    '''
    Use for TREC formatted collection
    '''
    def index(self, files_path):
        self.checkIndexExists()
        index = self.createIndexer()
        asList = self.createAsList(files_path)
        trecCol = TRECCollection(asList,"TrecDocTags","","")
        index.index([trecCol])
        self.index_called=True

class FilesIndexer(Indexer):
    '''
    Use for pdf, docx, txt etc files
    '''
    def index(self, files_path):
        self.checkIndexExists()
        index = self.createIndexer()
        asList = self.createAsList(files_path)
        simpleColl = SimpleFileCollection(asList,False)
        index.index([simpleColl])
        self.index_called=True
