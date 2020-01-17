from jnius import autoclass, cast, PythonJavaClass, java_method
from utils import *
import pandas as pd
import numpy as np

JavaDocCollection = autoclass("org.terrier.indexing.CollectionDocumentList")
stringReaderClass = autoclass("java.io.StringReader")
hashmapClass = autoclass("java.util.HashMap")
taggedDocClass = autoclass("org.terrier.indexing.TaggedDocument")
tokeniserClass = autoclass("org.terrier.indexing.tokenisation.Tokeniser")
trecColClass = autoclass("org.terrier.indexing.TRECCollection")
simpleColClass = autoclass("org.terrier.indexing.SimpleFileCollection")
basicIndexClass = autoclass("org.terrier.structures.indexing.classical.BasicIndexer")
collectionClass = autoclass("org.terrier.indexing.Collection")
array = autoclass("java.util.Arrays")
javaArray = autoclass('java.lang.reflect.Array')

class BasicIndex():
    def __init__(self,collection, path):
        def createCollection(docDataframe):
            lst = []
            for index, row in collection.iterrows():
                hashmap = hashmapClass()
                # all columns, except text are properties and add them to hashmap
                for column, value in row.iteritems():
                    if column!="text":
                        hashmap.put(column,value)
                tagDoc = taggedDocClass(stringReaderClass(row["text"]), hashmap, tokeniserClass.getTokeniser())
                lst.append(tagDoc)
            javaDocCollection = JavaDocCollection(lst, "null")
            return javaDocCollection

        # if collection is string assume it is path to corpus
        if type(collection) == type(""):
            asList = array.asList(collection)
            simpleColl = simpleColClass(asList,False)
            index = basicIndexClass("/home/alex/Documents/index_test","data")
            index.index([simpleColl])
        # if collection is a dataframe create a new collection object
        elif type(collection)==type(pd.DataFrame([])):
            col = PyCollection(collection)

            # if isinstance(col,autoclass("org.terrier.indexing.Collection")):
            #     print("\nCol is instance of org.terrier.indexing.Collection\n")
            # arr = javaArray.newInstance(collectionClass,1)
            # arr[0]=col
            # arr[0] = autoclass('o*rg.terrier.indexing.IndexTestUtils').makeCollection(["doc1"], ["the laxy brown hare jumped the fox"])
            # col = cast([collectionClass], col)
            # col = cast("java.lang.Object", col)
            # arr = [autoclass('org.terrier.indexing.IndexTestUtils').makeCollection(["doc1"], ["the laxy brown hare jumped the fox"])]
            # trecCol = trecColClass("/home/alex/Downloads/books/collection.spec")
            # index = basicIndexClass("/home/alex/Documents/index_test","data")
            # index.index([col])
            # col = PyCollection(collection)
            javaDocCollection = createCollection(collection)

            index = basicIndexClass(path, "data")
            index.index([javaDocCollection])

class PyCollection(PythonJavaClass):
    __javainterfaces__ = ['org/terrier/indexing/Collection',]

    def __init__(self, dataframe):
        # super().__init__(dataframe)
        # super(PyCollection,self).__init__()
        # super(PyCollection, self).__init__(dataframe)

        self.dataframe=dataframe
        lst = []


        for index, row in dataframe.iterrows():
            hashmap = hashmapClass()
            # all columns, except text are properties and add them to hashmap
            for column, value in dataframe.iteritems():
                if column!="text":
                    hashmap.put(column,value)
            tagDoc = taggedDocClass(stringReaderClass(row["text"]), hashmap, tokeniserClass.getTokeniser())
            lst.append(tagDoc)

        self.collection = lst
        self.index = 0

    @java_method("()Z")
    def endofCollection(self):
        return self.index>=len(self.collection)-1

    @java_method("()Lorg/terrier/indexing/Document")
    def getDocument(self):
        return self.collection[self.index]

    @java_method("()Z")
    def nextDocument(self):
        if self.endofCollection():
            return False
        else:
            self.index+=1
            return True

    @java_method("()V")
    def reset(self):
        index = 0
