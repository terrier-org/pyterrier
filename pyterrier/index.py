from jnius import autoclass, cast, PythonJavaClass, java_method
from utils import *
import pandas as pd
import numpy as np

# class Index():
#     def __init__(self, corpus, blocks=False, fields=[]):
#         print(corpus)
#     def addDocument(document): #??
#         print(document)
#     def saveIndex(path):
#         print(path)
#     def loadIndex(path):
#         print(path)


class BasicIndex():
    def __init__(self,collection):
        trecColClass = autoclass("org.terrier.indexing.TRECCollection")
        simpleColClass = autoclass("org.terrier.indexing.SimpleFileCollection")
        basicIndexClass = autoclass("org.terrier.structures.indexing.classical.BasicIndexer")
        collectionClass = autoclass("org.terrier.indexing.Collection")
        array = autoclass("java.util.Arrays")
        javaArray = autoclass('java.lang.reflect.Array')

        # if collection is string assume it is path to corpus
        if type(collection) == type(""):
            asList = array.asList(collection)
            simpleColl = simpleColClass(asList,False)
            index = basicIndexClass("/home/alex/Documents/index_test","data")
            index.index([simpleColl])
        # if collection is a dataframe create a new collection object
        elif type(collection)==type(pd.DataFrame([])):
            col = PyCollection(collection)

            if isinstance(col,collectionClass):
                print("\nCol is instance of org.terrier.indexing.Collection\n")
            # arr = javaArray.newInstance(javaArray,1)
            # arr[0]=col
            # col = cast(collectionClass, col)
            # col = cast("org.terrier.indexing.Collection", col)

            index = basicIndexClass("/home/alex/Documents/index_test","data")
            index.index([col])

class PyCollection(PythonJavaClass):
    __javainterfaces__ = ['org/terrier/indexing/Collection',]

    def __init__(self, dataframe):
        # super().__init__(dataframe)
        self.dataframe=dataframe
        lst = []
        stringReaderClass = autoclass("java.io.StringReader")
        hashmapClass = autoclass("java.util.HashMap")
        taggedDocClass = autoclass("org.terrier.indexing.TaggedDocument")
        tokeniserClass = autoclass("org.terrier.indexing.tokenisation.Tokeniser")

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
