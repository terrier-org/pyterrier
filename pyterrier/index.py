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
Collection = autoclass("org.terrier.indexing.Collection")
Arrays = autoclass("java.util.Arrays")
Array = autoclass('java.lang.reflect.Array')
ApplicationSetup = autoclass('org.terrier.utility.ApplicationSetup')
Properties = autoclass('java.util.Properties')


class BasicIndex():
    def createCollection(self, docDataframe):
        lst = []
        for index, row in docDataframe.iterrows():
            hashmap = HashMap()
            # all columns, except text are properties and add them to hashmap
            for column, value in row.iteritems():
                if column!="text":
                    hashmap.put(column,value)
            tagDoc = TaggedDocument(StringReader(row["text"]), hashmap, Tokeniser.getTokeniser())
            # print("DOCNO: " + str(hashmap.get("DOCNO")))
            lst.append(tagDoc)
        javaDocCollection = CollectionDocumentList(lst, "null")
        return javaDocCollection

    def __init__(self,collection, path):



        # if collection is string assume it is path to corpus
        if type(collection) == type(""):
            trec_props={
                "TrecDocTags.doctag":"DOC",
                "TrecDocTags.idtag":"DOCNO",
                "TrecDocTags.skip":"DOCHDR",
                "TrecDocTags.casesensitive":"false",
                "trec.collection.class": "TRECCollection"
            }
            props = Properties()
            for control,value in trec_props.items():
                props.put(control,value)

            ApplicationSetup.bootstrapInitialisation(props)
            print("trec.collection.class value: " + ApplicationSetup.getProperty("trec.collection.class","asd"))

            print("Index dir: "+ path)
            print("Collection: "+ collection )
            if (os.path.isfile(collection)):
                asList = Arrays.asList(collection)
                print(ApplicationSetup.getProperty("TrecDocTags.doctag",None))
                trecCol = TRECCollection(asList,"TrecDocTags","","")
                index = BasicIndexer(path,"data")
                index.index([trecCol])
            else:
                asList = Arrays.asList(collection)
                simpleColl = SimpleFileCollection(asList,False)
                index = BasicIndexer(path,"data")
                index.index([simpleColl])
        # if collection is a dataframe create a new collection object
        elif type(collection)==type(pd.DataFrame([])):
            # col = PyCollection(collection)

            # if isinstance(col,autoclass("org.terrier.indexing.Collection")):
            #     print("\nCol is instance of org.terrier.indexing.Collection\n")
            # arr = Array.newInstance(Collection,1)
            # arr[0]=col
            # arr[0] = autoclass('org.terrier.indexing.IndexTestUtils').makeCollection(["doc1"], ["the laxy brown hare jumped the fox"])
            # col = cast([Collection], col)
            # col = cast("java.lang.Object", col)
            # arr = [autoclass('org.terrier.indexing.IndexTestUtils').makeCollection(["doc1"], ["the laxy brown hare jumped the fox"])]
            # trecCol = TRECCollection("/home/alex/Downloads/books/collection.spec")
            # index = BasicIndexer("/home/alex/Documents/index_test","data")
            # index.index([col])
            # col = PyCollection(collection)

            print("Dataframe: ")
            print(collection)
            javaDocCollection = self.createCollection(collection)
            # print(javaDocCollection.nextDocument())
            # print("Collection: ")
            # while(not javaDocCollection.endOfCollection()):
            #     javaDocCollection.nextDocument()
            #     while(not javaDocCollection.getDocument().endOfDocument()):
            #         print(javaDocCollection.getDocument().getNextTerm(), end=' ')

                # print(javaDocCollection.getDocid())

            index = BasicIndexer(path, "data")
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
            hashmap = HashMap()
            # all columns, except text are properties and add them to hashmap
            for column, value in dataframe.iteritems():
                if column!="text":
                    print("Column: " + column)
                    print("Value: " + value)
                    hashmap.put(column,value)
            tagDoc = TaggedDocument(StringReader(row["text"]), hashmap, Tokeniser.getTokeniser())
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
