import jnius_config, os, pytrec_eval,json
import numpy as np
import pandas as pd
# from types import

jnius_config.set_classpath("terrier-project-5.1-jar-with-dependencies.jar")
from jnius import autoclass, cast

class Utils:
    @staticmethod
    def parse_trec_topics_file(file_path):
        system = autoclass("java.lang.System")
        system.setProperty("TrecQueryTags.doctag","TOP");
        system.setProperty("TrecQueryTags.idtag","NUM");
        system.setProperty("TrecQueryTags.process","TOP,NUM,TITLE");
        system.setProperty("TrecQueryTags.skip","DESC,NARR");

        trec = autoclass('org.terrier.applications.batchquerying.TRECQuery')
        tr = trec(file_path)
        topics_lst=[]
        while(tr.hasNext()):
            topic = tr.next()
            qid = tr.getQueryId()
            topics_lst.append([qid,topic])
        topics_dt = pd.DataFrame(topics_lst,columns=['qid','query'])
        return topics_dt

    # Convert dataframe with columns: [qid,docno,score] into a dict {qid1: {doc1:score,doc2:score } qid2:...}
    @staticmethod
    def convert_df_to_pytrec_eval(df, score_int=False):
        run_dict_pytrec_eval = {}
        for index, row in df.iterrows():
            if row['qid'] not in run_dict_pytrec_eval.keys():
                run_dict_pytrec_eval[row['qid']] = {}
            if score_int:
                run_dict_pytrec_eval[row['qid']][row['docno']] = int(row['score'])
            else:
                run_dict_pytrec_eval[row['qid']][row['docno']] = float(row['score'])
        return(run_dict_pytrec_eval)

    @staticmethod
    def parse_qrels(file_path):
        dph_results=[]
        with (open(file_path, 'r')) as qrels_file:
            for line in qrels_file:
                split_line=line.strip("\n").split(" ")
                dph_results.append([split_line[0], split_line[2],split_line[3]])
        res_dt = pd.DataFrame(dph_results,columns=['qid','docno','score'])
        return res_dt

    @staticmethod
    def evaluate(res,qrels):
        batch_retrieve_results_dict = Utils.convert_df_to_pytrec_eval(res)
        qrels_dic=Utils.convert_df_to_pytrec_eval(qrels, True)
        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dic, {'map', 'ndcg'})
        return json.dumps(evaluator.evaluate(batch_retrieve_results_dict), indent=1)


    @staticmethod
    def form_dataframe(query):
        if type(query)==type(pd.DataFrame()):
            return query
        elif type(query)==type(""):
            return pd.DataFrame([["1", query]],columns=['qid','query'])
        # if queries is a list or tuple
        elif type(query)==type([]) or type(query)==type(()):
            #if the list or tuple is made of strings
            if query!=[] and type(query[0])==type(""):
                indexed_query = []
                for i,item in enumerate(query):
                    # all elements must be of same type
                    assert type(item) is type(""), "%r is not a string" % item
                    indexed_query.append([str(i+1),item])
                return pd.DataFrame(indexed_query,columns=['qid','query'])


class BatchRetrieve:
    default_controls={
        "terrierql": "on",
        "parsecontrols": "on",
        "parseql": "on",
        "applypipeline": "on",
        "localmatching": "on",
        "filters": "on",
        "decorate":"on",
        "wmodel": "DPH",
    }

    default_properties={
        "querying.processes":"terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess",
        "querying.postfilters": "decorate:SimpleDecorate,site:SiteFilter,scope:Scope",
        "querying.default.controls": "wmodel:DPH,parsecontrols:on,parseql:on,applypipeline:on,terrierql:on,localmatching:on,filters:on,decorate:on",
        "querying.allowed.controls": "scope,qe,qemodel,start,end,site,scope",
        "termpipelines": "Stopwords,PorterStemmer"
    }

    def __init__(self, IndexRef, controls=None, properties=None):
        self.IndexRef=IndexRef
        self.appSetup = autoclass('org.terrier.utility.ApplicationSetup')

        if properties==None:
            self.properties=self.default_properties
        else:
            self.properties=properties

        props = autoclass('java.util.Properties')()
        for control,value in self.properties.items():
            props.put(control,value)
        self.appSetup.bootstrapInitialisation(props)

        # print(self.appSetup.getProperties().toString())

        if controls==None:
            self.controls=self.default_controls
        else:
            self.controls=controls

        MF = autoclass('org.terrier.querying.ManagerFactory')
        self.ManagerFactory = MF._from_(IndexRef)

    def transform(self,queries):
        results=[]
        queries=Utils.form_dataframe(queries)
        for index,row in queries.iterrows():
            srq = self.ManagerFactory.newSearchRequest(row['qid'],row['query'])
            for control,value in self.controls.items():
                srq.setControl(control,value)
            self.ManagerFactory.runSearchRequest(srq)
            result=srq.getResults()
            for item in result:
                res = [queries.iloc[index]['qid'],item.getMetadata("docno"),item.getScore()]
                results.append(res)
        res_dt=pd.DataFrame(results,columns=['qid','docno','score'])
        self.lastResult=res_dt
        return res_dt

    def saveResult(self, result, path):
        res_copy = result.copy()
        res_copy.insert(1, "Q0", "Q0")
        res_copy.insert(4, "wmodel", self.controls["wmodel"])
        res_copy.to_csv(path, sep=" ", header=False, index=False)

    def saveLastResult(self, path):
        self.saveResult(self.lastResult,path)

    def setControls(self, controls):
        self.controls=controls

    def setControl(self, control,value):
        self.controls[control]=value



class FeaturesBatchRetrieve(BatchRetrieve):
    default_controls={
        "terrierql": "on",
        "parsecontrols": "on",
        "parseql": "on",
        "applypipeline": "on",
        "localmatching": "on",
        "filters": "on",
        "decorate":"on",
        "wmodel": "DPH",
        "matching": "FatFeaturedScoringMatching,org.terrier.matching.daat.FatFull"
    }
    default_properties={
        "querying.processes":"terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess",
        "querying.postfilters": "decorate:SimpleDecorate,site:SiteFilter,scope:Scope",
        "querying.default.controls": "wmodel:DPH,parsecontrols:on,parseql:on,applypipeline:on,terrierql:on,localmatching:on,filters:on,decorate:on",
        "querying.allowed.controls": "scope,qe,qemodel,start,end,site,scope",
        "termpipelines": "Stopwords,PorterStemmer"
    }
    def __init__(self, IndexRef, features, controls=None, properties=None):
        self.IndexRef=IndexRef
        self.appSetup = autoclass('org.terrier.utility.ApplicationSetup')
        if properties==None:
            self.properties=self.default_properties
        else:
            self.properties=properties
        self.properties["fat.featured.scoring.matching.features"]=";".join(features)
        props = autoclass('java.util.Properties')()
        for control,value in self.properties.items():
            props.put(control,value)
        self.appSetup.bootstrapInitialisation(props)
        if controls==None:
            self.controls=self.default_controls
        else:
            self.controls=controls
        MF = autoclass('org.terrier.querying.ManagerFactory')
        self.ManagerFactory = MF._from_(IndexRef)


    def transform(self,topics):
        results=[]
        queries=Utils.form_dataframe(topics)
        for index,row in queries.iterrows():
            srq = self.ManagerFactory.newSearchRequest(row['qid'],row['query'])
            for control,value in self.controls.items():
                srq.setControl(control,value)
            self.ManagerFactory.runSearchRequest(srq)
            srq=cast('org.terrier.querying.Request',srq)
            fres=cast('org.terrier.learning.FeaturedResultSet', srq.getResultSet())
            feats = fres.getFeatureNames()
            for i in range(fres.getResultSize()):
                elem=[]
                elem.append(row['qid'])
                elem.append(fres.getMetaItems("docno")[i])
                elem.append(fres.getScores()[i])
                feats_array = np.array([])
                for feat in feats:
                    feats_array = np.append(feats_array, fres.getFeatureScores(feat)[i])
                elem.append(feats_array)
                results.append(elem)
        res_dt=pd.DataFrame(results, columns=["qid", "docno", "score", "features"])
        print(res_dt)

class Index():
    def __init__(self, corpus, blocks=False, fields=[]):
        print(corpus)
    def addDocument(document): #??
        print(document)
    def saveIndex(path):
        print(path)
    def loadIndex(path):
        print(path)

if __name__ == "__main__":
    JIR = autoclass('org.terrier.querying.IndexRef')
    indexref = JIR.of("./index/data.properties")
    topics = Utils.parse_trec_topics_file("./vaswani_npl/query-text.trec")
    topics_light = Utils.parse_trec_topics_file("./vaswani_npl/query_light.trec")

    retr = BatchRetrieve(indexref)
    features=["BM25","PL2"]
    feat_retrieve = FeaturesBatchRetrieve(indexref, ["WMODEL:BM25","WMODEL:PL2"])
    feat_res = feat_retrieve.transform(topics_light)
    print(feat_res)


    # batch_retrieve_results=retr.transform(topics_light)
    # print(batch_retrieve_results)
    # retr.saveLastResult("dph.res")
    # retr.saveResult(batch_retrieve_results,"/home/alex/Documents/Pyterrier/result.res")

    # qrels = Utils.parse_qrels("./vaswani_npl/qrels")
    # eval = Utils.evaluate(batch_retrieve_results,qrels)
    # print(eval)


#Alternative to pytrec_eval
