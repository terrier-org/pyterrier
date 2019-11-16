import jnius_config, os, pytrec_eval,json
import pandas as pd
# from types import

jnius_config.set_classpath("terrier-project-5.1-jar-with-dependencies.jar")
from jnius import autoclass

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


    def transform(self,queries):
        results=[]
        queries=BatchRetrieve.form_dataframe(queries)
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
        return res_dt

    def evaluate(self,res,qrels):
        batch_retrieve_results_dict = Utils.convert_df_to_pytrec_eval(res)
        qrels_dic=Utils.convert_df_to_pytrec_eval(qrels, True)
        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dic, {'map', 'ndcg'})
        return json.dumps(evaluator.evaluate(batch_retrieve_results_dict), indent=1)

    def setControls(controls):
        self.controls=controls

    def setControl(control,value):
        self.controls[control]=value

# set terrier home
# system = autoclass("java.lang.System")
# system.setProperty("terrier.home","/home/alex/Downloads/terrier-project-5.1");

if __name__ == "__main__":
    JIR = autoclass('org.terrier.querying.IndexRef')
    indexref = JIR.of("./index/data.properties")
    topics = Utils.parse_trec_topics_file("./vaswani_npl/query-text.trec")

    retr = BatchRetrieve(indexref)

    batch_retrieve_results=retr.transform(topics)
    qrels = Utils.parse_qrels("./vaswani_npl/qrels")
    eval = retr.evaluate(batch_retrieve_results,qrels)
    print(eval)
