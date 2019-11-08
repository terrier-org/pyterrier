import jnius_config, os, pytrec_eval,json
import pandas as pd

jnius_config.set_classpath("terrier-project-5.1-jar-with-dependencies.jar")
from jnius import autoclass

class Utils:
    @staticmethod
    def parse_trec_topics_file(file_path):
        trec = autoclass('org.terrier.applications.batchquerying.TRECQuery')
        tr = trec(file_path)
        topics_lst=[]
        while(tr.hasNext()):
            topics_lst.append([tr.getQueryId(),tr.next()])
        topics_dt = pd.DataFrame(topics_lst,columns=['qid','query'])
        return topics_dt

    # Convert dataframe with columns: [qid,docno,score] into a dict {qid1: {doc1:score,doc2:score } qid2:...}
    @staticmethod
    def qrels_to_pytrec_eval(df):
        run_dict_pytrec_eval = {}
        for index, row in df.iterrows():
            if row['qid'] not in run_dict_pytrec_eval.keys():
                run_dict_pytrec_eval[row['qid']] = {}
            run_dict_pytrec_eval[row['qid']][row['docno']] = int(row['score'])
        return(run_dict_pytrec_eval)

    # Convert dataframe with columns: [qid,docno,score] into a dict {qid1: {doc1:score,doc2:score } qid2:...}
    @staticmethod
    def run_to_pytrec_eval(df):
        run_dict_pytrec_eval = {}
        for index, row in df.iterrows():
            if str(row['qid']) not in run_dict_pytrec_eval.keys():
                run_dict_pytrec_eval[str(row['qid'])] = {}
            run_dict_pytrec_eval[str(row['qid'])][str(row['docno'])] = float(row['score'])
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

    # # reads a file with the results of terrier batchretrieve a returns a dataframe with columns: [qid,docno,score]
    # @staticmethod
    # def parse_dph(file_path):
    #     dph_results=[]
    #     with (open(file_path, 'r')) as dph_file:
    #         for line in dph_file:
    #             split_line=line.split(" ")
    #             dph_results.append([split_line[0], split_line[2],split_line[4]])
    #     res_dt = pd.DataFrame(dph_results,columns=['qid','docno','score'])
    #     return res_dt

class BatchRetrieve:
    default_controls={
        "terrierql": "on",
        "parsecontrols": "on",
        "parseql": "on",
        "applypipeline": "on",
        "localmatching": "on",
        "filters": "on",
        "decorate":"off",
        "wmodel": "DPH",
    }

    default_properties={
        "querying.processes":"terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess",
        "querying.postfilters": "decorate:SimpleDecorate,site:SiteFilter,scope:Scope",
        # "querying.default.controls": "wmodel:DPH,parsecontrols:on,parseql:on,applypipeline:on,terrierql:on,localmatching:on,filters:on,decorate:on"),
        "querying.allowed.controls": "scope,qe,qemodel,start,end,site,scope",
        "termpipelines": "Stopwords,PorterStemmer"
    }

    def __init__(self, IndexRef, controls=None, properties=None):
        self.IndexRef=IndexRef
        MF = autoclass('org.terrier.querying.ManagerFactory')
        self.ManagerFactory = MF._from_(IndexRef)
        self.appSetup = autoclass('org.terrier.utility.ApplicationSetup')

        if controls==None:
            self.controls=self.default_controls
        else:
            self.controls=controls

        if properties==None:
            self.properties=self.default_properties
        else:
            self.properties=properties

        for control,value in self.properties.items():
            self.appSetup.setProperty(control,value)

    def transform(self,query, qid=None):
        if type(query)==type(""):
            # srq = self.ManagerFactory.newSearchRequestFromQuery(query)
            srq = self.ManagerFactory.newSearchRequest(qid,query)
            for control,value in self.controls.items():
                srq.setControl(control,value)
            self.ManagerFactory.runSearchRequest(srq)
            return srq.getResults()
        else:
            results=[]
            objForNewDF=pd.DataFrame()
            for index,row in query.iterrows():
                for i, item in enumerate(retr.transform(row['query'], qid=row['qid'])):
                    # result = [query.iloc[index]['qid'],item.getDocid(),item.getScore()]
                    result = [query.iloc[index]['qid'],int(item.getDocid()),item.getScore()]
                    results.append(result)
            res_dt = pd.DataFrame(results,columns=['qid','docno','score'])
            return res_dt

    def setControls(controls):
        self.controls=controls

    def setControl(control,value):
        self.controls[control]=value

# set terrier home
system = autoclass("java.lang.System")
system.setProperty("terrier.home","/home/alex/Downloads/terrier-project-5.1");

JIR = autoclass('org.terrier.querying.IndexRef')
JMF = autoclass('org.terrier.querying.ManagerFactory')

topics = Utils.parse_trec_topics_file("./vaswani_npl/query-text.trec")
indexref = JIR.of("./index/data.properties")
retr = BatchRetrieve(indexref)

batch_retrieve_results=retr.transform(topics)
print(batch_retrieve_results)
qrels = Utils.parse_qrels("./vaswani_npl/qrels")
print(qrels)
batch_retrieve_results_dict = Utils.run_to_pytrec_eval(batch_retrieve_results)
qrels_dic=Utils.qrels_to_pytrec_eval(qrels)


evaluator = pytrec_eval.RelevanceEvaluator(qrels_dic, {'map', 'ndcg'})
print(json.dumps(evaluator.evaluate(batch_retrieve_results_dict), indent=1))


# batch_retrieve_results=retr.transform(pd.DataFrame([["1","light"]],columns=['qid','query']))
