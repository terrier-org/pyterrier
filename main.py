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
    def run_to_pytrec_eval(df):
        run_dict_pytrec_eval = {}
        for index, row in df.iterrows():
            if str(row['qid']) not in run_dict_pytrec_eval.keys():
                run_dict_pytrec_eval[str(row['qid'])] = {}
            run_dict_pytrec_eval[str(row['qid'])][str(row['docno'])] = int(float(row['score']))
        return(run_dict_pytrec_eval)

    # reads a file with the results of terrier batchretrieve a returns a dataframe with columns: [qid,docno,score]
    @staticmethod
    def parse_dph(file_path):
        dph_results=[]
        with (open(file_path, 'r')) as dph_file:
            for line in dph_file:
                split_line=line.split(" ")
                dph_results.append([split_line[0], split_line[2],split_line[4]])
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
        "decorate":"off",
        "wmodel": "DPH",
    }

    def __init__(self, IndexRef, controls=None):
        self.IndexRef=IndexRef
        MF = autoclass('org.terrier.querying.ManagerFactory')
        self.ManagerFactory = MF._from_(IndexRef)
        if controls==None:
            self.controls=self.default_controls

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
                    result = [query.iloc[index]['qid'],int(item.getDocid())+1,item.getScore()]
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
dph_res = Utils.parse_dph("./DPH_0.res")

batch_retrieve_results_dict = Utils.run_to_pytrec_eval(batch_retrieve_results)
dph_res_dict = Utils.run_to_pytrec_eval(dph_res)

evaluator = pytrec_eval.RelevanceEvaluator(dph_res_dict, {'map', 'ndcg'})
print(json.dumps(evaluator.evaluate(batch_retrieve_results_dict), indent=1))

# batch_retrieve_results=retr.transform(pd.DataFrame([["1","light"]],columns=['qid','query']))

# appSetup = autoclass('org.terrier.utility.ApplicationSetup')
# appSetup.setProperty("querying.processes","terrierql:TerrierQLParser,parsecontrols:TerrierQLToControls,parseql:TerrierQLToMatchingQueryTerms,matchopql:MatchingOpQLParser,applypipeline:ApplyTermPipeline,localmatching:LocalManager$ApplyLocalMatching,qe:QueryExpansion,labels:org.terrier.learning.LabelDecorator,filters:LocalManager$PostFilterProcess")
# appSetup.setProperty("querying.postfilters","decorate:SimpleDecorate,site:SiteFilter,scope:Scope")
# appSetup.setProperty("querying.default.controls","wmodel:DPH,parsecontrols:on,parseql:on,applypipeline:on,terrierql:on,localmatching:on,filters:on,decorate:on")
#
