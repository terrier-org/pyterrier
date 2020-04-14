import pandas as pd
import pytrec_eval
import json
import ast
import os

class Utils:

    @staticmethod
    def parse_query_result(filename):
        results=[]
        with open(filename, 'r') as file:
            for line in file:
                split_line=line.strip("\n").split(" ")
                results.append([split_line[1],float(split_line[2])])
        return results

    @staticmethod
    def parse_res_file(filename):
        results=[]
        with open(filename, 'r') as file:
            for line in file:
                split_line=line.strip("\n").split(" ")
                results.append([split_line[0],split_line[2],float(split_line[4])])
        return results

    @staticmethod
    def parse_singleline_topics_file(filepath, tokenise=True):
        """
        Parse a file containing topics, one per line

        Args:
            file_path(str): The path to the topics file
            tokenise(bool): whether the query should be tokenised, using Terrier's standard Tokeniser. 
                If you are using matchop formatted topics, this should be set to False.

        Returns:
            pandas.Dataframe with columns=['qid','query']
        """
        rows=[]
        from jnius import autoclass
        system = autoclass("java.lang.System")
        system.setProperty("SingleLineTRECQuery.tokenise", "true" if tokenise else "false")
        slqIter = autoclass("org.terrier.applications.batchquerying.SingleLineTRECQuery")(filepath)
        for q in slqIter:
            rows.append([slqIter.getQueryId(), q])
        return pd.DataFrame(rows, columns=["qid", "query"])

    @staticmethod
    def parse_trec_topics_file(file_path):
        """
        Parse a file containing topics in standard TREC format

        Args:
            file_path(str): The path to the topics file

        Returns:
            pandas.Dataframe with columns=['qid','query']
        """
        from jnius import autoclass
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

    @staticmethod
    def parse_qrels(file_path):
        """
        Parse a file containing qrels

        Args:
            file_path(str): The path to the qrels file

        Returns:
            pandas.Dataframe with columns=['qid','docno', 'label']
        """
        dph_results=[]
        with (open(file_path, 'r')) as qrels_file:
            for line in qrels_file:
                split_line=line.strip("\n").split(" ")
                dph_results.append([split_line[0], split_line[2],int(split_line[3])])
        res_dt = pd.DataFrame(dph_results,columns=['qid','docno','label'])
        return res_dt

    @staticmethod
    def convert_qrels_to_dict(df):
        """
        Convert a qrels dataframe to dictionary for use in pytrec_eval

        Args:
            df(pandas.Dataframe): The dataframe to convert

        Returns:
            dict: {qid:{docno:label,},}
        """
        run_dict_pytrec_eval = {}
        for index, row in df.iterrows():
            if row['qid'] not in run_dict_pytrec_eval.keys():
                run_dict_pytrec_eval[row['qid']] = {}
            run_dict_pytrec_eval[row['qid']][row['docno']] = int(row['label'])
        return(run_dict_pytrec_eval)

    @staticmethod
    def convert_res_to_dict(df):
        """
        Convert a result dataframe to dictionary for use in pytrec_eval

        Args:
            df(pandas.Dataframe): The dataframe to convert

        Returns:
            dict: {qid:{docno:score,},}
        """
        run_dict_pytrec_eval = {}
        for index, row in df.iterrows():
            if row['qid'] not in run_dict_pytrec_eval.keys():
                run_dict_pytrec_eval[row['qid']] = {}
            if "predicted" in df.columns:
                run_dict_pytrec_eval[row['qid']][row['docno']] = float(row['predicted'])
            else:
                run_dict_pytrec_eval[row['qid']][row['docno']] = float(row['score'])
        return(run_dict_pytrec_eval)

    @staticmethod
    def evaluate(res,qrels, metrics = ['map', 'ndcg'], perquery=False):
        """
        Evaluate the result dataframe with the given qrels

        Args:
            res: Either a dataframe with columns=['qid', 'docno', 'score'] or a dict {qid:{docno:score,},}
            qrels: Either a dataframe with columns=['qid','docno', 'label'] or a dict {qid:{docno:label,},}
            metrics(list): A list of strings specifying which evaluation metrics to use. Default=['map', 'ndcg']
            perquery(bool): If true return each metric for each query, else return mean metrics. Default=False
        """

        if (type(res)==type(pd.DataFrame())):
            batch_retrieve_results_dict = Utils.convert_res_to_dict(res)
        else:
            batch_retrieve_results_dict=res

        if (type(qrels)==type(pd.DataFrame())):
            qrels_dic=Utils.convert_qrels_to_dict(qrels)
        else:
            qrels_dic=qrels

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dic, set(metrics))
        result = evaluator.evaluate(batch_retrieve_results_dict)
        if perquery:
            return result
        else:
            measures_sum = {}
            mean_dict = {}
            for val in result.values():
                for measure, measure_val in val.items():
                    measures_sum[measure]=measures_sum.get(measure, 0.0)+measure_val
            for measure, value in measures_sum.items():
                mean_dict[measure]=value/len(result.values())
            return mean_dict

    # create a dataframe of string of queries or a list or tuple of strings of queries
    @staticmethod
    def form_dataframe(query):
        """
        Convert either a string or a list of strings to a dataframe for use as topics in retrieval.

        Args:
            query: Either a string or a list of strings

        Returns:
            dataframe with columns=['qid','query']
        """
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

    @staticmethod
    def get_files_in_dir(dir):
        """
        Returns all the files present in a directory and its subdirectories

        Args:
            dir(str): The directory containing the files

        Returns:
            paths(list): A list of the paths to the files
        """
        lst = []
        zip_paths = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            lst.append([dirpath,filenames])
        for sublist in lst:
            for zip in sublist[1]:
                zip_paths.append(os.path.join(sublist[0],zip))
        return zip_paths
