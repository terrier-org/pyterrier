import pandas as pd
import pytrec_eval
from collections import defaultdict
import os
import deprecation


@deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.autopen()")
def autoopen(filename, mode='rb'):
    if filename.endswith(".gz"):
        import gzip
        return gzip.open(filename, mode)
    elif filename.endswith(".bz2"):
        import bz2
        return bz2.open(filename, mode)
    return open(filename, mode)

class Utils:

    @staticmethod
    @deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.write_results(res, filename, format='letor')")
    def write_results_letor(res, filename, qrels=None, default_label=0):
        if qrels is not None:
            res = res.merge(qrels, on=['qid', 'docno'], how='left').fillna(default_label)
        with autoopen(filename, "wt") as f:
            for row in res.itertuples():
                values = row.features
                label = row.label if qrels is not None else default_label
                feat_str = ' '.join( [ '%i:%f' % (i+1,values[i]) for i in range(len(values)) ] )
                f.write("%d qid:%s %s # docno=%s\n" % (label, row.qid, feat_str, row.docno))
    
    @staticmethod
    @deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.write_results(res, filename, format='trec')")
    def write_results_trec(res, filename, run_name="pyterrier"):
        res_copy = res.copy()[["qid", "docno", "rank", "score"]]
        res_copy.insert(1, "Q0", "Q0")
        res_copy.insert(5, "run_name", run_name)
        res_copy.to_csv(filename, sep=" ", header=False, index=False)

    @staticmethod
    @deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.read_results(filename, format='letor')")
    def parse_letor_results_file(filename, labels=False):

        def _parse_line(l):
                # $line =~ s/(#.*)$//;
                # my $comment = $1;
                # my @parts = split /\s+/, $line;
                # my $label = shift @parts;
                # my %hash = map {split /:/, $_} @parts;
                # return ($label, $comment, %hash);
            import re
            import numpy as np
            line, comment = l.split("#")
            line = line.strip()
            parts = re.split(r'\s+|:', line)
            label = parts.pop(0)
            m = re.search(r'docno\s?=\s?(\S+)', comment)
            docno = m.group(1)
            kv = {}
            qid = None
            #print(parts)
            for i, k in enumerate(parts):
                if i % 2 == 0:
                    if k == "qid":
                        qid = parts[i+1]
                    else:
                        kv[int(k)] = float(parts[i+1])
            features = np.array([kv[i] for i in sorted(kv.keys())])
            return (label, qid, docno, features)       

        with autoopen(filename, 'rt') as f:
            rows = []
            for line in f:
                if line.startswith("#"):
                    continue
                (label, qid, docno, features) = _parse_line(line)
                if labels:
                    rows.append([qid, docno, features, label])
                else:
                    rows.append([qid, docno, features])
            return pd.DataFrame(rows, columns=["qid", "docno", "features", "label"] if labels else ["qid", "docno", "features"])

    @staticmethod
    @deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.read_results(filename, format='trec')")
    def parse_results_file(filename):
        results = []
        df = pd.read_csv(filename, sep=r'\s+', names=["qid", "iter", "docno", "rank", "score", "name"])
        df = df.drop(columns="iter")
        df["qid"] = df["qid"].astype(str)
        df["docno"] = df["docno"].astype(str)
        df["rank"] = df["rank"].astype(int)
        df["score"] = df["score"].astype(float)
        return df

    @staticmethod
    @deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.read_topics(filename, format='singleline')")
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
        rows = []
        from jnius import autoclass
        from . import check_version
        assert check_version("5.3")
        slqIter = autoclass("org.terrier.applications.batchquerying.SingleLineTRECQuery")(filepath, tokenise)
        for q in slqIter:
            rows.append([slqIter.getQueryId(), q])
        return pd.DataFrame(rows, columns=["qid", "query"])

    @staticmethod
    @deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.read_topics(filename, format='trec')")
    def parse_trec_topics_file(file_path, doc_tag="TOP", id_tag="NUM", whitelist=["TITLE"], blacklist=["DESC","NARR"]):
        """
        Parse a file containing topics in standard TREC format

        Args:
            file_path(str): The path to the topics file

        Returns:
            pandas.Dataframe with columns=['qid','query']
            both columns have type string
        """
        from jnius import autoclass
        from . import check_version
        assert check_version("5.3")
        trecquerysource = autoclass('org.terrier.applications.batchquerying.TRECQuery')
        tqs = trecquerysource(
            [file_path], doc_tag, id_tag, whitelist, blacklist,
            # help jnius select the correct constructor 
            signature="([Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;)V")
        topics_lst=[]
        while(tqs.hasNext()):
            topic = tqs.next()
            qid = tqs.getQueryId()
            topics_lst.append([qid,topic])
        topics_dt = pd.DataFrame(topics_lst,columns=['qid','query'])
        return topics_dt

    @staticmethod
    @deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.read_topics(filename, format='trecxml')")
    def parse_trecxml_topics_file(filename, tags=["query", "question", "narrative"], tokenise=True):
        """
        Parse a file containing topics in TREC-like XML format

        Args:
            filename(str): The path to the topics file

        Returns:
            pandas.Dataframe with columns=['qid','query']
        """
        import xml.etree.ElementTree as ET
        import pandas as pd
        tags=set(tags)
        topics=[]
        tree = ET.parse(filename)
        root = tree.getroot()
        from jnius import autoclass
        tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
        for child in root.iter('topic'):
            qid = child.attrib["number"]
            query = ""
            for tag in child:
                if tag.tag in tags:
                    query_text = tag.text
                    if tokenise:
                        query_text = " ".join(tokeniser.getTokens(query_text))
                    query += " " + query_text
            topics.append((str(qid), query)) 
        return pd.DataFrame(topics, columns=["qid", "query"])

    @staticmethod
    @deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.io.read_qrels(filename)")
    def parse_qrels(file_path):
        """
        Parse a file containing qrels

        Args:
            file_path(str): The path to the qrels file

        Returns:
            pandas.Dataframe with columns=['qid','docno', 'label']
            with column types string, string, and int
        """
        df = pd.read_csv(file_path, sep=r'\s+', names=["qid", "iter", "docno", "label"])
        df = df.drop(columns="iter")
        df["qid"] = df["qid"].astype(str)
        df["docno"] = df["docno"].astype(str)
        return df

    @staticmethod
    def convert_qrels_to_dict(df):
        """
        Convert a qrels dataframe to dictionary for use in pytrec_eval

        Args:
            df(pandas.Dataframe): The dataframe to convert

        Returns:
            dict: {qid:{docno:label,},}
        """
        run_dict_pytrec_eval = defaultdict(dict)
        for row in df.itertuples():
            run_dict_pytrec_eval[row.qid][row.docno] = int(row.label)
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
        run_dict_pytrec_eval = defaultdict(dict)
        for row in df.itertuples():
            run_dict_pytrec_eval[row.qid][row.docno] = float(row.score)
        return(run_dict_pytrec_eval)

    @staticmethod
    def evaluate(res, qrels, metrics=['map', 'ndcg'], perquery=False):
        """
        Evaluate the result dataframe with the given qrels

        Args:
            res: Either a dataframe with columns=['qid', 'docno', 'score'] or a dict {qid:{docno:score,},}
            qrels: Either a dataframe with columns=['qid','docno', 'label'] or a dict {qid:{docno:label,},}
            metrics(list): A list of strings specifying which evaluation metrics to use. Default=['map', 'ndcg']
            perquery(bool): If true return each metric for each query, else return mean metrics. Default=False
        """
        if isinstance(res, pd.DataFrame):
            batch_retrieve_results_dict = Utils.convert_res_to_dict(res)
        else:
            batch_retrieve_results_dict = res
        if isinstance(qrels, pd.DataFrame):
            qrels_dic = Utils.convert_qrels_to_dict(qrels)
        else:
            qrels_dic = qrels
        req_metrics = set()
        cutdown = False
        for m in metrics:
            if m.startswith("ndcg_cut_"):
                req_metrics.add("ndcg_cut")
                cutdown = True
            # elif m.startswith("P_"):
            #     req_metrics.add("P")
            else:
                req_metrics.add(m)

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_dic, req_metrics)

        result = evaluator.evaluate(batch_retrieve_results_dict)
        if perquery:
            if not cutdown:
                return result
            # user wanted metrics like ndcg_cut_5, but we had to request ndcg_cut
            # lets cutout the metrics they didnt want

            # get any arbitrary query
            q = next(iter(result.keys()))
            todel=[]
            for m in result[q]:
                if not m in metrics:
                    todel.append(m)
            for q in result:
                for m in todel:
                    del result[q][m]
            return result

        return Utils.mean_of_measures(result, metrics)

    @staticmethod
    def ensure(dictionary, measures, qids):
        missing = 0
        for q in qids:
            if q not in dictionary:
                dictionary[q] = { m : 0 for m in measures }
                missing+=1
            # for m in measures:
            #     if m not in dictionary[q][m]:
            #         dictionary[q][m] = 0
        return (dictionary, missing)

    @staticmethod
    def mean_of_measures(result, measures=None):
        import numpy as np
        measures_sum = {}
        mean_dict = {}
        if measures is None:
            measures = next(iter(result.values()))
        measures_no_mean = set(["num_q", "num_rel", "num_ret", "num_rel_ret"])
        if "iprec_at_recall" in measures:
            measures = measures + ["iprec_at_recall_%0.2f" % cutoff for cutoff in np.arange(0., 1.1, 0.1) ]
            measures.remove("iprec_at_recall")
        for val in result.values():
            for measure in measures:
                measure_val = val[measure]
                measures_sum[measure] = measures_sum.get(measure, 0.0) + measure_val
        for measure, value in measures_sum.items():
            mean_dict[measure] = value / (1 if measure in measures_no_mean else len(result.values()) )
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
        if isinstance(query, pd.DataFrame):
            return query
        elif isinstance(query, str):
            return pd.DataFrame([["1", query]], columns=['qid', 'query'])
        # if queries is a list or tuple
        elif isinstance(query, list) or isinstance(query, tuple):
            # if the list or tuple is made of strings
            if query != [] and isinstance(query[0], str):
                indexed_query = []
                for i, item in enumerate(query):
                    # all elements must be of same type
                    assert isinstance(item, str), f"{item} is not a string"
                    indexed_query.append([str(i + 1), item])
                return pd.DataFrame(indexed_query, columns=['qid', 'query'])

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
        files = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            for name in filenames:
                files.append(os.path.join(dirpath, name))
        return sorted(files)
