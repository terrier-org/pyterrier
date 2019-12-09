import jnius_config
jnius_config.add_classpath("../terrier-project-5.1-jar-with-dependencies.jar")
from jnius import autoclass, cast

import os, pytrec_eval,json
import numpy as np
import pandas as pd
from utils import *
from batchretrieve import *


if __name__ == "__main__":
    JIR = autoclass('org.terrier.querying.IndexRef')
    indexref = JIR.of("../index/data.properties")
    topics = Utils.parse_trec_topics_file("../vaswani_npl/query-text.trec")
    topics_light = Utils.parse_trec_topics_file("../vaswani_npl/query_light.trec")

    feat_retrieve = FeaturesBatchRetrieve(indexref, ["WMODEL:BM25","WMODEL:PL2"])
    feat_res = feat_retrieve.transform(topics_light)
    print(feat_res)

    # retr = BatchRetrieve(indexref)
    # batch_retrieve_results=retr.transform(topics_light)
    # print(batch_retrieve_results)
    # retr.saveLastResult("dph.res")
    # retr.saveResult(batch_retrieve_results,"/home/alex/Documents/Pyterrier/result.res")

    # qrels = Utils.parse_qrels("./vaswani_npl/qrels")
    # eval = Utils.evaluate(batch_retrieve_results,qrels)
    # print(eval)
