from pyterrier import Transformer
from typing import Dict, List, Literal, Optional, Tuple, Union, overload
from ._execution import _run_and_evaluate
from ._utils import _restore_state, _save_state
from . import MEASURE_TYPE, MEASURES_TYPE
from ir_measures import Measure
import pandas as pd
import pyterrier as pt

TRANSFORMER_PARAMETER_VALUE_TYPE = Union[str,float,int,str]
GRID_SCAN_PARAM_SETTING = Tuple[
            Transformer, 
            str, 
            TRANSFORMER_PARAMETER_VALUE_TYPE
        ]
GRID_SEARCH_RETURN_TYPE_SETTING = Tuple[
    float, 
    List[GRID_SCAN_PARAM_SETTING]
]

GRID_SEARCH_RETURN_TYPE_BOTH = Tuple[
    Transformer,
    float, 
    List[GRID_SCAN_PARAM_SETTING]
]


@overload
def GridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE,
        jobs : int,
        backend: str,
        verbose: bool ,
        batch_size : Optional[int],
        return_type : Literal['opt_pipeline'],
    ) -> Transformer: ...

@overload
def GridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE,
        jobs : int,
        backend: str,
        verbose: bool ,
        batch_size : Optional[int],
        return_type : Literal['best_setting'],
    ) -> GRID_SEARCH_RETURN_TYPE_SETTING: ...

@overload
def GridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE,
        jobs : int,
        backend: str,
        verbose: bool ,
        batch_size : Optional[int],
        return_type : Literal['both'],
    ) -> GRID_SEARCH_RETURN_TYPE_BOTH: ...

def GridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metric : MEASURE_TYPE = "map",
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size : Optional[int] = None,
        return_type : Literal['opt_pipeline', 'best_setting', 'both'] = "opt_pipeline"
    ) -> Union[Transformer,GRID_SEARCH_RETURN_TYPE_SETTING,GRID_SEARCH_RETURN_TYPE_BOTH]:
    """
    GridSearch is essentially, an argmax GridScan(), i.e. it returns an instance of the pipeline to tune
    with the best parameter settings among params, that were found that were obtained using the specified
    topics and qrels, and for the specified measure.

    :param pipeline: a transformer or pipeline to tune
    :param params: a two-level dictionary, mapping transformer to param name to a list of values
    :param topics: topics to tune upon
    :param qrels: qrels to tune upon       
    :param metric: name of the metric on which to determine the most effective setting. Defaults to "map".
    :param batch_size: If not None, evaluation is conducted in batches of batch_size topics. Default=None, which evaluates all topics at once. 
        Applying a batch_size is useful if you have large numbers of topics, and/or if your pipeline requires large amounts of temporary memory
        during a run. Default is None.
    :param jobs: Number of parallel jobs to run. Default is 1, which means sequentially.
    :param backend: Parallelisation backend to use. Defaults to "joblib". 
    :param verbose: whether to display progress bars or not
    :param return_type: whether to return the same transformer with optimal pipeline setting, and/or a setting of the
        higher metric value, and the resulting transformers and settings.
    """
    # save state
    initial_state = _save_state(params)

    if isinstance(metric, list):
        raise KeyError("GridSearch can only maximise ONE metric, but you passed a list (%s)." % str(metric))

    grid_outcomes = GridScan(
        pipeline, 
        params, 
        topics, 
        qrels, 
        [metric], 
        jobs, 
        backend, 
        verbose, 
        batch_size, 
        dataframe=False)
    assert not isinstance(grid_outcomes, pd.DataFrame)

    assert len(grid_outcomes) > 0, "GridScan returned 0 rows"
    max_measure = grid_outcomes[0][1][metric]
    max_setting = grid_outcomes[0][0]
    for setting, measures in grid_outcomes: # TODO what is the type of this iteration?
        if measures[metric] > max_measure:
            max_measure = measures[metric]
            max_setting = setting
    print("Best %s is %f" % (metric, max_measure))
    print("Best setting is %s" % str(["%s %s=%s" % (str(t), k, v) for t, k, v in max_setting]))

    if return_type == "opt_pipeline":
        for tran, param, value in max_setting:
            tran.set_parameter(param, value)
        return pipeline
    if return_type == "best_setting":
        _restore_state(initial_state)
        return max_measure, max_setting
    if return_type == "both":
        for tran, param, value in max_setting:
            tran.set_parameter(param, value)
        return (pipeline, max_measure, max_setting)
    raise ValueError("Unknown return_type option %s" % return_type)

def GridScan(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics : pd.DataFrame,
        qrels : pd.DataFrame,
        metrics : Union[MEASURE_TYPE,MEASURES_TYPE] = ["map"],
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size = None,
        dataframe = True,
    ) -> Union[pd.DataFrame, List [ Tuple [ List[ GRID_SCAN_PARAM_SETTING ], Dict[Union[str, Measure] ,float]  ]  ] ]:
    """
    GridScan applies a set of named parameters on a given pipeline and evaluates the outcome. The topics and qrels 
    must be specified. The trec_eval measure names can be optionally specified.
    The transformers being tuned, and their respective parameters are named in the param_dict. The parameter being
    varied must be changable using the :func:`set_parameter()` method. This means instance variables,
    as well as controls in the case of Retriever.

    :param pipeline: a transformer or pipeline
    :param params: a two-level dictionary, mapping transformer to param name to a list of values
    :param topics: topics to tune upon
    :param qrels: qrels to tune upon       
    :param metrics): name of the metrics to report for each setting. Defaults to ["map"].
    :param batch_size: If not None, evaluation is conducted in batches of batch_size topics. Default=None, which evaluates all topics at once. 
        Applying a batch_size is useful if you have large numbers of topics, and/or if your pipeline requires large amounts of temporary memory
        during a run. Default is None.
    :param jobs: Number of parallel jobs to run. Default is 1, which means sequentially.
    :param backend: Parallelisation backend to use. Defaults to "joblib". 
    :param verbose: whether to display progress bars or not
    :param dataframe: return a dataframe or a list

    :return: A dataframe showing the effectiveness of all evaluated settings, if dataframe=True
        A list of settings and resulting evaluation measures, if dataframe=False
    
    Raises:
        ValueError: if a specified transformer does not have such a parameter

    Example::

        # graph how PL2's c parameter affects MAP
        pl2 = pt.terrier.Retriever(index, wmodel="PL2", controls={'c' : 1})
        rtr = pt.GridScan(
            pl2, 
            {pl2 : {'c' : [0.1, 1, 5, 10, 20, 100]}}, 
            topics,
            qrels,
            ["map"]
        )
        import matplotlib.pyplot as plt
        plt.plot(rtr["tran_0_c"], rtr["map"])
        plt.xlabel("PL2's c value")
        plt.ylabel("MAP")
        plt.show()

    """
    import itertools

    if verbose and jobs > 1:
        from warnings import warn
        warn("Cannot provide progress on parallel job")
    if isinstance(metrics, str):
        metrics = [metrics]

    # Store the all parameter names and candidate values into a dictionary, keyed by a tuple of the transformer and the parameter name
    # such as {(Retriever, 'wmodel'): ['BM25', 'PL2'], (Retriever, 'c'): [0.1, 0.2, 0.3], (Bla, 'lr'): [0.001, 0.01, 0.1]}
    candi_dict: Dict[Tuple[Transformer, str], List[TRANSFORMER_PARAMETER_VALUE_TYPE]] = {}
    for tran, param_set in params.items():
        for param_name, values in param_set.items():
            candi_dict[ (tran, param_name) ] = values
    if len(candi_dict) == 0:
        raise ValueError("No parameters specified to optimise")
    for tran, param in candi_dict:
        try:
            tran.get_parameter(param)
        except Exception:
            raise ValueError("Transformer %s does not expose a parameter named %s" % (str(tran), param))
    
    keys, vals = zip(*candi_dict.items())
    combinations = list(itertools.product(*vals))
    assert len(combinations) > 0, "No combinations selected"

    def _evaluate_one_setting(keys, values):
        #'params' is every combination of candidates
        params = dict(zip(keys, values))
        parameter_list = []
        
        # Set the parameter value in the corresponding transformer of the pipeline
        for (tran, param_name), value in params.items():
            tran.set_parameter(param_name, value)
            # such as (Retriever, 'wmodel', 'BM25')
            parameter_list.append( (tran, param_name, value) )
            
        time, eval_scores = _run_and_evaluate(pipeline, topics, qrels, metrics, perquery=False, batch_size=batch_size)
        return parameter_list, eval_scores

    def _evaluate_several_settings(inputs : List[Tuple]):
        return [_evaluate_one_setting(k,v) for k, v in inputs]

    eval_list = []
    #for each combination of parameter values
    if jobs == 1:
        for v in pt.tqdm(combinations, total=len(combinations), desc="GridScan", mininterval=0.3) if verbose else combinations:
            parameter_list, eval_scores = _evaluate_one_setting(keys, v)
            eval_list.append( (parameter_list, eval_scores) )
    else:
        import itertools
        import more_itertools
        try:
            from pyterrier_alpha.parallel import parallel_lambda # type: ignore
        except ImportError as ie:
            raise ImportError("pyterrier-alpha[parallel] must be installed for jobs>1") from ie
    
        all_inputs = [(keys, values) for values in combinations]

        # how many jobs to distribute this to
        num_batches = int(len(combinations)/jobs) if len(combinations) >= jobs else len(combinations)

        # built the batches to distribute
        batched_inputs = list(more_itertools.chunked(all_inputs, num_batches))
        assert len(batched_inputs) > 0, "No inputs identified for parallel_lambda"
        eval_list = parallel_lambda(_evaluate_several_settings, batched_inputs, jobs, backend=backend)
        eval_list =  list(itertools.chain(*eval_list))
        assert len(eval_list) > 0, "parallel_lambda returned 0 rows" 
    
    # resulting eval_list has the form [ 
    #   ( [(BR, 'wmodel', 'BM25'), (BR, 'c', 0.2)]  ,   {"map" : 0.2654} )
    # ]
    # ie, a list of possible settings, combined with measure values    
    if not dataframe:
        return eval_list

    rtr=[]
    for setting, measures in eval_list:
        row={}
        for i, (tran, param, value) in enumerate(setting):
            row["tran_%d" % i]  = tran
            row["tran_%d_%s" % (i,param) ]  = value
        row.update(measures)
        rtr.append(row)
    # resulting dataframe looks like:
    #    tran_0  tran_0_c       map
    #0  BR(PL2)     0.1  0.104820
    #1  BR(PL2)     1.0  0.189274
    #2  BR(PL2)     5.0  0.230838
    return pd.DataFrame(rtr)

def KFoldGridSearch(
        pipeline : Transformer,
        params : Dict[Transformer,Dict[str,List[TRANSFORMER_PARAMETER_VALUE_TYPE]]],
        topics_list : List[pd.DataFrame],
        qrels : Union[pd.DataFrame,List[pd.DataFrame]],
        metric : MEASURE_TYPE = "map",
        jobs : int = 1,
        backend='joblib',
        verbose: bool = False,
        batch_size : Optional[int] = None) -> Tuple[pd.DataFrame, List[List[GRID_SCAN_PARAM_SETTING]]]:
    """
    Applies a GridSearch using different folds. It returns the *results* of the 
    tuned transformer pipeline on the test topics. The number of topics dataframes passed
    to topics_list defines the number of folds. For each fold, all but one of the dataframes
    is used as training, and the remainder used for testing. 

    The state of the transformers in the pipeline is restored after the KFoldGridSearch has
    been executed.

    :param pipeline: a transformer or pipeline to tune
    :param params: a two-level dictionary, mapping transformer to param name to a list of values
    :param topics_list: a *list* of topics dataframes to tune upon
    :param qrels: qrels to tune upon. A single dataframe, or a list for each fold.       
    :param metric: name of the metric on which to determine the most effective setting. Defaults to "map".
    :param batch_size: If not None, evaluation is conducted in batches of batch_size topics. Default=None, which evaluates all topics at once. 
        Applying a batch_size is useful if you have large numbers of topics, and/or if your pipeline requires large amounts of temporary memory
        during a run. Default is None.
    :param jobs: Number of parallel jobs to run. Default is 1, which means sequentially.
    :param backend: Parallelisation backend to use. Defaults to "joblib". 
    :param verbose(bool): whether to display progress bars or not

    :return: A tuple containing, firstly, the results of pipeline on the test topics after tuning, and secondly, a list of the best parameter settings for each fold.

    Consider tuning a terrier.Retriever PL2 where the folds of queries are pre-determined::

        pl2 = pt.terrier.Retriever(index, wmodel="PL2", controls={'c' : 1})
        tuned_pl2, _ = pt.KFoldGridSearch(
            pl2, 
            {pl2 : {'c' : [0.1, 1, 5, 10, 20, 100]}}, 
            [topicsf1, topicsf2],
            qrels,
            ["map"]
        )
        pt.Experiment([pl2, tuned_pl2], all_topics, qrels, ["map"])

    As 2 splits are defined, PL2 is first tuned on topicsf1 and tested on topicsf2, then 
    trained on topicsf2 and tested on topicsf1. The results dataframe of PL2 after tuning of the c
    parameter are returned by the KFoldGridSearch, and can be used directly in a pt.Experiment().
    """

    import pandas as pd
    num_folds = len(topics_list)
    if isinstance(qrels, pd.DataFrame):
        qrels = [qrels] * num_folds

    FOLDS=list(range(0, num_folds))
    results : List[pd.DataFrame] = []
    settings=[]

    # save state
    initial_state = _save_state(params)

    for fold in FOLDS:
        print("Fold %d" % (fold+1))

        train_indx = FOLDS.copy()
        train_indx.remove(fold)
        train_topics = pd.concat([topics_list[offset] for offset in train_indx])
        train_qrels = pd.concat([qrels[offset] for offset in train_indx])
        test_topics = topics_list[fold]
        #test_qrels arent needed
        #test_qrels = qrels[fold]

        # safety - give the GridSearch a stable initial setting
        _restore_state(initial_state)

        optPipe: Transformer
        max_measure: float
        max_setting: List[GRID_SCAN_PARAM_SETTING]
        optPipe, max_measure, max_setting = GridSearch(
            pipeline,
            params,
            train_topics,
            train_qrels,
            metric,
            jobs=jobs,
            backend=backend,
            verbose=verbose,
            batch_size=batch_size,
            return_type="both")
        results.append(optPipe.transform(test_topics))
        settings.append(max_setting)

    # restore state
    _restore_state(initial_state)

    return (pd.concat(results), settings)