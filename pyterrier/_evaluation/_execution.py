import pyterrier as pt
from pyterrier.model import coerce_dataframe_types
from ._rendering import _convert_measures
from . import MEASURES_TYPE, SYSTEM_OR_RESULTS_TYPE, SAVEMODE_TYPE, SAVEFORMAT_TYPE

from .. import Transformer
from .._ops import Compose

import ir_measures
import pandas as pd
import tqdm as tqdm_module
from ir_measures import Measure, Metric
import os
import sys
import types
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from warnings import warn

#list(iter_calc([ir_measures.AP], qrels, run))
#[Metric(query_id='Q0', measure=AP, value=1.0), Metric(query_id='Q1', measure=AP, value=1.0)]
def _ir_measures_to_dict(
        seq : Iterator[Metric], 
        measures: Sequence[Measure],
        rev_mapping : Dict[Measure,str], 
        num_q : int,
        perquery : bool = True,
        backfill_qids : Optional[Sequence[str]] = None) -> Union[ Dict[str, Dict[str, float]], Dict[str, float]]:
    from collections import defaultdict
    if perquery:
        # qid -> measure -> value
        rtr_perquery : Dict[str, Dict[str, float]] = defaultdict(dict)
        for metric in seq:
            measure = metric.measure
            measure_name = rev_mapping.get(measure, str(measure))
            rtr_perquery[metric.query_id][measure_name] = metric.value
        # When reporting per-query results, it can desirable to show something for topics that were executed
        # do not have corresponding qrels. If the caller passes in backfill_qids, we'll ensure that these
        # qids are present, and if not add placeholders with NaN values for all measures.
        if backfill_qids is not None:
            backfill_count = 0
            for qid in backfill_qids:
                if qid not in rtr_perquery:
                    backfill_count += 1
                    for m in measures:
                        rtr_perquery[qid][rev_mapping.get(m, str(m))] = float('NaN')
            if backfill_count > 0:
                warn(f'{backfill_count} topic(s) not found in qrels. Scores for these topics are given as NaN and should not contribute to averages.')
        return rtr_perquery
    assert backfill_qids is None, "backfill_qids only supported when perquery=True"
    
    metric_agg = {rev_mapping.get(m, str(m)): m.aggregator() for m in measures}
    for metric in seq:
        measure_name = rev_mapping.get(metric.measure, str(metric.measure))
        metric_agg[measure_name].add(metric.value)
    
    rtr_aggregated : Dict[str,float] = {} # measure -> value
    for m_name in metric_agg:
        rtr_aggregated[m_name] = metric_agg[m_name].result()
    return rtr_aggregated

def _validate_R_is_res(df : pd.DataFrame):
    found = []
    unfound = []
    for c in ["qid", "docno", "score", "rank"]:
        if c in df.columns:
            found.append(c)
        else:
            unfound.append(c)
    if len(unfound):
        raise TypeError("save_dir was set, but results dont look like R (expected and found %s, missing %s). You probably need to set save_format kwarg, "
                        "e.g. save_format=pickle" %
                        (str(found), str(unfound)))


def _run_and_evaluate(
        system : SYSTEM_OR_RESULTS_TYPE,
        topics : Optional[pd.DataFrame],
        qrels: pd.DataFrame,
        metrics : MEASURES_TYPE,
        pbar : Optional[tqdm_module.tqdm] = None,
        save_mode : Optional[SAVEMODE_TYPE] = None,
        save_file : Optional[str] = None,
        save_format : SAVEFORMAT_TYPE = 'trec',
        perquery : bool = False,
        batch_size : Optional[int] = None,
        backfill_qids : Optional[Sequence[str]] = None):

    from ..io import read_results, write_results

    if pbar is None:
        pbar = pt.tqdm(disable=True)

    metrics, rev_mapping = _convert_measures(metrics)
    qrels = pt.model.to_ir_measures(qrels)
    from timeit import default_timer as timer
    runtime : float = 0.
    num_q = qrels['query_id'].nunique()
    if save_file is not None and os.path.exists(save_file):
        if save_mode == 'reuse':
            if save_format == 'trec':
                system = read_results(save_file)
            elif isinstance(save_format, types.ModuleType):
                with pt.io.autoopen(save_file, 'rb') as fin:
                    system = save_format.load(fin)
            elif isinstance(save_format, tuple) and len(save_format) == 2:
                with pt.io.autoopen(save_file, 'rb') as fin:
                    system = save_format[0](fin)
            else:
                raise ValueError("Unknown save_format %s" % str(save_format))
        elif save_mode == 'overwrite':
            os.remove(save_file)
        elif save_mode == 'warn':
            warn(("save_dir is set, but the file '%s' already exists. If you are aware of are happy to reuse this " % save_file)+
                             "file to speed up evaluation, set save_mode='reuse'; if you want to overwrite it, set save_mode='overwrite'."+
                             " To make this condition an error, use save_mode='error'.")
        elif save_mode == 'error':
            raise ValueError(("save_dir is set, but the file '%s' already exists. If you are aware of are happy to reuse this " % save_file)+
                             "file to speed up evaluation, set save_mode='reuse'; if you want to overwrite it, set save_mode='overwrite'."+
                              "To make this condition a warning, use save_mode='warn'.")
        else:
            raise ValueError("Unknown save_mode argument '%s', valid options are 'error', 'warn', 'reuse' or 'overwrite'." % save_mode)

    res : pd.DataFrame
    # if its a DataFrame, use it as the results
    if isinstance(system, pd.DataFrame):
        res = system
        res = coerce_dataframe_types(res)
        if len(res) == 0:
            if topics is None:
                raise ValueError("No topics specified, and no results in dataframe")
            else:
                raise ValueError("%d topics, but no results in dataframe" % len(topics))
        evalMeasuresDict = _ir_measures_to_dict(
            ir_measures.iter_calc(metrics, qrels, pt.model.to_ir_measures(res)),
            metrics,
            rev_mapping,
            num_q,
            perquery,
            backfill_qids)
        pbar.update()

    elif batch_size is None:

        assert topics is not None, "topics must be specified"
        #transformer, evaluate all queries at once

        starttime = timer()
        res = system.transform(topics)
        endtime = timer()
        runtime =  float(endtime - starttime) * 1000.

        # write results to save_file; we can be sure this file does not exist
        if save_file is not None:
            if save_format == 'trec':
                _validate_R_is_res(res)
                write_results(res, save_file)
            elif isinstance(save_format, types.ModuleType):
                with pt.io.autoopen(save_file, 'wb') as fout:
                    save_format.dump(res, fout)
            elif isinstance(save_format, tuple) and len(save_format) == 2:
                with pt.io.autoopen(save_file, 'wb') as fout:
                    save_format[1](res, fout)
            else:
                raise ValueError("Unknown save_format %s" % str(save_format))
        res = coerce_dataframe_types(res)

        if len(res) == 0:
            raise ValueError("%d topics, but no results received from %s" % (len(topics), str(system)) )

        evalMeasuresDict = _ir_measures_to_dict(
            ir_measures.iter_calc(metrics, qrels, pt.model.to_ir_measures(res)),
            metrics,
            rev_mapping,
            num_q,
            perquery,
            backfill_qids)
        pbar.update()
    else:
        assert topics is not None, "topics must be specified"
        if save_file is not None:
            # only 
            assert save_format == 'trec', 'save_format=%s is not supported when save_dir is enabled and batch_size is not None' % str(save_format)

        #transformer, evaluate queries in batches
        assert batch_size > 0
        starttime = timer()
        evalMeasuresDict = {}
        remaining_qrel_qids = set(qrels.query_id)
        try:
            batch_topics : pd.DataFrame
            for i, (res, batch_topics) in enumerate( system.transform_gen(topics, batch_size=batch_size, output_topics=True)):
                if len(res) == 0:
                    raise ValueError("batch of %d topics, but no results received in batch %d from %s" % (len(batch_topics), i, str(system) ) )
                endtime = timer()
                runtime += (endtime - starttime) * 1000.

                # write results to save_file; we will append for subsequent batches
                if save_file is not None:
                    _validate_R_is_res(res)
                    write_results(res, save_file, append=True)

                res = coerce_dataframe_types(res)
                batch_qids = set(batch_topics.qid)
                batch_qrels = qrels[qrels.query_id.isin(batch_qids)] # filter qrels down to just the qids that appear in this batch
                remaining_qrel_qids.difference_update(batch_qids)
                batch_backfill = [qid for qid in backfill_qids if qid in batch_qids] if backfill_qids is not None else None
                evalMeasuresDict.update(_ir_measures_to_dict( # type: ignore[arg-type]
                    ir_measures.iter_calc(metrics, batch_qrels, pt.model.to_ir_measures(res)),
                    metrics,
                    rev_mapping,
                    num_q,
                    perquery=True,
                    backfill_qids=batch_backfill))
                pbar.update()
                starttime = timer()
        except:
            # if an error is thrown, we need to clean up our existing file
            if save_file is not None and os.path.exists(save_file):
                os.remove(save_file)
            raise
        if remaining_qrel_qids:
            # there are some qids in the qrels that were not in the topics. Get the default values for these and update evalMeasuresDict
            missing_qrels = qrels[qrels.query_id.isin(remaining_qrel_qids)]
            empty_res = pd.DataFrame([], columns=['query_id', 'doc_id', 'score'])
            evalMeasuresDict.update(_ir_measures_to_dict( # type: ignore[arg-type]
                ir_measures.iter_calc(metrics, missing_qrels, empty_res),
                metrics,
                rev_mapping,
                num_q,
                perquery=True))
        if not perquery:
            # aggregate measures if not in per query mode
            aggregators: Dict[str, Any] = {rev_mapping.get(m, str(m)): m.aggregator() for m in metrics}
            q : str
            for q in evalMeasuresDict:
                for metric in metrics:
                    s_metric = rev_mapping.get(metric, str(metric))
                    aggregators[s_metric].add(evalMeasuresDict[q][s_metric]) #type: ignore
            evalMeasuresDict = {m: agg.result() for m, agg in aggregators.items()}
    return (runtime, evalMeasuresDict)


def _identifyCommon(pipes : List[Union[pt.Transformer, pd.DataFrame]]) -> Tuple[Optional[pt.Transformer], List[Union[pt.Transformer,pd.DataFrame]]]:
    # constructs a common prefix pipeline across a list of pipelines, along with various suffices. 
    # pt.Transformer.identity() is used for a no-op suffix

    # no precomputation for single-system case
    if len(pipes) == 1:
        return None, pipes
    pipe_lists: List[List[pt.Transformer]] = []
    for p in pipes:
        # no optimisation possible for experiments involving dataframes as systems
        if isinstance(p, pd.DataFrame):
            return None, pipes
        if isinstance(p, Compose):
            pipe_lists.append(list(p._transformers))
        else:
            if not isinstance(p, pt.Transformer):
                raise ValueError("pt.Experiment has systems that are not either DataFrames or Transformers")
            pipe_lists.append([p])

    common_prefix, suffices  = _common_prefix(pipe_lists)

    if len(common_prefix) == 0:
        # no common prefix, return existing pipelines as-is
        return None, pipes

    def _construct(inp: List[pt.Transformer]) -> pt.Transformer:
        # use identify as a no-op
        if len(inp) == 0:
            return pt.Transformer.identity()
        # use transformer itself
        if len(inp) == 1:
            return inp[0]
        # more than 1, compose...
        return Compose(*inp)

    return (
        _construct(common_prefix), # prefix common to all 
        [ _construct(remainder) for remainder in suffices ] # individual suffices
    )


def _precomputation(
        retr_systems : List[SYSTEM_OR_RESULTS_TYPE],
        topics : pd.DataFrame,
        precompute_prefix : bool,
        verbose : bool,
        batch_size : Optional[int] = None
        ) -> Tuple[float, pd.DataFrame, List[SYSTEM_OR_RESULTS_TYPE]]:

    # this method identifies any common prefix to the pipelines in retr_systems,
    # and if precompute_prefix=True, then it computes the (partial) results of topics
    # on that prefix, and returns that, which is used later for evaluating the remainder
    # of each pipeline.

    tqdm_args_precompute: Dict[str, Any] = {
        'disable' : not verbose,
    }

    common_pipe, execution_retr_systems = _identifyCommon(retr_systems)
    precompute_time = 0.
    if precompute_prefix and common_pipe is not None:
        print("Precomputing results of %d topics on shared pipeline component %s" % (len(topics), str(common_pipe)), file=sys.stderr)

        tqdm_args_precompute['desc'] = "pt.Experiment precomputation"
        from timeit import default_timer as timer
        starttime = timer()
        if batch_size is not None:

            warn("precompute_prefix with batch_size is very experimental. Please report any problems")
            import math
            tqdm_args_precompute['unit'] = 'batches'
            # round number of batches up for each system
            tqdm_args_precompute['total'] = math.ceil((len(topics) / batch_size))
            with pt.tqdm(**tqdm_args_precompute) as pbar:
                precompute_results : List[pd.DataFrame] = []
                for r in common_pipe.transform_gen(topics, batch_size=batch_size):
                    assert isinstance(r, pd.DataFrame) # keep mypy happy
                    precompute_results.append(r)
                    pbar.update(1)
                execution_topics = pd.concat(precompute_results)

        else: # no batching  
            tqdm_args_precompute['total'] = 1
            tqdm_args_precompute['unit'] = "prefix pipeline"
            with pt.tqdm(**tqdm_args_precompute) as pbar:
                execution_topics = common_pipe(topics)
                pbar.update(1)

        endtime = timer()
        precompute_time = float(endtime - starttime) * 1000.

    elif precompute_prefix and common_pipe is None:
        warn('precompute_prefix was True for pt.Experiment, but no common pipeline prefix was found among %d pipelines' % len(retr_systems))
        execution_retr_systems = retr_systems
        execution_topics = topics

    else: # precomputation not requested
        execution_retr_systems = retr_systems
        execution_topics = topics

    return precompute_time, execution_topics, execution_retr_systems


def _common_prefix(pipes: List[List[Transformer]]) -> Tuple[List[Transformer], List[List[Transformer]]]:
    # finds the common prefix within a list of transformers
    assert len(pipes) > 1, "pipes must contain at least two lists"
    common_prefix = []
    for stage in zip(*pipes):
        elements = set(stage) # uses Transformer.__equals__ 
        if len(elements) == 1:
            common_prefix.append(next(iter(elements)))
        else:
            break
    suffixes = [p[len(common_prefix):] for p in pipes]
    return common_prefix, suffixes