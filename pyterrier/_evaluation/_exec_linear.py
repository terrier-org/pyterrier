import pyterrier as pt

from . import MEASURES_TYPE, SYSTEM_OR_RESULTS_TYPE, SAVEFORMAT_TYPE, SAVEMODE_TYPE
from ._execution import _run_and_evaluate
from .. import Transformer
from .._ops import Compose
import pandas as pd
import os
import sys
import types
from typing import Sequence, List, Optional, Tuple, Dict, Any, Union, Literal
from warnings import warn


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
        if common_pipe is not None:
            warn(
                "There are shared pipeline components among %d pipelines. Consider setting plan='tree' for faster experiment." % len(retr_systems))
        execution_retr_systems = retr_systems
        execution_topics = topics

    return precompute_time, execution_topics, execution_retr_systems


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

def linear_execution(renderer,retr_systems, 
                     topics : pd.DataFrame, 
                     qrels: pd.DataFrame,
                     eval_metrics : MEASURES_TYPE,
                     names: Sequence[str], 
                     precompute_prefix: bool = True, 
                     verbose : bool = False,
                     save_dir : Optional[str] = None,
                     save_mode : Optional[SAVEMODE_TYPE] = None,
                     save_format : SAVEFORMAT_TYPE ='trec',
                     batch_size : Optional[int]=None, 
                     perquery : bool = False):
    
    # split the transformers into a common prefix and individual suffixes, improved efficiency
    precompute_time, execution_topics, execution_retr_systems = _precomputation(retr_systems, topics, precompute_prefix, verbose, batch_size)
    # progress bar construction
    tqdm_args={
        'disable' : not verbose,
        'unit' : 'system',
        'total' : len(retr_systems),
        'desc' : 'pt.Experiment'
    }

    with pt.tqdm(**tqdm_args) as pbar:
        # run and evaluate each system

        all_topic_qids = topics["qid"].values
        for sysid, (name, system) in enumerate(zip(names, execution_retr_systems)):
            save_file = None
            if save_dir is not None:
                if save_format == 'trec':
                    save_ext = 'res.gz'
                elif isinstance(save_format, types.ModuleType):
                    save_ext = 'mod'
                elif isinstance(save_format, tuple):
                    save_ext = 'custom'
                else:
                    raise ValueError("Unrecognised save_mode %s" % str(save_format)) 
                save_file = os.path.join(save_dir, "%s.%s" % (name, save_ext))
            time, evalMeasuresDict = _run_and_evaluate(
                system, execution_topics, qrels, eval_metrics, 
                perquery=True, 
                batch_size=batch_size, 
                backfill_qids=all_topic_qids if perquery else None,
                save_file=save_file,
                save_mode=save_mode,
                save_format=save_format,
                pbar=pbar)
            renderer.add_metrics(sysid, evalMeasuresDict, precompute_time + time)