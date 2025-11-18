import pyterrier as pt

from ._rendering import _convert_measures
from . import MEASURES_TYPE

from .. import Transformer

from ._execution import _precomputation, _run_and_evaluate, _ir_measures_to_dict
import ir_measures
import pandas as pd
import os

from ._trie import RadixTree, decompose_pipelines
import types
from typing import Optional, Sequence, Tuple

def linear_execution(renderer,retr_systems, 
                     topics : pd.DataFrame, 
                     qrels: pd.DataFrame,
                     eval_metrics : MEASURES_TYPE,
                     names: Optional[Sequence[str]] = None, 
                     precompute_prefix: bool = False, 
                     verbose=False, 
                     save_dir=None, 
                     save_mode=None, 
                     save_format='trec',
                     batch_size=None, 
                     perquery=False):
    print("Using linear execution for pt.Experiment : ")
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
            renderer.add_metrics(sysid, evalMeasuresDict, time)


def tree_execution(renderer,retr_systems, 
                     topics : pd.DataFrame, 
                     qrels: pd.DataFrame,
                     eval_metrics : MEASURES_TYPE,
                     names: Optional[Sequence[str]] = None, 
                     precompute_prefix: bool = False, 
                     verbose=False, 
                     save_dir=None, 
                     save_mode=None, 
                     save_format='trec',
                     batch_size=None, 
                     perquery=False):
    # build radix tree from retr_systems

    print("Using tree execution for pt.Experiment : ")
    # keys: tuple of Transformer objects; values: system id (int)
    tree: RadixTree[Tuple[Transformer], int] = RadixTree()

    # Insert each system individually as (tuple(system), sysid)
    for sysid, system in enumerate(decompose_pipelines(retr_systems)):
        if isinstance(system, pd.DataFrame):
            system = [pt.transformer.from_df(system)]
        key = tuple(system)
        # Insert the key and associate it with sysid
        tree.insert(key, sysid)
    
    metrics, rev_mapping = _convert_measures(eval_metrics)
    qrels = pt.model.to_ir_measures(qrels)
    num_q = qrels['query_id'].nunique()
    all_topic_qids = topics["qid"].values


    def callback(res: pd.DataFrame, eval_index: int, cum_time: float):
        evalMeasuresDict = _ir_measures_to_dict(
                ir_measures.iter_calc(metrics, qrels, pt.model.to_ir_measures(res)),
                metrics,
                rev_mapping,
                num_q,
                True,  # Always use per-query for internal processing
                all_topic_qids  # Always provide backfill_qids for per-query
                )
        renderer.add_metrics(eval_index, evalMeasuresDict, cum_time)
    
    tree.root.traverse(topics, callback, 0.0)