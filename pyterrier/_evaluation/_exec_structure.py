import pyterrier as pt

from ._rendering import _convert_measures
from . import MEASURES_TYPE

from .. import Transformer

from ._execution import _precomputation, _run_and_evaluate, _ir_measures_to_dict
from pyterrier.model import coerce_dataframe_types

import ir_measures
import pandas as pd
import os
from IPython.display import HTML, display
import time
import logging
logging.basicConfig(level=logging.INFO)

from ._trie import RadixTree, decompose_pipelines
import types
from typing import Optional, Sequence, Tuple, Dict

def linear_execution(renderer,retr_systems, 
                     topics : pd.DataFrame, 
                     qrels: pd.DataFrame,
                     eval_metrics : MEASURES_TYPE,
                     names: Optional[Sequence[str]] = None, 
                     precompute_prefix: bool = True, 
                     verbose=False, 
                     save_dir=None, 
                     save_mode=None, 
                     save_format='trec',
                     batch_size=None, 
                     perquery=False):
    print("Using linear execution for pt.Experiment : ")
    
    # Warn if precomputation is turned off
    if not precompute_prefix:
        import warnings
        warnings.warn("Precomputation is turned off (precompute_prefix=False). "
                     "This may result in slower execution if pipelines share common prefixes. "
                     "Consider using precompute_prefix=True for better performance.", 
                     UserWarning)
    
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
                     precompute_prefix: bool = True,  
                     verbose=False, 
                     save_dir=None, 
                     save_mode=None, 
                     save_format='trec',
                     batch_size=None, 
                     perquery=False,
                     render_html = False):
    # build radix tree from retr_systems

    print("Using tree execution for pt.Experiment : ")
    # keys: tuple of Transformer objects; values: system id (int)
    tree: RadixTree[Tuple[Transformer], int] = RadixTree()

    for sysid, system in enumerate(decompose_pipelines(retr_systems)):
        key = tuple(system)
        tree.insert(key, sysid)
    
    if verbose:
        print("\nPipeline structure:")
        tree.print_live(names=names, clear_previous=False)
        print()

    if render_html:
        schematic = pt.schematic.radix_tree_schematic(tree, input_columns=["qid", "query"])
        display(HTML(pt.schematic.draw_html_schematic(schematic)))
        time.sleep(1.5)  # Allow time for display to render
    
    metrics, rev_mapping = _convert_measures(eval_metrics)
    qrels = pt.model.to_ir_measures(qrels)
    num_q = qrels['query_id'].nunique()
    all_topic_qids = topics["qid"].values

    assert topics is not None, "topics must be specified"
    def make_callback(batch_qrels: pd.DataFrame, backfill_qids):
    
        def callback(res: pd.DataFrame, sysid: int, cum_time: float):
            # Validate results
            if len(res) == 0:
                raise ValueError("%d topics, but no results received from system %d" % (len(topics), sysid))
            
            # Update live tree visualization if verbose
            if verbose:
                tree.print_live(names=names, clear_previous=True)
            
            # Print timing for each pipeline
            # pipeline_name = names[sysid] if names and sysid < len(names) else f"Pipeline {sysid}"
            # print(f"{pipeline_name}: {cum_time:.2f}ms")
            
            # Always use perquery=True here - renderer will handle aggregation if needed
            evalMeasuresDict = _ir_measures_to_dict(
                ir_measures.iter_calc(metrics, batch_qrels, pt.model.to_ir_measures(res)),
                metrics,
                rev_mapping,
                num_q,
                perquery=True,
                backfill_qids=backfill_qids)            
            renderer.add_metrics(sysid, evalMeasuresDict, cum_time)
        return callback

    if batch_size is None:
        # No batching - execute all queries at once   
        tree.root.traverse(topics, make_callback(qrels, all_topic_qids if perquery else None), 0.0)
    #not fully functional
    else:
        # Batch processing - evaluate queries in batches
        assert batch_size > 0
        topic_batches = pt.model.split_df(topics, batch_size=batch_size)
        
        # Track which qrels haven't been processed yet (for queries not in topics)
        # system_remaining_qrels = {sysid: set(qrels.query_id) for sysid in range(len(retr_systems))}
        
        for batch_idx, topic_batch in enumerate(topic_batches):
            if verbose:
                print(f"Processing batch {batch_idx + 1}/{len(topic_batches)}")
            
            # Get the query IDs in this batch for backfilling
            batch_qids = set(topic_batch.qid)
            batch_qrels = qrels[qrels.query_id.isin(batch_qids)]
            batch_backfill = [qid for qid in all_topic_qids if qid in batch_qids] if perquery else None

            
            tree.root.traverse(topic_batch, make_callback(batch_qrels, batch_backfill), 0.0)
        
        
        # Print final timing for each pipeline
        for sysid in range(len(retr_systems)):
            pipeline_name = names[sysid] if names and sysid < len(names) else f"Pipeline {sysid}"
            total_time = sum(renderer.mrts[sysid]) if isinstance(renderer.mrts[sysid], list) else renderer.mrts[sysid]
            print(f"{pipeline_name}: {total_time:.2f}ms")