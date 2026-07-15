import pyterrier as pt

from ._rendering import _convert_measures
from . import MEASURES_TYPE, SAVEFORMAT_TYPE, SAVEMODE_TYPE
from ._execution import _ir_measures_to_dict
from ._rendering import RenderFromPerQuery
from ._trie import RadixNode, RadixTree
from pyterrier._ops import Compose

import ir_measures
import pandas as pd
from typing import List, Optional, Union, Sequence, Tuple, Callable, cast as tcast, Literal
from time import perf_counter as timer
import uuid

def emit_js(node_id, state):
    from IPython.display import Javascript, display # type: ignore
    display(Javascript(f"""
        setNodeState("{node_id}", "{state}");
    """))


def decompose_pipelines(pipes: List[Union[pd.DataFrame, pt.Transformer]]) -> List[List[pt.Transformer]]:
    """
    Decomposes a list of pipelines (which can be DataFrames, Transformers, or Composes) into a list of lists of Transformers.
    Each inner list represents a sequence of Transformers that should be applied in order.
    """
    pipe_lists: List[List[pt.Transformer]] = []
    for p in pipes:
        # Convert DataFrames to Transformers
        if isinstance(p, pd.DataFrame):
            pipe_lists.append([pt.Transformer.from_df(p)])
        elif isinstance(p, Compose):
            transformers = []
            for t in p._transformers:
                if isinstance(t, pd.DataFrame):
                    transformers.append(pt.Transformer.from_df(t))
                else:
                    transformers.append(t)
            pipe_lists.append(transformers)
        else:
            if not isinstance(p, pt.Transformer):
                raise ValueError("pt.Experiment has systems that are not either DataFrames or Transformers, found: %s" % type(p))
            pipe_lists.append([p])
    return pipe_lists


TREE_KEY_TYPE = Tuple[pt.Transformer, ...]
  
class TransformerRadixNode(RadixNode[TREE_KEY_TYPE, int]):
    def __init__(self):
        super().__init__()
        self.execution_state: Literal['pending', 'running', 'done'] = 'pending'
        self.node_id = str(uuid.uuid4())

    def traverse(self, 
                 inp: pd.DataFrame, 
                 key: Union[TREE_KEY_TYPE, None],
                 exec_callback: Optional[Callable] = None,
                 eval_callback: Optional[Callable] = None, 
                 cum_time: float = 0.0):
        
        # Apply transformation and get time from visit
        res, transform_time = self.visit(inp, 
            key,
            exec_callback = exec_callback, 
        )
        total_time = cum_time + transform_time

        # Invoke callback if this is a terminal node
        if self.value is not None:
            if eval_callback is not None:
                eval_callback(res, self.value, total_time)

        # Recurse into children, adding current node to parents stack
        for childkey, childnode in tcast(List[Tuple[TREE_KEY_TYPE, 'TransformerRadixNode']], self.children.items()):
            childnode.traverse(res, key = childkey,
                exec_callback = exec_callback, 
                eval_callback = eval_callback, 
                cum_time = total_time, 
                )

    def visit(self, inp: pd.DataFrame, key: Union[TREE_KEY_TYPE, None], exec_callback: Optional[Callable] = None) -> Tuple[pd.DataFrame, float]:        
        if key is None:
            assert len(self.children) > 0, "Root node should have children"
            return inp, 0.0  # Root node - no transformation
        
        if isinstance(key, tuple):
            transformers = list(key)
        else:
            transformers = [key]
            
        if len(transformers) == 1:
            self.execution_state = 'running'
            if exec_callback is not None:
                exec_callback(self.node_id, self)

            start = timer()
            result = transformers[0].transform(inp)
            end = timer()
            transform_time = (end - start)*1000.0
            
            # Mark as completed after execution
            self.execution_state = 'done'
            if exec_callback is not None:
                exec_callback(self.node_id, self)
        
        elif len(transformers) > 1:
            result = inp
            transform_time = 0.0

            for i, transformer in enumerate(transformers):
                self.execution_state = 'running'
                if exec_callback is not None:
                    exec_callback(f"{self.node_id}:{i}", self)

                start = timer()
                result = transformer.transform(result)
                end = timer()
                transform_time += (end - start) * 1000.0

                self.execution_state = 'done'
                if exec_callback is not None:
                    exec_callback(f"{self.node_id}:{i}", self)

        else:
            assert False, "No transformers found for this node"

        return result, transform_time



class TransformerRadixTree(RadixTree[TREE_KEY_TYPE, int]):

    def __init__(self):
        super().__init__(node_clz=TransformerRadixNode)

    def traverse(self, 
                 inp: pd.DataFrame, 
                 exec_callback: Optional[Callable] = None,
                 eval_callback: Optional[Callable] = None, 
                 cum_time: float = 0.0, 
                 ):
        tcast(TransformerRadixNode, self.root).traverse(
            inp, 
            None, # root node has no key
            exec_callback = exec_callback,
            eval_callback = eval_callback,
            cum_time = cum_time)

    def reset_status(self):
        def _recurse_set_status(node: TransformerRadixNode):
            node.execution_state = 'pending'
            for child in tcast(List[TransformerRadixNode], node.children.values()):
                _recurse_set_status(child)
        _recurse_set_status(tcast(TransformerRadixNode, self.root))
    
    def describe_tree_structure(self) -> List:
        """Return a structured representation of the radix tree for debugging.
        
        Returns a list of [edge_label, eval_index, children] for each top-level child.
        - eval_index is None for non-terminal nodes, or the index for terminal nodes
        - children is [] for leaf nodes, or a list of [edge_label, eval_index, children] for internal nodes
        """
        def dfs(node: TransformerRadixNode) -> List:
            children_repr: List = []
            for edge_label, child in sorted(node.children.items(), key=lambda x: str(x[0])):
                child_struct = dfs(tcast(TransformerRadixNode, child))
                children_repr.append([
                    edge_label,
                    child.value,
                    child_struct if child_struct else []
                ])
            return children_repr

        # Build the top-level structure from root's children
        result: List = []
        for edge_label, child in sorted(self.root.children.items(), key=lambda x: str(x[0])):
            child_struct = dfs(tcast(TransformerRadixNode, child))
            result.append([
                edge_label,
                child.value,
                child_struct if child_struct else []
            ])
        return result  

    def pretty_print(self, names: Optional[List[str]] = None, colored: bool = False) -> str:
        
        # ANSI color codes
        RED = '\033[91m'      # Pending (not yet executed)
        ORANGE = '\033[93m'   # Running (currently executing)
        GREEN = '\033[38;2;19;136;8m'    # Completed
        RESET = '\033[0m'     
        
        lines = ["Root"]
        
        def format_transformers(edge_label: Tuple[pt.Transformer, ...]) -> str:
            if isinstance(edge_label, tuple):
                if len(edge_label) == 1:
                    return f"{str(edge_label[0])}"
                return " >> ".join(str(t) for t in edge_label)
            return str(edge_label)
        
        def get_color(state: str) -> str:
            """Get ANSI color code based on execution state."""
            if not colored:
                return ""
            if state == 'pending':
                return RED
            elif state == 'running':
                return ORANGE
            elif state == 'done' or state == 'completed':
                return GREEN
            return ""
        
        # Recursion to traverse the tree and build lines with connectors
        def traverse_node(node: TransformerRadixNode, prefix: str):
            children_list = sorted(node.children.items(), key=lambda x: str(x[0]))
            
            for i, (edge_label, _child) in enumerate(children_list):
                child = tcast(TransformerRadixNode, _child)
                is_last_child = (i == len(children_list) - 1)
                
                # Draw the connector
                connector = "└─ " if is_last_child else "├─ "
                
                # Apply color based on execution state
                color = get_color(child.execution_state)
                reset = RESET if colored else ""
                
                lines.append(prefix + connector + color + format_transformers(edge_label) + reset)
                
                # Recurse with updated prefix
                extension = "   " if is_last_child else "│  "
                traverse_node(child, prefix + extension)
        
        # Start traversal from root
        traverse_node(tcast(TransformerRadixNode, self.root), "")
        return "\n".join(lines)
    
    def print_live(self, names: Optional[List[str]] = None, clear_previous: bool = True):

        import sys
        
        if clear_previous:
            def count_lines(node: TransformerRadixNode) -> int:
                count = len(node.children)
                for child in node.children.values():
                    count += count_lines(tcast(TransformerRadixNode, child))
                return count
            
            num_lines = count_lines(tcast(TransformerRadixNode, self.root)) + 1  
            sys.stdout.write(f'\033[{num_lines}A')  
            sys.stdout.write('\033[J')  
        
        print(self.pretty_print(names=names, colored=True))
        sys.stdout.flush()

def tree_execution(renderer : RenderFromPerQuery, 
                    retr_systems, 
                    topics : pd.DataFrame, 
                    qrels: pd.DataFrame,
                    eval_metrics : MEASURES_TYPE,
                    names: Sequence[str],
                    verbose : Literal['notebook', 'terminal', False], 
                    save_dir : Optional[str] = None, 
                    save_mode : Optional[SAVEMODE_TYPE] = None, 
                    save_format : SAVEFORMAT_TYPE = 'trec',
                    batch_size : Optional[int]=None, 
                    perquery : bool = False):
    

    # keys: tuple of Transformer objects; values: system id (int)
    tree: TransformerRadixTree = TransformerRadixTree()

    # build radix tree from retr_systems
    for sysid, system in enumerate(decompose_pipelines(retr_systems)):
        key = tuple(system)
        tree.insert(key, sysid)
    
    exec_cb = None
    if verbose == 'terminal':
        print("\nPipeline structure:")
        tree.print_live(names=list(names), clear_previous=False)
        print()
    elif verbose == 'notebook':
        from IPython.display import HTML, display # type: ignore
        schematic = pt.schematic.radix_tree_schematic(tree, input_columns=["qid", "query"])
        display(HTML(pt.schematic.draw_html_schematic(schematic)))
        exec_cb = lambda node_id, node: emit_js(node_id, node.execution_state) # noqa: E731
    elif verbose is False:
        pass
    else:
        assert False, "verbose must be either False, 'notebook' or 'terminal', found %s" % str(verbose)
        
    metrics, rev_mapping = _convert_measures(eval_metrics)
    qrels = pt.model.to_ir_measures(qrels)
    num_q = qrels['query_id'].nunique()
    all_topic_qids = topics["qid"].values

    assert topics is not None, "topics must be specified"

    def _evaluate_results(
        res: pd.DataFrame,
        eval_qrels: pd.DataFrame,
        backfill_qids: Optional[Sequence[str]],
        error_message: str
    ) -> dict:
        if len(res) == 0:
            raise ValueError(error_message)

        if verbose == 'terminal':
            tree.print_live(names=list(names), clear_previous=True)

        return _ir_measures_to_dict(
            ir_measures.iter_calc(metrics, eval_qrels, pt.model.to_ir_measures(res)),
            metrics,
            rev_mapping,
            num_q,
            perquery=True,
            backfill_qids=backfill_qids)

    def make_eval_callback(
        eval_qrels: pd.DataFrame,
        backfill_qids: Optional[Sequence[str]],
        error_message: str,
        accumulated_metrics: Optional[dict[int, dict]] = None,
        system_times: Optional[dict[int, float]] = None,
    ) -> Callable:
        def callback(res: pd.DataFrame, sysid: int, cum_time: float):
            eval_measures = _evaluate_results(
                res,
                eval_qrels,
                backfill_qids,
                error_message % sysid)

            # accumulated_metrics and system_times are used for batch processing to accumulate results across batches
            if accumulated_metrics is None or system_times is None:
                # If not in batch mode, add metrics directly to the renderer.
                # eval_measures should have one entry for each num_q; 
                # cum_time is the total time for all queries, so we divide by num_q to get the average time per query.
                renderer.add_metrics(sysid, eval_measures, cum_time / num_q)
            else:
                # In batch mode, accumulate metrics and times for each system across batches
                if sysid not in accumulated_metrics:
                    accumulated_metrics[sysid] = {}
                    system_times[sysid] = 0
                accumulated_metrics[sysid].update(eval_measures)
                system_times[sysid] += cum_time

        return callback

    if batch_size is None:
        # No batching - execute all queries at once

        tree.traverse(
            topics,
            exec_callback=exec_cb,
            eval_callback=make_eval_callback(
                qrels,
                all_topic_qids if perquery else None,
                f"{len(topics)} topics, but no results received from system %d"),
            cum_time=0.)
    else:
        # Batch processing - evaluate queries in batches and accumulate per-system results
        assert batch_size > 0
        topic_batches = pt.model.split_df(topics, batch_size=batch_size)

        accumulated_metrics: dict[int, dict] = {}
        system_times: dict[int, float] = {}
        processed_qids = set()

        iter_batch = enumerate(topic_batches)
        if verbose == 'terminal':
            iter_batch = pt.tqdm(iter_batch, total=len(topic_batches), desc="Processing batches", unit="batch")

        for batch_idx, topic_batch in iter_batch:
            tree.reset_status()

            batch_qids = set(topic_batch.qid)
            processed_qids.update(batch_qids)
            batch_qrels = qrels[qrels.query_id.isin(batch_qids)]
            batch_backfill = [qid for qid in all_topic_qids if qid in batch_qids] if perquery else None

            tree.traverse(
                topic_batch,
                exec_callback=exec_cb,
                eval_callback=make_eval_callback(
                    batch_qrels,
                    batch_backfill,
                    "batch of %d topics, but no results received in batch %d from system %%d" % (len(topic_batch), batch_idx),
                    accumulated_metrics=accumulated_metrics,
                    system_times=system_times),
                cum_time=0.)

        # Handle qids in qrels that were not in topics (same behavior as linear execution)
        remaining_qids = set(qrels.query_id) - processed_qids
        if remaining_qids and accumulated_metrics:
            remaining_qrels = qrels[qrels.query_id.isin(remaining_qids)]
            empty_res = pd.DataFrame([], columns=['query_id', 'doc_id', 'score'])
            missing_metrics = _ir_measures_to_dict(
                ir_measures.iter_calc(metrics, remaining_qrels, empty_res),
                metrics,
                rev_mapping,
                num_q,
                perquery=True)
            for sysid in accumulated_metrics:
                accumulated_metrics[sysid].update(missing_metrics)

        # only once all batches have been processed, add the accumulated metrics to the renderer
        for sysid, eval_measures in accumulated_metrics.items():
            renderer.add_metrics(sysid, eval_measures, system_times[sysid] / num_q)
