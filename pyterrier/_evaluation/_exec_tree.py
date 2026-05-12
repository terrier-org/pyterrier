import pyterrier as pt

from ._rendering import _convert_measures
from . import MEASURES_TYPE
from ._execution import _ir_measures_to_dict
from ._trie import RadixNode, RadixTree
from pyterrier._ops import Compose

import ir_measures
import pandas as pd
from typing import List, Optional, Union, Sequence, Tuple, Callable, cast, Literal
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
        self.execution_state: Literal['pending', 'running', 'done'] = 'pending'  # 'pending', 'running', 'done'
        self.node_id = str(uuid.uuid4())

    def traverse(self, 
                 inp: pd.DataFrame, 
                 callback: Optional[Callable] = None, 
                 cum_time: float = 0.0, 
                 parents: Optional[List['TransformerRadixNode']] = None):
        if parents is None:
            parents = []
        
        # Apply transformation and get time from visit
        res, transform_time = self.visit(inp, parents)
        total_time = cum_time + transform_time

        # Invoke callback if this is a terminal node
        if self.value is not None:
            if callback is not None:
                callback(res, self.value, total_time)

        # Recurse into children, adding current node to parents stack
        if self.children:
            parents.append(self) 
            for child in cast(List['TransformerRadixNode'], self.children.values()):
                child.traverse(res, callback, total_time, parents)
            parents.pop()

    def visit(self, inp: pd.DataFrame, parents: List['TransformerRadixNode']) -> Tuple[pd.DataFrame, float]:        
        if not parents:
            return inp, 0.0  # Root node - no transformation
        
        # Search for key in parent's children
        transformers = []
        last_parent = parents[-1]
        for key, child in last_parent.children.items():
            if child is self:
                if isinstance(key, tuple):
                    transformers = list(key)
                else:
                    transformers = [key]
                break
        
        # Apply transformers and measure time only during execution
        if transformers:
            # Mark as running before execution
            # this is for a single node, so we can update self.execution_state directly
            if len(transformers) ==1:
                # Can remove execution_states? 
                self.execution_state = 'running'
                emit_js(self.node_id, self.execution_state)
                start = timer()
                result = transformers[0].transform(inp)
                end = timer()
                transform_time = (end - start)*1000.0
                print(f"{transformers[0]}: {transform_time:.2f} s")
                # Mark as completed after execution
                self.execution_state = 'done'
                emit_js(self.node_id, self.execution_state)
                
                return result, transform_time
            
            for i in range(len(transformers)):
                new_id = f"{self.node_id}:{i}"
                self.execution_state = 'running'
                emit_js(new_id, self.execution_state)
            
            start = timer()
            if len(transformers) == 1:
                result = transformers[0].transform(inp)
            else:
                result = Compose(*transformers).transform(inp)
            end = timer()
            transform_time = (end - start) * 1000.0
            print('Transformer time:', transform_time)                        
            for i in range(len(transformers)):
                new_id = f"{self.node_id}:{i}"
            # Mark as completed after execution
                self.execution_state = 'done'
                emit_js(new_id, self.execution_state)
            
            return result, transform_time
        else:
            return inp, 0.0



class TransformerRadixTree(RadixTree[TREE_KEY_TYPE, int]):

    def __init__(self):
        super().__init__(node_clz=TransformerRadixNode)

    def traverse(self, 
                 inp: pd.DataFrame, 
                 callback: Optional[Callable] = None, 
                 cum_time: float = 0.0, 
                 parents: Optional[List['TransformerRadixNode']] = None):
        cast(TransformerRadixNode, self.root).traverse(inp, callback, cum_time, parents)

    def describe_tree_structure(self) -> List:
        """Return a structured representation of the radix tree for debugging.
        
        Returns a list of [edge_label, eval_index, children] for each top-level child.
        - eval_index is None for non-terminal nodes, or the index for terminal nodes
        - children is [] for leaf nodes, or a list of [edge_label, eval_index, children] for internal nodes
        """
        def dfs(node: TransformerRadixNode) -> List:
            children_repr: List = []
            for edge_label, child in sorted(node.children.items(), key=lambda x: str(x[0])):
                child_struct = dfs(cast(TransformerRadixNode, child))
                children_repr.append([
                    edge_label,
                    child.value,
                    child_struct if child_struct else []
                ])
            return children_repr

        # Build the top-level structure from root's children
        result: List = []
        for edge_label, child in sorted(self.root.children.items(), key=lambda x: str(x[0])):
            child_struct = dfs(cast(TransformerRadixNode, child))
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
                child = cast(TransformerRadixNode, _child)
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
        traverse_node(cast(TransformerRadixNode, self.root), "")
        return "\n".join(lines)
    
    def print_live(self, names: Optional[List[str]] = None, clear_previous: bool = True):

        import sys
        
        if clear_previous:
            def count_lines(node: TransformerRadixNode) -> int:
                count = len(node.children)
                for child in node.children.values():
                    count += count_lines(cast(TransformerRadixNode, child))
                return count
            
            num_lines = count_lines(cast(TransformerRadixNode, self.root)) + 1  
            sys.stdout.write(f'\033[{num_lines}A')  
            sys.stdout.write('\033[J')  
        
        print(self.pretty_print(names=names, colored=True))
        sys.stdout.flush()

def tree_execution(renderer,retr_systems, 
                     topics : pd.DataFrame, 
                     qrels: pd.DataFrame,
                     eval_metrics : MEASURES_TYPE,
                     names: Sequence[str],
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
    tree: TransformerRadixTree = TransformerRadixTree()

    for sysid, system in enumerate(decompose_pipelines(retr_systems)):
        key = tuple(system)
        tree.insert(key, sysid)
    
    if verbose:
        print("\nPipeline structure:")
        tree.print_live(names=list(names), clear_previous=False)
        print()

    if render_html:
        from IPython.display import HTML, display # type: ignore
        schematic = pt.schematic.radix_tree_schematic(tree, input_columns=["qid", "query"])
        display(HTML(pt.schematic.draw_html_schematic(schematic)))
    
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
                tree.print_live(names=list(names), clear_previous=True)
            
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
        tree.traverse(topics, make_callback(qrels, all_topic_qids if perquery else None), 0.0)
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

            tree.traverse(topic_batch, make_callback(batch_qrels, batch_backfill), 0.0)
