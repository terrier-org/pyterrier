from typing import List, Tuple, Union, Optional, Callable, TypeVar, Generic, Protocol, overload
from collections.abc import Sequence

from time import perf_counter as timer
import pandas as pd
import pyterrier as pt
from pyterrier._ops import Compose
from IPython.display import Javascript, display
import uuid


T = TypeVar('T')  # Generic type for values stored in the tree
K = TypeVar('K', bound="Sliceable")  # Generic type for keys (tuples of transformers)
E = TypeVar("E", covariant=True)

class Sliceable(Protocol[E]):
    def __len__(self : K) -> int: ...

    @overload
    def __getitem__(self : K, index: int) -> E: ...

    @overload
    def __getitem__(self: K, index: slice) -> K: ...

def emit_js(node_id, state):
    display(Javascript(f"""
        
        setNodeState("{node_id}", "{state}");
    """))



class RadixNode(Generic[K, T]):
    def __init__(self):
        self.children: dict[K, 'RadixNode[K, T]'] = {}  # key -> RadixNode
        self.value: Optional[T] = None  # (e.g., sysid) -> evaluation_index
        self.execution_state: str = 'pending'  # 'pending', 'running', 'done'
        self.node_id = str(uuid.uuid4())

    def traverse(self, inp: pd.DataFrame, callback: Optional[Callable] = None, cum_time: float = 0.0, parents: Optional[List['RadixNode[K, T]']] = None):
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
            for child in self.children.values():
                child.traverse(res, callback, total_time, parents)
            parents.pop()
    


    def visit(self, inp: pd.DataFrame, parents: List['RadixNode[K, T]']) -> Tuple[pd.DataFrame, float]:        
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


  
class RadixTree(Generic[K, T]):
    def __init__(self):
        self.root: RadixNode[K, T] = RadixNode()


    def insert(self, pipeline: K, value: Optional[T] = None) -> None:
        node = self.root
        remaining = pipeline
        while remaining:
            found = False
            for edge_label, child in list(node.children.items()):
                # Find longest common prefix between edge_label and remaining
                common_len = 0
                for i in range(min(len(edge_label), len(remaining))):
                    if edge_label[i] == remaining[i]:
                        common_len += 1
                    else:
                        break
                
                if common_len > 0:
                    found = True
                    
                    if common_len == len(edge_label) == len(remaining):
                        # Exact match - pipeline already exists or is being re-inserted
                        #child.is_end_of_pipeline = True
                        child.value = value  
                        return
                    elif common_len == len(edge_label):
                        # Full edge matches, continue down this path
                        remaining = remaining[common_len:]
                        node = child
                    else:
                        # Need to split the edge (common_len < len(edge_label))
                        # Create new intermediate node
                        new_node : RadixNode[K, T] = RadixNode()
                        # Old edge becomes: common_prefix -> new_node -> rest_of_edge -> old_child
                        rest_of_edge = edge_label[common_len:]
                        new_node.children[rest_of_edge] = child
                        # Update parent to point to new_node with common prefix
                        del node.children[edge_label]
                        common_prefix = edge_label[:common_len]
                        node.children[common_prefix] = new_node
                        remaining = remaining[common_len:]
                        node = new_node
                    
                    break
            
            if not found:
                # No matching edge, create new child with remaining string
                new_child : RadixNode[K, T] = RadixNode()
                new_child.value = value  
                node.children[remaining] = new_child
                return
        
        # If we've consumed all of remaining, mark current node as end
        node.value = value  


    

    def pretty_print(self, names: Optional[List[str]] = None, colored: bool = False) -> str:
        
        # ANSI color codes
        RED = '\033[91m'      # Pending (not yet executed)
        ORANGE = '\033[93m'   # Running (currently executing)
        GREEN = '\033[38;2;19;136;8m'    # Completed
        RESET = '\033[0m'     
        
        lines = ["Root"]
        
        def format_transformers(edge_label: K) -> str:
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
        def traverse_node(node: RadixNode[K, T], prefix: str, is_last: bool):
            children_list = sorted(node.children.items(), key=lambda x: str(x[0]))
            
            for i, (edge_label, child) in enumerate(children_list):
                is_last_child = (i == len(children_list) - 1)
                
                # Draw the connector
                connector = "└─ " if is_last_child else "├─ "
                
                # Apply color based on execution state
                color = get_color(child.execution_state)
                reset = RESET if colored else ""
                
                lines.append(prefix + connector + color + format_transformers(edge_label) + reset)
                
                # Recurse with updated prefix
                extension = "   " if is_last_child else "│  "
                traverse_node(child, prefix + extension, is_last_child)
        
        # Start traversal from root
        traverse_node(self.root, "", True)
        return "\n".join(lines)
    
    def print_live(self, names: Optional[List[str]] = None, clear_previous: bool = True):

        import sys
        
        if clear_previous:
            def count_lines(node: RadixNode[K, T]) -> int:
                count = len(node.children)
                for child in node.children.values():
                    count += count_lines(child)
                return count
            
            num_lines = count_lines(self.root) + 1  
            sys.stdout.write(f'\033[{num_lines}A')  
            sys.stdout.write('\033[J')  
        
        print(self.pretty_print(names=names, colored=True))
        sys.stdout.flush()


def decompose_pipelines(pipes: List[Union[pd.DataFrame, pt.Transformer]]) -> List[List[pt.Transformer]]:
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
                raise ValueError("pt.Experiment has systems that are not either DataFrames or Transformers")
            pipe_lists.append([p])
    return pipe_lists


