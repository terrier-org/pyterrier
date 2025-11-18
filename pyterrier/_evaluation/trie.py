from typing import List, Tuple, Union, Optional, Callable, TypeVar, Generic, Any
from time import perf_counter as timer
import pandas as pd
import pyterrier as pt
from pyterrier._ops import Compose

T = TypeVar('T')  # Generic type for values stored in the tree
K = TypeVar('K')  # Generic type for edge labels (keys)


class RadixNode(Generic[K, T]):
    def __init__(self):
        self.children: dict[K, 'RadixNode[K, T]'] = {}  # edge_label -> RadixNode
        self.is_end_of_word: bool = False
        self.eval_index: Optional[int] = None  # index of evaluation step if terminal
        self.value: Optional[T] = None  # payload stored at nodes (e.g., transformer)

    def traverse(self, inp: pd.DataFrame, callback: Optional[Callable] = None, cum_time: float = 0.0, parents: Optional[List['RadixNode[K, T]']] = None):
        """Traverse node and its descendants, applying transformations and tracking cumulative time.

        When `self.eval_index` is set (not None) the `callback` will be invoked as
            callback(result, self.eval_index, total_time)

        Args:
            inp: input object (e.g., DataFrame) passed into transformers
            callback: optional callable receiving (result, eval_index, total_time_ms)
            cum_time: cumulative time (ms) carried from ancestors
            parents: stack of parent nodes from root to current (initially empty)
        """
        if parents is None:
            parents = []
        
        start = timer()
        res = self.visit(inp, parents)
        end = timer()

        transform_time = (end - start) * 1000.0
        total_time = cum_time + transform_time

        if self.eval_index is not None:
            assert callback is not None, "evaluation_index is set but no callback was provided"
            callback(res, self.eval_index, total_time)

        # Recurse into children, adding current node to parents stack
        new_parents = parents + [self]
        for child in self.children.values():
            child.traverse(res, callback, total_time, new_parents)
    
    def visit(self, inp: pd.DataFrame, parents: List['RadixNode[K, T]']) -> pd.DataFrame:
        """Visit the node and apply its transformation to the input data.

        The key insight: instead of storing redundant transformer values at each node,
        we reconstruct the pipeline from the stored value which represents the complete
        pipeline path up to this point.

        Args:
            inp: Input data to transform
            parents: Stack of parent nodes from root to current node
        """
        # If this node has a value, it represents the complete pipeline to this point
        if self.value is not None:
            if isinstance(self.value, tuple):
                # Value is a tuple of transformers - compose them
                if len(self.value) == 1:
                    return self.value[0].transform(inp)
                else:
                    composed = Compose(*self.value)
                    return composed.transform(inp)
            else:
                # Value is a single transformer
                return self.value.transform(inp)
        else:
            # No value at this node, return input unchanged
            return inp
    
    # def get_children(self) -> List[Tuple[K, 'RadixNode[K]']]:
    #     """
    #     Returns:
    #         List of tuples containing edge labels and corresponding child RadixNode instances.
    #     """
    #     return list(self.children.items())
class RadixTree(Generic[K, T]):
    def __init__(self):
        self.root: RadixNode[K, T] = RadixNode()
        self.eval_index: int = 0  # Counter for assigning eval_index to terminal nodes

    def insert(self, word: K, value: Optional[T] = None) -> None:
        """Insert a word into the radix tree, optionally attaching a payload.

        Args:
            word: The full key/path to insert.
            value: Optional payload to store at the terminal node. If None, defaults to `word`.
        """
        node = self.root
        remaining = word
        # print(node.children)
        while remaining:
            # Find a child edge that shares a prefix with remaining
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
                        # Exact match - word already exists or is being re-inserted
                        child.is_end_of_word = True
                        child.value = word if value is None else value
                        if child.eval_index is None:  # Only assign if not already set
                            child.eval_index = self.eval_index
                            self.eval_index += 1
                        return
                    elif common_len == len(edge_label):
                        # Full edge matches, continue down this path
                        remaining = remaining[common_len:]
                        node = child
                    else:
                        # Need to split the edge (common_len < len(edge_label))
                        # Create new intermediate node
                        new_node = RadixNode()
                        
                        # Old edge becomes: common_prefix -> new_node -> rest_of_edge -> old_child
                        rest_of_edge = edge_label[common_len:]
                        new_node.children[rest_of_edge] = child
                        # Update parent to point to new_node with common prefix
                        del node.children[edge_label]
                        node.children[edge_label[:common_len]] = new_node
                        
                        # Continue with remaining part
                        remaining = remaining[common_len:]
                        node = new_node
                    
                    break
            
            if not found:
                # No matching edge, create new child with remaining string
                new_child = RadixNode()
                new_child.is_end_of_word = True
                new_child.value = word if value is None else value
                new_child.eval_index = self.eval_index
                self.eval_index += 1
                node.children[remaining] = new_child
                return
        
        # If we've consumed all of remaining, mark current node as end
        node.is_end_of_word = True
        node.value = word if value is None else value
        if node.eval_index is None:  # Only assign if not already set
            node.eval_index = self.eval_index
            self.eval_index += 1


    def search(self, word: K) -> bool:
        """Check if a word exists in the radix tree."""
        node = self.root
        remaining = word
        
        while remaining:
            found = False
            for edge_label, child in node.children.items():
                if remaining.startswith(edge_label):  # type: ignore[union-attr]
                    remaining = remaining[len(edge_label):]  # type: ignore[index,arg-type]
                    node = child
                    found = True
                    break
            
            if not found:
                return False
        
        return node.is_end_of_word

    # def pretty_print(self):
    #     """Print the radix tree structure with indentation.

    #     Each edge label is printed on its own line, indented by depth.
    #     Nodes that mark the end of a word are annotated with a '*'.
    #     """
    #     def dfs(node: RadixNode, depth: int):
    #         for edge_label, child in sorted(node.children.items()):
    #             end_marker = '*' if child.is_end_of_word else ''
    #             print("  " * depth + f"'{edge_label}'{end_marker}")
    #             dfs(child, depth + 1)

    #     dfs(self.root, 0)
    def insert_all(self, items: Union[List[K], List[Tuple[K, T]]]) -> None:
        """Insert multiple entries.

        Accepts either a list of keys or a list of (key, value) tuples.
        When passing just keys, the stored value defaults to the key itself.
        Lists are automatically converted to tuples to make them hashable.
        """
        if not items:
            return
        first = items[0]
        print("first:", first)
        if isinstance(first, tuple) and len(first) == 2:
            for key, val in items:  # type: ignore[misc]
                # Convert key to tuple if it's a list
                hashable_key = tuple(key) if isinstance(key, list) else key
                self.insert(hashable_key, val)  # type: ignore[arg-type]
        else:
            for key in items:  # type: ignore[assignment]
                # Convert key to tuple if it's a list
                hashable_key = tuple(key) if isinstance(key, list) else key
                self.insert(hashable_key, hashable_key)  # type: ignore[arg-type]

    
    def describe_tree_structure(self) -> List:
        """Return a structured representation of the radix tree for debugging.
        
        Returns a list of [edge_label, eval_index, children] for each top-level child.
        - eval_index is None for non-terminal nodes, or the index for terminal nodes
        - children is [] for leaf nodes, or a list of [edge_label, eval_index, children] for internal nodes
        """
        def dfs(node: RadixNode[K, T]) -> List:
            children_repr: List = []
            for edge_label, child in sorted(node.children.items(), key=lambda x: str(x[0])):
                child_struct = dfs(child)
                children_repr.append([
                    edge_label,
                    child.eval_index if child.is_end_of_word else None,
                    child_struct if child_struct else []
                ])
            return children_repr

        # Build the top-level structure from root's children
        result: List = []
        for edge_label, child in sorted(self.root.children.items(), key=lambda x: str(x[0])):
            child_struct = dfs(child)
            result.append([
                edge_label,
                child.eval_index if child.is_end_of_word else None,
                child_struct if child_struct else []
            ])
        return result  

def decompose_pipelines(pipes: List[Union[pd.DataFrame, pt.Transformer]]) -> List[List[pt.Transformer]]:
    if len(pipes) == 1:
        return pipes
    pipe_lists: List[List[pt.Transformer]] = []
    for p in pipes:
        # no optimisation possible for experiments involving dataframes as systems
        if isinstance(p, pd.DataFrame):
            return pipes
        if isinstance(p, Compose):
            pipe_lists.append(list(p._transformers))
        else:
            if not isinstance(p, pt.Transformer):
                raise ValueError("pt.Experiment has systems that are not either DataFrames or Transformers")
            pipe_lists.append([p])
    return pipe_lists
    # return pipe_lists[0][:-1], [pl[-1] for pl in pipe_lists]


tree = RadixTree()
tree.insert('abc')
tree.insert('abd')
print(tree.describe_tree_structure())
