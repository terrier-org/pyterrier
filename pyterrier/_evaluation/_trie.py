from typing import List, Tuple, Union, Optional, Callable, TypeVar, Generic
from time import perf_counter as timer
import pandas as pd
import pyterrier as pt
from pyterrier._ops import Compose

T = TypeVar('T')  # Generic type for values stored in the tree
K = TypeVar('K')  # Generic type for keys (tuples of transformers)


class RadixNode(Generic[K, T]):
    def __init__(self):
        self.children: dict[K, 'RadixNode[K, T]'] = {}  # key -> RadixNode
        self.is_end_of_word: bool = False
        self.value: Optional[T] = None  # (e.g., sysid)
        self.cached_result: Optional[pd.DataFrame] = None  # Cached result for nodes with 2+ children
        self._key: Optional[K] = None  # Cache the key from parent to this node

    def traverse(self, inp: pd.DataFrame, callback: Optional[Callable] = None, cum_time: float = 0.0, parents: Optional[List['RadixNode[K, T]']] = None):
        """Traverse node and its descendants, applying transformations and tracking cumulative time.

        When `self.value` is set (not None) the `callback` will be invoked as
            callback(result, self.value, total_time)

        Args:
            inp: input object (e.g., DataFrame) passed into transformers
            callback: optional callable receiving (result, value, total_time_ms)
            cum_time: cumulative time (ms) carried from ancestors
            parents: stack of parent nodes from root to current (initially empty)
        """
        if parents is None:
            parents = []
        
        # Check if result is already cached (for nodes with 2+ children)
        if self.cached_result is not None:
            res = self.cached_result
            total_time = cum_time  # Time already included when cached
        else:
            start = timer()
            res = self.visit(inp, parents)
            end = timer()

            transform_time = (end - start) * 1000.0
            total_time = cum_time + transform_time
            
            # Cache result if this node will be reused:
            # - Has 2+ children (branching to multiple paths), OR
            # - Is a terminal node with children (extends to longer pipelines)
            if len(self.children) >= 2 or (self.value is not None and len(self.children) >= 1):
                self.cached_result = res

        # Invoke callback if this is a terminal node
        if self.value is not None:
            callback(res, self.value, total_time)

        # Recurse into children, adding current node to parents stack
        if self.children:
            parents.append(self)  # Modify in place instead of creating new list
            for child in self.children.values():
                child.traverse(res, callback, total_time, parents)
            parents.pop()
    
    def visit(self, inp: pd.DataFrame, parents: List['RadixNode[K, T]']) -> pd.DataFrame:
        """Apply transformers from the key leading to this node."""
        
        # Use cached key if available (avoids dictionary search)
        if self._key:
            if isinstance(self._key, tuple):
                transformers = list(self._key)
            else:
                transformers = [self._key]
        elif parents:
            # Fallback: search for key
            transformers = []
            last_parent = parents[-1]
            for key, child in last_parent.children.items():
                if child is self:
                    if isinstance(key, tuple):
                        transformers = list(key)
                    else:
                        transformers = [key]
                    self._key = key  # Cache it for next time
                    break
        else:
            return inp  # Root node
        
        # Apply transformers directly
        if transformers:
            if len(transformers) == 1:
                return transformers[0].transform(inp)
            else:
                return Compose(*transformers).transform(inp)
        else:
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
                        child.value = value  # Store the sysid directly
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
                        child._key = rest_of_edge  # Cache key
                        # Update parent to point to new_node with common prefix
                        del node.children[edge_label]
                        common_prefix = edge_label[:common_len]
                        node.children[common_prefix] = new_node
                        new_node._key = common_prefix  # Cache key
                        remaining = remaining[common_len:]
                        node = new_node
                    
                    break
            
            if not found:
                # No matching edge, create new child with remaining string
                new_child = RadixNode()
                new_child.is_end_of_word = True
                new_child.value = value  # Store the sysid directly
                new_child._key = remaining  # Cache key
                node.children[remaining] = new_child
                return
        
        # If we've consumed all of remaining, mark current node as end
        node.is_end_of_word = True
        node.value = value  # Store the sysid directly

    
    def insert_all(self, items: Union[List[K], List[Tuple[K, T]]]) -> None:
        """Insert multiple entries.

        Accepts either a list of keys or a list of (key, value) tuples.
        When passing just keys, the stored value defaults to the key itself.
        Lists are automatically converted to tuples to make them hashable.
        """
        if not items:
            return
        first = items[0]
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
        
        Returns a list of [edge_label, value, children] for each top-level child.
        - value is None for non-terminal nodes, or the stored value (sysid) for terminal nodes
        - children is [] for leaf nodes, or a list of [edge_label, value, children] for internal nodes
        """
        def dfs(node: RadixNode[K, T]) -> List:
            children_repr: List = []
            for edge_label, child in sorted(node.children.items(), key=lambda x: str(x[0])):
                child_struct = dfs(child)
                children_repr.append([
                    edge_label,
                    child.value if child.is_end_of_word else None,
                    child_struct if child_struct else []
                ])
            return children_repr

        # Build the top-level structure from root's children
        result: List = []
        for edge_label, child in sorted(self.root.children.items(), key=lambda x: str(x[0])):
            child_struct = dfs(child)
            result.append([
                edge_label,
                child.value if child.is_end_of_word else None,
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


# tree = RadixTree()

# Test compression with transformer sequences
# print("=== Testing Linear Chain Compression ===")

# # Example sequences that should preserve branching but compress linear chains
# test_sequences = [
#     ('bm25', '%10', 'monoT5'),  # Should become: bm25 -> (%10 >> monoT5)
#     ('bm25', '%100'),           # Should become: bm25 -> %100
# ]

# print("Inserting sequences:")
# for i, seq in enumerate(test_sequences):
#     tree.insert(seq)
#     print(f"  {i+1}: {seq}")

# print("\nBefore compression:")
# print(tree.describe_tree_structure())

# # Apply linear chain compression
# tree.compress_linear_chains()

# print("\nAfter compression:")
# structure = tree.describe_tree_structure()
# print(structure)

