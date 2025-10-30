from typing import Any, List, Tuple, Union, Optional, Callable
from time import perf_counter as timer
import pandas as pd


class RadixNode:
    def __init__(self):
        self.children = {}  # edge_label -> RadixNode
        self.is_end_of_word = False
        self.value: Any = None  # payload stored at terminal nodes (e.g., transformer later)
        # evaluation order index for terminal nodes (None until set via insert_all)
        self.eval_index: "Optional[int]" = None
    # need to come back to self.value

    def traverse(self, inp: pd.DataFrame, callback: Optional[Callable] = None, cum_time: float = 0.0):
        """Traverse node and its descendants, applying transformations and tracking cumulative time.


        When `self.eval_index` is set (not None) the `callback` will be invoked as
            callback(result, self.eval_index, total_time)

        Args:
            inp: input object (e.g., DataFrame) passed into transformers
            callback: optional callable receiving (result, eval_index, total_time_ms)
            cum_time: cumulative time (ms) carried from ancestors
        """
        start = timer()
        res = self.visit(inp)
        end = timer()

        transform_time = (end - start) * 1000.0
        total_time = cum_time + transform_time

        if self.eval_index is not None:
            assert callback is not None, "evaluation_index is set but no callback was provided"
            callback(res, self.eval_index, total_time)

        # Recurse into children
        for child in self.children.values():
            child.traverse(res, callback, total_time)
    
    def visit(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Visit the node and apply its transformation to the input data.

        Args:
            inp: Input data to transform"""
        
        return self.value.transform(inp)
class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def insert(self, word: str, value: Any = None, eval_index: "Optional[int]" = None):
        """Insert a word into the radix tree, optionally attaching a payload.

        Args:
            word: The full key/path to insert.
            value: Optional payload to store at the terminal node. If None, defaults to `word`.
        """
        if not word:
            raise ValueError(f"empty or falsy key not allowed: {word!r}")
        node = self.root
        remaining = word
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
                        # set eval_index on terminal node if provided
                        if eval_index is not None:
                            child.eval_index = eval_index
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
                if eval_index is not None:
                    new_child.eval_index = eval_index
                node.children[remaining] = new_child
                return
        
        # If we've consumed all of remaining, mark current node as end
        node.is_end_of_word = True
        node.value = word if value is None else value
        if node.is_end_of_word and eval_index is not None:
            node.eval_index = eval_index

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

    def describe_tree_structure(self):
        """Return a structured representation of the radix tree for debugging."""
        def dfs(node: RadixNode, prefix: str):
            children_repr = []
            for edge_label, child in sorted(node.children.items()):
                full_label = edge_label
                sub_repr = dfs(child, full_label)
                child_repr = sub_repr if sub_repr else {}
                children_repr.append([
                    f"{full_label}",
                    child.eval_index if child.is_end_of_word else None,
                    child_repr
                ])
            return children_repr

        return dfs(self.root, "")  


    # -----------------------------
    # Bulk operations and utilities
    # -----------------------------
    def insert_all(self, items: Union[List[str], List[Tuple[str, Any]]]):
        """Insert multiple entries.

        Accepts either a list of strings (keys) or a list of (key, value) tuples.
        When passing just strings, the stored value defaults to the string itself.
        """
        if not items:
            return
        for i, item in enumerate(items):
            if isinstance(item, tuple) and len(item) == 2:
                key, val = item  # type: ignore[misc]
                self.insert(str(key), val, eval_index=i)
            else:
                self.insert(str(item), item, eval_index=i)

# Example usage
radix_tree = RadixTree()


radix_tree.insert_all(["AB", "ABC", "D"])
# radix_tree.insert_all(['AB','ABCD'])

# # Print the radix tree structure
# print("\nRadix tree structure:")
# radix_tree.pretty_print()

structure = radix_tree.describe_tree_structure()
print(structure)