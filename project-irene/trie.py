from typing import Any, List, Tuple, Union, Optional, Callable, TypeVar, Generic, Protocol
from time import perf_counter as timer
import pandas as pd

# just checking
class RadixNode:
    def __init__(self):
        self.children = {}  # edge_label -> RadixNode
        self.is_end_of_word = False
        self.value: Any = None  # payload stored at terminal nodes (e.g., transformer later)
    # need to come back to self.value
        self.eval_index: Optional[int] = None  # index of evaluation step if terminal

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
    
    # def get_children(self) -> List[Tuple[K, 'RadixNode[K]']]:
    #     """
    #     Returns:
    #         List of tuples containing edge labels and corresponding child RadixNode instances.
    #     """
    #     return list(self.children.items())
class RadixTree:
    def __init__(self):
        self.root = RadixNode()
        self.eval_index = 0  # Counter for assigning eval_index to terminal nodes

    def insert(self, word: str, value: Any = None):
        """Insert a word into the radix tree, optionally attaching a payload.

        Args:
            word: The full key/path to insert.
            value: Optional payload to store at the terminal node. If None, defaults to `word`.
        """
        node = self.root
        remaining = word
        print(node.children)
        while remaining:
            # Find a child edge that shares a prefix with remaining
            found = False
            for edge_label, child in list(node.children.items()):
                print("edge_label:", edge_label, "remaining:", remaining)
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
                        print(new_node.children)
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


    def search(self, word):
        """Check if a word exists in the radix tree."""
        node = self.root
        remaining = word
        
        while remaining:
            found = False
            for edge_label, child in node.children.items():
                if remaining.startswith(edge_label):
                    remaining = remaining[len(edge_label):]
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
    def insert_all(self, items: Union[List[str], List[Tuple[str, Any]]]):
        """Insert multiple entries.

        Accepts either a list of strings (keys) or a list of (key, value) tuples.
        When passing just strings, the stored value defaults to the string itself.
        """
        if not items:
            return
        first = items[0]
        print("first:", first)
        if isinstance(first, tuple) and len(first) == 2:
            for key, val in items:  # type: ignore[misc]
                self.insert(str(key), val)
        else:
            for key in items:  # type: ignore[assignment]
                self.insert(str(key), key)

    
    def describe_tree_structure(self):
        """Return a structured representation of the radix tree for debugging.
        
        Returns a list of [edge_label, eval_index, children] for each top-level child.
        - eval_index is None for non-terminal nodes, or the index for terminal nodes
        - children is {} for leaf nodes, or a list of [edge_label, eval_index, children] for internal nodes
        """
        def dfs(node):
            children_repr = []
            for edge_label, child in sorted(node.children.items(), key=lambda x: str(x[0])):
                child_struct = dfs(child)
                children_repr.append([
                    edge_label,
                    child.eval_index if child.is_end_of_word else None,
                    child_struct if child_struct else {}
                ])
            return children_repr

        # Build the top-level structure from root's children
        result = []
        for edge_label, child in sorted(self.root.children.items(), key=lambda x: str(x[0])):
            child_struct = dfs(child)
            result.append([
                edge_label,
                child.eval_index if child.is_end_of_word else None,
                child_struct if child_struct else {}
            ])
        return result  


# Example usage
radix_tree = RadixTree()
# radix_tree.insert("ueue")
# radix_tree.insert("eue")
# radix_tree.insert("ue")
# radix_tree.insert("e")
radix_tree.insert_all(['ABCD', 'ABE', 'D'])

# print("Search 'ABC':", radix_tree.search("ABC"))
# print("Search 'ABD':", radix_tree.search("ABD"))
# print("Search 'AB':", radix_tree.search("AB"))
# print("Starts with 'AB':", radix_tree.starts_with("AB"))

# Test find_lcp_node
# node, lcp, remaining = radix_tree.find_lcp_node("ABD")
# print(f"\nFor 'ABD': LCP='{lcp}', remaining='{remaining}', is_end={node.is_end_of_word}")

# node, lcp, remaining = radix_tree.find_lcp_node("ABE")
# print(f"For 'ABE': LCP='{lcp}', remaining='{remaining}', is_end={node.is_end_of_word}")

# Print the radix tree structure
print("\nRadix tree structure:")
print(radix_tree.describe_tree_structure())