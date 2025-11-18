from collections import defaultdict

# pipelines = ['A', 'AB', 'D', 'ABC', 'BCDE', 'BCD', 'CDE']
pipelines = ['ABCDE','ABC','ABD']
prefix_dict = defaultdict(list)
print(prefix_dict)

# create a map from prefixes to the list of pipelines that share that prefix
for pipeline in pipelines:
    for i in range(1, len(pipeline)+1):
        prefix = pipeline[:i]
        prefix_dict[prefix].append(pipeline)
        # print(prefix, prefix_map)

# print(prefix_dict.keys())
# print(prefix_dict.items())
# Find all prefixes that are shared by more than one pipeline
common_prefixes = {p: group for p, group in prefix_dict.items() if len(group) > 1}
print("Common prefixes and their groups:", common_prefixes)

# Group prefixes by their starting characters
prefix_groups = defaultdict(list)
for prefix in common_prefixes:
    prefix_groups[prefix[0]].append(prefix)

# For each group, find the longest common prefix among shared prefixes
longest_common_prefixes = set()
for group in prefix_groups.values():
    max_prefix = None
    max_shared = 0
    
    # First, find which prefixes are actually shared by multiple full pipeline names
    shared_prefixes = {}
    for prefix in group:
        # Get the full pipeline names that contain this prefix
        containing_pipelines = [p for p in pipelines if p.startswith(prefix)]
        if len(containing_pipelines) > 1:  # if more than one pipeline starts with this prefix
            shared_prefixes[prefix] = containing_pipelines
    
    # From the shared prefixes, find the longest one
    if shared_prefixes:
        longest_prefix = max(shared_prefixes.keys(), key=len)
        longest_common_prefixes.add(longest_prefix)

print("\nLongest common prefixes:", longest_common_prefixes)
for prefix in longest_common_prefixes:
    sharing = [p for p in pipelines if p.startswith(prefix)]
    print(f"Prefix '{prefix}' is shared by pipelines: {sharing}")







from typing import Any, List, Optional, Tuple, Union


class RadixNode:
    def __init__(self):
        self.children = {}  # edge_label -> RadixNode
        self.is_end_of_word = False
        self.value: Any = None  # payload stored at terminal nodes (e.g., transformer later)
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
    
    def get_children(self) -> List[Tuple[K, 'RadixNode[K]']]:
        """
        Returns:
            List of tuples containing edge labels and corresponding child RadixNode instances.
        """
        return list(self.children.items())
class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def insert(self, word: str, value: Any = None):
        """Insert a word into the radix tree, optionally attaching a payload.

        Args:
            word: The full key/path to insert.
            value: Optional payload to store at the terminal node. If None, defaults to `word`.
        """
        if not word:
            raise ValueError(f"empty or falsy key not allowed: {word!r}")
        # ////////////////////////////////////////////////////////////////////////////
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
                node.children[remaining] = new_child
                return
        
        # If we've consumed all of remaining, mark current node as end
        node.is_end_of_word = True
        node.value = word if value is None else value


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

    def pretty_print(self):
        """Print the radix tree structure with indentation.

        Each edge label is printed on its own line, indented by depth.
        Nodes that mark the end of a word are annotated with a '*'.
        """
        def dfs(node: RadixNode, depth: int):
            for edge_label, child in sorted(node.children.items()):
                end_marker = '*' if child.is_end_of_word else ''
                print("  " * depth + f"'{edge_label}'{end_marker}")
                dfs(child, depth + 1)

        dfs(self.root, 0)

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
        first = items[0]
        print("first:", first)
        if isinstance(first, tuple) and len(first) == 2:
            for key, val in items:  # type: ignore[misc]
                self.insert(str(key), val)
        else:
            for key in items:  # type: ignore[assignment]
                self.insert(str(key), key)

    def list_words(self) -> List[str]:
        """Return all stored keys (words) in the radix tree."""
        out: List[str] = []

        def dfs(node: RadixNode, prefix: str):
            if node.is_end_of_word:
                out.append(prefix)
            for edge_label, child in node.children.items():
                dfs(child, prefix + edge_label)

        dfs(self.root, "")
        return out
    
    def describe_tree_structure(self):
        """Return a structured representation of the radix tree for debugging."""
        def dfs(node: RadixNode[K], prefix: str):
            children_repr = []
            for edge_label, child in sorted(node.children.items(), key=lambda x: str(x[0])):
                full_label = str(edge_label)
                sub_repr = dfs(child, full_label)
                child_repr = sub_repr if sub_repr else {}
                children_repr.append([
                    f"{full_label}",
                    child.eval_index if child.is_end_of_word else None,
                    child_repr
                ])
            return children_repr

        return dfs(self.root, "")  

    # def list_items(self) -> List[Tuple[str, Any]]:
    #     """Return all (key, value) pairs stored in the radix tree."""
    #     out: List[Tuple[str, Any]] = []

    #     def dfs(node: RadixNode, prefix: str):
    #         if node.is_end_of_word:
    #             out.append((prefix, node.value))
    #         for edge_label, child in node.children.items():
    #             dfs(child, prefix + edge_label)

    #     dfs(self.root, "")
    #     return out
    def list_items(self) -> List[Tuple[str, "Optional[int]", dict]]:
        """Return list of (key, eval_index, children_map) for terminal nodes.

        children_map maps immediate child edge_label -> child's eval_index (or None).
        """
        out: List[Tuple[str, "Optional[int]", dict]] = []

        def dfs(node: RadixNode, prefix: str):
            if node.is_end_of_word:
                # build immediate children map
                children = {edge_label: child.eval_index for edge_label, child in node.children.items()}
                out.append((prefix, node.eval_index, children))
            for edge_label, child in node.children.items():
                dfs(child, prefix + edge_label)

        dfs(self.root, "")
        return out

# Example usage
radix_tree = RadixTree()
# radix_tree.insert("ueue")
# radix_tree.insert("eue")
# radix_tree.insert("ue")
# radix_tree.insert("e")
radix_tree.insert_all(['ABCD', 'AB'])

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
radix_tree.pretty_print()
