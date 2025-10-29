from typing import Any, List, Tuple, Union


class RadixNode:
    def __init__(self):
        self.children = {}  # edge_label -> RadixNode
        self.is_end_of_word = False
        self.value: Any = None  # payload stored at terminal nodes (e.g., transformer later)
    # need to come back to self.value
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

    def list_items(self) -> List[Tuple[str, Any]]:
        """Return all (key, value) pairs stored in the radix tree."""
        out: List[Tuple[str, Any]] = []

        def dfs(node: RadixNode, prefix: str):
            if node.is_end_of_word:
                out.append((prefix, node.value))
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