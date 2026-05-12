from typing import Optional, Callable, TypeVar, Generic, Protocol, overload
T = TypeVar('T')  # Generic type for values stored in the tree
K = TypeVar('K', bound="Sliceable")  # Generic type for keys (tuples of transformers)
E = TypeVar("E", covariant=True)

class Sliceable(Protocol[E]):
    def __len__(self : K) -> int: ...

    @overload
    def __getitem__(self : K, index: int) -> E: ...

    @overload
    def __getitem__(self: K, index: slice) -> K: ...



class RadixNode(Generic[K, T]):
    def __init__(self):
        self.children: dict[K, 'RadixNode[K, T]'] = {}  # key -> RadixNode
        self.value: Optional[T] = None  # (e.g., sysid) -> evaluation_index

class RadixTree(Generic[K, T]):
    def __init__(self, node_clz: Callable[[], RadixNode[K, T]] = RadixNode):
        self.node_clz = node_clz
        self.root: RadixNode[K, T] = node_clz()


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
                        new_node : RadixNode[K, T] = self.node_clz()
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
                new_child : RadixNode[K, T] = self.node_clz()
                new_child.value = value  
                node.children[remaining] = new_child
                return
        
        # If we've consumed all of remaining, mark current node as end
        node.value = value  

