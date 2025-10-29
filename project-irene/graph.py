from typing import List, Optional, Callable, Dict, Any
from timeit import default_timer as timer
import pyterrier as pt

all_eval_results: Dict[int, Any] = {}

def _eval_callback(res, eval_index: int, qrels, metrics: List[str], cum_time: float = 0.0):
    """Callback function for evaluating results using pyterrier.
    
    Args:
        res: The result to evaluate
        eval_index: Index to store the evaluation results
        qrels: Query relevance judgments
        metrics: List of evaluation metrics to compute
        cum_time: Cumulative transformation time in milliseconds
    """
    eval_out = pt.Evaluate(res, qrels, metrics)
    all_eval_results[eval_index] = eval_out
    all_eval_results[eval_index]['cum_time'] = cum_time

class Node:
    def __init__(self, me, children: Optional[List['Node']] = None, evaluation_index: Optional[int] = None):
        self.me = me
        self.children = children if children is not None else []
        self.evaluation_index = evaluation_index

    def add_child(self, child_node: "Node"):
        assert isinstance(child_node, Node), f"child_node must be a Node, got {type(child_node).__name__}"
        self.children.append(child_node)

    # def get_children(self) -> Union[List["Node"], None]:
    #     """Return the list of child Node objects or None if no children exist."""
    #     return list(self.children) if self.children else None
    def get_children(self) -> list:
        """Return the list of child Node objects."""
        return list(self.children)

    def get_children_me(self) -> list:
        """Return the list of child node me values."""
        return [child.me for child in self.children]
    
    
    def traverse(self, inp, callback: Optional[Callable] = None, cum_time: float = 0.0):
        """Traverse the graph, applying transformations and tracking cumulative time.
        
        Args:
            inp: Input data to transform
            callback: Optional callback function for evaluation
            cum_time: Cumulative transformation time in milliseconds (default 0.0)
        """
        starttime = timer()
        res = self.me.transform(inp)
        endtime = timer()
        transform_time = (endtime - starttime) * 1000.0  # Convert to milliseconds
        total_time = cum_time + transform_time
        
        if self.evaluation_index is not None:
            assert callback is not None, "evaluation_index is set but no callback was provided"
            callback(res, self.evaluation_index, total_time)
        for child in self.children:
            child.traverse(res, callback, total_time)

    def __repr__(self):
        children_repr = ', '.join(repr(child) for child in self.children)
        return f"Node({self.me}, [{children_repr}])"

class DAG:
    # anything with _me is the code replaced for name
    def get_root_nodes(self) -> list:
        """Return a list of root nodes (nodes with no parents)."""
        all_children = set()
        for node in self.nodes.values():
            all_children.update(child.me for child in node.children)
        return [node for node in self.nodes.values() if node.me not in all_children]
    def __init__(self):
        self.nodes = {}

    def add_node(self, me):
        if me not in self.nodes:
            self.nodes[me] = Node(me)
        return self.nodes[me]

    def add_edge(self, parent, child_me):
        parent = self.add_node(parent)
        child = self.add_node(child_me)
        parent.add_child(child)

    def get_children(self, parent: str) -> list:
        """Return the child Node objects for the node with the given me value.

        Raises KeyError if the parent node does not exist.
        """
        parent = self.nodes[parent]
        return parent.get_children()

    def get_children_me(self, parent: str) -> list:
        # this is a get_children names method, need to come up with a better name
        """Return the child node me values for the node with the given me value.

        Raises KeyError if the parent node does not exist.
        """
        parent = self.nodes[parent]
        return parent.get_children_me()



# Create the DAG
dag = DAG()

# Add edges for the first component
dag.add_edge("AB", "C")
dag.add_edge("AB", "D")
dag.add_edge("C", "E")


# Print the list of root nodes (disconnected components)
print(dag.get_root_nodes())