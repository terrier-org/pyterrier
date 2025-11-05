"""
Test and demonstration of generic RadixTree and TransformerRadixTree.
"""
from trie import RadixTree, TransformerRadixTree
import pandas as pd


class DummyTransformer:
    """Mock transformer for testing - has a name and a transform method."""
    def __init__(self, name: str):
        self.name = name
    
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Mock transform that adds a column."""
        inp[self.name] = f"transformed_by_{self.name}"
        return inp
    
    def __repr__(self):
        return f"DummyTransformer({self.name})"


# ============================================================================
# Test 1: String-based RadixTree (original behavior)
# ============================================================================
print("=" * 70)
print("Test 1: String-based RadixTree")
print("=" * 70)

string_tree = RadixTree[str]()
string_tree.insert_all(["AB", "ABC", "D"])
structure = string_tree.describe_tree_structure()
print("String tree structure:")
print(structure)
print()


# ============================================================================
# Test 2: TransformerRadixTree with pipelines
# ============================================================================
print("=" * 70)
print("Test 2: TransformerRadixTree with Transformer pipelines")
print("=" * 70)

# Create some dummy transformers
t1 = DummyTransformer("retrieval")
t2 = DummyTransformer("rerank1")
t3 = DummyTransformer("rerank2")

transformer_tree = TransformerRadixTree()

# Insert different pipeline configurations
# Key insight: Each pipeline is identified by its transformers.
# We can store the entire pipeline tuple as the value, or just metadata.
# For traverse to work, we need actual transformer objects that have .transform()

# Strategy: Store a composite transformer that applies all in sequence
class CompositeTransformer:
    def __init__(self, transformers):
        self.transformers = transformers
    
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result = inp.copy()
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result
    
    def __repr__(self):
        return f"CompositeTransformer({len(self.transformers)} stages)"

# Insert different pipeline configurations
pipelines = [
    ((t1,), CompositeTransformer([t1])),
    ((t1, t2), CompositeTransformer([t1, t2])),
    ((t1, t2, t3), CompositeTransformer([t1, t2, t3])),
]

for i, (pipeline_key, pipeline_value) in enumerate(pipelines):
    transformer_tree.insert(pipeline_key, pipeline_value, eval_index=i)

print("Transformer tree structure (showing which pipelines are cached):")
structure = transformer_tree.describe_tree_structure()
print(structure)
print()


# ============================================================================
# Test 3: Verify traverse functionality with mock data
# ============================================================================
print("=" * 70)
print("Test 3: Traverse with transformers (mock execution)")
print("=" * 70)

# Create a test DataFrame
test_data = pd.DataFrame({
    'query': ['test_query_1', 'test_query_2'],
    'doc_id': ['doc1', 'doc2'],
    'score': [0.95, 0.87]
})

print("Input DataFrame:")
print(test_data)
print()

# Define a callback to capture results
results = {}
def capture_result(result, eval_index, total_time):
    results[eval_index] = {
        'result': result,
        'time_ms': total_time
    }
    print(f"  Evaluation {eval_index} completed in {total_time:.2f}ms")

print("Running traverse on transformer tree (from first child):")
# Get the first child node (first pipeline)
first_child = list(transformer_tree.root.children.values())[0]
first_child.traverse(test_data, capture_result)
print()

print("Results from traverse:")
for idx, data in results.items():
    print(f"\n  Index {idx} - Time: {data['time_ms']:.2f}ms")
    print(f"  Result shape: {data['result'].shape}")
    print(f"  Columns: {list(data['result'].columns)}")


# ============================================================================
# Test 4: Generic RadixTree with tuple keys (sequence of strings)
# ============================================================================
print("\n" + "=" * 70)
print("Test 4: Generic RadixTree with tuple keys")
print("=" * 70)

tuple_tree = RadixTree[tuple]()

# Insert sequences of strings
sequences = [
    (("step1",), "just_step1"),
    (("step1", "step2"), "step1_and_step2"),
    (("step1", "step2", "step3"), "full_sequence"),
]

for seq_key, value in sequences:
    tuple_tree.insert(seq_key, value)

print("Tuple-key tree structure:")
structure = tuple_tree.describe_tree_structure()
print(structure)
print()


print("=" * 70)
print("✓ All tests completed successfully!")
print("=" * 70)
