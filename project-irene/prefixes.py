from collections import defaultdict

pipelines = ['AB','AC']
# pipelines = ['ABCDE','ABC','ABD']
# pipelines = ['A', 'AB', 'DE', 'ABC', 'BCDE', 'BCD', 'B']
# pipelines = []
def compute_final_prefixes(pipelines):
    prefix_dict = defaultdict(list)
    for pipeline in pipelines:
        for i in range(1, len(pipeline)+1):
            prefix = pipeline[:i]
            prefix_dict[prefix].append(pipeline)

    first_letter_groups = defaultdict(list)
    for key in prefix_dict:
        first_letter_groups[key[0]].append(key) 

    # Process each group to find the key with most elements in its value
    final_prefixes = {}
    for letter, keys in first_letter_groups.items():
        if len(keys) == 1:
            # Only one key with this letter, keep it
            key = keys[0]
            final_prefixes[key] = prefix_dict[key]
        else:
            # Multiple keys starting with same letter
            max_elements = 0
            best_key = None

            for key in keys:
                elements = len(prefix_dict[key])
                current_value_set = frozenset(prefix_dict[key])
                # If current key has more elements, it's automatically better
                if elements > max_elements:
                    max_elements = elements
                    best_key = key
                # If same number of elements, check if values are identical
                elif elements == max_elements and best_key:
                    best_value_set = frozenset(prefix_dict[best_key])
                    # If values are identical, prefer longer prefix
                    if current_value_set == best_value_set and len(key) > len(best_key):
                        best_key = key
                    # If values are different but same length, keep existing best_key
                
            if best_key:
                final_prefixes[best_key] = prefix_dict[best_key]

    return final_prefixes

print(compute_final_prefixes(pipelines))