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

