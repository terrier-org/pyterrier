# This file has useful methods for using the Pyterrier Pandas datamodel

# the first rank SHOULD be 0, see the standard "Welcome to TREC email"
FIRST_RANK = 0

def add_ranks(rtr):
    rtr.drop(columns=["rank"], errors="ignore")
    # -1 assures that first rank will be 0
    rtr["rank"] = rtr.groupby("qid").rank(ascending=False, method="first")["score"].astype(int) -1
    return rtr