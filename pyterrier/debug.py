from .transformer import TransformerBase
from typing import List

def print_columns(by_query : bool = False, message : str = None) -> TransformerBase:
    """
    Returns a transformer that can be inserted into pipelines that can print the column names of the dataframe
    at this stage in the pipeline:

    Arguments:
     - by_query(bool): whether to display for each query. Defaults to False.
     - message(str): whether to display a message before printing. Defaults to None, which means no message. This
     is useful when print_columns() is being used multiple times within a pipeline 
     

    Example::
    
        pipe = (
            bm25
            >> pt.debug.print_columns() 
            >> pt.rewrite.RM3() 
            >> pt.debug.print_columns()
            bm25

    When the above pipeline is executed, two sets of columns will be displayed
     - `["qid", "query", "docno", "rank", "score"]`  - the output of BM25, a ranking of documents
     - `["qid", "query", "query_0"]`   - the output of RM3, a reformulated query
    
        
    """
    import pyterrier as pt
    def _do_print(df):
        if message is not None:
            print(message)
        print(df.columns)
        return df
    return pt.apply.by_query(_do_print) if by_query else pt.apply.generic(_do_print) 

def print_rows(
        by_query : bool = True, 
        jupyter: bool = True, 
        head : int = 2, 
        message : str = None, 
        columns : List[str] = None) -> TransformerBase:
    """
    Returns a transformer that can be inserted into pipelines that can print some of the dataframe
    at this stage in the pipeline:

    Arguments:
     - by_query(bool): whether to display for each query. Defaults to True.
     - jupyter(bool): Whether to use IPython's display function to display the dataframe. Defaults to True.
     - head(int): The number of rows to display. None means all rows.
     - columns(List[str]): Limit the columns for which data is displayed. Default of None displays all columns.
     - message(str): whether to display a message before printing. Defaults to None, which means no message. This
       is useful when print_rows() is being used multiple times within a pipeline 

    Example::

        pipe = (
            bm25
            >> pt.debug.print_rows() 
            >> pt.rewrite.RM3() 
            >> pt.debug.print_rows()
            bm25
     
    """
    import pyterrier as pt
    def _do_print(df):
        if message is not None:
            print(message)
        render = df if head is None else df.head(head)
        if columns is not None:
            render = render[columns]
        if jupyter:
            from IPython.display import display
            display(render)
        else:
            print(render)
        return df
    return pt.apply.by_query(_do_print) if by_query else pt.apply.generic(_do_print) 