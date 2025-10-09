from deprecated import deprecated
import pyterrier as pt

@deprecated(version='1.0.0', reason="use pyterrier_alpha.fusion.PerQueryMaxMinScore() instead")
class PerQueryMaxMinScoreTransformer(pt.Transformer):
    '''
    applies per-query maxmin scaling on the input scores
    '''
    
    def transform(self, topics_and_res):
        from sklearn.preprocessing import minmax_scale
        topics_and_res = topics_and_res.copy()
        topics_and_res["score"] = topics_and_res.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return topics_and_res

Experiment = deprecated(version='1.0.0', reason="use pt.Experiment() instead")(pt.Experiment)
Evaluate = deprecated(version='1.0.0', reason="use pt.Evaluate() instead")(pt.Experiment)
GridScan = deprecated(version='1.0.0', reason="use pt.GridScan() instead")(pt.GridScan)
GridSearch = deprecated(version='1.0.0', reason="use pt.GridSearch() instead")(pt.GridScan)
KFoldGridSearch = deprecated(version='1.0.0', reason="use pt.KFoldGridSearch() instead")(pt.KFoldGridSearch)
