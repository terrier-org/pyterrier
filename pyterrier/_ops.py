from .transformer import Transformer, Estimator, get_transformer, SupportsFuseFeatureUnion, SupportsFuseRankCutoff, SupportsFuseRight, SupportsFuseLeft
from .model import add_ranks
from collections import deque
from warnings import warn
from typing import Optional, Iterable, Tuple, Iterator, List
from itertools import chain
import pandas as pd
import pyterrier as pt

class NAryTransformerBase(Transformer):
    """
        A base class for all operator transformers that can combine the input of 2 or more transformers. 
    """
    def __init__(self, *transformers: Transformer):
        assert len(transformers) > 0
        # Flatten out multiple layers of the same NAryTransformer into one
        transformers = _flatten(transformers, type(self))
        # Coerce datatypes
        self._transformers = tuple(get_transformer(x, stacklevel=6) for x in transformers)

    def __getitem__(self, number) -> Transformer:
        """
            Allows access to the ith transformer.
        """
        return self._transformers[number]

    def __len__(self) -> int:
        """
            Returns the number of transformers in the operator.
        """
        return len(self._transformers)

    def __iter__(self) -> Iterator[Transformer]:
        """
            Returns an iterator over the transformers in this pipeline.
        """
        return iter(self._transformers)

def _flatten(transformers: Iterable[Transformer], cls: type) -> Tuple[Transformer, ...]:
    return tuple(chain.from_iterable(
        (t._transformers if isinstance(t, cls) else [t]) # type: ignore[attr-defined]
        for t in transformers
    ))

class SetUnion(Transformer):
    """      
        This operator makes a retrieval set that includes documents that occur in the union (either) of both retrieval sets. 
        For instance, let left and right be pandas dataframes, both with the columns = [qid, query, docno, score], 
        left = [1, "text1", doc1, 0.42] and right = [1, "text1", doc2, 0.24]. 
        Then, left | right will be a dataframe with only the columns [qid, query, docno] and two rows = [[1, "text1", doc1], [1, "text1", doc2]].
                
        In case of duplicated both containing (qid, docno), only the first occurrence will be used.
    """
    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    schematic = {'label': 'SetUnion |', 'inner_pipelines_mode': 'linked'}

    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)
        import pandas as pd
        assert isinstance(res1, pd.DataFrame)
        assert isinstance(res2, pd.DataFrame)
        rtr = pd.concat([res1, res2])
        
        on_cols = ["qid", "docno"]     
        rtr = rtr.drop_duplicates(subset=on_cols)
        rtr = rtr.sort_values(by=on_cols)
        rtr.drop(columns=["score", "rank"], inplace=True, errors='ignore')
        return rtr

class SetIntersection(Transformer):
    """
        This operator makes a retrieval set that only includes documents that occur in the intersection of both retrieval sets. 
        For instance, let left and right be pandas dataframes, both with the columns = [qid, query, docno, score], 
        left = [[1, "text1", doc1, 0.42]] (one row) and right = [[1, "text1", doc1, 0.24],[1, "text1", doc2, 0.24]] (two rows).
        Then, left & right will be a dataframe with only the columns [qid, query, docno] and one single row = [[1, "text1", doc1]].
                
        For columns other than (qid, docno), only the left value will be used.
    """
    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    schematic = {'label': 'SetIntersection &', 'inner_pipelines_mode': 'linked'}

    def transform(self, topics):
        res1 = self.left.transform(topics)
        res2 = self.right.transform(topics)  
        
        on_cols = ["qid", "docno"]
        rtr = res1.merge(res2, on=on_cols, suffixes=('','_y'))
        rtr.drop(columns=["score", "rank", "score_y", "rank_y", "query_y"], inplace=True, errors='ignore')
        for col in rtr.columns:
            if '_y' not in col:
                continue
            new_name = col.replace('_y', '')
            if new_name in rtr.columns:
                # duplicated column, drop
                rtr.drop(columns=[col], inplace=True)
                continue
            # column only from RHS, keep, but rename by removing '_y' suffix
            rtr.rename(columns={col:new_name}, inplace=True)

        return rtr

class Sum(Transformer):
    """
        Adds the scores of documents from two different retrieval transformers.
        Documents not present in one transformer are given a score of 0.
    """
    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    schematic = {'label': 'Sum +', 'inner_pipelines_mode': 'linked'}

    def transform(self, topics_and_res):
        res1 = self.left.transform(topics_and_res)
        res2 = self.right.transform(topics_and_res)
        both_cols = set(res1.columns) & set(res2.columns)
        both_cols.remove("qid")
        both_cols.remove("docno")
        merged = res1.merge(res2, on=["qid", "docno"], suffixes=[None, "_r"], how='outer')
        merged["score"] = merged["score"].fillna(0) + merged["score_r"].fillna(0)
        merged = merged.drop(columns=["%s_r" % col for col in both_cols])
        merged = add_ranks(merged)
        return merged

class Concatenate(Transformer):
    epsilon = 0.0001

    def __init__(self, left: Transformer, right: Transformer):
        self.left = left
        self.right = right

    def transform(self, topics_and_res):
        import pandas as pd
        # take the first set as the top of the ranking
        res1 = self.left.transform(topics_and_res)
        # identify the lowest score for each query
        last_scores = res1[['qid', 'score']].groupby('qid').min().rename(columns={"score" : "_lastscore"})

        # the right hand side will provide the rest of the ranking        
        res2 = self.right.transform(topics_and_res)

        
        intersection = pd.merge(res1[["qid", "docno"]], res2[["qid", "docno"]].reset_index())
        remainder = res2.drop(intersection["index"])

        # we will append documents from remainder to res1
        # but we need to offset the score from each remaining document based on the last score in res1
        # explanation: remainder["score"] - remainder["_firstscore"] - self.epsilon ensures that the
        # first document in remainder has a score of -epsilon; we then add the score of the last document
        # from res1
        first_scores = remainder[['qid', 'score']].groupby('qid').max().rename(columns={"score" : "_firstscore"})

        remainder = remainder.merge(last_scores, on=["qid"]).merge(first_scores, on=["qid"])
        remainder["score"] = remainder["score"] - remainder["_firstscore"] + remainder["_lastscore"] - self.epsilon
        remainder = remainder.drop(columns=["_lastscore",  "_firstscore"])

        # now bring together and re-sort
        # this sort should match trec_eval
        rtr = pd.concat([res1, remainder]).sort_values(by=["qid", "score", "docno"], ascending=[True, False, True]) 

        # recompute the ranks
        rtr = add_ranks(rtr)
        return rtr

class ScalarProduct(Transformer):
    """
        Multiplies the retrieval score by a scalar
    """
    def __init__(self, scalar: float):
        self.scalar = scalar

    def transform(self, inp):
        out = inp.assign(score=inp["score"] * self.scalar)
        if self.scalar < 0:
            out = add_ranks(out)
        return out
    
    def schematic(self, *, input_columns = None): 
        return {'label': f'* {self.scalar}'}

    def __repr__(self):
        return f'ScalarProduct({self.scalar!r})'

    def __eq__(self, other):
        if not isinstance(other, ScalarProduct):
            return NotImplemented
        return self.scalar == other.scalar

    def __hash__(self):
        return hash(('ScalarProduct', self.scalar))

class RankCutoff(Transformer):
    """
        Filters the input by rank<k for each query in the input
    """
    def __init__(self, k: int = 1000):
        self.k = k

    def transform(self, inp):
        assert 'rank' in inp.columns, "require rank to be present in the result set"
        res = inp[inp["rank"] < self.k]
        res = res.reset_index(drop=True)
        return res

    def __repr__(self):
        return f'RankCutoff({self.k!r})'

    def __eq__(self, other):
        if not isinstance(other, RankCutoff):
            return NotImplemented
        return self.k == other.k

    def __hash__(self):
        return hash(('RankCutoff', self.k))

    def schematic(self, *, input_columns = None): 
        return {'label': f'% {self.k}'}

    def fuse_left(self, left: Transformer) -> Optional[Transformer]:
        # If the preceding component supports a native rank cutoff (via fuse_rank_cutoff), apply it.
        if isinstance(left, SupportsFuseRankCutoff):
            return left.fuse_rank_cutoff(self.k)
        return None

class _TrajectoryStoppingRule(Transformer):
    """Applies a stopping rule to each ranked, labelled review trajectory."""

    def _stop_position(self, labels):
        raise NotImplementedError

    def _stop_position_for_results(self, ranked, labels):
        return self._stop_position(labels)

    def _stopping_decisions(self, inp):
        pt.validate.columns(inp, includes=['qid', 'rank', 'label'], context=self)
        for qid, query_results in inp.groupby('qid', sort=False):
            ranked = query_results.sort_values('rank', kind='stable')
            labels = pd.to_numeric(ranked['label'], errors='raise').fillna(0).to_numpy()
            yield qid, ranked, self._stop_position_for_results(ranked, labels)

    def transform(self, inp):
        if len(inp) == 0:
            pt.validate.columns(inp, includes=['qid', 'rank', 'label'], context=self)
            return inp.copy()
        output = [ranked if stop is None else ranked.iloc[:stop] for _, ranked, stop in self._stopping_decisions(inp)]
        return pd.concat(output, ignore_index=True)

    def stop_report(self, inp):
        """Return one row per query with the stopping decision and its full-review cost.

        ``stop`` is the number of ranked documents retained. ``fired``
        distinguishes a rule which stopped at the final document from one that
        never certified a stop. ``control_cost`` records separately screened
        documents for control-set rules and is zero otherwise.
        """
        rows = []
        control_cost = getattr(self, 'control_cost', None)
        for qid, ranked, stop in self._stopping_decisions(inp):
            rows.append({
                'qid': qid,
                'stop': len(ranked) if stop is None else int(stop),
                'fired': stop is not None,
                'n_ranked': len(ranked),
                'control_cost': 0 if control_cost is None else int(control_cost(qid)),
            })
        return pd.DataFrame(rows, columns=['qid', 'stop', 'fired', 'n_ranked', 'control_cost'])


def _batch_positions(size, batch_size, initial_documents):
    if size < initial_documents:
        return ()
    return range(initial_documents, size + 1, batch_size)


def _knee_slope_ratio(cumulative):
    import numpy as np

    stop = len(cumulative)
    if stop < 2 or cumulative[-1] == 0:
        return 0.0
    positions = np.arange(1, stop)
    knee = np.argmax(stop * cumulative[:-1] - positions * cumulative[-1]) + 1
    return (cumulative[knee - 1] / knee) / ((cumulative[-1] - cumulative[knee - 1] + 1) / (stop - knee))


def _validate_collection_size(labels, collection_size):
    if len(labels) > collection_size:
        raise ValueError("collection_size cannot be smaller than the review trajectory")


class Kneedle(_TrajectoryStoppingRule):
    """Filters each query's ranking at the first TAR Knee-method stopping point.

    The input must contain ``qid``, ``rank``, and ``label`` columns. It uses only
    labels at or above each candidate rank, so a complete labelled ranking can
    be used to reproduce an offline priority-screening evaluation. If the rule
    does not stop, the complete ranking is retained.
    """
    def __init__(self, min_documents: int = 1000, batch_size: int = 200, initial_documents: int = 1):
        if min_documents < 2:
            raise ValueError("min_documents must be at least 2")
        if batch_size < 1 or initial_documents < 1:
            raise ValueError("batch_size and initial_documents must be positive")
        self.min_documents = min_documents
        self.batch_size = batch_size
        self.initial_documents = initial_documents

    def _check_positions(self, size):
        from math import ceil

        bmi_batch = bmi_size = self.initial_documents
        positions = set()
        while bmi_size < size:
            bmi_batch += ceil(bmi_batch / 10)
            bmi_size += bmi_batch
            if bmi_size >= self.min_documents:
                fixed = self.initial_documents + ceil((bmi_size - self.initial_documents) / self.batch_size) * self.batch_size
                if fixed <= size:
                    positions.add(fixed)
        return sorted(positions)

    def _stop_position(self, labels):
        cumulative = (labels > 0).cumsum()
        for stop in self._check_positions(len(cumulative)):
            found = cumulative[stop - 1]
            if _knee_slope_ratio(cumulative[:stop]) >= 156 - min(found, 150):
                return stop
        return None

    def __repr__(self):
        return f'Kneedle(min_documents={self.min_documents!r}, batch_size={self.batch_size!r}, initial_documents={self.initial_documents!r})'

    def __eq__(self, other):
        if not isinstance(other, Kneedle):
            return NotImplemented
        return (self.min_documents, self.batch_size, self.initial_documents) == (other.min_documents, other.batch_size, other.initial_documents)

    def __hash__(self):
        return hash(('Kneedle', self.min_documents, self.batch_size, self.initial_documents))

    def schematic(self, *, input_columns = None):
        return {'label': 'Kneedle'}


class FixedRound(_TrajectoryStoppingRule):
    """Stops after a fixed number of review rounds."""

    def __init__(self, max_round: int, batch_size: int = 200, initial_documents: int = 1):
        if max_round < 0 or batch_size < 1 or initial_documents < 1:
            raise ValueError("max_round must be non-negative and batch sizes must be positive")
        self.max_round = max_round
        self.batch_size = batch_size
        self.initial_documents = initial_documents

    def _stop_position(self, labels):
        stop = self.initial_documents + self.max_round * self.batch_size
        return stop if stop <= len(labels) else None


class BatchPrecision(_TrajectoryStoppingRule):
    """Stops after ``patience`` low-precision review batches."""

    def __init__(self, precision_cutoff: float = 5 / 200, patience: int = 1, batch_size: int = 200, initial_documents: int = 1):
        if not 0 <= precision_cutoff <= 1 or patience < 1 or batch_size < 1 or initial_documents < 1:
            raise ValueError("precision_cutoff must be in [0, 1] and counts must be positive")
        self.precision_cutoff = precision_cutoff
        self.patience = patience
        self.batch_size = batch_size
        self.initial_documents = initial_documents

    def _stop_position(self, labels):
        start = low_precision_batches = 0
        for stop in _batch_positions(len(labels), self.batch_size, self.initial_documents):
            precision = (labels[start:stop] > 0).mean()
            low_precision_batches = low_precision_batches + 1 if precision <= self.precision_cutoff else 0
            if low_precision_batches >= self.patience:
                return stop
            start = stop
        return None


class Rule2399(_TrajectoryStoppingRule):
    """Stops when reviewed documents reach ``2399 + 1.2 * positives``."""

    def __init__(self, batch_size: int = 200, initial_documents: int = 1):
        if batch_size < 1 or initial_documents < 1:
            raise ValueError("batch_size and initial_documents must be positive")
        self.batch_size = batch_size
        self.initial_documents = initial_documents

    def _stop_position(self, labels):
        cumulative = (labels > 0).cumsum()
        for stop in _batch_positions(len(labels), self.batch_size, self.initial_documents):
            if stop >= 2399 + 1.2 * cumulative[stop - 1]:
                return stop
        return None


class ReviewHalf(_TrajectoryStoppingRule):
    """Stops after reviewing half of a known collection."""

    def __init__(self, collection_size: int, batch_size: int = 200, initial_documents: int = 1):
        if collection_size < 1 or batch_size < 1 or initial_documents < 1:
            raise ValueError("collection_size and batch sizes must be positive")
        self.collection_size = collection_size
        self.batch_size = batch_size
        self.initial_documents = initial_documents

    def _stop_position(self, labels):
        _validate_collection_size(labels, self.collection_size)
        for stop in _batch_positions(len(labels), self.batch_size, self.initial_documents):
            if stop >= self.collection_size // 2:
                return stop
        return None


class Budget(_TrajectoryStoppingRule):
    """Cormack and Grossman's budget stopping rule for a known collection."""

    def __init__(self, collection_size: int, batch_size: int = 200, initial_documents: int = 1):
        if collection_size < 1 or batch_size < 1 or initial_documents < 1:
            raise ValueError("collection_size and batch sizes must be positive")
        self.collection_size = collection_size
        self.batch_size = batch_size
        self.initial_documents = initial_documents

    def _stop_position(self, labels):
        _validate_collection_size(labels, self.collection_size)
        cumulative = (labels > 0).cumsum()
        for stop in _batch_positions(len(labels), self.batch_size, self.initial_documents):
            if stop >= .75 * self.collection_size:
                return stop
            found = cumulative[stop - 1]
            if found and stop >= 10 * self.collection_size / found and _knee_slope_ratio(cumulative[:stop]) >= 6:
                return stop
        return None


class CMHHeuristic(_TrajectoryStoppingRule):
    """Callaghan and Müller-Hansen's one-phase CMH stopping heuristic."""

    def __init__(self, target_recall: float, collection_size: int, alpha: float = .05, batch_size: int = 200, initial_documents: int = 1):
        if not 0 < target_recall <= 1 or not 0 < alpha < 1 or collection_size < 1 or batch_size < 1 or initial_documents < 1:
            raise ValueError("target_recall and alpha must be in (0, 1), and sizes must be positive")
        self.target_recall = target_recall
        self.collection_size = collection_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.initial_documents = initial_documents

    def _stop_position(self, labels):
        from scipy.stats import hypergeom

        _validate_collection_size(labels, self.collection_size)
        batch_ends = list(_batch_positions(len(labels), self.batch_size, self.initial_documents))
        previous = 0
        positives = []
        for stop in batch_ends:
            positives.append((labels[previous:stop] > 0).sum())
            previous = stop
            if len(positives) < 3:
                continue
            positive_total = sum(positives)
            for split in range(1, len(positives) - 1):
                positive_before = sum(positives[:split + 1])
                reviewed_before = batch_ends[split]
                p_value = hypergeom.cdf(
                    positive_total - positive_before,
                    self.collection_size - reviewed_before,
                    int(positive_total / self.target_recall - positive_before + 1),
                    stop - reviewed_before,
                )
                if p_value < self.alpha:
                    return stop
        return None


class PoissonPoint(_TrajectoryStoppingRule):
    """Inhomogeneous Poisson power-law stopping from Bin-Hezam and Stevenson.

    The defaults reproduce the released IP-P configuration: ten windows, 2.5%
    checkpoints, a dynamic 20-positive gate, and a 0.1 fit-error cutoff.
    """

    def __init__(self, collection_size: int, target_recall: float = .8, confidence: float = .95,
                 initial_fraction: float = .025, check_fraction: float = .025,
                 n_windows: int = 10, fit_error_threshold: float = .1,
                 initial_min_relevant: int = 20):
        if (collection_size < 1 or not 0 < target_recall <= 1 or not 0 < confidence < 1
                or not 0 < initial_fraction <= 1 or not 0 < check_fraction <= 1
                or n_windows < 2 or fit_error_threshold < 0 or initial_min_relevant < 0):
            raise ValueError("invalid Poisson point-process parameter")
        self.collection_size = collection_size
        self.target_recall = target_recall
        self.confidence = confidence
        self.initial_fraction = initial_fraction
        self.check_fraction = check_fraction
        self.n_windows = n_windows
        self.fit_error_threshold = fit_error_threshold
        self.initial_min_relevant = initial_min_relevant

    def _sample_sizes(self):
        import numpy as np

        proportions = np.arange(self.initial_fraction, 1 + self.check_fraction, self.check_fraction).round(3)
        return [int(round(self.collection_size * proportion)) for proportion in proportions]

    @staticmethod
    def _power_law(x, a, k):
        return a * x ** k

    @staticmethod
    def _poisson_upper_bound(mean, confidence, maximum):
        from scipy.stats import poisson

        if mean < 0 or not maximum:
            return None
        bound = poisson.ppf(confidence, mean)
        return None if not pd.notna(bound) else min(int(bound), maximum)

    def _stop_position(self, labels):
        import numpy as np
        from scipy.optimize import curve_fit

        _validate_collection_size(labels, self.collection_size)
        minimum_relevant = self.initial_min_relevant
        for sample_size in self._sample_sizes():
            if sample_size > len(labels):
                break
            if sample_size < self.n_windows:
                continue
            observed = labels[:sample_size] > 0
            found = int(observed.sum())
            if found >= minimum_relevant:
                windows = np.array_split(np.arange(sample_size), self.n_windows)
                window_size = len(windows[0]) - 1
                if window_size > 0:
                    x = np.array([window[0] + 1 for window in windows])
                    y = np.array([observed[window[0]:window[-1]].sum() / window_size for window in windows])
                    if y[5:].sum() == 0:
                        return sample_size
                    try:
                        parameters, _ = curve_fit(self._power_law, x, y, p0=[.1, .001])
                        predicted = self._power_law(x, *parameters)
                        scale = y.max() - y.min()
                        error = np.square(y - predicted).sum() / scale if scale else np.inf
                        if np.isfinite(error) and error < self.fit_error_threshold:
                            a, k = parameters
                            residual_mean = a / (k + 1) * (
                                self.collection_size ** (k + 1) - sample_size ** (k + 1)
                            )
                            residual = self._poisson_upper_bound(residual_mean, self.confidence, self.collection_size - sample_size)
                            if residual is not None and found >= self.target_recall * (found + residual):
                                return sample_size
                    except (FloatingPointError, RuntimeError, ValueError, ZeroDivisionError):
                        pass
            if minimum_relevant:
                minimum_relevant = int(minimum_relevant - sample_size / self.collection_size * minimum_relevant)
        return None


class _ControlSetStoppingRule(_TrajectoryStoppingRule):
    """Base for stopping rules certified by a separately screened control set."""

    def __init__(self, control: pd.DataFrame):
        required = ['qid', 'docno', 'label']
        if not isinstance(control, pd.DataFrame) or any(column not in control.columns for column in required):
            raise ValueError("control must be a DataFrame with qid, docno, and label columns")
        if control[required[:2]].isna().any().any() or control.duplicated(required[:2]).any():
            raise ValueError("control qid/docno pairs must be present and unique")
        self.control = control.loc[:, required].copy()
        self.control['label'] = pd.to_numeric(self.control['label'], errors='raise').fillna(0)

    def control_cost(self, qid=None):
        """Number of separately screened control documents, optionally for one query."""
        return len(self.control) if qid is None else int((self.control['qid'] == qid).sum())

    def _positive_control_positions(self, ranked):
        if ranked['docno'].duplicated().any():
            raise ValueError("ranked results must not duplicate docno within a query")
        positives = self.control.loc[(self.control['qid'] == ranked['qid'].iloc[0]) & (self.control['label'] > 0), 'docno']
        if len(positives) == 0:
            return ()
        positions = pd.Series(range(1, len(ranked) + 1), index=ranked['docno']).reindex(positives)
        return None if positions.isna().any() else tuple(sorted(positions.astype(int)))


class TargetRecapture(_ControlSetStoppingRule):
    """Target-set recapture stopping with an explicit independently screened control set.

    ``control`` must contain every separately screened document, including its
    labels. Its screening cost is available through :meth:`control_cost` and is
    intentionally not hidden in the ranked-review output.
    """

    def __init__(self, control: pd.DataFrame, target_recall: float = .7, confidence: float = .95):
        from math import ceil, log

        if not 0 < target_recall < 1 or not 0 < confidence < 1:
            raise ValueError("target_recall and confidence must be in (0, 1)")
        super().__init__(control)
        self.target_recall = target_recall
        self.confidence = confidence
        self.target_size = ceil(-log(1 - confidence) / (1 - target_recall))

    def _stop_position_for_results(self, ranked, labels):
        positions = self._positive_control_positions(ranked)
        if positions is None or len(positions) < self.target_size:
            return None
        return positions[-1]


class QBCB(_ControlSetStoppingRule):
    """Quantile Binomial Confidence Bound with an explicit control set."""

    def __init__(self, control: pd.DataFrame, target_recall: float = .8, confidence: float = .95):
        if not 0 < target_recall < 1 or not 0 < confidence < 1:
            raise ValueError("target_recall and confidence must be in (0, 1)")
        super().__init__(control)
        self.target_recall = target_recall
        self.confidence = confidence

    def _required_control_rank(self, positive_control_count):
        from scipy.stats import binom

        for rank in range(1, positive_control_count + 1):
            if binom.cdf(rank - 1, positive_control_count, self.target_recall) >= self.confidence:
                return rank
        return positive_control_count + 1

    def _stop_position_for_results(self, ranked, labels):
        positions = self._positive_control_positions(ranked)
        if positions is None:
            return None
        required_rank = self._required_control_rank(len(positions))
        return positions[required_rank - 1] if required_rank <= len(positions) else None

class FeatureUnion(NAryTransformerBase):
    """
        Implements the feature union operator.

        Example::
            cands = pt.terrier.Retriever(index wmodel="BM25")
            pl2f = pt.terrier.Retriever(index wmodel="PL2F")
            bm25f = pt.terrier.Retriever(index wmodel="BM25F")
            pipe = cands >> (pl2f ** bm25f)
    """
    schematic = {'inner_pipelines_mode': 'linked', 'label': 'FeatureUnion **'}

    def transform_inputs(self) -> Optional[List[List[str]]]:
        this_minimum : List[Optional[List[List[str]]]] = [[['qid', 'docno']]]
        return pt.inspect._minimal_inputs(this_minimum + [ 
            pt.inspect.transformer_inputs(t) for t in self._transformers
        ])

    def transform(self, inputRes):
        pt.validate.result_frame(inputRes, context=self)

        num_results = len(inputRes)
        import numpy as np

        # a parent could be a feature union, but it still passes the inputRes directly, so inputRes should never have a features column
        if "features" in inputRes.columns:
            raise ValueError("FeatureUnion operates as a re-ranker. They can be nested, but input "
                "should not contain a features column; found columns were %s" %  str(inputRes.columns))
        
        all_results = []

        for i, m in enumerate(self._transformers):
            # IMPORTANT this .copy() is important, in case an operand transformer changes inputRes
            results = m.transform(inputRes.copy())
            if len(results) == 0 and num_results != 0:
                raise ValueError("Got no results from %s, expected %d" % (repr(m), num_results) )
            assert "features_x" not in results.columns 
            assert "features_y" not in results.columns
            all_results.append( results )

    
        for i, (m, res) in enumerate(zip(self._transformers, all_results)):
            # IMPORTANT: dont do this BEFORE calling subsequent feature unions
            if "features" not in res.columns:
                if "score" not in res.columns:
                    raise ValueError("Results from %s did not include either score or features columns, found columns were %s" % (repr(m), str(res.columns)) )

                if len(res) != num_results:
                    warn(
                        "Got number of results different expected from %s, expected %d received %d, feature scores for any "
                        "missing documents be 0, extraneous documents will be removed" % (repr(m), num_results, len(res)))
                    all_results[i] = res = inputRes[["qid", "docno"]].merge(res, on=["qid", "docno"], how="left")
                    res["score"] = res["score"].fillna(value=0)

                if len(res) == 0:
                    res["features"] = pd.Series([], dtype='float64')
                else:
                    res["features"] = res.apply(lambda row : np.array([row["score"]]), axis=1)
                res.drop(columns=["score"], inplace=True)
            assert "features" in res.columns
            #print("%d got %d features from operand %d" % ( id(self) ,   len(results.iloc[0]["features"]), i))

        def _concat_features(row):
            assert isinstance(row["features_x"], np.ndarray)
            assert isinstance(row["features_y"], np.ndarray)
            
            left_features = row["features_x"]
            right_features = row["features_y"]
            return np.concatenate((left_features, right_features))
        
        def _reduce_fn(left, right):
            import pandas as pd
            both_cols = set(left.columns) & set(right.columns)
            both_cols.remove("qid")
            both_cols.remove("docno")
            both_cols.remove("features")
            rtr = pd.merge(left, right, on=["qid", "docno"])
            rtr["features"] = rtr.apply(_concat_features, axis=1, result_type='reduce')
            rtr.rename(columns={"%s_x" % col : col for col in both_cols}, inplace=True)
            rtr.drop(columns=["features_x", "features_y"] + ["%s_y" % col for col in both_cols], inplace=True)
            return rtr
        
        from functools import reduce
        final_DF = reduce(_reduce_fn, all_results)

        # final_DF should have the features column
        assert "features" in final_DF.columns

        # we used .copy() earlier, inputRes should still have no features column
        assert "features" not in inputRes.columns

        # final merge - this brings us the score attribute from any previous transformer
        both_cols = set(inputRes.columns) & set(final_DF.columns)
        both_cols.remove("qid")
        both_cols.remove("docno")
        final_DF = inputRes.merge(final_DF, on=["qid", "docno"])
        final_DF.rename(columns={"%s_x" % col : col for col in both_cols}, inplace=True)
        final_DF.drop(columns=["%s_y" % col for col in both_cols], inplace=True)
        # remove the duplicated columns
        #final_DF = final_DF.loc[:,~final_DF.columns.duplicated()]
        assert "features_x" not in final_DF.columns 
        assert "features_y" not in final_DF.columns 
        return final_DF

    def compile(self) -> Transformer:
        """
            Returns a new transformer that fuses feature unions where possible.
        """
        out : deque = deque()
        inp = deque([t.compile() for t in self._transformers])
        while inp:
            right = inp.popleft()
            if out and isinstance(out[-1], SupportsFuseFeatureUnion) and (fused := out[-1].fuse_feature_union(right, is_left=True)) is not None:
                out.pop()
                inp.appendleft(fused)
            elif out and isinstance(right, SupportsFuseFeatureUnion) and (fused := right.fuse_feature_union(out[-1], is_left=False)) is not None:
                out.pop()
                inp.appendleft(fused)
            else:
                out.append(right)
        if len(out) == 1:
            return out[0]
        return FeatureUnion(*out)

    def __repr__(self):
        return '(' + ' ** '.join([str(t) for t in self._transformers]) + ')'

    def __eq__(self, other):
        if not isinstance(other, FeatureUnion):
            return NotImplemented
        return self._transformers == other._transformers

    def __hash__(self):
        return hash(('FeatureUnion', self._transformers))

class Compose(NAryTransformerBase):
    """ 
        This class allows pipeline components to be chained together using the "then" operator.

        :Example:

        >>> comp = ComposedPipeline([ DPH_br, ApplyGenericTransformer(lambda res : res[res["rank"] < 2])])
        >>> # OR
        >>> # we can even use lambdas as transformers
        >>> comp = ComposedPipeline([DPH_br, lambda res : res[res["rank"] < 2]])
        >>> #this is equivelent
        >>> #comp = DPH_br >> lambda res : res[res["rank"] < 2]]
    """
    name = "Compose"

    def index(self, iter : pt.model.IterDict, batch_size=None):
        """
        This methods implements indexing pipelines. It is responsible for calling the transform_iter() method of its 
        constituent transformers (except the last one) on batches of records, and the index() method on the last transformer.
        """
        from more_itertools import chunked
        
        prev_transformer = Compose(*self._transformers[0:-1])
        last_transformer = self._transformers[-1]

        # guess a good batch size from the batch_size of individual components earlier in the pipeline
        if batch_size is None:
            batch_size = 100 # default to 100 as a reasonable minimum (and fallback if no batch sizes found)
            for tr in prev_transformer:
                if hasattr(tr, 'batch_size') and isinstance(tr.batch_size, int) and tr.batch_size > batch_size:
                    batch_size = tr.batch_size

        def gen():
            for batch in chunked(iter, batch_size):
                yield from prev_transformer.transform_iter(batch)
        return last_transformer.index(gen()) 

    def transform_iter(self, inp: pt.model.IterDict) -> pt.model.IterDict:
        out = inp
        for transformer in self._transformers:
            out = transformer.transform_iter(out)
        return out
    
    def transform(self, inp : pd.DataFrame) -> pd.DataFrame:
        out = inp
        for m in self._transformers:
            out = m.transform(out)
        return out

    def fit(self, topics_or_res_tr, qrels_tr, topics_or_res_va=None, qrels_va=None):
        """
        This is a default implementation for fitting a pipeline. The assumption is that
        all EstimatorBase be composed with a TransformerBase. It will execute any pre-requisite
        transformers BEFORE executing the fitting the stage.
        """
        for m in self._transformers:
            if isinstance(m, Estimator):
                m.fit(topics_or_res_tr, qrels_tr, topics_or_res_va, qrels_va)
            else:
                topics_or_res_tr = m.transform(topics_or_res_tr)
                # validation is optional for some learners
                if topics_or_res_va is not None:
                    topics_or_res_va = m.transform(topics_or_res_va)

    def __repr__(self):
        return '(' + ' >> '.join([str(t) for t in self._transformers]) + ')'

    def __eq__(self, other):
        if not isinstance(other, Compose):
            return NotImplemented
        return self._transformers == other._transformers

    def __hash__(self):
        return hash(('Compose', self._transformers))

    def compile(self, verbose: bool = False) -> Transformer:
        """Returns a new transformer that iteratively fuses adjacent transformers to form a more efficient pipeline."""
        # compile constituent transformers (flatten allows compile() to return Compose pipelines)
        inp = deque(_flatten((t.compile() for t in self._transformers), Compose))
        out : deque = deque()
        counter = 1
        while inp:
            if verbose:
                print(counter, list(out), list(inp))
            counter +=1 
            right = inp.popleft()
            if out and isinstance(out[-1], SupportsFuseRight) and (fused := out[-1].fuse_right(right)) is not None:
                if verbose:
                    print(f"  fuse_right {out[-1]} >> {right}  == {fused}")
                out.pop()
                # add the fused pipeline to the start of the input queue so it will be processed next (must be done in reverse due to how extendleft works)
                inp.extendleft(reversed(_flatten([fused], Compose)))
            elif out and isinstance(right, SupportsFuseLeft) and (fused := right.fuse_left(out[-1])) is not None:
                if verbose:
                    print(f"  fuse_left {out[-1]} >> {right}  == {fused}")
                out.pop()
                # add the fused pipeline to the start of the input queue so it will be processed next (must be done in reverse due to how extendleft works)
                inp.extendleft(reversed(_flatten([fused], Compose)))
            else:
                out.append(right)
            if counter == MAX_COMPILE_ITER:
                raise OverflowError()
        if len(out) == 1:
            return out[0]
        return Compose(*out)

    def transform_inputs(self):
        # The first transformer in the pipeline may accept multiple input configurations, but not all of these
        # may work for the rest of the pipeline. So find out which (if any) of the input configurations work, and
        # prioritise those.
        io_configurations = [
            {
                'input_columns': input_columns,
                'output_columns': input_columns,
            }
            for input_columns in pt.inspect.transformer_inputs(self._transformers[0])
        ]
        for transformer in self:
            for configuration in io_configurations:
                if configuration['output_columns'] is None:
                    continue
                configuration['output_columns'] = pt.inspect.transformer_outputs(transformer, configuration['output_columns'], strict=False)
        return [io_cfg['input_columns'] for io_cfg in sorted(io_configurations, key=lambda x: x['output_columns'] is None)]

    def transform_outputs(self, input_columns):
        # Figure out the output columns for the given input columns. This is a more direct and robust way of getting the outputs
        # for a composed pipeline than using inspect's default implementation (running an empty dataframe through the whole pipeline)
        # since it can leverage `transform_outputs` implementation of each transformer.
        output_columns = input_columns
        for transformer in self:
            output_columns = pt.inspect.transformer_outputs(transformer, output_columns)
        return output_columns

    def schematic(self, *, input_columns):
        pipeline = []
        columns = input_columns
        for transformer in self:
            schematic = pt.schematic.transformer_schematic(transformer, input_columns=columns)
            pipeline.append(schematic)
            columns = schematic['output_columns']
        return {
            'type': 'pipeline',
            'input_columns': pipeline[0]['input_columns'] if pipeline else None,
            'output_columns': pipeline[-1]['output_columns'] if pipeline else None,
            'title': None,
            'transformers': pipeline,
        }


MAX_COMPILE_ITER = 10_000
