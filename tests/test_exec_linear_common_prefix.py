import pyterrier as pt
import pyterrier._evaluation._exec_linear


class EqOnlyTransformer(pt.Transformer):
    def transform(self, inp):
        return inp

    def __eq__(self, other):
        return isinstance(other, EqOnlyTransformer)


def test_identify_common_unhashable_transformer():
    pipe_a = EqOnlyTransformer()
    pipe_b = EqOnlyTransformer()
    pipes = [pipe_a, pipe_b]
    common, suffices = pyterrier._evaluation._exec_linear._identifyCommon(pipes)
    assert common is None
    assert suffices == pipes
