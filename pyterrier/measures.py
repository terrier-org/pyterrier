import ir_measures
for k, v in ir_measures.measures.registry.items():
    v.__doc__ = v.__doc__
    v.__module__ = __name__
    #v.__name__ = k
    globals()[k] = v

