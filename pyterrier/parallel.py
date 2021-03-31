from .transformer import TransformerBase
import pandas as pd
import pyterrier as pt

SUPPORTED_BACKENDS=["joblib", "ray"]

class PoolParallelTransformer(TransformerBase):

    def __init__(self, parent, n_jobs, backend='joblib', **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.n_jobs = n_jobs
        self.backend = backend
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError("Backend of %s unknown, only %s supported." % str(SUPPORTED_BACKENDS))

    def _transform_joblib(self, splits):
        def with_initializer(p, f_init):
            # Overwrite initializer hook in the Loky ProcessPoolExecutor
            # https://github.com/tomMoral/loky/blob/f4739e123acb711781e46581d5ed31ed8201c7a9/loky/process_executor.py#L850
            hasattr(p._backend, '_workers') or p.__enter__()
            origin_init = p._backend._workers._initializer
            def new_init():
                origin_init()
                f_init()
            p._backend._workers._initializer = new_init if callable(origin_init) else f_init
            return p
       
        from joblib import Parallel, delayed
        with Parallel(n_jobs=self.n_jobs) as parallel:
            results = with_initializer(parallel, lambda: pt.init(**pt.init_args))(delayed(self.parent)(topics) for topics in splits)
            return pd.concat(results)
        
    def _transform_ray(self, splits):
        from ray.util.multiprocessing import Pool
        with Pool(self.n_jobs, lambda: pt.init(**pt.init_args)) as pool:
            results = pool.map(lambda topics : self.parent(topics), splits)
            return pd.concat(results)

    def transform(self, topics_and_res):
        from .model import split_df
        splits = split_df(topics_and_res, self.n_jobs)
        
        rtr = None
        if self.backend == 'joblib':
            rtr =  self._transform_joblib(splits)
        if self.backend == 'ray':
            rtr = self._transform_ray(splits)
        return rtr