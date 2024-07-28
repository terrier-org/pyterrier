from . import Transformer
import pandas as pd
import pyterrier as pt

SUPPORTED_BACKENDS=["joblib", "ray"]

#see https://stackoverflow.com/a/55566003
def _joblib_with_initializer(p, _f_init, args=None):
    # Overwrite initializer hook in the Loky ProcessPoolExecutor
    # https://github.com/tomMoral/loky/blob/f4739e123acb711781e46581d5ed31ed8201c7a9/loky/process_executor.py#L850
    hasattr(p._backend, '_workers') or p.__enter__()
    if hasattr(p, 'with_initializer'):
        return p
    origin_init = p._backend._workers._initializer
    if args is None:
        f_init = _f_init
    else:
        def f_init():
            _f_init(*args)
    def new_init():
        origin_init()
        f_init()
    p.with_initializer = True
    p._backend._workers._initializer = new_init if callable(origin_init) else f_init
    return p

def _pt_init(args):
    import pyterrier as pt
    if not pt.started():
        pt.init(no_download=True, **args)
    else:
        from warnings import warn
        warn("Avoiding reinit of PyTerrier")


def _check_ray():
    try:
        import ray
    except:
        raise NotImplementedError("ray is not installed. Run pip install ray")
    if not ray.is_initialized():
        raise ValueError("ray needs to be initialised. Run ray.init() first")


def parallel_lambda(function, inputs, jobs, backend='joblib'):
    if pt.java.bootstrap.is_windows():
        raise ValueError("No support for parallelisation on Windows")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError("Backend of %s unknown, only %s supported." % str(SUPPORTED_BACKENDS))
    if backend == 'ray':
        return _parallel_lambda_ray(function, inputs, jobs)
    if backend == 'joblib':
        return _parallel_lambda_joblib(function, inputs, jobs)

def _parallel_lambda_ray(function, inputs, jobs):
    from ray.util.multiprocessing import Pool
    with Pool(jobs, pt.java.parallel_init, pt.java.parallel_init_args()) as pool:
        return pool.map(function, inputs)

def _parallel_lambda_joblib(function, inputs, jobs):
    from joblib import Parallel, delayed
    with Parallel(n_jobs=jobs) as parallel:
        parallel_mp = _joblib_with_initializer(parallel, pt.java.parallel_init, pt.java.parallel_init_args())
        return parallel_mp(
            delayed(function)(input) for input in inputs)
        

class PoolParallelTransformer(Transformer):

    def __init__(self, parent, n_jobs, backend='joblib', **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.n_jobs = n_jobs
        self.backend = backend
        if pt.java.bootstrap.is_windows():
            raise ValueError("No support for parallelisation on Windows")
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError("Backend of %s unknown, only %s supported." % str(SUPPORTED_BACKENDS))
        if self.backend == 'ray':
            _check_ray()

    def _transform_joblib(self, splits):
        from joblib import Parallel, delayed
        with Parallel(n_jobs=self.n_jobs) as parallel:
            results = _joblib_with_initializer(parallel, pt.java.parallel_init, pt.java.parallel_init_args())(delayed(self.parent)(topics) for topics in splits)
            return pd.concat(results)
        
    def _transform_ray(self, splits):
        from ray.util.multiprocessing import Pool
        with Pool(self.n_jobs, pt.java.parallel_init, pt.java.parallel_init_args()) as pool:
            results = pool.map(lambda topics : self.parent(topics), splits)
            return pd.concat(results)

    def transform(self, topics_and_res):
        splits = pt.model.split_df(topics_and_res, self.n_jobs)
        
        rtr = None
        if self.backend == 'joblib':
            rtr =  self._transform_joblib(splits)
        elif self.backend == 'ray':
            rtr = self._transform_ray(splits)
        else:
            raise RuntimeError(f'Unsupported backend: {self.backend}')
        return rtr

    def __repr__(self):
        return "PoolParallelTransformer("+self.parent.__repr__()+")"

    def __str__(self):
        return "PoolParallelTransformer("+str(self.parent)+")"
