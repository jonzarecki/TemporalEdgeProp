import itertools
import logging
import multiprocessing
import sys
from typing import Callable, Iterable, List

from pathos.pools import ProcessPool as Pool
from tqdm.autonotebook import tqdm

from typing import List, TypeVar

_S = TypeVar('_S')


def flatten(l: List[List[_S]]) -> List[_S]:
    return list(itertools.chain.from_iterable([(i if isinstance(i, list) else [i]) for i in l]))


i = 0
proc_count = 1
force_serial = False


class LoggerWriter:
    def __init__(self, level=logging.debug):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        self.level(sys.stderr)


def _chunk_spawn_fun(args_list):
    return [_spawn_fun(args) for args in args_list]


def _spawn_fun(args):
    input, func, proc_i, keep_child_tqdm = args
    import random, numpy
    import sys
    old_err = sys.stderr
    if not keep_child_tqdm:
        sys.stderr = LoggerWriter()
    random.seed(1554 + proc_i)
    numpy.random.seed(42 + proc_i)  # set random seeds
    global force_serial
    force_serial = True

    try:
        res = func(input)
        return res
    except:
        import traceback
        traceback.print_exc(file=sys.stdout)
        raise  # re-raise exception
    finally:
        sys.stderr = old_err


def chunk_iterator(itr: Iterable, chunk_size: int) -> Iterable:
    """
    Returns the values of the iterator in chunks of size $chunk_size

    Args:
        itr: The iterator we want to split into chunks
        chunk_size: The chunk size

    Returns:
        A new iterator which returns the values of $iter in chunks
    """
    itr = iter(itr)
    for _ in itertools.count():
        chunk = []
        for _ in range(chunk_size):
            try:
                chunk.append(next(itr))
            except StopIteration:
                break  # finished

        yield chunk

        if len(chunk) < chunk_size:  # no more in iterator
            return


def parmap(f: Callable, X: List[object], nprocs=multiprocessing.cpu_count(), force_parallel=False,
           chunk_size=1, use_tqdm=False, keep_child_tqdm=True, **tqdm_kwargs) -> list:
    """
    Utility function for doing parallel calculations with multiprocessing.
    Splits the parameters into chunks (if wanted) and calls.
    Equivalent to list(map(func, params_iter))
    Args:
        f: The function we want to calculate for each element
        X: The parameters for the function (each element ins a list)
        chunk_size: Optional, the chunk size for the workers to work on
        nprocs: The number of procs to use (defaults for all cores)
        use_tqdm: Whether to use tqdm (default to False)
        tqdm_kwargs: kwargs passed to tqdm

    Returns:
        The list of results after applying func to each element

    Has problems with using self.___ as variables in f (causes self to be pickled)
    """
    if len(X) == 0:
        return []  # like map
    if nprocs != multiprocessing.cpu_count() and len(X) < nprocs * chunk_size:
        chunk_size = 1  # use chunk_size = 1 if there is enough procs for a batch size of 1

    nprocs = int(max(1, min(nprocs, len(X) / chunk_size)))  # at least 1
    if len(X) < nprocs:
        if nprocs != multiprocessing.cpu_count(): print("parmap too much procs")
        nprocs = len(X)  # too much procs

    args = list(zip(X, [f] * len(X), range(len(X)), [keep_child_tqdm] * len(X)))
    if chunk_size > 1:
        args = list(chunk_iterator(args, chunk_size))
        s_fun = _chunk_spawn_fun  # spawn fun
    else:
        s_fun = _spawn_fun  # spawn fun

    if (nprocs == 1 and not force_parallel) or force_serial:  # we want it serial (maybe for profiling)
        return list(map(f, tqdm(X, disable=not use_tqdm, **tqdm_kwargs)))

    try:  # try-catch hides bugs
        global proc_count
        old_proc_count = proc_count
        proc_count = nprocs
        p = Pool(nprocs)
        p.restart(force=True)
        # can throw if current proc is daemon
        if use_tqdm:
            retval_par = tqdm(p.imap(lambda arg: s_fun(arg), args), total=int(len(X)/chunk_size),
                              **tqdm_kwargs)
        else:
            # import  pdb
            # pdb.set_trace()
            retval_par = p.map(lambda arg: s_fun(arg), args)

        retval = list(retval_par)  # make it like the original map
        if chunk_size > 1:
            retval = flatten(retval)


        p.terminate()
        proc_count = old_proc_count
        global i
        i += 1
    except AssertionError as e:
        # if e == "daemonic processes are not allowed to have children":
        retval = list(map(f, tqdm(X, disable=not use_tqdm, **tqdm_kwargs)))  # can't have pool inside pool
    return retval