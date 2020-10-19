import typing as t
from collections import namedtuple
from itertools import chain, combinations, count, starmap

import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

from protmc.pipeline import Pipeline, Summary


class Search:
    def __init__(self, pipeline: Pipeline, grid: t.Dict[str, t.List[t.Any]]):
        self.pipeline = pipeline
        self.grid = grid
        self.results = None

    def _grid_generator(self):
        params = chain.from_iterable(
            ((k, v) for v in values) for k, values in self.grid.items())
        comb = filter(
            lambda c: len(set(x[0] for x in c)) == len(self.grid),
            combinations(params, r=len(self.grid)))
        yield from comb

    def grid_search(self, dump_results: bool = False, verbose: bool = True) -> pd.DataFrame:
        SearchSummary = namedtuple('SearchSummary', list(self.grid) + list(Summary._fields))
        param_comb = self._grid_generator()
        if verbose:
            param_comb = tqdm(param_comb, desc='Grid_Search ')
        self.results = []
        for i, comb in enumerate(param_comb):
            self.pipeline.setup(mc_config_changes=list(comb))
            summary = self.pipeline.run(
                dump_results=dump_results,
                dump_name=f'SUMMARY_grid{i}.tsv')
            self.results.append(SearchSummary(
                *(val for param, val in comb),
                *summary))
        return pd.DataFrame(self.results)

    def grid_search_parallel(self, verbose: bool = True, n: int = 2) -> pd.DataFrame:
        SearchSummary = namedtuple('SearchSummary', list(self.grid) + list(Summary._fields))
        param_comb = self._grid_generator()
        if verbose:
            param_comb = tqdm(param_comb, desc='Grid_Search:>')
        pipes = (self.pipeline.copy() for _ in count())
        if n > 1:
            with Pool(n) as workers:
                results = workers.starmap(self._run_pipe, zip(pipes, param_comb))
        else:
            results = starmap(self._run_pipe, zip(pipes, param_comb))
        self.results = [
            SearchSummary(
                *(val for _, val in comb), *s)
            for comb, s in zip(self._grid_generator(), results)
        ]
        return pd.DataFrame(self.results)

    @staticmethod
    def _run_pipe(pipe: Pipeline, param_changes):
        dirname = "_".join([f'{x[0]}-{x[1]}' for x in param_changes])
        pipe.base_dir = f'{pipe.base_dir}/{pipe.exp_dir_name}'
        pipe.exp_dir_name = dirname
        pipe.setup(mc_config_changes=list(param_changes))
        summary = pipe.run()
        return summary


if __name__ == '__main__':
    raise RuntimeError
