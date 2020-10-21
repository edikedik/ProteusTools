import typing as t
from collections import namedtuple
from functools import partial
from itertools import chain, combinations, count, starmap

import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

from protmc.pipeline import Pipeline, Summary


class ParamSearch:
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


class AffinitySearch:
    def __init__(
            self, positions,
            apo_setup,
            holo_setup,
            exe_path, base_dir):
        self.positions = positions
        self.apo_setup = apo_setup
        self.holo_setup = holo_setup
        self.exe_path = exe_path
        self.base_dir = base_dir


class AffinityWorker:
    def __init__(
            self, apo_setup, holo_setup, base_post_cfg, active_pos, mut_space_size, exe_path, run_dir,
            adapt_dir_name: str = 'ADAPT', mc_dir_name: str = 'MC'):
        self.apo_adapt_cfg, self.apo_mc_cfg, self.apo_matrix_path = apo_setup
        self.holo_adapt_cfg, self.holo_mc_cfg, self.holo_matrix_path = holo_setup
        self.base_post_cfg = base_post_cfg
        self.active_pos = active_pos
        self.mut_space_size = mut_space_size
        self.exe_path = exe_path
        self.run_dir = run_dir
        self.adapt_dir_name, self.mc_dir_name = adapt_dir_name, mc_dir_name
        self.apo_adapt_pipe, self.holo_adapt_pipe = None, None
        self.apo_mc_pipe, self.holo_mc_pipe = None, None
        self.apo_adapt_summary, self.holo_adapt_summary = None, None
        self.apo_mc_summary, self.holo_mc_summary = None, None
        self.ran_setup = False

    def setup(self) -> None:
        def create_pipe(cfg, base_dir, exp_dir, energy_dir):
            return Pipeline(
                base_post_conf=self.base_post_cfg, exe_path=self.exe_path,
                mut_space_size=self.mut_space_size, active_pos=self.active_pos,
                base_mc_conf=cfg, base_dir=base_dir, exp_dir_name=exp_dir, energy_dir=energy_dir)
        base_apo, base_holo = f'{self.run_dir}/apo', f'{self.run_dir}/holo'

        self.apo_adapt_pipe = create_pipe(
            cfg=self.apo_adapt_cfg, base_dir=base_apo, exp_dir=self.adapt_dir_name, energy_dir=self.apo_matrix_path)
        self.holo_adapt_pipe = create_pipe(
            cfg=self.holo_adapt_cfg, base_dir=base_holo, exp_dir=self.adapt_dir_name, energy_dir=self.holo_matrix_path)
        self.apo_mc_pipe = create_pipe(
            cfg=self.apo_mc_cfg, base_dir=base_apo, exp_dir=self.mc_dir_name, energy_dir=self.apo_matrix_path)
        self.holo_mc_pipe = create_pipe(
            cfg=self.holo_mc_cfg, base_dir=base_holo, exp_dir=self.mc_dir_name, energy_dir=self.holo_matrix_path)

        self.apo_adapt_pipe.setup()
        self.holo_adapt_pipe.setup()
        self.apo_mc_pipe.setup()
        self.holo_mc_pipe.setup()

        self.ran_setup = True

    def run(self, parallel: bool = True) -> None:
        if not self.ran_setup:
            self.setup()
        if parallel:
            with Pool(2) as workers:
                self.apo_adapt_summary, self.apo_mc_summary = workers.map(
                    lambda p: p.run(), [self.apo_adapt_pipe, self.holo_adapt_pipe])
            with Pool(2) as workers:
                self.apo_mc_summary, self.holo_mc_summary = workers.map(
                    lambda p: p.run(), [self.apo_mc_pipe, self.holo_mc_pipe])
        else:
            self.apo_adapt_summary, self.apo_mc_summary, self.holo_adapt_summary, self.holo_mc_summary = map(
                lambda p: p.run(), [self.apo_adapt_pipe, self.apo_mc_pipe, self.holo_adapt_pipe, self.holo_mc_pipe])


if __name__ == '__main__':
    raise RuntimeError
