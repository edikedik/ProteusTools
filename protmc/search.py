import typing as t
from collections import namedtuple
from itertools import chain, combinations, count, starmap
from pathlib import Path

import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

from protmc.pipeline import Pipeline, Summary
from protmc.affinity import affinity
from protmc.utils import get_reference, get_bias_state
from protmc import config


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
            count_threshold: int = 10, adapt_dir_name: str = 'ADAPT', mc_dir_name: str = 'MC'):
        self.apo_adapt_cfg, self.apo_mc_cfg, self.apo_matrix_path = apo_setup
        self.holo_adapt_cfg, self.holo_mc_cfg, self.holo_matrix_path = holo_setup
        self.base_post_cfg = base_post_cfg
        self.active_pos = active_pos
        self.mut_space_size = mut_space_size
        self.exe_path = exe_path
        self.run_dir = run_dir
        self.count_threshold = count_threshold
        self.adapt_dir_name, self.mc_dir_name = adapt_dir_name, mc_dir_name
        self.apo_adapt_pipe, self.holo_adapt_pipe = None, None
        self.apo_mc_pipe, self.holo_mc_pipe = None, None
        self.apo_adapt_summary, self.holo_adapt_summary = None, None
        self.apo_mc_summary, self.holo_mc_summary = None, None
        self.ran_setup, self.ran_pipes = False, False
        self.temperature = None
        self.aff = None

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
        self.temperature = self._get_temperature()
        self._copy_configs()

        self.ran_setup = True

    def run(self, parallel: bool = True) -> None:
        if not self.ran_setup:
            self.setup()
        if parallel:
            # TODO: CAREFUL! -- results and summary are not stored internally in any of the pipes
            with Pool(2) as adapt_workers:
                self.apo_adapt_summary, self.apo_mc_summary = adapt_workers.map(
                    lambda p: p.run(), [self.apo_adapt_pipe, self.holo_adapt_pipe])
            self._transfer_biases()
            with Pool(2) as mc_workers:
                self.apo_mc_summary, self.holo_mc_summary = mc_workers.map(
                    lambda p: p.run(), [self.apo_mc_pipe, self.holo_mc_pipe])
        else:
            self.apo_adapt_summary, self.apo_mc_summary = map(
                lambda p: p.run(), [self.apo_adapt_pipe, self.holo_adapt_pipe])
            self._transfer_biases()
            self.apo_mc_summary, self.holo_mc_summary = map(
                lambda p: p.run(), [self.apo_mc_pipe, self.holo_mc_pipe])
        self.ran_pipes = True

    def _get_temperature(self):
        # Run only after pipes has been setup
        pipes = [self.apo_adapt_pipe, self.apo_mc_pipe, self.holo_adapt_pipe, self.holo_mc_pipe]
        temps = set(p.mc_conf.get_field_value('Temperature') for p in pipes)
        if len(temps) != 1:
            raise ValueError(f'Every pipe should have the same `Temperature` parameter. Got {temps}')
        return float(temps.pop())

    def _transfer_biases(self):
        # Copy last bias state from ADAPT to MC
        def transfer_bias(adapt_cfg: config.ProtMCconfig, mc_pipe: Pipeline):
            bias_path = adapt_cfg.get_field_value('Adapt_Output_File')
            output_period = adapt_cfg.get_field_value('Adapt_Output_Period')
            n_steps = adapt_cfg.get_field_value('Trajectory_Length')
            last_step = (n_steps // output_period) * output_period
            bias = get_bias_state(bias_path, last_step)
            bias_path_new = f'{mc_pipe.exp_dir}/ADAPT.inp'
            with open(bias_path_new, 'w') as f:
                print(bias.rstrip(), file=f)
            mc_pipe.mc_conf['MC_IO']['Bias_Input_File'] = bias_path_new
            return None

        transfer_bias(self.apo_adapt_pipe.mc_conf, self.apo_mc_pipe)
        transfer_bias(self.holo_adapt_pipe.mc_conf, self.holo_mc_pipe)
        self._copy_configs()

    def _copy_configs(self):
        self.apo_adapt_cfg = self.apo_adapt_pipe.mc_conf.copy()
        self.holo_adapt_cfg = self.holo_adapt_pipe.mc_conf.copy()
        self.apo_mc_cfg = self.apo_mc_pipe.mc_conf.copy()
        self.holo_mc_cfg = self.holo_mc_pipe.mc_conf.copy()

    def affinity(self, position_list: t.Optional[str] = None):
        if not self.ran_pipes:
            self.run()
        active = list(map(str, self.active_pos))
        if not position_list:
            apo_pos = f'{self.apo_matrix_path}/position_list.dat'
            holo_pos = f'{self.holo_matrix_path}/position_list.dat'
            if Path(apo_pos).exists():
                position_list = apo_pos
            elif Path(holo_pos).exists():
                position_list = holo_pos
            else:
                raise RuntimeError('Could not find position_list.dat in matrix directories.')
        ref_seq = get_reference(position_list, active)
        if not ref_seq:
            raise RuntimeError('Reference sequence is empty')

        # Handle biases
        apo_bias = self.apo_mc_cfg.get_field_value('Bias_Input_File')
        holo_bias = self.holo_mc_cfg.get_field_value('Bias_Input_File') or apo_bias

        # Find the populations
        pop_apo, pop_holo = self.apo_mc_pipe.results, self.holo_mc_pipe.results

        if pop_apo is None:
            pop_apo_path = f'{self.apo_mc_pipe.exp_dir}/{self.apo_mc_pipe.default_results_dump_name}'
            if not Path(pop_apo_path).exists():
                raise RuntimeError(f'Could not find the population at {pop_apo_path}')
            pop_apo = pd.read_csv(pop_apo_path, sep='\t')
        if pop_holo is None:
            pop_holo_path = f'{self.holo_mc_pipe.exp_dir}/{self.holo_mc_pipe.default_results_dump_name}'
            if not Path(pop_holo_path).exists():
                raise RuntimeError(f'Could not find the population at {pop_holo_path}')
            pop_holo = pd.read_csv(pop_holo_path, sep='\t')

        # Compute the affinity
        self.aff = affinity(
            reference_seq=ref_seq,
            pop_unbound=pop_apo,
            pop_bound=pop_holo,
            bias_unbound=apo_bias,
            bias_bound=holo_bias,
            temperature=self.temperature,
            threshold=self.count_threshold,
            positions=active)
        return self.aff


if __name__ == '__main__':
    raise RuntimeError
