import typing as t
from collections import namedtuple
from copy import deepcopy
from itertools import chain, combinations, count, starmap
from pathlib import Path
from multiprocess.pool import Pool

import pandas as pd
from tqdm import tqdm

from protmc import config
from protmc.post import analyze
from protmc.runner import Runner

Summary = namedtuple('summary', [
    'num_unique', 'coverage',
    'seq_prob_mean', 'seq_prob_std', 'seq_prob_rss'])


def setup_exp_dir(base_dir: str, name: str) -> str:
    """
    Create an experiment directory and return its path
    :param base_dir: a path to a base directory holding experiment
    :param name: experiment name
    :return: path to an experiment directory
    """
    path = f'{base_dir}/{name}'
    Path(path).mkdir(exist_ok=True, parents=True)
    return path


def setup_mc_io(cfg: config.ProtMCconfig, exp_dir: str, energy_dir: str):
    """
    Setup MC/ADAPT config file IO parameters
    :param cfg: a config object
    :param exp_dir: a path to the experiment dir
    :param energy_dir: a path to the matrix directory
    :return: a copy of a config with IO paths within the `exp_dir`
    """
    cfg = cfg.copy()
    mode = cfg.mode.field_values[0]
    if mode == 'ADAPT':
        cfg.change_field('Adapt_Output_File', f'{exp_dir}/{mode}.dat')
    cfg.change_field('Energy_Directory', energy_dir)
    cfg.change_field('Seq_Output_File', f'{exp_dir}/{mode}.seq')
    cfg.change_field('Ener_Output_File', f'{exp_dir}/{mode}.ener')
    return cfg


def setup_post_io(cfg: config.ProtMCconfig, cfg_mc: config.ProtMCconfig, exp_dir: str):
    """
    Setup POSTPROCESS config IO parameters.
    Depends on MC/ADAPT config to infer
    :param cfg: config object for POSTPROCESS mode
    :param cfg_mc: config object for MC/ADAPT mode
    :param exp_dir: a path to a directory of the experiment
    :return:
    """
    cfg = cfg.copy()
    cfg.change_field(
        'Seq_Input_File',
        cfg_mc.get_field_value('Seq_Output_File'))
    cfg.change_field(
        'Energy_Directory',
        cfg_mc.get_field_value('Energy_Directory'))
    cfg.change_field('Fasta_File', f'{exp_dir}/POSTPROCESS.rich')
    return cfg


class Pipeline:
    """
    Basic protMC pipeline (ADAPT/MC -> POSTPROCESS -> analyze -> summary).
    Analysis results include (1) a summary of the run, and (2) an aggregated DataFrame with unique sequences,
        a number of times each sequence was sampled, etc.
    """

    def __init__(
            self, base_mc_conf: config.ProtMCconfig, base_post_conf: config.ProtMCconfig,
            exe_path: str, base_dir: str, exp_dir_name: str, energy_dir: str,
            active_pos: t.Optional[t.Iterable[int]] = None):
        """
            :param base_mc_conf: base config for the MC/ADAPT mode
            :param base_post_conf: base config for the POSTPROCESS mode
            :param exe_path: a path to a protMC executable
            :param base_dir: a name of a base directory for the experiment
            :param exp_dir_name: a name of the experiment
            :param energy_dir: a path to an energy directory with matrices
            :param active_pos: active positions (allowed to mutate)
            :return:
            """
        self.base_mc_conf = base_mc_conf
        self.base_post_conf = base_post_conf
        self.exe_path = exe_path
        self.base_dir = base_dir
        self.exp_dir_name = exp_dir_name
        self.energy_dir = energy_dir
        self.active_pos = active_pos
        self.mc_conf, self.post_conf, self.exp_dir = None, None, None
        self.mc_runner, self.post_runner = None, None
        self.results, self.summary = None, None
        self.ran_setup = False

    def copy(self):
        return deepcopy(self)

    def _change_config(self, changes: t.List[t.Tuple[str, t.Any]], conf_type: str = 'MC'):
        conf = self.mc_conf if conf_type else self.post_conf
        for f_name, f_value in changes:
            conf.change_field(f_name, f_value)

    def setup(self, mc_config_changes: t.Optional[t.List[t.Tuple[str, t.Any]]] = None):
        # setup working directory and configs
        self.exp_dir = setup_exp_dir(self.base_dir, self.exp_dir_name)
        mc_conf = setup_mc_io(self.base_mc_conf, self.exp_dir, self.energy_dir)
        # modify base config values
        if mc_config_changes:
            self._change_config(changes=mc_config_changes)
        self.mc_conf = mc_conf
        self.post_conf = setup_post_io(self.base_post_conf, self.mc_conf, self.exp_dir)
        self.ran_setup = True

    def run(self, dump_results: bool = True, dump_name: str = 'SUMMARY.tsv') -> Summary:
        if not self.ran_setup:
            self.setup()

        self.mc_runner = Runner(run_dir=self.exp_dir, exe_path=self.exe_path, config=self.mc_conf)
        self.post_runner = Runner(run_dir=self.exp_dir, exe_path=self.exe_path, config=self.post_conf)

        self.mc_runner.run()
        self.post_runner.run()

        # attempt to infer active positions
        if self.active_pos is None:
            space = self.base_mc_conf.get_field_value('Adapt_Space')
            if space is None:
                active = None
            else:
                active = [int(x) for x in set(
                    chain.from_iterable(x.split('-') for x in space))]
        else:
            active = self.active_pos

        # aggregate the results
        df = analyze(
            seq=self.post_conf.get_field_value('Seq_Input_File'),
            rich=self.post_conf.get_field_value('Fasta_File'),
            steps=int(self.mc_conf.get_field_value('Trajectory_Length')),
            active=active)
        df['exp'] = self.exp_dir_name

        # optionally dump the results into the experiment directory
        if dump_results:
            df.to_csv(f'{self.base_dir}/{self.exp_dir_name}/{dump_name}', sep='\t', index=False)

        # store the results internally
        self.results = df

        # compose the summary
        self.summary = Summary(
            num_unique=len(df),
            coverage=len(df) / 26 ** len(active) if active else len(df['seq'][0]),
            seq_prob_mean=df['seq_prob'].mean(),
            seq_prob_std=df['seq_prob'].std(),
            seq_prob_rss=((df['seq_prob'] - 1 / len(df)) ** 2).sum()
        )
        return self.summary


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

    def grid_search(self, dump_results: bool = False, verbose: bool = True):
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

    def grid_search_parallel(self, verbose: bool = True, n: int = 2):
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
