import typing as t
from copy import deepcopy
from glob import glob
from itertools import chain, filterfalse
from os import remove
from pathlib import Path

import pandas as pd
from multiprocess.pool import Pool

from protmc import config
from protmc.post import analyze_seq_no_rich, compose_summary, Summary
from protmc.runner import Runner
from protmc.utils import get_bias_state, infer_mut_space

PipelineOutput = t.NamedTuple('PipelineOutput', [('results', pd.DataFrame), ('summary', Summary)])


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
            self, base_mc_conf: config.ProtMCconfig, base_post_conf: t.Optional[config.ProtMCconfig],
            exe_path: str, base_dir: str, exp_dir_name: str, energy_dir: str,
            active_pos: t.Sequence[int], mut_space_n_types: int = 18):
        """
        :param base_mc_conf: base config for the MC/ADAPT mode
        :param base_post_conf: base config for the POSTPROCESS mode
        :param exe_path: a path to a protMC executable
        :param base_dir: a name of a base directory for the experiment
        :param exp_dir_name: a name of the experiment
        :param energy_dir: a path to an energy directory with matrices
        :param active_pos: active positions (allowed to mutate)
        :param mut_space_n_types: the number of available types in the mutation space.
        :return:
        """
        self.base_mc_conf: config.ProtMCconfig = base_mc_conf
        self.base_post_conf: config.ProtMCconfig = base_post_conf
        self.exe_path: str = exe_path
        self.base_dir: str = base_dir
        self.exp_dir_name: str = exp_dir_name
        self.energy_dir: str = energy_dir
        self.mut_space_n_types = mut_space_n_types
        self.active_pos = self._infer_active(active_pos, base_mc_conf)
        self.mut_space_size = self._infer_mut_space(mut_space_n_types, base_mc_conf)
        self.mc_conf: t.Optional[config.ProtMCconfig] = None
        self.post_conf: t.Optional[config.ProtMCconfig] = None
        self.exp_dir: t.Optional[str] = None
        self.mc_runner: t.Optional[Runner] = None
        self.post_runner: t.Optional[Runner] = None
        self.results: t.Optional[PipelineOutput] = None
        self.ran_setup: bool = False
        self.default_results_dump_name: str = 'RESULTS.tsv'

    def copy(self):
        return deepcopy(self)

    def _change_config(self, changes: t.Iterable[t.Tuple[str, t.Any]], conf_type: str = 'MC') -> None:
        """
        Implement changes in either `mc_conf` or `post_conf` attributes.
        Handling of changes lies on the `change_field` method of the `ProtMCConfig`
        :param changes: a list of changes [(field_name, field_value), ...]
        :param conf_type: a type of the config (`MC` or `POST`)
        :return:
        """
        conf = self.mc_conf if conf_type == 'MC' else self.post_conf
        for f_name, f_value in changes:
            conf.change_field(f_name, f_value)

    @staticmethod
    def _infer_active(active: t.Optional, conf: config.ProtMCconfig):
        """
        Attempt to infer active positions (from Adapt_Space, if exists)
        :return:
        """
        if active is None:
            space = conf.get_field_value('Adapt_Space')
            if space is None:
                return
            else:
                return [int(x) for x in set(
                    chain.from_iterable(x.split('-') for x in space))]
        return active

    def _infer_mut_space(self, n_types, conf: config.ProtMCconfig):
        existing_constraints = conf.get_field_value('Space_Constraints')
        return infer_mut_space(
            n_types, len(self.active_pos),
            [existing_constraints] if isinstance(existing_constraints, str) else existing_constraints)

    def setup(self, mc_config_changes: t.Optional[t.Iterable[t.Tuple[str, t.Any]]] = None, continuation: bool = False):
        """
        Prepare for the run: handle config's IO, implement changes if necessary,
        dump configs into the experiment directory (method will create in case it doesn't exist).
        :param mc_config_changes: an iterable of field_name - field_values pairs,
        constituting changes applied to the `mc_conf` attribute.
        :param continuation: indicates whether the setup should be initialized from the existing setup.
        This is mainly for internal use (by `continue_run` method).
        :return:
        """
        # setup working directory and configs
        self.exp_dir = setup_exp_dir(self.base_dir, self.exp_dir_name)
        self.mc_conf = setup_mc_io(
            self.mc_conf if continuation else self.base_mc_conf,
            self.exp_dir, self.energy_dir)

        # modify base config values
        if mc_config_changes:
            self._change_config(changes=mc_config_changes)
        if self.base_post_conf:
            self.post_conf = setup_post_io(self.base_post_conf, self.mc_conf, self.exp_dir)

        # init runners (this will also dump config files)
        self.mc_runner = Runner(run_dir=self.exp_dir, exe_path=self.exe_path, config=self.mc_conf)
        if self.base_post_conf or self.post_conf:
            self.post_runner = Runner(run_dir=self.exp_dir, exe_path=self.exe_path, config=self.post_conf)

        # re-infer active positions and mutation space size from the new config
        self.active_pos = self._infer_active(self.active_pos, self.mc_conf)
        self.mut_space_size = self._infer_mut_space(self.mut_space_n_types, self.mc_conf)

        # flip the flag on
        self.ran_setup = True

    def run(self, dump_results: bool = True, dump_name: t.Optional[str] = None,
            parallel: bool = True) -> PipelineOutput:
        """
        Run the pipeline.
        If ran prior to `setup` method, the latter will be called first (with the default arguments).
        :param dump_results: flag whether to dump the results DataFrame
        :param dump_name:
        :param parallel: if number of walkers exceeds 1, aggregate the results in parallel
        :return:
        """
        if not self.ran_setup:
            self.setup()
        if not dump_name:
            dump_name = self.default_results_dump_name

        # Run the calculations
        self.mc_runner.run()
        if self.base_post_conf or self.post_conf:
            self.post_runner.run()

        # Infer the max number of walkers
        n_walkers = [1, self.mc_conf.get_field_value('Replica_Number'),
                     self.mc_conf.get_field_value('Trajectory_Number')]
        n_walkers = max(map(int, filterfalse(lambda x: x is None, n_walkers)))

        # aggregate the results
        if parallel and n_walkers > 1:
            with Pool(n_walkers) as pool:
                results = pool.map(
                    lambda n: self._agg_walker(self.active_pos, dump_results, dump_name, n),
                    range(n_walkers))
        elif not parallel and n_walkers > 1:
            results = [self._agg_walker(self.active_pos, dump_results, dump_name, n) for n in range(n_walkers)]
        else:
            results = [self._agg_walker(self.active_pos, dump_results, dump_name)]

        # Either add the results to existing ones or store as new
        if self.results is not None and len(results) == len(self.results):
            dfs = [pd.concat([r_old.results, r_new]) for r_old, r_new in zip(self.results, results)]
        else:
            dfs = results

        # Concatenate results
        summaries = pd.DataFrame(compose_summary(x, self.mut_space_size) for x in dfs)
        summaries['walker'] = list(range(n_walkers))
        dfs = pd.concat(dfs)

        # Return the results
        return PipelineOutput(dfs, summaries)

    def _agg_walker(
            self, active, dump_results: bool = True, dump_name: t.Optional[str] = None,
            n_walker: t.Optional[int] = None) -> pd.DataFrame:
        """
        The function exists to aggregate a single walker. It basically handles the suffixes [_0, _1, ...]
            protMC assigns depending on the number of trajectories (or replicas).
        :param active: active positions
        :param dump_results: dump results or not
        :param dump_name: file name to dump results
        :param n_walker: walker number -- can be either empty string or "0", "1" and so on.
        :return: DataFrame of aggregation results
        """
        # aggregate the results
        n_walker = '' if n_walker is None else f'_{n_walker}'
        df = analyze_seq_no_rich(
            seq_path=self.mc_conf.get_field_value('Seq_Output_File') + n_walker,
            matrix_bb=f'{self.energy_dir}/matrix.bb', active=active,
            steps=int(self.mc_conf.get_field_value('Trajectory_Length')))
        df['exp'] = self.exp_dir_name
        df['walker'] = n_walker

        # optionally dump the results into the experiment directory
        if dump_results:
            df.to_csv(f'{self.base_dir}/{self.exp_dir_name}/{dump_name}' + n_walker, sep='\t', index=False)

        return df

    def continue_run(
            self, new_exp_name: str, bias_step: int, n_walker: str = '',
            dump_results: bool = True, parallel: bool = False, dump_name: t.Optional[str] = None,
            mc_config_changes: t.Optional[t.List[t.Tuple[str, t.Any]]] = None) -> Summary:
        """
        Continue the run using previously developed bias.
        The results of the previous and the current run will be concatenated,
        and the summary will be computed on this concatenation.

        :param new_exp_name: the name of the new experiment (must be different from the previous experiment name)
        :param bias_step: the step to continue from
        :param n_walker: the walker number to get the bias from
        (in case of continuing multi-replica or multi-trajectory run)
        :param parallel: call `run` method in parallel mode
        :param dump_results: dump results of the current (!) run in the experiment directory
        :param dump_name: name of the dump file (passed to the `run` method)
        :param mc_config_changes: changes passed to the `run` method
        :return: Summary namedtuple with basic computed over the results
        """
        self.exp_dir_name = new_exp_name
        bias_path = self.mc_conf.get_field_value('Adapt_Output_File') + n_walker
        bias = get_bias_state(bias_path, bias_step)
        if not bias:
            raise ValueError(f'Could not find the step {bias_step} in the {bias_path}')
        mode = self.mc_conf.mode.field_values[0]

        Path(f'{self.base_dir}/{self.exp_dir_name}').mkdir(exist_ok=True, parents=True)
        bias_path = f'{self.base_dir}/{self.exp_dir_name}/{mode}.inp.dat'
        with open(bias_path, 'w') as f:
            print(bias, file=f)

        self.mc_conf['MC_IO']['Bias_Input_File'] = config.ProtMCfield(
            field_name='Bias_Input_File',
            field_values=bias_path,
            comment='Path to a file with existing bias potential'
        )
        self.setup(mc_config_changes=mc_config_changes, continuation=True)
        return self.run(dump_results=dump_results, dump_name=dump_name, parallel=parallel)

    def cleanup(self, leave_ext: t.Tuple[str, ...] = ('dat', 'conf', 'tsv'), leave_names: t.Tuple[str, ...] = ()):
        files = glob(f'{self.exp_dir}/*')
        if not files:
            raise ValueError(f'No files are found in experiment directory {self.exp_dir}')
        for p in files:
            in_leave = any(n in p.split('/')[-1] for n in leave_names)
            ext_valid = any(n in p.split('.')[-1] for n in leave_ext)
            if not (in_leave or ext_valid):
                remove(p)


if __name__ == '__main__':
    raise RuntimeError
