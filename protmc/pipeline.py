import typing as t
from copy import deepcopy
from glob import glob
from itertools import chain
from os import remove
from pathlib import Path

import pandas as pd

from protmc import config
from protmc.post import analyze_seq_no_rich, compose_summary, Summary
from protmc.runner import Runner
from protmc.utils import get_bias_state


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
            active_pos: t.Optional[t.Iterable[int]] = None, mut_space_size: int = 18):
        """
        :param base_mc_conf: base config for the MC/ADAPT mode
        :param base_post_conf: base config for the POSTPROCESS mode
        :param exe_path: a path to a protMC executable
        :param base_dir: a name of a base directory for the experiment
        :param exp_dir_name: a name of the experiment
        :param energy_dir: a path to an energy directory with matrices
        :param active_pos: active positions (allowed to mutate)
        :param mut_space_size: the number of available types in the mutation space, excluding protonation states.
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
        self.mutation_space_size = mut_space_size

    def copy(self):
        return deepcopy(self)

    def _change_config(self, changes: t.List[t.Tuple[str, t.Any]], conf_type: str = 'MC'):
        conf = self.mc_conf if conf_type == 'MC' else self.post_conf
        for f_name, f_value in changes:
            conf.change_field(f_name, f_value)

    def setup(self, mc_config_changes: t.Optional[t.List[t.Tuple[str, t.Any]]] = None, continuation: bool = False):
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
        self.ran_setup = True

    def run(self, dump_results: bool = True, dump_name: str = 'RESULTS.tsv') -> Summary:
        if not self.ran_setup:
            self.setup()

        self.mc_runner = Runner(run_dir=self.exp_dir, exe_path=self.exe_path, config=self.mc_conf)
        self.mc_runner.run()
        if self.base_post_conf or self.post_conf:
            self.post_runner = Runner(run_dir=self.exp_dir, exe_path=self.exe_path, config=self.post_conf)
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
        df = analyze_seq_no_rich(
            seq_path=self.mc_conf.get_field_value('Seq_Output_File'),
            matrix_bb=f'{self.energy_dir}/matrix.bb', active=active,
            steps=int(self.mc_conf.get_field_value('Trajectory_Length')))
        df['exp'] = self.exp_dir_name

        # optionally dump the results into the experiment directory
        if dump_results:
            df.to_csv(f'{self.base_dir}/{self.exp_dir_name}/{dump_name}', sep='\t', index=False)

        # store the results internally
        self.results = pd.concat([self.results, df]) if isinstance(self.results, pd.DataFrame) else df

        # compose the summary
        return compose_summary(results=self.results, mut_space_size=self.mutation_space_size, num_active=len(active))

    def continue_run(
            self, new_exp_name: str, bias_step: int,
            dump_results: bool = True, dump_name: str = 'SUMMARY.tsv',
            mc_config_changes: t.Optional[t.List[t.Tuple[str, t.Any]]] = None) -> Summary:
        """
        Continue the run using previously developed bias.
        The results of the previous and the current run will be concatenated,
        and the summary will be computed on this concatenation.

        :param new_exp_name: the name of the new experiment (must be different from the previous experiment name)
        :param bias_step: the step to continue from
        :param dump_results: dump results of the current (!) run in the experiment directory
        :param dump_name: name of the dump file (passed to the `run` method)
        :param mc_config_changes: changes passed to the `run` method
        :return: Summary namedtuple with basic computed over the results
        """
        self.exp_dir_name = new_exp_name
        bias_path = self.mc_conf.get_field_value('Adapt_Output_File')
        bias = get_bias_state(bias_path, bias_step)
        if not bias:
            raise ValueError(f'Could not find the step {bias_step} in the {bias_path}')
        mode = self.mc_conf.mode.field_values[0]

        Path(f'{self.base_dir}/{self.exp_dir_name}').mkdir(exist_ok=True, parents=True)
        bias_path = f'{self.base_dir}/{self.exp_dir_name}/{mode}.dat.inp'
        with open(bias_path, 'w') as f:
            print(bias, file=f)

        self.mc_conf['MC_IO']['Bias_Input_File'] = config.ProtMCfield(
            field_name='Bias_Input_File',
            field_values=bias_path,
            comment='Path to a file with existing bias potential'
        )
        self.setup(mc_config_changes=mc_config_changes, continuation=True)
        return self.run(dump_results=dump_results, dump_name=dump_name)

    def cleanup(self, leave_ext: t.Tuple[str, ...] = ('dat', 'conf', 'tsv'), leave_names: t.Tuple[str, ...] = ()):
        files = glob(f'{self.exp_dir}/*')
        if not files:
            raise ValueError(f'No files are found in experiment directory {self.exp_dir}')
        for p in files:
            in_leave = p.split('/')[-1] in leave_names
            ext_valid = p.split('.')[-1] in leave_ext
            if not (in_leave or ext_valid):
                remove(p)


if __name__ == '__main__':
    raise RuntimeError
