import logging
import subprocess as sp
import typing as t
from abc import ABCMeta
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from protmc.basic import analyze_seq_no_rich, compose_summary
from protmc.basic.config import ProtMCconfig, ConfigValues
from protmc.common import mut_space_size
from protmc.common.base import Summary
from protmc.common.utils import clean_dir


@dataclass
class WorkerParams:
    # TODO: allow dumping; create separate function for reading
    working_dir: str
    protmc_exe_path: str
    energy_dir_path: str
    active_pos: t.List[int]
    config: ProtMCconfig
    mut_space_number_of_types: int
    last_bias_name: str = 'ADAPT.last.dat'
    input_bias_name: str = 'ADAPT.inp.dat'
    results_name: str = 'RESULTS.tsv'


class Worker(metaclass=ABCMeta):
    def __init__(self, params: WorkerParams, id_: t.Optional[t.Union[int, str]] = None,
                 seqs: t.Optional[pd.DataFrame] = None, summary: t.Optional[Summary] = None):
        self._params = params
        if not self.validate_config(params.config):
            raise RuntimeError(f'Worker {self._id}: failed to validate config')
        self._config = params.config.copy()
        self._id = id_ or id(self)
        self._mode = self._params.config.mode.field_values[0]
        self._seqs, self._summary = seqs, summary

    @property
    def seqs(self) -> t.Optional[pd.DataFrame]:
        return self._seqs

    @property
    def summary(self):
        return self._summary

    @property
    def mut_space_size(self) -> int:
        constraints = self._config.get_field_value('Space_Constraints')
        if isinstance(constraints, str):
            constraints = [constraints]
        if constraints is not None:
            constraints = list(filter(lambda c: int(c.split()[0]) in self._params.active_pos, constraints))
        return mut_space_size(
            self._params.mut_space_number_of_types, len(self._params.active_pos), constraints)

    def validate_config(self, config: ProtMCconfig) -> bool:
        trajectory_number = config.get_field_value('Trajectory_Number')
        replica_number = config.get_field_value('Replica_Number')
        if not (trajectory_number is None or trajectory_number == 1):
            raise NotImplementedError(f'Worker {self._id}: multiple trajectories are not supported; '
                                      f'got {trajectory_number}')
        if not (replica_number is None or replica_number == 1):
            raise NotImplementedError(f'Worker {self._id}: multiple replicas are not supported; '
                                      f'got {replica_number}')
        return True

    def modify_config(
            self, field_changes: t.Optional[t.Iterable[t.Tuple[str, ConfigValues]]] = None,
            field_setters: t.Optional[t.Iterable[t.Tuple[str, str, ConfigValues]]] = None,
            dump: bool = False) -> None:
        if field_changes is not None:
            for field_name, field_values in field_changes:
                self._config.change_field(field_name, field_values)
                logging.debug(f'Worker {self._id}: changed field {field_name}')
        if field_setters is not None:
            for group_name, field_name, field_values in field_setters:
                self._config.set_field(group_name, field_name, field_values)
                logging.debug(f'Worker {self._id}: set up new field {field_name}')
        if dump:
            self._config.dump(f'{self._params.working_dir}/{self._mode}.conf')
        return None

    def setup_io(self, dump: bool = True):
        Path(self._params.working_dir).mkdir(exist_ok=True, parents=True)
        mode = self._config.mode.field_values[0]
        setters = [
            ('MC_IO', 'Energy_Directory', self._params.energy_dir_path),
            ('MC_IO', 'Seq_Output_File', f'{self._params.working_dir}/{mode}.seq'),
            ('MC_IO', 'Ener_Output_File', f'{self._params.working_dir}/{mode}.ener')
        ]
        if mode == 'ADAPT':
            setters.append(
                ('ADAPT_IO', 'Adapt_Output_File', f'{self._params.working_dir}/{mode}.dat'))
        changes = [x[1:] for x in setters if x[1] in self._config.fields]
        setters = [x for x in setters if x[1] not in changes]
        self.modify_config(field_changes=changes, field_setters=setters, dump=dump)
        logging.info(f'Worker {self._id}: finished setting up IO')

    def run(self) -> sp.CompletedProcess:
        cmd = f'{self._params.protmc_exe_path} ' \
              f'< {self._config.last_dump_path} ' \
              f'> {self._params.working_dir}/{self._mode}.log'
        logging.info(f'Worker {self._id}: started running protmc')
        try:
            proc = sp.run(cmd, shell=True, check=True)
        except sp.CalledProcessError as e:
            output = sp.run(cmd, shell=True, check=False, capture_output=True, text=True)
            raise RuntimeError(f'Failed to run protMC with an error {e} and output {output}')
        logging.info(f'Worker {self._id}: finished running protmc')
        return proc

    def collect_seqs(self, dump: bool = True, overwrite: bool = True, combine: bool = False) -> pd.DataFrame:
        logging.info(f'Worker {self._id}: parsing sequences')
        seqs = analyze_seq_no_rich(
            seq_path=self._config.get_field_value('Seq_Output_File'),
            matrix_bb=f'{self._params.energy_dir_path}/matrix.bb',
            active=self._params.active_pos,
            steps=int(self._config.get_field_value('Trajectory_Length')),
            simple_output=True)
        logging.info(f'Worker {self._id}: finished parsing sequences')
        if combine:
            if self._seqs is None:
                logging.warning(f'Worker {self._id}: `combine=True` but no sequences are stored in memory')
            else:
                seqs = pd.concat([self.seqs, seqs]).groupby('seq', as_index=False).agg({'total_count': 'sum'})
                logging.debug(f'Worker {self._id}: combined seqs')
        if overwrite:
            self._seqs = seqs
        if dump:
            path = f'{self._params.working_dir}/{self._params.results_name}'
            seqs.to_csv(path, index=False, sep='\t')
            logging.info(f'Worker {self._id}: stored results to {path}')
        return seqs

    def compose_summary(self, overwrite: bool = True):
        if self.seqs is None:
            raise RuntimeError('Cannot create run summary without seqs being in memory')
        summary = compose_summary(self.seqs, self.mut_space_size)
        if overwrite:
            self._summary = summary
        return summary

    def cleanup(self, leave_ext: t.Tuple[str, ...] = ('dat', 'conf', 'tsv'), leave_names: t.Tuple[str, ...] = ()):
        clean_dir(self._params.working_dir, leave_ext, leave_names)
        logging.info(f'Worker {self._id}: cleaned up')


class MC(Worker):
    def __init__(self, params: WorkerParams, id_: t.Optional[t.Union[int, str]] = None):
        if params.config.mode.field_values[0] != 'MC':
            raise ValueError(f'Incorrect mode in config: {params.config.mode.field_values[0]} != MC')
        super().__init__(params, id_)


class ADAPT(Worker):
    def __init__(self, params: WorkerParams, id_: t.Optional[t.Union[int, str]] = None):
        if params.config.mode.field_values[0] != 'ADAPT':
            raise ValueError(f'Incorrect mode in config: {params.config.mode.field_values[0]} != ADAPT')
        super().__init__(params, id_)
