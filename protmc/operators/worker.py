import logging
import subprocess as sp
import typing as t
from pathlib import Path

import pandas as pd

from protmc.basic import analyze_seq_no_rich, compose_summary, Bias
from protmc.basic.config import ProtMCconfig, ConfigValues
from protmc.common import mut_space_size
from protmc.common.base import Summary, AbstractWorker, Id, WorkerParams
from protmc.common.utils import clean_dir


class Worker(AbstractWorker):
    def __init__(self, params: WorkerParams, id_: Id = None,
                 seqs: t.Optional[pd.DataFrame] = None,
                 summary: t.Optional[Summary] = None):
        super().__init__(id_)
        self._params = params
        self._id = id_ or id(self)
        if not self.validate_config(params.config):
            raise RuntimeError(f'Worker {self._id}: failed to validate config')
        self._mode = self._params.config.mode.field_values[0]
        self._seqs, self._summary = seqs, summary

    @property
    def seqs(self) -> t.Optional[pd.DataFrame]:
        return self._seqs

    def change_seqs(self, new: pd.DataFrame) -> None:
        for c in ['seq', 'total_count']:
            if c not in new.columns:
                raise ValueError(f'{c} must be in columns of provided DataFrame')
        self._seqs = new

    @property
    def summary(self):
        return self._summary

    @property
    def params(self):
        return self._params

    @property
    def mut_space_size(self) -> int:
        constraints = self.params.config.get_field_value('Space_Constraints')
        if isinstance(constraints, str):
            constraints = [constraints]
        if constraints is not None:
            constraints = list(filter(lambda c: int(c.split()[0]) in self.params.active_pos, constraints))
        return mut_space_size(
            self.params.mut_space_number_of_types, len(self.params.active_pos), constraints)

    def validate_config(self, config: ProtMCconfig) -> bool:
        trajectory_number = config.get_field_value('Trajectory_Number')
        replica_number = config.get_field_value('Replica_Number')
        if not (trajectory_number is None or trajectory_number != 1):
            raise NotImplementedError(f'Worker {self.id}: multiple trajectories are not supported; '
                                      f'got {trajectory_number}')
        if not (replica_number is None or replica_number != 1):
            raise NotImplementedError(f'Worker {self.id}: multiple replicas are not supported; '
                                      f'got {replica_number}')
        return True

    def modify_config(
            self, field_changes: t.Optional[t.Iterable[t.Tuple[str, ConfigValues]]] = None,
            field_setters: t.Optional[t.Iterable[t.Tuple[str, str, ConfigValues]]] = None,
            dump: bool = False) -> None:
        if field_changes is not None:
            for field_name, field_values in field_changes:
                self.params.config.change_field(field_name, field_values)
                logging.debug(f'Worker {self.id}: changed field {field_name}')
        if field_setters is not None:
            for group_name, field_name, field_values in field_setters:
                self.params.config.set_field(group_name, field_name, field_values)
                logging.debug(f'Worker {self.id}: set up new field {field_name}')
        if dump:
            self.params.config.dump(f'{self.params.working_dir}/{self._mode}.conf')
        return None

    def setup_io(self, dump: bool = True):
        Path(self.params.working_dir).mkdir(exist_ok=True, parents=True)
        mode = self.params.config.mode.field_values[0]
        setters = [
            ('MC_IO', 'Energy_Directory', self.params.energy_dir_path),
            ('MC_IO', 'Seq_Output_File', f'{self.params.working_dir}/{mode}.seq'),
            ('MC_IO', 'Ener_Output_File', f'{self.params.working_dir}/{mode}.ener')
        ]
        if mode == 'ADAPT':
            setters.append(
                ('ADAPT_IO', 'Adapt_Output_File', f'{self.params.working_dir}/{mode}.dat'))
        changes = [x[1:] for x in setters if x[1] in self.params.config.fields]
        setters = [x for x in setters if x[1] not in changes]
        self.modify_config(field_changes=changes, field_setters=setters, dump=dump)
        logging.info(f'Worker {self.id}: finished setting up IO')

    def run(self) -> sp.CompletedProcess:

        cmd = f'{self.params.protmc_exe_path} ' \
              f'< {self.params.config.last_dump_path} '

        logging.info(f'Worker {self.id}: started running protmc')
        # TODO: blocking via `run` is a bit lame. Should find a working strategy to use `Popen` object
        with open(f'{self.params.working_dir}/{self._mode}.log', 'w') as out:
            try:
                proc = sp.run(cmd, shell=True, check=True, text=True, stdout=out, stderr=sp.PIPE)
            except sp.CalledProcessError as e:
                proc = sp.run(cmd, shell=True, check=False, text=True, stdout=sp.PIPE, stderr=sp.PIPE)
                raise ValueError(f'Command {cmd} failed with an error {e}, '
                                 f'stdout {proc.stdout}, and stderr {proc.stderr}')
        return proc

    def collect_seqs(self, dump: bool = True, overwrite: bool = True, combine: bool = False) -> pd.DataFrame:
        logging.info(f'Worker {self.id}: parsing sequences')
        seqs = analyze_seq_no_rich(
            seq_path=self.params.config.get_field_value('Seq_Output_File'),
            matrix_bb=f'{self.params.energy_dir_path}/matrix.bb',
            active=self.params.active_pos,
            steps=int(self.params.config.get_field_value('Trajectory_Length')),
            simple_output=True)
        logging.info(f'Worker {self.id}: finished parsing sequences')
        if combine:
            if self._seqs is None:
                logging.warning(f'Worker {self.id}: `combine=True` but no sequences are stored in memory')
            else:
                seqs = pd.concat([self.seqs, seqs]).groupby('seq', as_index=False).agg({'total_count': 'sum'})
                logging.debug(f'Worker {self.id}: combined seqs')
        if overwrite:
            self._seqs = seqs
        if dump:
            path = f'{self.params.working_dir}/{self.params.results_name}'
            seqs.to_csv(path, index=False, sep='\t')
            logging.info(f'Worker {self.id}: stored results to {path}')
        return seqs

    def compose_summary(self, overwrite: bool = True, count_threshold: int = 1):
        if self.seqs is None:
            raise RuntimeError('Cannot create run summary without seqs being in memory')
        summary = compose_summary(
            self.seqs[self.seqs['total_count'] >= count_threshold],
            self.mut_space_size)
        if overwrite:
            self._summary = summary
        return summary

    def cleanup(self, leave_ext: t.Tuple[str, ...] = ('dat', 'conf', 'tsv'), leave_names: t.Tuple[str, ...] = ()):
        clean_dir(self.params.working_dir, leave_ext, leave_names)
        logging.info(f'Worker {self.id}: cleaned up the {self.params.working_dir}, '
                     f'leaving {leave_ext} extensions and {leave_names} specific files')


class MC(Worker):
    def __init__(self, params: WorkerParams, id_: Id = None):
        if params.config.mode.field_values[0] != 'MONTECARLO':
            raise ValueError(f'Incorrect mode in config: {params.config.mode.field_values[0]} != MC')
        super().__init__(params, id_)


class ADAPT(Worker):
    def __init__(self, params: WorkerParams, id_: Id = None,
                 bias: t.Union[str, Bias, None] = None, bias_input_type: str = 'tsv'):
        super().__init__(params, id_)
        if params.config.mode.field_values[0] != 'ADAPT':
            raise ValueError(f'Incorrect mode in config: {params.config.mode.field_values[0]} != ADAPT')
        self.bias = bias
        if isinstance(bias, str):
            self.bias = Bias()
            if bias_input_type == 'tsv':
                self.bias.read_bias_df(bias, overwrite=True)
            elif bias_input_type == 'dat':
                self.bias.read_adapt_output(bias, overwrite=True)
            else:
                raise ValueError(f'Expected `bias_input_type` either `tsv` or `dat`. Got {bias_input_type} instead')
        logging.debug(f'ADAPT {self.id}: init')

    def store_bias(self):
        bias_path = self.params.config.get_field_value('Adapt_Output_File')
        if bias_path is None:
            raise RuntimeError(f'ADAPT {self.id}: `Adapt_Output_File` is empty in the config')
        if not Path(bias_path).exists():
            raise RuntimeError(f'ADAPT {self.id}: `Adapt_Output_File` {bias_path} does not exist')
        bias = Bias().read_adapt_output(bias_path, overwrite=True)
        if self.bias is None or self.bias.bias is None:
            self.bias = bias
            logging.info(f'ADAPT {self.id}: stored new bias {bias_path}')
        else:
            self.bias.update(bias, overwrite=True)
            logging.info(f'ADAPT {self.id}: updated existing bias with {bias_path}')


if __name__ == '__main__':
    raise RuntimeError
