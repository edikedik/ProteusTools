import logging
import subprocess as sp
import typing as t
from pathlib import Path

import pandas as pd

from ..basic.bias import Bias
from ..basic.config import ProtMCconfig
from ..basic.post import analyze_seq_no_rich, compose_summary
from ..common.base import ConfigValues, Summary, AbstractWorker, Id, WorkerParams
from ..common.utils import mut_space_size, clean_dir


class Worker(AbstractWorker):
    """Base class defining common operations for ADAPT and MC workers."""
    def __init__(self, params: WorkerParams, id_: Id = None,
                 seqs: t.Optional[pd.DataFrame] = None,
                 summary: t.Optional[Summary] = None):
        """
        :param params: A dataclass containing all the parameters necessary to run the Worker.
        :param id_: Unique identifier of a Worker. If None will default to `id(self)`.
        :param seqs: Optional `DataFrame` containing the sampling results.
        :param summary: Optional namedtuple containing sampling summary.
        :raises ValueError: If the `validate_config` method returns `False`.
        """
        super().__init__(id_)
        self._params = params
        self._id = id_ or id(self)
        if not self.validate_config(params.config):
            raise ValueError(f'Worker {self._id}: failed to validate config')
        self._mode = self._params.config.mode.field_values[0]
        self._seqs, self._summary = seqs, summary

    def __repr__(self):
        return f'WorkerID={self._id},Params={self._params}'

    @property
    def seqs(self) -> t.Optional[pd.DataFrame]:
        """
        Typically an output of `analyze_seq_no_rich` residing in `basic.post`.
        Contains at least two columns: `seq` and `total_count`.
        """
        return self._seqs

    def set_seqs(self, new: pd.DataFrame) -> None:
        """
        Replace `seqs` with new DataFrame containing at least two columns: `seq` and `total_count`.
        """
        for c in ['seq', 'total_count']:
            if c not in new.columns:
                raise ValueError(f'{c} must be in columns of provided DataFrame')
        self._seqs = new

    @property
    def summary(self) -> t.Optional[Summary]:
        """
        Summary containing the number of types and the coverage
        (accounting for mutation space defined in config)
        """
        return self._summary

    @property
    def params(self) -> WorkerParams:
        """
        A dataclass holding the parameters of a Worker.
        """
        return self._params

    @property
    def mut_space_size(self) -> int:
        """
        Calculates mutation space size based on
        (1) number of total types in the mutation space, and
        (2) space constraints defined in config.
        """
        constraints = self.params.config.get_field_value('Space_Constraints')
        if isinstance(constraints, str):
            constraints = [constraints]
        if constraints is not None:
            constraints = list(filter(lambda c: int(c.split()[0]) in self.params.active_pos, constraints))
        return mut_space_size(
            self.params.mut_space_number_of_types, len(self.params.active_pos), constraints)

    def validate_config(self, config: ProtMCconfig) -> bool:
        """
        Returns `True` if provided `config` complies with the `Worker`'s limitations.
        Namely, the `Worker` is assumed to be single-threaded:
        having single trajectory and replica (which is True by default).
        """
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
        """
        An interface function to modify existing or create new config fields.
        :param field_changes: An iterable over tuples, where the first element
        is the field name and the second element is (are) the field value(s).
        :param field_setters: An iterable over tuples, where the first element
        is the group name, the second element is the field name,
        and the last element is (are) the field value(s).
        :param dump: Whether to dump modified config into the working directory of a `Worker`.
        """
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
        """
        Modifies config so that all file paths point to the `Worker`
        working directory, and have correct names.
        :param dump: Whether to dump modified config into the working directory of a `Worker`.
        """
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
        """
        Run the `Worker` using the current config.
        Make sure to call `setup_io` first.
        Note -- this is a blocking operation; the spawned subprocess will exist until
        (1) it has finished, (2) it raised an error, or (3) it is killed manually.
        :return: `subprocess.CompletedProcess` instance.
        """

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
        """
        Collect the sampling results into a `Dataframe`.
        :param dump: Dump the collected sequences into the working directory.
        :param overwrite: Overwrite existing `seqs` attribute.
        :param combine: Combined with the previously stored sequences,
        summing up the sequence counts.
        :return: The collected (and potentially combined) sequences.
        """
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

    def compose_summary(self, overwrite: bool = True, count_threshold: int = 1) -> Summary:
        """
        Compose summary of the sampling -- the number of unique sequences
        and the coverage of the mutation space -- based on the
        information stored in `seqs` DataFrame.
        :param overwrite: Overwrite existing `summary` attribute.
        :param count_threshold: Sequences with counts above this threshold
        are used to compute the coverage.
        :return:
        """
        if self.seqs is None:
            raise RuntimeError('Cannot create run summary without seqs being in memory')
        summary = compose_summary(
            self.seqs[self.seqs['total_count'] >= count_threshold],
            self.mut_space_size)
        if overwrite:
            self._summary = summary
        return summary

    def cleanup(self, leave_ext: t.Tuple[str, ...] = ('dat', 'conf', 'tsv'), leave_names: t.Tuple[str, ...] = ()):
        """
        Cleanup the current working directory.
        By default, preserves configs, dat, and tsv files (with counts, summary, etc.).
        :param leave_ext: Extension of the files to leave untouched.
        :param leave_names: Concrete names of the files to leave untouched.
        """
        clean_dir(self.params.working_dir, leave_ext, leave_names)
        logging.info(f'Worker {self.id}: cleaned up the {self.params.working_dir}, '
                     f'leaving {leave_ext} extensions and {leave_names} specific files')


class MC(Worker):
    """
    A subclass of a `Worker` where the mode `MONTECARLO` is explicitly specified in config.
    """
    def __init__(self, params: WorkerParams, id_: Id = None):
        if params.config.mode.field_values[0] != 'MONTECARLO':
            raise ValueError(f'Incorrect mode in config: {params.config.mode.field_values[0]} != MC')
        super().__init__(params, id_)


class ADAPT(Worker):
    """
    A subclass of a `Worker` where the mode `ADAPT` is explicitly specified in config.
    Compared to base `Worker`, has one extra attribute and attribute to store and manipulate bias:
    `bias` and `store_bias`.
    """
    def __init__(self, params: WorkerParams, id_: Id = None,
                 bias: t.Union[str, Bias, None] = None, bias_input_type: str = 'tsv'):
        """
        :param params: A dataclass containing all the parameters necessary to run the Worker.
        :param id_: Unique identifier of a Worker. If None will default to `id(self)`.
        :param bias: Explicitly provide bias.
        :param bias_input_type: Type of the object provided in `bias` argument.
        Can be either `tsv` or `dat`.
        :raises ValueError: If the `validate_config` method returns `False`.
        """
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
        """
        Parse and store bias.
        Path of to the bias provided via config and automatically created during the `setup_io` call.
        :raises ValueError: If no bias path is found in `config` or the path does not exist.
        """
        bias_path = self.params.config.get_field_value('Adapt_Output_File')
        if bias_path is None:
            raise ValueError(f'ADAPT {self.id}: `Adapt_Output_File` is empty in the config')
        if not Path(bias_path).exists():
            raise ValueError(f'ADAPT {self.id}: `Adapt_Output_File` {bias_path} does not exist')
        bias = Bias().read_adapt_output(bias_path, overwrite=True)
        if self.bias is None or self.bias.bias is None:
            self.bias = bias
            logging.info(f'ADAPT {self.id}: stored new bias {bias_path}')
        else:
            self.bias.update(bias, overwrite=True)
            logging.info(f'ADAPT {self.id}: updated existing bias with {bias_path}')


if __name__ == '__main__':
    raise RuntimeError
