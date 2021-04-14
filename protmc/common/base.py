import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import pandas as pd

Summary = t.NamedTuple('Summary', [
    ('num_unique', int), ('num_unique_merged', int), ('coverage', float),
    ('seq_prob_mean', float), ('seq_prob_std', float), ('seq_prob_rss', float)])
ShortSummary = t.NamedTuple('ShortSummary', [('num_unique', int), ('num_unique_merged', int), ('coverage', float)])
MCState = t.NamedTuple(
    'MCState', [('Summary', t.Union[Summary, ShortSummary]), ('Bias', pd.DataFrame), ('Seqs', pd.DataFrame)])
Population_element = t.NamedTuple('Population_element', [('seq', str), ('count', int)])
AA_pair = t.NamedTuple('AA_pair', [('pos_i', str), ('pos_j', str), ('aa_i', str), ('aa_j', str)])
PairBias = t.NamedTuple('PairBias', [('aa_pair', AA_pair), ('bias', float)])
AffinityResult = t.NamedTuple('AffinityResults', [('seq', str), ('affinity', float)])
AffinityResults = t.NamedTuple('AffinityResults', [('run_dir', str), ('affinity', float)])
PipelineOutput = t.NamedTuple('PipelineOutput', [('seqs', pd.DataFrame), ('summary', pd.DataFrame)])
ParsedEntry = t.NamedTuple('ParsedEntry', [('seq', str), ('counts', int), ('energy', float)])

_AA_DICT = """ALA A ACT
CYS C ACT
THR T ACT
GLU E ED
GLH e ED
ASP D ED
ASH d ED
PHE F FW
TRP W FW
ILE I IVL
VAL V IVL
LEU L IVL
LYS K K
LYN k K
MET M M
ASN N NQ
GLN Q NQ
SER S S
ARG R R
TYR Y Y
TYD y Y
HID h H
HIE j H
HIP H H
PRO P PG
GLY G PG"""

Id = t.Optional[t.Union[int, str]]


@dataclass
class WorkerParams:
    working_dir: str
    protmc_exe_path: str
    energy_dir_path: str
    active_pos: t.List[int]
    config: "ProtMCConfig"
    mut_space_number_of_types: int
    last_bias_name: str = 'ADAPT.last.dat'
    input_bias_name: str = 'ADAPT.inp.dat'
    results_name: str = 'RESULTS.tsv'


class AminoAcidDict:
    def __init__(self, inp: str = _AA_DICT):
        self._aa_dict = self._parse_dict(inp)

    @staticmethod
    def _parse_dict(inp):
        inp_split = [x.split() for x in inp.split('\n')]
        return {
            **{line[0]: line[1] for line in inp_split},
            **{line[1]: line[0] for line in inp_split}}

    @property
    def aa_dict(self) -> t.Dict[str, str]:
        return self._aa_dict

    @property
    def proto_mapping(self) -> t.Dict[str, str]:
        return {'e': 'E', 'd': 'D', 'k': 'K', 'y': 'Y', 'j': 'H', 'h': 'H',
                'GLH': 'GLU', 'ASH': 'ASP', 'LYN': 'LYS', 'TYD': 'TYR', 'HID': 'HIP', 'HIE': 'HIP'}


class NoReferenceError(Exception):
    pass


class AbstractWorker(metaclass=ABCMeta):
    def __init__(self, id_: Id = None):
        self._id = id_ or id(self)

    @property
    def id(self) -> Id:
        return self._id

    @property
    @abstractmethod
    def seqs(self) -> pd.DataFrame:
        raise NotImplementedError

    @property
    @abstractmethod
    def summary(self) -> t.Union[Summary, ShortSummary]:
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self) -> WorkerParams:
        raise NotImplementedError

    @abstractmethod
    def setup_io(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> t.Any:
        raise NotImplementedError

    @abstractmethod
    def collect_seqs(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def compose_summary(self) -> t.Union[Summary, ShortSummary]:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> t.Any:
        raise NotImplementedError


class AbstractExecutor(metaclass=ABCMeta):
    def __init__(self, id_: Id = None):
        self._id = id_ or id(self)

    @property
    def id(self):
        return self._id

    @abstractmethod
    def __call__(self, worker: AbstractWorker) -> AbstractWorker:
        raise NotImplementedError


class AbstractCallback(metaclass=ABCMeta):
    def __init__(self, id_: Id = None):
        self._id = id_ or id(self)

    @property
    def id(self):
        return self._id

    @abstractmethod
    def __call__(self, worker: AbstractWorker) -> AbstractWorker:
        raise NotImplementedError


class AbstractPoolExecutor(metaclass=ABCMeta):
    def __init__(self, id_: Id = None):
        self._id = id_ or id(self)

    @property
    def id(self):
        return self._id

    def __call__(self, workers: t.Sequence[AbstractWorker]) -> t.Sequence[AbstractWorker]:
        raise NotImplementedError


class AbstractAggregator(metaclass=ABCMeta):
    def __init__(self, id_: Id = None):
        self._id = id_ or id(self)

    @property
    def id(self):
        return self._id

    @abstractmethod
    def aggregate(self, workers: t.Collection[AbstractWorker]):
        raise NotImplementedError


class AbstractManager(metaclass=ABCMeta):
    def __init__(self, id_: Id = None):
        self._id = id_ or id(self)


if __name__ == '__main__':
    raise RuntimeError
