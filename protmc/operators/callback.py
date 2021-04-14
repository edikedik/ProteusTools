from collections import defaultdict
import typing as t
import logging
from pathlib import Path

import pandas as pd

from protmc.common.base import MCState
from protmc.basic import Bias
from protmc.common.base import AbstractCallback, AbstractWorker, Id
from protmc.operators import ADAPT, Worker, MC
from protmc.common.utils import extract_constraints, intersect_constraints
from protmc.common.base import AminoAcidDict


class Tracker(AbstractCallback):
    def __init__(
            self, id_, keep_in_memory: bool, dump_to_workdir: True,
            dump_name: str):
        super().__init__(id_)
        self.keep_in_memory = keep_in_memory
        self.dump_to_workdir = dump_to_workdir
        self.dump_name = dump_name
        self._memory = defaultdict(list)

    @property
    def memory(self):
        return self._memory

    def flush_memory(self):
        self._memory = defaultdict(list)

    def __call__(self, worker: AbstractWorker) -> AbstractWorker:
        raise NotImplementedError


class SummaryTracker(Tracker):
    def __init__(self, id_: Id = None, keep_in_memory: bool = True, dump_to_workdir: bool = True,
                 dump_name: str = 'summary_history.tsv'):
        super().__init__(id_, keep_in_memory, dump_to_workdir, dump_name)

    def __call__(self, worker: AbstractWorker) -> AbstractWorker:
        if worker.summary is None:
            return worker
        if self.keep_in_memory:
            self._memory[worker.id].append(worker.summary)
        if self.dump_to_workdir:
            with open(f'{worker.params.working_dir}/{self.dump_name}', 'a+') as f:
                print(*worker.summary, sep='\t', file=f)
        return worker


class BiasTracker(Tracker):
    def __init__(self, id_: Id = None, keep_in_memory: bool = True, dump_to_workdir: bool = True,
                 dump_name: str = 'bias_history.tsv', overwrite: bool = True):
        self.overwrite = overwrite
        super().__init__(id_, keep_in_memory, dump_to_workdir, dump_name)

    def __call__(self, adapt: ADAPT) -> ADAPT:
        if adapt.bias is None or adapt.bias.bias is None:
            return adapt
        if self.keep_in_memory:
            self.memory[adapt.id].append(adapt.bias.bias)
        if self.dump_to_workdir:
            mode = 'w' if self.overwrite else 'a+'
            adapt.bias.bias.to_csv(
                f'{adapt.params.working_dir}/{self.dump_name}', sep='\t',
                index=False, header=mode == 'w', mode=mode)
        return adapt


class NotAPairDesign(Exception):
    def __init__(self, positions, pair_pos: int):
        super().__init__(
            f'Constraints can be applied only in the pair design contexts. '
            f'Got {positions} values on the {pair_pos} position')


class NoResidues(Exception):
    def __init__(self, pos: str, threshold):
        super().__init__(f'No biases < {threshold} for position {pos}')


class Constrainer(AbstractCallback):
    def __init__(self, pos_order: t.Mapping[int, int], id_: Id = None,
                 bias_threshold: float = 10, count_threshold: int = 1,
                 ref_states: t.Optional[t.Dict[str, str]] = None):
        super().__init__(id_)
        self.bias_threshold = bias_threshold
        self.count_threshold = count_threshold
        self.pos_order = pos_order
        self.ref_states = ref_states

    def get_bias(self, worker: Worker) -> t.Optional[pd.DataFrame]:
        raise NotImplementedError

    def __call__(self, worker: Worker) -> Worker:
        def viable_types(df_: pd.DataFrame, first: bool = True):
            p_idx, a_idx, pos = (0, 1, 1) if first else (2, 3, 2)
            p = set(df['var'].apply(lambda x: x.split('-')[p_idx]))
            if len(p) > 1:
                raise NotAPairDesign(p, pos)
            p = p.pop()
            df_['a'] = df['var'].apply(lambda x: x.split('-')[a_idx])
            df_ = df_.groupby('a', as_index=False).agg(
                passes=pd.NamedAgg(
                    column='bias',
                    aggfunc=lambda x: any(b < self.bias_threshold for b in x)
                ))
            ts = " ".join(df_.loc[df_.passes, 'a'].unique())
            if not ts:
                # raise ValueError(f'Worker {worker.id}, ts {ts}, p {p} df {df_}')
                raise NoResidues(p1, self.bias_threshold)
            return p, ts

        # Get bias from worker. Subclasses must implement this function.
        df = self.get_bias(worker)
        if df is None:
            logging.debug(f'Constrainer {self.id} found no bias for worker {worker.id}')
            return worker

        # Find `ts1` -- a collection of types passing the threshold for
        # the first position (`p1`), and `ts2` -- for the second position.
        df = df[df.step == max(df.step)][['var', 'bias']]
        p1, ts1 = viable_types(df.copy(), first=True)
        p2, ts2 = viable_types(df.copy(), first=False)

        # Encode these types into the `Space_Constraints` config parameter
        existing = [] or extract_constraints([worker.params.config])
        constraints = intersect_constraints(existing + [f'{p1} {ts1}', f'{p2} {ts2}'])
        worker.modify_config(
            field_setters=[('MC_PARAMS', 'Space_Constraints', constraints)],
            dump=True)
        logging.debug(f'Constrainer {self.id} placed constraints {constraints} '
                      f'into the config of a worker {worker.id}')

        # Filter the sequence counts of the worker  (if present)
        # and recalculate the summary (if present)
        if worker.seqs is not None:
            aa_mappings = AminoAcidDict().aa_dict
            p1, p2 = map(lambda x: self.pos_order[int(x)], [p1, p2])
            ts1, ts2 = map(lambda ts: [aa_mappings[aa] for aa in ts.split()], [ts1, ts2])
            df = worker.seqs
            logging.debug(f'Constrainer {self.id} -- filtering `seqs` with {len(df)} '
                          f'records for worker {worker.id}')
            df = df[df['seq'].apply(lambda x: x[p1] in ts1)]
            df = df[df['seq'].apply(lambda x: x[p2] in ts2)]
            worker.change_seqs(df)
            logging.debug(f'Constrainer {self.id} -- wrote new `seqs` with {len(df)} '
                          f'records for worker {worker.id}')
            if worker.summary is not None:
                worker.compose_summary(overwrite=True, count_threshold=self.count_threshold)
                logging.debug(f'Constrainer {self.id} -- composed new summary '
                              f'for worker {worker.id}')
        return worker


class AdaptConstrainer(Constrainer):
    def get_bias(self, worker: ADAPT) -> t.Optional[pd.DataFrame]:
        if worker.bias is None or worker.bias.bias is None:
            logging.debug(f'Constrainer {self.id} -- no bias for worker {worker.id}')
            return None
        bias = worker.bias
        if self.ref_states is not None:
            bias = bias.center_at_ref(self.ref_states, overwrite=False)
        return bias.bias


class McConstrainer(Constrainer):
    def get_bias(self, worker: MC) -> t.Optional[pd.DataFrame]:
        bias_path = worker.params.config.get_field_value('Bias_Input_File')
        if not bias_path:
            logging.debug(f'Constrainer {self.id} -- no bias for worker {worker.id}')
            return None
        bias = Bias().read_adapt_output(bias_path)
        if self.ref_states is not None:
            bias = bias.center_at_ref(self.ref_states, overwrite=False)
        return bias.bias


class BestStateKeeper(AbstractCallback):
    def __init__(
            self, id_=None, dump_to_workdir: bool = True,
            dump_name_bias: str = 'ADAPT.best.dat',
            dump_name_seq_count: str = 'RESULTS.best.tsv'):
        super().__init__(id_)
        self.dump_to_workdir = dump_to_workdir
        self.dump_name_bias = dump_name_bias
        self.dump_name_seq_count = dump_name_seq_count
        self._memory: t.Dict[str, MCState] = {}

    def __call__(self, worker: MC) -> MC:
        if worker.summary is None:
            return worker
        bias_path = worker.params.config.get_field_value('Bias_Input_File')
        if not bias_path or not Path(bias_path).exists():
            logging.warning(f'BestStateKeeper {self.id} -- no bias for worker {worker.id}')
            return worker
        bias = Bias().read_adapt_output(bias_path)
        if worker.seqs is None:
            logging.warning(f'BestStateKeeper {self.id} -- no seqs for worker {worker.id}')
            return worker
        if worker.id not in self._memory:
            self._memory[worker.id] = MCState(worker.summary, bias.bias, worker.seqs)
            return worker
        prev_cov = self._memory[worker.id].Summary.coverage
        curr_cov = worker.summary.coverage
        if curr_cov > prev_cov:
            self._memory[worker.id] = MCState(worker.summary, bias.bias, worker.seqs)
        return worker


if __name__ == '__main__':
    raise RuntimeError
