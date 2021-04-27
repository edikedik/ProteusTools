import logging
import typing as t
from dataclasses import dataclass
from glob import glob
from itertools import chain, groupby, combinations, product, starmap
from warnings import warn

import pandas as pd
import ray
from more_itertools import chunked
from tqdm import tqdm

from ..basic.config import ProtMCconfig, parse_config
from ..basic.stability import stability
from ..common.base import Id, NoReferenceError, WorkerParams
from ..common.utils import extract_constraints, space_constraints
from ..operators.aggregator import AffinityAggregator
from ..operators.executor import GenericExecutor
from ..operators.flattener import Flattener, flatten_pair
from ..operators.worker import Worker, ADAPT, MC
from ..resources import ADAPT_CONF_PATH, MC_CONF_PATH, PROTMC_PATH

_UnorderedQuadruplet = t.Tuple[Worker, Worker, Worker, Worker]
_OrderedQuadruplet = t.Tuple[ADAPT, MC, ADAPT, MC]
_WorkerPair = t.NamedTuple('WorkerPair', [('adapt', ADAPT), ('mc', MC)])
Param_set = t.Tuple[t.Tuple[int, ...], t.Tuple[str, str], t.Tuple[str, ProtMCconfig, t.Union[ADAPT, MC]]]
RayActor = t.TypeVar('RayActor')


@dataclass
class FlattenerSetup:
    """Dataclass holding parameters for the `Flattener` operator."""
    executor_adapt: t.Optional[GenericExecutor] = None
    executor_mc: t.Optional[GenericExecutor] = None
    predicate_adapt: t.Callable[[ADAPT], bool] = None
    predicate_mc: t.Callable[[MC], bool] = None
    reference: t.Optional[str] = None
    count_threshold: int = 1
    step_threshold: int = 100


@dataclass
class AffinitySetup:
    """Dataclass holding parameters for the `AffinitySearch`."""
    combinations: t.Optional[t.List[t.Tuple[int, ...]]]
    base_dir: str
    apo_energy_dir: str
    holo_energy_dir: str
    active_pos: t.List[int]
    mut_space_number_of_types: int
    mut_space_path: str
    reference_seq: str
    apo_avg_ref_ener: str
    holo_avg_ref_ener: str
    bias_singletons: bool = False
    bias_pairs: bool = True
    temperature: float = 0.6
    count_threshold: int = 100
    apo_dir_name: str = 'apo'
    holo_dir_name: str = 'holo'
    mc_dir_name: str = 'MC'
    adapt_base_conf: str = ADAPT_CONF_PATH
    mc_base_conf: str = MC_CONF_PATH
    protmc_exe_path: str = PROTMC_PATH
    adapt_dir_name: str = 'ADAPT'
    last_bias_name: str = 'ADAPT.last.dat'
    input_bias_name: str = 'ADAPT.inp.dat'
    results_name: str = 'RESULTS.tsv'

    def setup_conf(self, combination: t.Tuple[int, ...], config: ProtMCconfig) -> ProtMCconfig:
        conf = config.copy()
        constraints = space_constraints(
            reference=self.reference_seq,
            subset=combination, active=self.active_pos,
            mutation_space=self.mut_space_path,
            existing_constraints=extract_constraints([config]))
        conf.set_field('MC_PARAMS', 'Space_Constraints', constraints)
        if conf.mode.field_values[0] == 'ADAPT':
            space = []
            if self.bias_pairs:
                space += [f'{p1}-{p2}' for p1, p2 in sorted(combinations(combination, 2))]
            if self.bias_singletons:
                space += [f'{p}-{p}' for p in sorted(set(combination))]
            if not space:
                raise RuntimeError('Empty `Adapt_Space`')
            conf.set_field('ADAPT_PARAMS', 'Adapt_Space', space)
        return conf

    def setup_avg_energy(self, worker: t.Union[ADAPT, MC], system_dir: str):
        avg_ener_path = self.apo_avg_ref_ener if system_dir == self.apo_dir_name else self.holo_avg_ref_ener
        with open(avg_ener_path) as f:
            avg_ener = [x.rstrip() for x in f if x != '\n']
        setter = ('MC_PARAMS', 'Ref_Ener', avg_ener)
        worker.modify_config(field_setters=[setter])

    def setup_worker(self, combination: t.Tuple[int, ...], system: t.Tuple[str, str],
                     mode: t.Tuple[str, ProtMCconfig, t.Union[ADAPT, MC]],
                     combination_base_dir: t.Optional[str] = None) -> t.Union[ADAPT, MC]:
        system_dir, energy_dir = system
        workdir, conf, operator = mode
        if combination_base_dir is None:
            combination_base_dir = "-".join(map(str, combination))
        params = WorkerParams(
            working_dir=f'{self.base_dir}/{combination_base_dir}/{system_dir}/{workdir}',
            config=self.setup_conf(combination, conf),
            protmc_exe_path=self.protmc_exe_path,
            energy_dir_path=energy_dir,
            active_pos=self.active_pos,
            mut_space_number_of_types=self.mut_space_number_of_types,
            last_bias_name=self.last_bias_name,
            input_bias_name=self.input_bias_name,
            results_name=self.results_name
        )
        worker = operator(params, f'{combination_base_dir}_{system_dir}_{workdir}')
        self.setup_avg_energy(worker, system_dir)
        worker.setup_io(dump=True)
        return worker

    def prepare_params(self) -> t.Iterator[t.Tuple[Param_set, Param_set]]:
        with open(self.adapt_base_conf) as f:
            adapt_conf = parse_config(f.read())
        with open(self.mc_base_conf) as f:
            mc_conf = parse_config(f.read())

        combs = self.combinations
        if combs is None:
            combs = chain(
                ((p, p) for p in self.active_pos),  # singletons
                combinations(self.active_pos, 2)  # pairs
            )

        params = product(
            combs,
            [(self.apo_dir_name, self.apo_energy_dir), (self.holo_dir_name, self.holo_energy_dir)],
            [(self.adapt_dir_name, adapt_conf, ADAPT), (self.mc_dir_name, mc_conf, MC)])
        return chunked(params, 2)

    def setup_workers(self) -> t.Tuple[t.Tuple[ADAPT, MC]]:
        param_pairs = self.prepare_params()
        workers = tuple(tuple(starmap(self.setup_worker, pair)) for pair in param_pairs)

        if any(not (isinstance(p[0], ADAPT) or isinstance([1], MC)) for p in workers):
            raise ValueError('Each pair must be (ADAPT, MC)')
        return workers


class AffinitySearch:
    def __init__(self, setup: AffinitySetup, workers: t.Optional[t.List[t.Tuple[ADAPT, MC]]] = None):
        self.setup = setup
        workers = workers or setup.setup_workers()
        self._workers = {(adapt.id, mc.id): (adapt, mc) for adapt, mc in workers}
        self._done_ids: t.Set[t.Tuple[str, str]] = set()

    @property
    def done_ids(self) -> t.Set[t.Tuple[str, str]]:
        return self._done_ids

    def flush_done(self):
        self._done_ids = set()

    @property
    def not_done_ids(self) -> t.Set[t.Tuple[str, str]]:
        return set(self._workers) - self._done_ids

    @property
    def workers(self) -> t.Dict[t.Tuple[Id, Id], t.Tuple[ADAPT, MC]]:
        return self._workers

    @property
    def categories(self) -> t.Dict[str, t.Tuple[str, int]]:
        return {
            f'{self.setup.apo_dir_name}_{self.setup.adapt_dir_name}': ('apo_adapt', 0),
            f'{self.setup.apo_dir_name}_{self.setup.mc_dir_name}': ('apo_mc', 1),
            f'{self.setup.holo_dir_name}_{self.setup.adapt_dir_name}': ('holo_adapt', 2),
            f'{self.setup.holo_dir_name}_{self.setup.mc_dir_name}': ('holo_mc', 3),
        }

    def _get_quadruplets(self, pairs: t.List[t.Tuple[Worker]]):
        def categorize(worker: Worker) -> t.Tuple[str, int]:
            return self.categories['_'.join(worker.id.split('_')[1:])]

        def order(quadruplet: _UnorderedQuadruplet) -> t.Tuple[Worker, ...]:
            return tuple(sorted(quadruplet, key=lambda w: categorize(w)[1]))

        workers = sorted(chain.from_iterable(pairs), key=lambda x: x.id.split('_')[0])
        groups = (tuple(gg) for g, gg in groupby(workers, lambda x: x.id.split('_')[0]))
        return map(order, filter(lambda gg: len(gg) == 4, groups))

    def flatten(self, setup: t.Union[FlattenerSetup, t.List[FlattenerSetup]], stop_on_fail: bool = False,
                ids: t.Optional[t.Sequence[t.Tuple[Id, Id]]] = None, update: bool = True) \
            -> t.List[t.Tuple[ADAPT, MC]]:
        workers = self._workers
        if ids:
            workers = {key: v for key, v in workers.items() if key in ids}
        logging.info(f'Will flatten {len(workers)} workers')

        if isinstance(setup, FlattenerSetup):
            fs = [Flattener(id_=i, **setup.__dict__) for i in range(len(workers))]
        else:
            assert len(setup) == len(workers)
            fs = [Flattener(id_=i, **s.__dict__) for i, s in enumerate(setup)]

        handles = [flatten_pair.remote(p, f) for p, f in zip(workers.values(), fs)]
        all_ids = set(workers)
        while handles:
            remain = all_ids - self._done_ids
            msg = f'Remaining {len(remain)} our of {len(all_ids)}'
            if len(remain) <= 10:
                msg += f': {remain}'
            logging.info(msg)

            pair_done, handles = ray.wait(handles)
            adapt, mc, step = ray.get(pair_done[0])
            if not step:
                logging.warning(f'Failed on {adapt.id} and {mc.id}')
                if stop_on_fail:
                    return [(adapt, mc) for adapt, mc in workers.values() if (adapt.id, mc.id) in self._done_ids]
            else:
                if update:
                    self._workers[(adapt.id, mc.id)] = adapt, mc
                logging.info(f'Flattened {adapt.id} and {mc.id} in {step} steps')
            self._done_ids |= {(adapt.id, mc.id)}

        return [(adapt, mc) for adapt, mc in workers.values() if (adapt.id, mc.id) in self._done_ids]

    def aggregate(self, aggregator: AffinityAggregator, ids: t.Collection[Id]):
        raise NotImplementedError


def aggregate_from_base(
        base_dir: str, ref_seq: str, ref_pos: t.Sequence[int],
        pos_parser: t.Callable[[str], t.List[str]] = lambda x: x.split('-'),
        temperature: float = 0.6, count_threshold: int = 100,
        holo: str = 'holo', apo: str = 'apo', mc: str = 'MC',
        bias_name: str = 'ADAPT.inp.dat', seqs_name: str = 'RESULTS.tsv') -> pd.DataFrame:
    """
    Aggregate the `AffinitySearch` results.
    It assumes the following directory structure:
    `base_dir`/`pair dir`/`apo` or `holo`/`adapt` or `mc`.
    It also assumes that `pos_parser` can obtain a list of positions from the `pair dir`.
    :param base_dir: "Root" directory path.
    :param ref_seq: Reference sequence.
    :param ref_pos: Positions of amino acids in the reference sequence in the same order.
    :param pos_parser: Callable accepting a name of the `pair dir` and returning a list of positions.
    :param temperature: Simulation temperature.
    :param count_threshold: Sequences above this threshold will be excluded during aggregation.
    :param holo: A name of the `holo` dir.
    :param apo: A name of the `apo` dir.
    :param mc: A name of the `mc` dir.
    :param bias_name: A name of the bias file. Assumed to reside in the `mc` dir.
    :param seqs_name: A name of the "tsv" file with the sequence counts.
    :return: A DataFrame with aggregated sequences, stability of each `apo` and `holo` systems,
    and affinity -- a stability difference between `holo` and `apo` systems.
    """

    ref_pos_str = list(map(str, ref_pos))
    ref_pos_mapping = {p: i for i, p in enumerate(ref_pos_str)}

    def affinity_df(pair_base):
        pop_apo = pd.read_csv(f'{pair_base}/{apo}/{mc}/{seqs_name}', sep='\t')
        pop_holo = pd.read_csv(f'{pair_base}/{holo}/{mc}/{seqs_name}', sep='\t')
        bias_apo = f'{pair_base}/{apo}/{mc}/{bias_name}'
        bias_holo = f'{pair_base}/{holo}/{mc}/{bias_name}'
        stability_apo = stability(pop_apo, bias_apo, ref_seq, temperature, count_threshold, ref_pos_str)
        stability_holo = stability(pop_holo, bias_holo, ref_seq, temperature, count_threshold, ref_pos_str)
        df = pd.merge(stability_apo, stability_holo, on='seq', how='outer', suffixes=['_apo', '_holo'])
        df['affinity'] = df['stability_holo'] - df['stability_apo']
        positions = pos_parser(pair_base)
        df['seq_subset'] = df['seq'].apply(lambda s: ''.join(s[ref_pos_mapping[p]] for p in positions))
        df['pos'] = '-'.join(positions)
        return df

    paths = tqdm(glob(f'{base_dir}/*'), desc='Aggregating workers')
    dfs = []
    for p in paths:
        try:
            dfs.append(affinity_df(p))
        except (NoReferenceError, ValueError, KeyError) as e:
            warn(f'Could not aggregate worker {p} due to {e}')

    return pd.concat(dfs)


if __name__ == '__main__':
    raise RuntimeError
