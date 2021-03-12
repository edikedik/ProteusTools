import logging
import typing as t
from dataclasses import dataclass
from itertools import chain, groupby, combinations, product, starmap

import pandas as pd
import ray
from more_itertools import chunked
from ray.exceptions import RayTaskError, RayActorError

import protmc.common.utils as u
import protmc.operators as ops
from protmc.basic.config import ProtMCconfig, parse_config
from protmc.common.base import Id

RemoteFlattener = ray.remote(ops.Flattener)
_UnorderedQuadruplet = t.Tuple[ops.Worker, ops.Worker, ops.Worker, ops.Worker]
_OrderedQuadruplet = t.Tuple[ops.ADAPT, ops.MC, ops.ADAPT, ops.MC]
_WorkerPair = t.NamedTuple('WorkerPair', [('adapt', ops.ADAPT), ('mc', ops.MC)])
RayActor = t.TypeVar('RayActor')


@dataclass
class AffinitySetup:
    combinations: t.Optional[t.List[t.Tuple[int, ...]]]
    protmc_exe_path: str
    base_dir: str
    apo_energy_dir: str
    holo_energy_dir: str
    active_pos: t.List[int]
    mut_space_number_of_types: int
    mut_space_path: str
    adapt_base_conf: str
    mc_base_conf: str
    reference_seq: str
    bias_singletons: bool = False
    bias_pairs: bool = True
    temperature: float = 0.6
    count_threshold: int = 100
    apo_dir_name: str = 'apo'
    holo_dir_name: str = 'holo'
    mc_dir_name: str = 'MC'
    adapt_dir_name: str = 'ADAPT'
    last_bias_name: str = 'ADAPT.last.dat'
    input_bias_name: str = 'ADAPT.inp.dat'
    results_name: str = 'RESULTS.tsv'

    def _setup_conf(self, combination: t.Tuple[int, ...], config: ProtMCconfig) -> ProtMCconfig:
        conf = config.copy()
        constraints = u.space_constraints(
            reference=self.reference_seq,
            subset=combination, active=self.active_pos,
            mutation_space=self.mut_space_path,
            existing_constraints=u.extract_constraints([config]))
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

    def _setup_worker(self, combination: t.Tuple[int, ...], system: t.Tuple[str, str],
                      mode: t.Tuple[str, ProtMCconfig, t.Union[ops.ADAPT, ops.MC]]) \
            -> t.Union[ops.ADAPT, ops.MC]:
        system_dir, energy_dir = system
        workdir, conf, operator = mode
        positions = "-".join(map(str, combination))
        params = ops.WorkerParams(
            working_dir=f'{self.base_dir}/{positions}/{system_dir}/{workdir}',
            config=self._setup_conf(combination, conf),
            protmc_exe_path=self.protmc_exe_path,
            energy_dir_path=energy_dir,
            active_pos=self.active_pos,
            mut_space_number_of_types=self.mut_space_number_of_types,
            last_bias_name=self.last_bias_name,
            input_bias_name=self.input_bias_name,
            results_name=self.results_name
        )
        worker = operator(params, f'{positions}_{system_dir}_{workdir}')
        worker.setup_io(dump=True)
        return worker

    def setup_workers(self) -> t.Tuple[t.Tuple[ops.ADAPT, ops.MC]]:

        with open(self.adapt_base_conf) as f:
            adapt_conf = parse_config(f.read())
        with open(self.mc_base_conf) as f:
            mc_conf = parse_config(f.read())

        combs = self.combinations
        if combs is None:
            combs = chain(((p, p) for p in self.active_pos), combinations(self.active_pos, 2))

        params = product(
            combs,
            [(self.apo_dir_name, self.apo_energy_dir), (self.holo_dir_name, self.holo_energy_dir)],
            [(self.adapt_dir_name, adapt_conf, ops.ADAPT), (self.mc_dir_name, mc_conf, ops.MC)])
        param_pairs = chunked(params, 2)

        workers = tuple(tuple(starmap(self._setup_worker, pair)) for pair in param_pairs)

        if any(not (isinstance(p[0], ops.ADAPT) or isinstance([1], ops.MC)) for p in workers):
            raise ValueError('Each pair must be (ADAPT, MC)')
        return workers


@ray.remote
class AffinitySearch:
    def __init__(self, setup: AffinitySetup, log_path: t.Optional[str] = None,
                 actors_log_path: t.Optional[str] = None, actors_log_level=logging.INFO):
        self.setup = setup
        self.log_path = log_path
        self.actors_log_path = actors_log_path
        self.actors_log_level = actors_log_level
        self._workers: t.Dict[t.Tuple[Id, Id], t.Tuple[ops.ADAPT, ops.MC]] = {
            (adapt.id, mc.id): (adapt, mc) for adapt, mc in setup.setup_workers()}
        self._actors = None
        self._done_ids: t.List[t.Tuple[str, str]] = []

    def get_done_ids(self):
        return self._done_ids

    def get_workers(self) -> t.Dict[t.Tuple[Id, Id], t.Tuple[ops.ADAPT, ops.MC]]:
        return self._workers

    def get_actors(self):
        return self._actors

    @property
    def categories(self) -> t.Dict[str, t.Tuple[str, int]]:
        return {
            f'{self.setup.apo_dir_name}_{self.setup.adapt_dir_name}': ('apo_adapt', 0),
            f'{self.setup.apo_dir_name}_{self.setup.mc_dir_name}': ('apo_mc', 1),
            f'{self.setup.holo_dir_name}_{self.setup.adapt_dir_name}': ('holo_adapt', 2),
            f'{self.setup.holo_dir_name}_{self.setup.mc_dir_name}': ('holo_mc', 3),
        }

    def _get_quadruplets(self, pairs: t.List[t.Tuple[ops.Worker]]):
        def categorize(worker: ops.Worker) -> t.Tuple[str, int]:
            return self.categories['_'.join(worker.id.split('_')[1:])]

        def order(quadruplet: _UnorderedQuadruplet) -> t.Tuple[ops.Worker, ...]:
            return tuple(sorted(quadruplet, key=lambda w: categorize(w)[1]))

        workers = sorted(chain.from_iterable(pairs), key=lambda x: x.id.split('_')[0])
        groups = (tuple(gg) for g, gg in groupby(workers, lambda x: x.id.split('_')[0]))
        return map(order, filter(lambda gg: len(gg) == 4, groups))

    def _spawn_actors(self, n: int, predicate_mc, predicate_adapt) -> t.List[RayActor]:
        self._actors = [
            RemoteFlattener.remote(
                f'FLATTENER-{i}', reference=self.setup.reference_seq,
                predicate_mc=predicate_mc, predicate_adapt=predicate_adapt,
                log_path=self.actors_log_path, loglevel=self.actors_log_level)
            for i in range(1, n + 1)]
        return self._actors

    def _log_message(self, msg) -> None:
        if self.log_path:
            with open(self.log_path, 'a+') as f:
                print(msg, file=f)

    def run(self, aggregator: ops.AffinityAggregator,
            adapt_executor: t.Optional[ops.GenericExecutor],
            mc_executor: t.Optional[ops.GenericExecutor] = None,
            predicate_adapt: t.Callable[[ops.ADAPT], bool] = None,
            predicate_mc: t.Callable[[ops.MC], bool] = None,
            num_cpus: t.Optional[int] = None,
            stop_run_on_actors_fail: bool = True) \
            -> t.Tuple[t.List[t.Tuple[ops.ADAPT, ops.MC]], t.Dict[str, pd.DataFrame]]:
        # TODO: is it hard to allow non-blocking behavior?
        # This would allow combining execution of different AffinitySearch instances in a
        # single run time.

        def flatten_safely(actor, pair):
            try:
                result = actor.apply.remote(adapt_executor, pair, mc_executor)
                return result
            except (RayTaskError, RayActorError, ValueError) as e:
                actor_id = ray.get(actor.get_id.remote())
                msg = f'Actor {actor_id} failed on {pair[0].id} and {pair[1].id} with an error {e}'
                self._log_message(msg)
                if stop_run_on_actors_fail:
                    raise RuntimeError(msg)
            return None, None

        if num_cpus is None:
            num_cpus = int(ray.available_resources()['CPU'])

        # Spawn actors
        actors = self._spawn_actors(num_cpus - 1, predicate_mc, predicate_adapt)
        self._log_message(f'Set up {len(actors)} actors')

        pool = ray.util.ActorPool(actors)
        pairs_flattened, quadruplets = [], {}
        iter_results = pool.map_unordered(flatten_safely, list(self._workers.values()))
        for i, (adapt, mc) in enumerate(iter_results, start=1):
            if adapt is not None and mc is not None:
                pairs_flattened.append((adapt, mc))
                self._done_ids.append((adapt.id, mc.id))
                self._workers[(adapt.id, mc.id)] = (adapt, mc)
                self._log_message(f'Flattened {adapt.id} and {mc.id}. '
                                  f'Remaining: {len(self._workers) - i} pairs')
                for q in self._get_quadruplets(pairs_flattened):
                    id_ = q[0].id.split('_')[0]
                    if id_ in quadruplets:
                        continue
                    quadruplets[id_] = aggregator.aggregate((q[1], q[3]))
                    self._log_message(f'Aggregated {id_}')
        return pairs_flattened, quadruplets


if __name__ == '__main__':
    raise RuntimeError
