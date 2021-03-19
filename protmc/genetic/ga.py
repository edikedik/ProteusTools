import typing as t
from functools import partial
from itertools import chain, groupby, tee
from math import floor
from random import sample
from statistics import mean

import genetic.operators as ops
import numpy as np
import ray
from genetic.base import Operators, Callback
from genetic.evolvers import GenericEvolver
from more_itertools import random_permutation, unzip, partition, peekable, take, distribute
from tqdm import tqdm

from .base import Gene, GeneticParams, Record
from .individual import GenericIndividual
from .mutator import Mutator, BucketMutator
from .score import score


def filter_genes(genes: t.Iterable[Gene], coupling_threshold: float) -> t.Iterator[Gene]:
    """
    Filters `genes` so that a graph induced by them abides structural constraints.
    """

    def process_group(genes_: t.Iterable[Gene]):
        weak, strong = (partition(lambda g: g.C >= coupling_threshold, genes_))
        strong = peekable(strong)
        if strong.peek(None) is not None:
            return strong
        return take(1, weak)

    groups = groupby(sorted(genes), lambda g: (g.P1, g.P2))
    return chain.from_iterable(process_group(gg) for g, gg in groups)


def _check_mating_group(mating_group: t.List[t.Tuple[GenericIndividual, Record]]):
    # Validate the operation
    if len(mating_group) < 2:
        raise RuntimeError(f'Mating group must contain at least 2 individuals. Got {len(mating_group)}')
    ts = set(indiv.coupling_threshold for indiv, _ in mating_group)
    if len(ts) > 1:
        raise RuntimeError(f'All Individuals in the mating group must have the same coupling threshold. Got {ts}')
    coupling_threshold = ts.pop()
    types = set(indiv.__class__ for indiv, _, in mating_group)
    if len(types) > 1:
        raise RuntimeError(f'All Individuals in the mating group must be of a single type. '
                           f'Got {ts} for mating group {mating_group}')
    ind_type = types.pop()
    max_spaces = set(indiv.max_mut_space for indiv, _ in mating_group)
    if len(max_spaces) > 1:
        raise RuntimeError(f'Multiple values of `max_mut_space` attribute in the mating group.')
    # Perform the recombination
    max_space = max_spaces.pop()
    return coupling_threshold, ind_type, max_space


def recombine_genes_uniformly(mating_group: t.List[t.Tuple[GenericIndividual, Record]],
                              brood_size: int) -> t.List[GenericIndividual]:
    """
    Combines genes of individuals in the `mating_group` in a single pool,
    and uniformly divides these genes into `brood_size` number of individuals.
    :param mating_group: A group of individuals selected to give progeny.
    :param brood_size: A number of offsprings.
    :return: List of offsprings.
    """
    coupling_threshold, ind_type, max_space = _check_mating_group(mating_group)
    pool = random_permutation(chain.from_iterable(indiv.genes() for indiv, _ in mating_group))
    chunks = take(brood_size, distribute(len(mating_group), pool))
    return [ind_type(list(genes), coupling_threshold, max_space) for genes in chunks]


def recombine_into(mating_group: t.List[t.Tuple[GenericIndividual, Record]],
                   brood_size: int) -> t.List[GenericIndividual]:
    coupling_threshold, ind_type, max_space = _check_mating_group(mating_group)
    pool = random_permutation(chain.from_iterable(indiv.genes() for indiv, _ in mating_group))
    chunks = distribute(brood_size, pool)
    return [ind_type(list(genes), coupling_threshold, max_space) for genes in chunks]


def exchange_fraction(mating_group: t.List[t.Tuple[GenericIndividual, Record]],
                      brood_size: int, fraction: float) -> t.List[GenericIndividual]:
    if brood_size > len(mating_group):
        raise ValueError(f'Brood size {brood_size} cannot be larger than the mating group {len(mating_group)}')
    staged_comb, staged_pool = tee(((ind, floor(len(ind) * fraction)) for ind, _ in mating_group), 2)
    pool_samples, samples = tee((take(n, ind.genes()) for ind, n in staged_pool), 2)
    pool = random_permutation(chain.from_iterable(pool_samples))
    recombined = (
        ind.copy().remove_genes(s).add_genes(take(n, pool))
        for (ind, n), s in zip(staged_comb, samples))
    return list(take(brood_size, recombined))


class GA:
    def __init__(self, genetic_params: GeneticParams,
                 populations: t.Optional[t.List[t.List[GenericIndividual]]] = None,
                 records: t.Optional[t.List[t.List[Record]]] = None):
        self.genetic_params = genetic_params

        # Setup helpers for operators
        self.score_func = partial(
            score,
            **genetic_params.Score_kwargs)
        self.selector_fn = partial(
            ops.ktournament, genetic_params.Tournaments_selection, lambda x: x.score,
            genetic_params.Number_of_mates, replace=True)
        self.policy_fn = partial(ops.ktournament, genetic_params.Tournaments_policy, lambda x: x.score)
        self.crossover_fn = partial(recombine_genes_uniformly, brood_size=genetic_params.Brood_size)

        # Setup operators
        self._estimator = ops.GenericEstimator(func=self.score_func)
        self._recorder = ops.GenericRecorder(
            start=lambda indiv, score_: Record(0, score_),
            update=lambda indiv, rec: Record(rec.age + 1, rec.score))
        self._selector = ops.GenericSelector(
            selector=self.selector_fn,
            nmates=genetic_params.Number_of_mates)
        self._crossover = ops.GenericCrossover(
            crossover=self.crossover_fn,
            nmates=genetic_params.Number_of_mates,
            broodsize=genetic_params.Brood_size)
        if genetic_params.Use_BucketMutator:
            self._mutator = BucketMutator(
                genetic_params.Gene_pool, genetic_params.Mutable_fraction)
        else:
            self._mutator = Mutator(
                genetic_params.Gene_pool, genetic_params.Mutable_fraction,
                genetic_params.Deletion_size, genetic_params.Acquisition_size,
                ps=genetic_params.Probabilities)
        self._policy = ops.GenericPolicy(selector=self.policy_fn)
        self._evolver = GenericEvolver()
        self.ops = Operators(self._recorder, self._estimator, self._selector,
                             self._crossover, self._mutator, self._policy)

        # Setup placeholders
        self.populations = populations
        self.records = records

    def spawn_populations(self, n: int, overwrite: bool = True, individual_type=GenericIndividual):
        populations = ray.get([_spawn_remote.remote(self.genetic_params, individual_type) for _ in range(n)])
        if overwrite:
            self.populations, self.records = populations, None
        return populations

    def flatten(self):
        if self.populations is None:
            raise ValueError('No populations')
        return (list(chain.from_iterable(self.populations)),
                list(chain.from_iterable(self.records)) if self.records else [None] * len(self.populations))

    def select_n_best(self, n: int, overwrite: bool = True):
        """
        Selects `n` best individuals per population based on the score function value.
        :param n: Number of individuals to take per population.
        :param overwrite: Overwrite current population with the selection.
        :return: A top-n selection.
        """

        def sel_population(pop, recs):
            if recs is None:
                recs = [None] * len(pop)
            sel = sorted(zip(pop, recs), key=lambda x: self.score_func(x[0]), reverse=True)
            return [sel[i][0] for i in range(n)], [sel[i][1] for i in range(n)]

        if self.populations is None:
            raise ValueError('No populations')

        records = [None] * len(self.populations) if self.records is None else self.records

        selections = [sel_population(p, r) for p, r in zip(self.populations, records)]
        populations, records = [x[0] for x in selections], [x[1] for x in selections]
        if self.records is None:
            records = None
        if overwrite:
            self.populations, self.records = populations, records
        return populations, records

    def evolve(self, num_gen: int, overwrite: bool = True,
               callbacks: t.Optional[t.List[Callback]] = None) \
            -> t.Tuple[t.List[t.List[np.ndarray]], t.List[t.List[Record]], t.Optional[t.List[Callback]], t.List[int]]:
        """
        Evolve populations currently held in `populations` attribute.
        Requires `ray` initialized externally to run.
        :param num_gen: Number of generations to evolve each population.
        :param callbacks: Optional callbacks. Internal state will be preserved.
        :param overwrite: Overwrite `populations` and `records` attributes with the results of the run.
        :return: A tuple of
        (1) a list of evolved populations
        (2) a list with lists of individuals' records,
        (3) a list of callbacks applied to a population throughout the simulation, and
        (4) a list with number of generations each population ran
        """
        data_to_evolve = zip(self.populations, self.records or [None] * len(self.populations))
        handles = [
            _evolve_remote.remote(
                num_gen, self._evolver, x[0], x[1], self.ops, self.genetic_params, callbacks)
            for x in data_to_evolve]
        results = []
        bar = tqdm(total=len(self.populations), desc='Evolving populations')
        while handles:
            done, handles = ray.wait(handles)
            results.append(ray.get(done[0]))
            bar.update(1)
        bar.close()
        pops, recs, callbacks, gens = map(list, unzip(results))
        if overwrite:
            self.populations, self.records = pops, recs
        return pops, recs, callbacks, gens

    def evolve_local(self, num_gen: int, overwrite: bool = True,
                     callbacks: t.Optional[t.List[Callback]] = None):
        """
        Evolve populations sequentially, using a single processor.
        No early stopping applied.
        Exists mainly for testing purposes.
        """
        data_to_evolve = zip(self.populations, self.records or [None] * len(self.populations))
        operators = self.ops
        results = []
        for individuals, records in tqdm(data_to_evolve, total=len(self.populations), desc='Evolving population'):
            individuals, records = self._evolver.evolve(
                num_gen, self.ops, self.genetic_params.Population_size, individuals, records)
            if callbacks:
                individuals, records, operators = self._evolver.call_callbacks(
                    callbacks, individuals, records, operators)
            results.append((individuals, records, callbacks))
        populations, records, callbacks = map(list, unzip(results))
        if overwrite:
            self.populations, self.records = populations, records
        return populations, records, callbacks


@ray.remote
def _spawn_remote(genetic_params: GeneticParams, individual_type):
    return [
        individual_type(
            sample(genetic_params.Gene_pool, genetic_params.Individual_base_size),
            genetic_params.Coupling_threshold, genetic_params.Max_mut_space)
        for _ in range(genetic_params.Population_size)]


@ray.remote
def _evolve_remote(
        num_gen: int, evolver: GenericEvolver, individuals: t.List[GenericIndividual],
        records: t.List[t.Optional[Record]], operators: Operators, genetic_params: GeneticParams,
        callbacks: t.Optional[t.List[Callback]]) \
        -> t.Tuple[t.List[GenericIndividual], t.List[Record], t.Optional[t.List[Callback]], int]:
    gen = 0
    counter = 0
    previous = 0
    selector = {
        'max': (lambda recs: max(r.score for r in recs)),
        'mean': (lambda recs: mean(r.score for r in recs))
    }[genetic_params.Early_Stopping.Selector]

    for gen in range(1, num_gen + 1):
        individuals, records = evolver.evolve_generation(
            operators, genetic_params.Population_size, individuals, records)
        if callbacks:
            individuals, records, operators = evolver.call_callbacks(
                callbacks, individuals, records, operators)
        current = selector(records)
        if current - previous < genetic_params.Early_Stopping.ScoreImprovement:
            counter += 1
            if counter >= genetic_params.Early_Stopping.Rounds:
                return individuals, records, callbacks, gen
        else:
            counter = 0
            previous = current
    return individuals, records, callbacks, gen


if __name__ == '__main__':
    raise RuntimeError
