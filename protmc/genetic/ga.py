import typing as t
from copy import deepcopy
from functools import partial
from itertools import chain
from math import floor
from random import sample
from statistics import mean
from warnings import warn

import genetic.operators as ops
import numpy as np
import ray
from genetic.base import Operators, Callback
from genetic.evolvers import GenericEvolver
from more_itertools import random_permutation, distribute, unzip
from tqdm import tqdm

from .base import Gene, GeneticParams, Record
from .individual import GenericIndividual
from .score import score


def recombine_fractions(
        mating_group: t.List[t.Tuple[GenericIndividual, Record]], brood_size: int,
        min_donate: float = 0.01, max_donate: float = 0.5):
    def donate_genes(indiv: GenericIndividual):
        indiv = deepcopy(indiv)
        indiv_size = len(indiv)
        num_donate = np.random.randint(
            np.floor(min_donate * indiv_size), np.floor(max_donate * indiv_size))
        donated = sample(indiv.genes(), num_donate)
        indiv.remove_genes(donated, update=False)
        return indiv, donated

    def accept_donation(indiv: GenericIndividual, donation: t.Iterable[Gene]):
        indiv.add_genes(donation)
        return indiv

    individuals, donations = unzip(donate_genes(indiv) for indiv, _ in mating_group)
    pool = random_permutation(chain(donations))
    chunks = filter(bool, distribute(len(mating_group), pool))
    return [accept_donation(indiv, chunk) for indiv, chunk in zip(individuals, chunks)]


def recombine_genes_uniformly(mating_group: t.List[t.Tuple[GenericIndividual, Record]], brood_size: int):
    ts = set(indiv.coupling_threshold for indiv, _ in mating_group)
    if len(ts) > 1:
        raise RuntimeError('...')
    coupling_threshold = ts.pop()
    pool = random_permutation(chain.from_iterable(indiv.genes() for indiv, _ in mating_group))
    chunks = distribute(brood_size, pool)
    return [GenericIndividual(genes, coupling_threshold) for genes in chunks]


class Mutator:
    def __init__(self, pool: t.Sequence[Gene], mutable_fraction: float, deletion_size: int, acquisition_size: int,
                 ps: t.Tuple[float, float, float]):
        self.pool = pool
        self.mutable_fraction = mutable_fraction
        self.deletion_size = deletion_size
        self.acquisition_size = acquisition_size
        self.ps = ps

    def mutation(self, individual: GenericIndividual) -> GenericIndividual:
        num_mut = floor(len(individual) * self.mutable_fraction)
        if not num_mut:
            return individual
        del_genes = sample(individual.genes(), num_mut)
        new_genes = sample(self.pool, num_mut)
        return individual.remove_genes(del_genes, update=False).add_genes(new_genes)

    def deletion(self, individual: GenericIndividual) -> GenericIndividual:
        if len(individual) <= self.deletion_size:
            warn(f"Can't delete {self.deletion_size} genes from {len(individual)}-sized GenericIndividual")
            return individual
        del_genes = sample(individual.genes(), self.deletion_size)
        return individual.remove_genes(del_genes)

    def acquisition(self, individual: GenericIndividual) -> GenericIndividual:
        new_genes = sample(self.pool, self.acquisition_size)
        return individual.add_genes(new_genes)

    def _choose(self):
        return np.random.choice([self.mutation, self.deletion, self.acquisition], p=self.ps)

    def __call__(self, individuals: t.List[GenericIndividual]) -> t.List[GenericIndividual]:
        mutators = [self._choose() for _ in range(len(individuals))]
        return [f(individual) for f, individual in zip(mutators, individuals)]


class ProgressSaver(Callback):
    def __init__(self, freq: int):
        self.freq = freq
        self.generation = 0
        self.acc = []

    def __call__(self, individuals: t.List[GenericIndividual], records: t.List[Record], operators: Operators) \
            -> t.Tuple[t.List[GenericIndividual], t.List[Record], Operators]:
        self.generation += 1
        if self.generation % self.freq == 0:
            scores = [r.score for r in records]
            self.acc.append((
                self.generation,
                mean(scores),
                max(scores)))
        return individuals, records, operators


class GA:
    def __init__(self, genetic_params: GeneticParams):
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
        self._mutator = Mutator(
            genetic_params.Gene_pool, genetic_params.Mutable_fraction,
            genetic_params.Deletion_size, genetic_params.Acquisition_size,
            ps=genetic_params.Probabilities)
        self._policy = ops.GenericPolicy(selector=self.policy_fn)
        self._evolver = GenericEvolver()
        self.ops = Operators(self._recorder, self._estimator, self._selector,
                             self._crossover, self._mutator, self._policy)

        # Setup placeholders
        self.populations: t.Optional[t.List[t.List[np.ndarray]]] = None
        self.records: t.Optional[t.List[t.List[Record]]] = None

    def spawn_population(self, individual_type=GenericIndividual) -> t.List[GenericIndividual]:
        return [
            individual_type(
                sample(self.genetic_params.Gene_pool, self.genetic_params.Individual_base_size),
                self.genetic_params.Coupling_threshold)
            for _ in range(self.genetic_params.Population_size)]

    def spawn_populations(self, n: int, overwrite: bool = True, individual_type=GenericIndividual):
        populations = [self.spawn_population(individual_type) for _ in range(n)]
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
        :param num_gen: Number of generations to evolve each population.
        :param callbacks: Optional callbacks. Internal state won't be preserved.
        Not really usable as of now.
        :param overwrite: Overwrite `populations` and `records` attributes with the results of the run.
        :return: A tuple with (1) a list of evolved populations
        (where each population is a list of individuals, i.e. numpy arrays),
        and (2) a list with lists of individuals' records.
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


@ray.remote
def _evolve_remote(
        num_gen: int, evolver: GenericEvolver, individuals: t.List[GenericIndividual],
        records: t.List[t.Optional[Record]], operators: Operators, genetic_params: GeneticParams,
        callbacks: t.Optional[t.List[Callback]]) \
        -> t.Tuple[t.List[GenericIndividual], t.List[Record], t.Optional[t.List[Callback]], int]:
    gen = 0
    stopping_counter = 0
    stopping_previous = 0
    stopping_func = {
        'max': (lambda recs: max(r.score for r in recs)),
        'mean': (lambda recs: mean(r.score for r in recs))
    }[genetic_params.Early_Stopping.Selector]

    for gen in range(1, num_gen + 1):
        individuals, records = evolver.evolve_generation(
            operators, genetic_params.Population_size, individuals, records)
        if callbacks:
            individuals, records, operators = evolver.call_callbacks(
                callbacks, individuals, records, operators)
        stopping_current = stopping_func(records)
        if abs(stopping_current - stopping_previous) < genetic_params.Early_Stopping.ScoreImprovement:
            stopping_counter += 1
            if stopping_counter >= genetic_params.Early_Stopping.Rounds:
                return individuals, records, callbacks, gen
        else:
            stopping_counter = 0
            stopping_previous = stopping_current
    return individuals, records, callbacks, gen


if __name__ == '__main__':
    raise RuntimeError
