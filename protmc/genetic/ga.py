import typing as t
from functools import partial
from itertools import chain
from random import sample
from statistics import mean

import genetic.operators as ops
import numpy as np
import ray
from genetic.base import Operators, Callback
from genetic.evolvers import GenericEvolver
from more_itertools import unzip
from tqdm import tqdm

from .base import GeneticParams, Record, EdgeGene
from .crossover import recombine_genes_uniformly, take_unchanged
from .individual import GraphIndividual, SeqIndividual, Individual, _GraphIndividual
from .mutator import Mutator
from .score import score


class GA:
    """
    Abstractly defined genetic algorithm. The core workflow is based on the `genetic` package.
    Based on the provided params, sets up the following operators: `Estimator` (estimating fitness),
    `Recorder` (managing individuals' records), `Selector` (selecting individuals for mating),
    `Crossover` (combining individuals' genes), `Mutator` (mutating newborn individuals;
    the whole operator is redefined for this version of the GA),
    `Policy` (selecting individuals for the next generation), and `Evolver` (defining the
    GA's workflow and checking operators' validity). Use `ops` attribute to access initialized operators.

    A few things are hard-coded: (1) type of the selection strategy for `Selector` and `Policy` operators
    is `ktournament`, (2) the fitness is assumed to be direct (i.e., better fitness value is better),
    (3) `Record` used to keep individuals' records is a minimalistic namedtuple object with two fields:
    age and score, (4) score is a `score` function defined in eponymous module (the actual score (fitness)
    computation is assigned to individuals).
    """

    def __init__(self, genetic_params: GeneticParams,
                 populations: t.Optional[t.List[t.List[Individual]]] = None,
                 records: t.Optional[t.List[t.List[Record]]] = None):
        """
        :param genetic_params: A dataclass holding parameters of the GA.
        :param populations: An optional list of populations, where each population is a list of Individuals.
        Can be provided afterwards.
        :param records: For each population, a list of individuals' records.
        """
        crossovers = {'recombine_genes_uniformly': recombine_genes_uniformly, 'take_unchanged': take_unchanged}
        self.genetic_params = genetic_params

        # Setup helpers for operators
        self.score_func = partial(
            score,
            **genetic_params.Score_kwargs)
        self.selector_fn = partial(
            ops.ktournament, genetic_params.Tournaments_selection, lambda x: x.score,
            genetic_params.Number_of_mates, replace=True)
        self.policy_fn = partial(ops.ktournament, genetic_params.Tournaments_policy, lambda x: x.score)
        self.crossover_fn = partial(crossovers[genetic_params.Crossover_mode], brood_size=genetic_params.Brood_size)

        # Setup operators
        self._estimator = ops.GenericEstimator(func=self.score_func)
        self._recorder = ops.GenericRecorder(
            start=lambda ind, score_: Record(0, score_),
            update=lambda ind, rec: Record(rec.age + 1, rec.score))
        self._selector = ops.GenericSelector(
            selector=self.selector_fn,
            nmates=genetic_params.Number_of_mates)
        self._crossover = ops.GenericCrossover(
            crossover=self.crossover_fn,
            nmates=genetic_params.Number_of_mates,
            broodsize=genetic_params.Brood_size)
        self._mutator = Mutator(
            genetic_params.Gene_pool, genetic_params.Mutation_size,
            genetic_params.Deletion_size, genetic_params.Acquisition_size,
            ps=genetic_params.Probabilities)
        self._policy = ops.GenericPolicy(selector=self.policy_fn)
        self._evolver = GenericEvolver()
        self.ops = Operators(self._recorder, self._estimator, self._selector,
                             self._crossover, self._mutator, self._policy)

        # Setup placeholders
        self.populations = populations
        self.records = records

    def flatten(self) -> t.Tuple[t.List[Individual], t.List[Record]]:
        """Flatten population into and records into lists."""
        if self.populations is None:
            raise ValueError('No populations')
        return (list(chain.from_iterable(self.populations)),
                list(chain.from_iterable(self.records)) if self.records else [None] * len(self.populations))

    def select_n_best(self, n: int, overwrite: bool = True):
        """
        Selects `n` best individuals per population based on the score function value.
        :param n: Number of individuals to take per population.
        :param overwrite: Overwrite current populations with the selection.
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
        Each population is evolved in a separate process.
        Single process can be used via `evolve_local` method.
        :param num_gen: Number of generations to evolve each population.
        :param callbacks: Optional callbacks. Internal state will be preserved.
        Applied separately to each evolving population following `Policy` operator.
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


def spawn_graph_populations(
        n: int, genetic_params: GeneticParams, pool: t.Optional[t.Collection[EdgeGene]] = None,
        individual_type: _GraphIndividual = GraphIndividual) -> t.List[t.List[_GraphIndividual]]:
    """
    Spawn `n` populations of `GraphIndividual`s with genes randomly sampled from `pool` without replacement.
    If `genetic_params.GenePool` attribute is empty, the provided `pool` will be used.
    """
    return [spawn_graph_individual(genetic_params, individual_type, pool) for _ in tqdm(
        range(n), total=n, desc='Spawning populations')]


def spawn_graph_individual(genetic_params: GeneticParams, individual_type, pool=None):
    """Spawn a single individual by randomly sampling genes from either
    `genetic_params.Gene_pool` or the provided `pool`."""
    return [
        individual_type(
            sample(pool or genetic_params.Gene_pool, genetic_params.Individual_base_size),
            genetic_params.Coupling_threshold, genetic_params.Max_mut_space, genetic_params.Max_num_pos)
        for _ in range(genetic_params.Population_size)]


def spawn_seq_populations(n: int, genetic_params: GeneticParams):
    """Spawn `n` populations of `SeqIndividual`s by randomly sampling genes
    from the `genetic_params.Gene_pool`."""
    return [spawn_seq_individual(genetic_params) for _ in range(n)]


def spawn_seq_individual(genetic_params: GeneticParams):
    return [SeqIndividual(sample(genetic_params.Gene_pool, genetic_params.Individual_base_size))
            for _ in range(genetic_params.Population_size)]


@ray.remote
def _evolve_remote(
        num_gen: int, evolver: GenericEvolver, individuals: t.List[GraphIndividual],
        records: t.List[t.Optional[Record]], operators: Operators, genetic_params: GeneticParams,
        callbacks: t.Optional[t.List[Callback]]) \
        -> t.Tuple[t.List[GraphIndividual], t.List[Record], t.Optional[t.List[Callback]], int]:
    """A helper function to evolve a population of `individuals` in parallel."""
    gen = 0
    counter = 0
    previous = 0
    selector = {
        'max': (lambda recs: max(r.score for r in recs)),
        'mean': (lambda recs: mean(r.score for r in recs))
    }[genetic_params.Early_stopping.Selector]

    # For each generation
    for gen in range(1, num_gen + 1):
        # Evolve generation
        individuals, records = evolver.evolve_generation(
            operators, genetic_params.Population_size, individuals, records)
        # Apply callbacks, if any
        if callbacks:
            individuals, records, operators = evolver.call_callbacks(
                callbacks, individuals, records, operators)
        # Either terminate based on early stopping criteria or increment and continue
        current = selector(records)
        if current - previous < genetic_params.Early_stopping.ScoreImprovement:
            counter += 1
            if counter >= genetic_params.Early_stopping.Rounds:
                return individuals, records, callbacks, gen
        else:
            counter = 0
            previous = current
    return individuals, records, callbacks, gen


if __name__ == '__main__':
    raise RuntimeError
