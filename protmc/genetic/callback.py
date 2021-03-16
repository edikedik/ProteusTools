import operator as op
import typing as t
from functools import reduce
from itertools import chain, tee
from statistics import mean

from genetic.base import Operators, Callback

from .base import Record
from .individual import GenericIndividual


class Accumulator(Callback):
    def __init__(self, freq: int):
        self.freq = freq
        self.generation = 0
        self.acc = []

    def __call__(self, individuals, records, operators):
        raise NotImplementedError


class ProgressSaver(Accumulator):

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


class PopulationGeneCounter(Accumulator):
    """Collects the tuples of the form: (generation, num_genes, num_new_genes, num_old_genes)"""

    def __init__(self, freq: int):
        super().__init__(freq)
        self.last_genes = set()

    def __call__(self, individuals: t.List[GenericIndividual], records: t.List[Record], operators: Operators) \
            -> t.Tuple[t.List[GenericIndividual], t.List[Record], Operators]:
        self.generation += 1
        if self.generation % self.freq == 0:
            current_genes = set(chain.from_iterable(ind.genes() for ind in individuals))
            self.acc.append((
                self.generation, len(current_genes),
                len(current_genes - self.last_genes),
                len(self.last_genes - current_genes)))
            self.last_genes = current_genes
        return individuals, records, operators


class UniqueGeneCounter(Accumulator):

    def __call__(self, individuals: t.List[GenericIndividual], records: t.List[Record], operators: Operators) \
            -> t.Tuple[t.List[GenericIndividual], t.List[Record], Operators]:
        self.generation += 1
        if self.generation % self.freq == 0:
            ind_genes = [set(ind.genes()) for ind in individuals]
            # Compare against the whole population
            unique_against_population = (
                len(g - reduce(op.or_, chain(ind_genes[:i], ind_genes[i + 1:])))
                for i, g in enumerate(ind_genes)
            )
            u1, u2 = tee(unique_against_population, 2)
            mean_genes_pop = mean(u1)
            mean_genes_pop_percent = mean(g / len(ind) * 100 for g, ind in zip(u2, individuals))
            # Compare against all individuals
            unique_against_individuals = (
                mean(len(g - o) for o in chain(ind_genes[:i], ind_genes[i + 1:]))
                for i, g in enumerate(ind_genes)
            )
            u1, u2 = tee(unique_against_individuals, 2)
            mean_genes_ind = mean(u1)
            mean_genes_ind_percent = mean(g / len(ind) * 100 for g, ind in zip(u2, individuals))

            self.acc.append(
                (self.generation,
                 mean_genes_pop, mean_genes_pop_percent,
                 mean_genes_ind, mean_genes_ind_percent))
        return individuals, records, operators


if __name__ == '__main__':
    raise RuntimeError
