import typing as t
from itertools import chain, groupby
from math import floor
from random import sample
from warnings import warn

import numpy as np

from .base import Gene
from .individual import GenericIndividual


class Mutator:
    def __init__(self, pool: t.Sequence[Gene], mutable_fraction: float, deletion_size: int, acquisition_size: int,
                 ps: t.Tuple[float, float, float], copy_individuals: bool = False):
        self.pool = pool
        self.mutable_fraction = mutable_fraction
        self.deletion_size = deletion_size
        self.acquisition_size = acquisition_size
        self.ps = ps
        self.copy = copy_individuals

    def mutation(self, individual: GenericIndividual) -> GenericIndividual:
        if self.copy:
            individual = individual.copy()
        num_mut = floor(len(individual) * self.mutable_fraction)
        if not num_mut:
            return individual
        del_genes = sample(individual.genes(), num_mut)
        new_genes = sample(self.pool, num_mut)
        return individual.remove_genes(del_genes, update=False).add_genes(new_genes)

    def deletion(self, individual: GenericIndividual) -> GenericIndividual:
        if self.copy:
            individual = individual.copy()
        if len(individual) <= self.deletion_size:
            warn(f"Can't delete {self.deletion_size} genes from {len(individual)}-sized GenericIndividual")
            return individual
        del_genes = sample(individual.genes(), self.deletion_size)
        return individual.remove_genes(del_genes)

    def acquisition(self, individual: GenericIndividual) -> GenericIndividual:
        if self.copy:
            individual = individual.copy()
        new_genes = sample(self.pool, self.acquisition_size)
        return individual.add_genes(new_genes)

    def _choose(self):
        return np.random.choice([self.mutation, self.deletion, self.acquisition], p=self.ps)

    def __call__(self, individuals: t.List[GenericIndividual]) -> t.List[GenericIndividual]:
        mutators = [self._choose() for _ in range(len(individuals))]
        return [f(individual) for f, individual in zip(mutators, individuals)]


class BucketMutator:
    def __init__(self, pool: t.Sequence[Gene], mutable_fraction: float, copy_individuals: bool = False):
        self.mutable_fraction = mutable_fraction
        self.pool = pool
        self.copy = copy_individuals
        self.buckets: t.Dict[t.Tuple[int, int], t.Set[Gene]] = dict(self.bucket_genes(pool))

    @staticmethod
    def bucket_genes(genes: t.Iterable[Gene]):
        return ((g, set(gg)) for g, gg in groupby(sorted(genes), lambda x: (x.P1, x.P2)))

    def mutation(self, individual: GenericIndividual) -> GenericIndividual:
        if self.copy:
            individual = individual.copy()
        num_mut = floor(len(individual) * self.mutable_fraction)
        if not num_mut:
            return individual
        del_genes = sample(individual.genes(), num_mut)
        new_genes = chain.from_iterable(
            (sample(genes_to_sample, min([len(genes_to_del), len(genes_to_sample)]))
             for genes_to_del, genes_to_sample in (
                 (gg, self.buckets[g]) for g, gg in self.bucket_genes(del_genes))))
        return individual.remove_genes(del_genes, update=False).add_genes(list(new_genes))

    def __call__(self, individuals: t.List[GenericIndividual]) -> t.List[GenericIndividual]:
        return list(map(self.mutation, individuals))


if __name__ == '__main__':
    raise RuntimeError
