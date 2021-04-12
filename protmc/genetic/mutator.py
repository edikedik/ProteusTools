import typing as t
from itertools import chain, groupby
from random import sample
from warnings import warn

import numpy as np

from .base import EdgeGene
from .individual import GraphIndividual


class Mutator:
    def __init__(self, pool: t.Sequence[EdgeGene], mutation_size: int, deletion_size: int, acquisition_size: int,
                 ps: t.Tuple[float, float, float], copy_individuals: bool = False):
        self.pool = pool
        self.mutation_size = mutation_size
        self.deletion_size = deletion_size
        self.acquisition_size = acquisition_size
        self.ps = ps
        self.copy = copy_individuals

    def mutation(self, individual: GraphIndividual) -> GraphIndividual:
        if self.copy:
            individual = individual.copy()
        del_genes = sample(individual.genes(), self.mutation_size)
        new_genes = sample(self.pool, self.mutation_size)
        return individual.remove_genes(del_genes).add_genes(new_genes)

    def deletion(self, individual: GraphIndividual) -> GraphIndividual:
        if self.copy:
            individual = individual.copy()
        if len(individual) <= self.deletion_size:
            warn(f"Can't delete {self.deletion_size} genes from {len(individual)}-sized GraphIndividual")
            return individual
        del_genes = sample(individual.genes(), self.deletion_size)
        return individual.remove_genes(del_genes)

    def acquisition(self, individual: GraphIndividual) -> GraphIndividual:
        if self.copy:
            individual = individual.copy()
        new_genes = sample(self.pool, self.acquisition_size)
        return individual.add_genes(new_genes)

    def _choose(self):
        return np.random.choice([self.mutation, self.deletion, self.acquisition], p=self.ps)

    def __call__(self, individuals: t.List[GraphIndividual]) -> t.List[GraphIndividual]:
        mutators = [self._choose() for _ in range(len(individuals))]
        return [f(individual) for f, individual in zip(mutators, individuals)]


class BucketMutator:
    def __init__(self, pool: t.Sequence[EdgeGene], mutation_size: int, copy_individuals: bool = False):
        self.mutation_size = mutation_size
        self.pool = pool
        self.copy = copy_individuals
        self.buckets: t.Dict[t.Tuple[int, int], t.Set[EdgeGene]] = dict(self.bucket_genes(pool))

    @staticmethod
    def bucket_genes(genes: t.Iterable[EdgeGene]):
        return ((g, set(gg)) for g, gg in groupby(sorted(genes), lambda x: (x.P1, x.P2)))

    def mutation(self, individual: GraphIndividual) -> GraphIndividual:
        if self.copy:
            individual = individual.copy()
        del_genes = sample(individual.genes(), self.mutation_size)
        new_genes = chain.from_iterable(
            (sample(genes_to_sample, min([len(genes_to_del), len(genes_to_sample)]))
             for genes_to_del, genes_to_sample in (
                 (gg, self.buckets[g]) for g, gg in self.bucket_genes(del_genes))))
        return individual.remove_genes(del_genes, update=False).add_genes(list(new_genes))

    def __call__(self, individuals: t.List[GraphIndividual]) -> t.List[GraphIndividual]:
        return list(map(self.mutation, individuals))


if __name__ == '__main__':
    raise RuntimeError
