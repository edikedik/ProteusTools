from random import sample

import numpy as np
import pytest
from more_itertools import distribute, partition

from ProteusTools.genetic import EdgeGene, GraphIndividual
from ProteusTools.genetic.base import Record
from ProteusTools.genetic.crossover import exchange_fraction, recombine_genes_uniformly, take_unchanged
from ProteusTools.genetic.mutator import Mutator, BucketMutator
from .fixtures import random_graph_genes


def test_mutator(random_genes):
    pool = random_graph_genes
    ind = GraphIndividual(sample(pool, len(pool) // 2))
    num_del = np.random.randint(1, len(ind))
    num_acq = np.random.randint(1, len(ind))
    mut_size = np.random.randint(1, len(ind))
    mutator = Mutator(pool, mut_size, num_del, num_acq, (0.1, 0.45, 0.45), True)
    mutated = mutator.deletion(ind)
    assert len(mutated) == len(ind) - num_del
    mutated = mutator.acquisition(ind)
    assert len(mutated) >= len(ind)
    mutated = mutator.mutation(ind)
    assert len(mutated) <= len(ind)
    assert mutated.validate()
    dozen = [ind.copy() for _ in range(12)]
    mutated = mutator(dozen)
    # This actually doesn't have a guarantee, but it'll most likely pass
    assert any(set(m.genes()) != set(ind.genes()) for m in mutated)
    mutator = Mutator(pool, mut_size, num_del, num_acq, (0.1, 0.45, 0.45), False)
    mutated = mutator.deletion(ind)
    assert set(mutated.genes()) == set(ind.genes())


def test_bucket_mutator():
    genes = [EdgeGene(1, 2, 'A', 'B', 0.1, 0.1), EdgeGene(1, 2, 'A', 'C', 1.0, 0.1)]

    # One-edge individual, one edge in the pool -> replacement
    ind = GraphIndividual([genes[0]])
    assert len(ind.genes()) == 1
    mutator = BucketMutator([genes[1]], 1, True)
    new_ind = mutator([ind])[0]
    assert len(new_ind.genes()) == 1
    assert new_ind.genes()[0] == genes[1]

    # Test correct bucketing behavior
    chars = ['A', 'B', 'C', 'D', 'E']
    large_pool = [
        EdgeGene(np.random.randint(3, 10), np.random.randint(3, 10),
                 np.random.choice(chars), np.random.choice(chars), 1.0, 1.0)
        for _ in range(10)]
    mutator = BucketMutator(large_pool, 1, True)
    with pytest.raises(KeyError):
        #  No buckets under (1, 2) pair of nodes within the pool
        mutator([ind])
    # Again -- replacement -- since there is only one (1, 2) edge in a pool
    mutator = BucketMutator(large_pool + [genes[0]], 1, True)
    new_ind = mutator([ind])[0]
    assert len(new_ind.genes()) == 1
    assert new_ind.genes()[0] == genes[0]


def test_recombine_genes_uniformly():
    genes1 = [EdgeGene(1, 2, 'A', 'B', 0.1, 0.1), EdgeGene(1, 2, 'A', 'C', 1.0, 0.1)]
    genes2 = [EdgeGene(1, 3, 'A', 'B', 0.1, 0.1), EdgeGene(1, 3, 'A', 'C', 1.0, 0.1)]
    ind1 = GraphIndividual(genes1)
    ind2 = GraphIndividual(genes2)
    mated = recombine_genes_uniformly([(ind1, Record(0, 0)), (ind2, Record(0, 0))], 1)
    assert len(mated) == 1
    assert len(mated[0].genes()) == 2
    mated = recombine_genes_uniformly([(ind1, Record(0, 0)), (ind2, Record(0, 0))], 2)
    assert len(mated) == 2
    assert set(genes1 + genes2) == set(mated[0].genes() + mated[1].genes())


def test_exchange_fraction(random_genes):
    g1, g2, g3, g4 = (
        EdgeGene(1, 2, 'A', 'B', 1, 1), EdgeGene(1, 2, 'A', 'C', 1, 1),
        EdgeGene(3, 4, 'A', 'B', 1, 1), EdgeGene(3, 4, 'A', 'C', 1, 1))
    ind1 = GraphIndividual([g1])
    res = exchange_fraction([(ind1, Record(0, 0))], 1, 1.0)
    assert len(res) == 1
    ind1_ = res[0]
    assert set(ind1_.genes()) == {g1}
    ind2 = GraphIndividual([g1])
    ind1_, ind2_ = exchange_fraction([(ind1, Record(0, 0)), (ind2, Record(0, 0))], 2, 1.0)
    assert len(ind1_) == len(ind2_) == 1
    assert set(ind1_.genes()) == set(ind2_.genes()) == {g1}
    ind2 = GraphIndividual([g2])
    ind1_, ind2_ = exchange_fraction([(ind1, Record(0, 0)), (ind2, Record(0, 0))], 2, 0.9)
    assert set(ind1_.genes()) == {g1} and set(ind2_.genes()) == {g2}
    ind1 = GraphIndividual([g1, g2])
    ind2 = GraphIndividual([g3, g4])
    res = exchange_fraction([(ind1, Record(0, 0)), (ind2, Record(0, 0))], 2, 0.51)
    assert len(res) == 2
    ind1_, ind2_ = res
    assert len(ind1_) == len(ind2_) == 2

    genes1, genes2 = distribute(2, random_graph_genes)
    ind1, ind2 = GraphIndividual(genes1), GraphIndividual(genes2)
    res = exchange_fraction([(ind1, Record(0, 0)), (ind2, Record(0, 0))], 1, 0.5)
    assert len(res) == 1
    ind = res[0]
    old_genes, new_genes = map(list, partition(lambda g: g in genes1, ind.genes()))
    assert len(old_genes) >= len(new_genes)
    assert set(new_genes).issubset(genes2)


def test_take_unchanged(random_genes):
    l = len(random_graph_genes) // 2
    ind1 = GraphIndividual(random_graph_genes[:l])
    ind2 = GraphIndividual(random_graph_genes[l:])
    res = take_unchanged([(ind1, Record(0, 0)), (ind2, Record(0, 0))], brood_size=1)
    assert len(res) == 1
    ind = res[0]
    assert set(ind.genes()) == set(ind1.genes()) or set(ind.genes()) == set(ind2.genes())
    res = take_unchanged([(ind1, Record(0, 0)), (ind2, Record(0, 0))], brood_size=2)
    assert len(res) == 2
    ind1_, ind2_ = res
    assert set(ind1.genes()) == set(ind1_.genes()) or set(ind1.genes()) == set(ind2_.genes())
    assert set(ind2.genes()) == set(ind1_.genes()) or set(ind2.genes()) == set(ind2_.genes())
