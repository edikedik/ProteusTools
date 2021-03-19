from itertools import groupby
from math import log
from statistics import median

import numpy as np
import pytest

from protmc.genetic.base import Gene, GenePool
from protmc.genetic.individual import GenericIndividual, AverageIndividual


def generate_genes(
        n_min=50, n_max=200, ps_min=1, ps_max=10,
        coupling_min=0.1, coupling_max=1.0,
        score_min=0.1, score_max=1.0) -> GenePool:
    types = ['A', 'B', 'C', 'D', 'E', 'F']
    n = np.random.randint(n_min, n_max)
    ps = np.random.randint(ps_min, ps_max, size=(n, 2))
    ts = np.random.choice(types, size=(n, 2))
    return list(
        Gene(p[0], p[1], a[0], a[1],
             np.random.uniform(score_min, score_max),
             np.random.uniform(coupling_min, coupling_max))
        for p, a in zip(ps, ts) if p[0] <= p[1])


@pytest.fixture()
def random_genes():
    return generate_genes()


@pytest.fixture()
def random_individual() -> GenericIndividual:
    return GenericIndividual(generate_genes())


@pytest.fixture()
def strongly_linked_individual() -> GenericIndividual:
    threshold = np.random.uniform(0.1, 0.9)
    return GenericIndividual(generate_genes(coupling_min=threshold + 0.05), coupling_threshold=threshold)


@pytest.fixture()
def weakly_linked_individual() -> GenericIndividual:
    threshold = np.random.uniform(0.1, 0.9)
    return GenericIndividual(generate_genes(coupling_max=threshold), coupling_threshold=threshold)


def test_basic_init(random_genes):
    individual = GenericIndividual(random_genes)
    # since coupling threshold is zero, all genes must be "strong"
    gene_set = set((g.P1, g.P2, g.A1 + g.A2) for g in random_genes)
    assert len(individual.genes()) == len(gene_set)
    assert len(list(individual.strong_links())) == len(gene_set)
    assert len(list(individual.weak_links())) == 0

    coupling_threshold = median(g.C for g in random_genes)
    strong_genes = set((g.P1, g.P2, g.A1 + g.A2) for g in random_genes if g.C >= coupling_threshold)
    individual = GenericIndividual(random_genes, coupling_threshold)
    assert len(individual.genes()) != len(gene_set)
    assert len(list(individual.strong_links())) != len(gene_set)
    assert len(list(individual.weak_links())) != 0
    # all of the strong genes are present
    assert len(set(individual.strong_links())) == len(strong_genes)

    edges = sorted(individual.graph.edges)
    groups = groupby(edges, lambda e: e[:2])
    num_genes = 0
    for _, gg in groups:
        group = list(gg)
        is_weak_group = all(individual.graph.edges[e]['gene'].C < coupling_threshold for e in group)
        is_strong_group = all(individual.graph.edges[e]['gene'].C >= coupling_threshold for e in group)
        assert is_weak_group or is_strong_group
        if is_weak_group:
            assert len(group) == 1
        num_genes += len(group)
    assert len(individual.genes()) == num_genes

    assert individual.validate() is True

    g = random_genes[np.random.randint(0, len(random_genes))]
    g = Gene(g.P1, g.P2, 'X', 'Y', g.S + 0.1, coupling_threshold - 0.1)
    individual._graph.add_edge(g.P1, g.P2, g.A1 + g.A2, gene=g)
    assert individual.validate() is False


def test_score(random_genes):
    # Remove redundancy
    edges, genes = [], []
    for g in random_genes:
        e = (g.P1, g.P2, g.A1 + g.A2)
        if e not in edges:
            edges.append(e)
            genes.append(g)
    genes = sorted(genes)
    # No threshold
    individual = GenericIndividual(genes, 0)
    assert individual.score == round(sum(g.S for g in genes), 4)

    # Introduce threshold
    coupling_threshold = median(g.C for g in genes)
    strong_genes_score = round(sum(g.S for g in genes if g.C >= coupling_threshold), 4)
    individual = GenericIndividual(genes, coupling_threshold)
    strong_genes_score_ = round(sum(individual.graph.edges[e]['gene'].S for e in individual.strong_links()), 4)
    assert strong_genes_score == strong_genes_score_
    score = individual.score
    for e in individual.weak_links():
        score -= individual.graph.edges[e]['gene'].S
    assert round(score, 3) == round(strong_genes_score_, 3)

    # Test average scoring scheme
    genes = [Gene(1, 2, 'A', 'B', 0.5, 0), Gene(1, 2, 'A', 'C', 1.5, 0.5), Gene(2, 3, 'A', 'B', 0.5, 0.5)]
    individual = AverageIndividual(genes)
    assert individual.score == 1.5
    individual = AverageIndividual(genes, 0.6)
    assert individual.score == 1.0


def test_mut_space():
    genes = [Gene(1, 2, 'A', 'B', 0.5, 1.0), Gene(1, 2, 'A', 'C', 1.5, 1.0),
             Gene(2, 3, 'A', 'B', 0.5, 0.1), Gene(3, 4, 'B', 'B', 0.5, 1.0)]
    # 1 - A, 2 - ABC, 3 - B, 4 - B
    individual = GenericIndividual(genes)
    assert individual.mut_space_size == 3 * log(1) + log(3)
    # 1 - A, 2 - BC, 3 - B, 4 - B
    individual = GenericIndividual(genes, 0.2)
    assert individual.mut_space_size == 3 * log(1) + log(2)
    individual = GenericIndividual(genes, 10)
    assert individual.mut_space_size == 0
    # Adding a weak link into one of the induced subgraphs does not change the outcome
    individual = GenericIndividual(genes + [Gene(1, 2, 'A', 'B', 0.5, 0.1)], 0.2)
    assert individual.mut_space_size == 3 * log(1) + log(2)


def test_add_genes():
    # No coupling threshold
    genes = [Gene(1, 2, 'A', 'B', 0.5, 1.0), Gene(1, 2, 'A', 'C', 1.5, 1.0), Gene(2, 3, 'A', 'B', 0.5, 0.1)]
    scores = sum(g.S for g in genes)
    individual = GenericIndividual(genes)
    # Such edge exists -> nothing added
    individual.add_gene(Gene(1, 2, 'A', 'B', 100, 100))
    assert individual.score == scores
    # Such edge does not exist (different key) -> add edge
    individual.add_gene(Gene(1, 2, 'B', 'A', 1.0, 1.0))
    assert len(individual.genes()) == 4
    # Without explicit update after calling `add_gene`
    # the scores are the same
    assert individual.score == scores
    individual.upd()
    assert individual.score == scores + 1
    # These are different edges due to a key (BC != CB)
    individual.add_genes([Gene(3, 4, 'B', 'C', 1, 1), Gene(4, 3, 'C', 'B', 1, 1)])
    assert len(individual.genes()) == 6
    assert individual.score == scores + 3

    # Introduce coupling threshold
    individual = GenericIndividual(genes, 0.2)
    # Both edges sequentially replace the weak edge
    individual.add_genes([Gene(2, 3, 'A', 'C', 1.0, 0.1), Gene(2, 3, 'A', 'D', 1.0, 1.0)])
    # -> the total number of edges remains the same
    assert len(individual.genes()) == 3
    # -> only one edge -- AD -- between 2 and 3
    edges = list(individual.graph[2][3])
    assert len(edges) == 1
    assert edges[0] == 'AD'
    # Adding one more strong edge is possible
    individual.add_genes([Gene(2, 3, 'A', 'C', 1.0, 1.0)])
    edges = list(individual.graph[2][3])
    assert len(edges) == 2
    assert 'AC' in edges and 'AD' in edges

    # Only the strong edge is added
    individual.add_genes([Gene(1, 2, 'A', 'D', 1.0, 1.0), Gene(1, 2, 'A', 'E', 1.0, 0.1)])
    edges = list(individual.graph[1][2])
    assert len(edges) == 3
    assert 'AD' in edges
    assert 'AE' not in edges
