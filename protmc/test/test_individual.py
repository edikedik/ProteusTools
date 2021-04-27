import operator as op
from functools import reduce
from itertools import groupby
from math import log
from statistics import median

import numpy as np

from protmc.genetic.base import EdgeGene, SeqGene
from protmc.genetic.individual import GraphIndividual, AverageIndividual, SeqIndividual
from .fixtures import random_graph_genes, random_seq_genes


def test_basic_init(random_graph_genes):
    individual = GraphIndividual(random_graph_genes)
    # since coupling threshold is zero, all genes must be "strong"
    gene_set = set((g.P1, g.P2, g.A1 + g.A2) for g in random_graph_genes)
    assert len(individual.genes()) == len(gene_set)
    assert len(list(individual.strong_links())) == len(gene_set)
    assert len(list(individual.weak_links())) == 0

    coupling_threshold = median(g.C for g in random_graph_genes)
    strong_genes = set((g.P1, g.P2, g.A1 + g.A2) for g in random_graph_genes if g.C >= coupling_threshold)
    individual = GraphIndividual(random_graph_genes, coupling_threshold)
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

    g = random_graph_genes[np.random.randint(0, len(random_graph_genes))]
    g = EdgeGene(g.P1, g.P2, 'X', 'Y', g.S + 0.1, coupling_threshold - 0.1)
    individual._graph.add_edge(g.P1, g.P2, g.A1 + g.A2, gene=g)
    assert individual.validate() is False


def test_score(random_graph_genes):
    # Remove redundancy
    edges, genes = [], []
    for g in random_graph_genes:
        e = (g.P1, g.P2, g.A1 + g.A2)
        if e not in edges:
            edges.append(e)
            genes.append(g)
    genes = sorted(genes)
    # No threshold
    individual = GraphIndividual(genes, 0)
    assert individual.score == round(sum(g.S for g in genes), 4)

    # Introduce threshold
    coupling_threshold = median(g.C for g in genes)
    strong_genes_score = round(sum(g.S for g in genes if g.C >= coupling_threshold), 4)
    individual = GraphIndividual(genes, coupling_threshold)
    strong_genes_score_ = round(sum(individual.graph.edges[e]['gene'].S for e in individual.strong_links()), 4)
    assert strong_genes_score == strong_genes_score_
    score = individual.score
    for e in individual.weak_links():
        score -= individual.graph.edges[e]['gene'].S
    assert round(score, 3) == round(strong_genes_score_, 3)

    # Test average scoring scheme
    genes = [EdgeGene(1, 2, 'A', 'B', 0.5, 0), EdgeGene(1, 2, 'A', 'C', 1.5, 0.5), EdgeGene(2, 3, 'A', 'B', 0.5, 0.5)]
    individual = AverageIndividual(genes)
    assert individual.score == 1.5
    individual = AverageIndividual(genes, 0.6)
    assert individual.score == 1.0


def test_mut_space():
    genes = [EdgeGene(1, 2, 'A', 'B', 0.5, 1.0), EdgeGene(1, 2, 'A', 'C', 1.5, 1.0),
             EdgeGene(2, 3, 'A', 'B', 0.5, 0.1), EdgeGene(3, 4, 'B', 'B', 0.5, 1.0)]
    # 1 - A, 2 - ABC, 3 - B, 4 - B
    individual = GraphIndividual(genes)
    assert individual.mut_space_size == 3 * log(1) + log(3)
    # 1 - A, 2 - BC, 3 - B, 4 - B
    individual = GraphIndividual(genes, 0.2)
    assert individual.mut_space_size == 3 * log(1) + log(2)
    individual = GraphIndividual(genes, 10)
    assert individual.mut_space_size == 0
    # Adding a weak link into one of the induced subgraphs does not change the outcome
    individual = GraphIndividual(genes + [EdgeGene(1, 2, 'A', 'B', 0.5, 0.1)], 0.2)
    assert individual.mut_space_size == 3 * log(1) + log(2)


def test_add_genes():
    # No coupling threshold
    genes = [EdgeGene(1, 2, 'A', 'B', 0.5, 1.0), EdgeGene(1, 2, 'A', 'C', 1.5, 1.0), EdgeGene(2, 3, 'A', 'B', 0.5, 0.1)]
    scores = sum(g.S for g in genes)
    individual = GraphIndividual(genes)
    # Such edge exists -> nothing added
    print(individual.genes())
    individual._add_gene(EdgeGene(1, 2, 'A', 'B', 100, 100))
    print(individual.genes())
    assert len(individual.genes()) == 3
    assert individual.score == scores
    # Such edge does not exist (different key) -> add edge
    individual._add_gene(EdgeGene(1, 2, 'B', 'A', 1.0, 1.0))
    assert len(individual.genes()) == 4
    # Without explicit update after calling `add_gene`
    # the scores are the same
    assert individual.score == scores
    individual.upd()
    assert individual.score == scores + 1
    # These are different edges due to a key (BC != CB)
    individual.add_genes([EdgeGene(3, 4, 'B', 'C', 1, 1), EdgeGene(4, 3, 'C', 'B', 1, 1)])
    assert len(individual.genes()) == 6
    assert individual.score == scores + 3

    # Introduce coupling threshold
    individual = GraphIndividual(genes, 0.2)
    # Both edges sequentially replace the weak edge
    individual.add_genes([EdgeGene(2, 3, 'A', 'C', 1.0, 0.1), EdgeGene(2, 3, 'A', 'D', 1.0, 1.0)])
    # -> the total number of edges remains the same
    assert len(individual.genes()) == 3
    # -> only one edge -- AD -- between 2 and 3
    edges = list(individual.graph[2][3])
    assert len(edges) == 1
    assert edges[0] == 'AD'
    # Adding one more strong edge is possible
    individual.add_genes([EdgeGene(2, 3, 'A', 'C', 1.0, 1.0)])
    edges = list(individual.graph[2][3])
    assert len(edges) == 2
    assert 'AC' in edges and 'AD' in edges

    # Only the strong edge is added
    individual.add_genes([EdgeGene(1, 2, 'A', 'D', 1.0, 1.0), EdgeGene(1, 2, 'A', 'E', 1.0, 0.1)])
    edges = list(individual.graph[1][2])
    assert len(edges) == 3
    assert 'AD' in edges
    assert 'AE' not in edges

    # Test node space constraints used when adding weak edge
    g1, g2, g3 = EdgeGene(1, 2, 'A', 'B', 1.0, 1.0), EdgeGene(2, 3, 'A', 'C', 1.0, 0.1), EdgeGene(2, 3, 'B', 'C', 1.0,
                                                                                                  0.1)
    individual = GraphIndividual([g1], 0.5)
    individual._add_gene(g2)
    # No B for node 2 -> nothing happens
    assert len(individual.genes()) == 1
    assert individual.genes()[0] == g1
    # B exists for node 2 -> addition
    individual._add_gene(g3)
    assert len(individual.genes()) == 2
    assert set(individual.genes()) == {g1, g3}
    # No edge between 2 and 3
    g4 = EdgeGene(3, 4, 'C', 'B', 1.0, 1.0)
    individual = GraphIndividual([g1, g4], 0.5)
    # No B for node 2 -> nothing happens
    individual._add_gene(g2)
    assert len(individual.genes()) == 2 and g2 not in individual.genes()
    # Both node keys match -> addition
    individual._add_gene(g3)
    assert len(individual.genes()) == 3 and g3 in individual.genes()
    individual = GraphIndividual([g1, g4], 0.0)
    # g2 becomes strong edge -> addition
    individual._add_gene(g2)
    assert len(individual.genes()) == 3 and g2 in individual.genes()


def test_seq_individual(random_seq_genes):
    genes = [SeqGene('AA', (1, 2), 10), SeqGene('AA', (2, 3), 10)]
    ind = SeqIndividual(genes)
    assert ind.pos_overlap()
    assert ind.score == 0
    genes = [SeqGene('AA', (1, 2), 10), SeqGene('AA', (3,), 10)]
    ind = SeqIndividual(genes)
    assert not ind.pos_overlap()
    assert ind.score == 20

    ind = SeqIndividual(random_seq_genes, False)
    assert len(ind) == len(random_seq_genes)
    assert ind.score == 0
    assert ind.n_pos == 0
    ind.upd()
    assert ind._n_pos == len(reduce(op.or_, (set(g.Pos) for g in random_seq_genes)))
    if not ind.pos_overlap():
        assert round(ind.score, 2) == round(sum(g.S for g in random_seq_genes), 2)
    else:
        assert ind.score == 0
