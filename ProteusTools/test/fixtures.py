import typing as t
from random import choices

import numpy as np
import pytest

from ..genetic.base import EdgeGene, SeqGene
from ..genetic.individual import GraphIndividual


def generate_graph_genes(
        n_min=50, n_max=200, ps_min=1, ps_max=10,
        coupling_min=0.1, coupling_max=1.0,
        score_min=0.1, score_max=1.0) -> t.List[EdgeGene]:
    types = ['A', 'B', 'C', 'D', 'E', 'F']
    n = np.random.randint(n_min, n_max)
    ps = np.random.randint(ps_min, ps_max, size=(n, 2))
    ts = np.random.choice(types, size=(n, 2))
    return list(
        EdgeGene(p[0], p[1], a[0], a[1],
                 np.random.uniform(score_min, score_max),
                 np.random.uniform(coupling_min, coupling_max))
        for p, a in zip(ps, ts) if p[0] <= p[1])


def generate_seq_genes(n_min=2, n_max=20, s_min=2, s_max=20, ps_min=1, ps_max=10,
                       score_min=0.1, score_max=1.0) -> t.List[SeqGene]:
    types = ['A', 'B', 'C', 'D', 'E', 'F']
    n = np.random.randint(n_min, n_max)
    seqs = ["".join(choices(types, k=np.random.randint(s_min, s_max))) for _ in range(n)]
    ps = [tuple(np.random.randint(ps_min, ps_max, len(s))) for s in seqs]
    scores = np.random.uniform(score_min, score_max, n).round(2)
    return list(SeqGene(*x) for x in zip(seqs, ps, scores))


@pytest.fixture()
def random_graph_genes():
    return generate_graph_genes()


@pytest.fixture()
def random_seq_genes():
    return generate_seq_genes()


@pytest.fixture()
def random_individual() -> GraphIndividual:
    return GraphIndividual(generate_graph_genes())


@pytest.fixture()
def strongly_linked_individual() -> GraphIndividual:
    threshold = np.random.uniform(0.1, 0.9)
    return GraphIndividual(generate_graph_genes(coupling_min=threshold + 0.05), coupling_threshold=threshold)


@pytest.fixture()
def weakly_linked_individual() -> GraphIndividual:
    threshold = np.random.uniform(0.1, 0.9)
    return GraphIndividual(generate_graph_genes(coupling_max=threshold), coupling_threshold=threshold)
