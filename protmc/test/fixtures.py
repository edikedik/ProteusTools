import numpy as np
import pytest

from protmc.genetic.base import Gene, GenePool
from protmc.genetic.individual import GenericIndividual


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
