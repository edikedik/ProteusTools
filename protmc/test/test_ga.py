from more_itertools import take

from protmc.genetic.base import GeneticParams
from protmc.genetic.ga import GA
from protmc.genetic.individual import GraphIndividual
from .fixtures import generate_graph_genes


def test_ga():
    """Use for profiling the code"""
    pool = generate_graph_genes(1000, 5000)
    populations = [[GraphIndividual(list(take(50, pool))) for _ in range(20)] for _ in range(20)]
    params = GeneticParams(
        Population_size=100,
        Coupling_threshold=0.3,
        Gene_pool=list(pool),
        Tournaments_selection=20,
        Tournaments_policy=20,
        Deletion_size=3,
        Acquisition_size=3,
        Probabilities=(0.1, 0.45, 0.45),
        Score_kwargs=dict(
            max_size=300))
    ga = GA(params, populations=populations)
    ga.evolve_local(10)
