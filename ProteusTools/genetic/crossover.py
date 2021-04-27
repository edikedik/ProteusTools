import typing as t
from itertools import chain, tee
from math import floor

from more_itertools import random_permutation, take, distribute, unzip

from .base import Record
from .individual import GraphIndividual, Individual, SeqIndividual


def _check_mating_group(mating_group: t.List[t.Tuple[GraphIndividual, Record]]):
    # Validate the operation
    if len(mating_group) < 2:
        raise RuntimeError(f'Mating group must contain at least 2 individuals. Got {len(mating_group)}')
    ts = set(ind.coupling_threshold for ind, _ in mating_group)
    if len(ts) > 1:
        raise RuntimeError(f'All Individuals in the mating group must have the same coupling threshold. Got {ts}')
    coupling_threshold = ts.pop()
    types = set(ind.__class__ for ind, _, in mating_group)
    if len(types) > 1:
        raise RuntimeError(f'All Individuals in the mating group must be of a single type. '
                           f'Got {ts} for mating group {mating_group}')
    ind_type = types.pop()
    max_spaces = set(ind.max_mut_space for ind, _ in mating_group)
    if len(max_spaces) > 1:
        raise RuntimeError(f'Multiple values of `max_mut_space` attribute in the mating group.')
    max_space = max_spaces.pop()
    max_poss = set(ind.max_num_positions for ind, _ in mating_group)
    if len(max_spaces) > 1:
        raise RuntimeError(f'Multiple values of `max_num_pos` attribute in the mating group.')
    # Perform the recombination
    max_pos = max_poss.pop()
    return coupling_threshold, ind_type, max_space, max_pos


def _init_chunks(mating_group, chunks):
    ind_example = mating_group[0][0]
    if isinstance(ind_example, GraphIndividual):
        coupling_threshold, ind_type, max_space, max_pos = _check_mating_group(mating_group)
        return [ind_type(list(genes), coupling_threshold, max_space, max_pos) for genes in chunks]
    if isinstance(ind_example, SeqIndividual):
        return [SeqIndividual(list(genes), upd_on_init=True) for genes in chunks]
    else:
        raise ValueError(f'Unrecognized individual type {ind_example.__class__}')


def recombine_genes_uniformly(mating_group: t.List[t.Tuple[Individual, Record]],
                              brood_size: int) -> t.List[Individual]:
    """
    Combines genes of individuals in the `mating_group` in a single pool,
    and uniformly divides these genes into `brood_size` number of individuals.
    :param mating_group: A group of individuals selected to give progeny.
    :param brood_size: A number of offsprings.
    :return: List of offsprings.
    """
    pool = random_permutation(chain.from_iterable(ind.genes() for ind, _ in mating_group))
    chunks = take(brood_size, distribute(len(mating_group), pool))
    return _init_chunks(mating_group, chunks)


def recombine_into(mating_group: t.List[t.Tuple[GraphIndividual, Record]],
                   brood_size: int) -> t.List[GraphIndividual]:
    """
    Take all genes and distribute them into progeny.
    For two individuals with N genes and `brood_size=1`, the single offspring will have 2N genes.
    :param mating_group: A group of individuals selected to give progeny.
    :param brood_size: A number of offsprings.
    :return: List of offsprings.
    """
    pool = random_permutation(chain.from_iterable(ind.genes() for ind, _ in mating_group))
    chunks = distribute(brood_size, pool)
    return _init_chunks(mating_group, chunks)


def exchange_fraction(mating_group: t.List[t.Tuple[GraphIndividual, Record]],
                      brood_size: int, fraction: float = 0.1) -> t.List[GraphIndividual]:
    """
    Takes `fraction` of genes from each Individual.
    Aggregates all taken fractions into a single pool.
    Samples from the pool the same number of genes an Individual has donated.
    :param mating_group: A group of individuals selected to give progeny.
    :param brood_size: A number of offsprings.
    :param fraction: A fraction of genes to take from an Individual.
    :return: List of offsprings.
    """
    staged_comb, staged_pool = tee(((ind, floor(len(ind) * fraction)) for ind, _ in mating_group), 2)
    pool_samples, samples = tee((take(n, ind.genes()) for ind, n in staged_pool), 2)
    pool = random_permutation(chain.from_iterable(pool_samples))
    recombined = (
        ind.copy().remove_genes(s).add_genes(take(n, pool))
        for (ind, n), s in zip(staged_comb, samples))
    return list(take(brood_size, recombined))


def take_unchanged(mating_group: t.List[t.Tuple[GraphIndividual, Record]],
                   brood_size: int) -> t.List[GraphIndividual]:
    """
    Randomly takes `brood_size` number of individuals from the mating groups.
    :param mating_group: A group of individuals selected to give progeny.
    :param brood_size: A number of offsprings.
    :return: List of offsprings -- copies of the parents.
    """
    individuals, _ = unzip(mating_group)
    return list(take(brood_size, (ind.copy() for ind in random_permutation(individuals))))


if __name__ == '__main__':
    raise RuntimeError
