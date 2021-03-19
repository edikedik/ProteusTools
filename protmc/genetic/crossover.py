import typing as t
from itertools import chain, groupby, tee
from math import floor

from more_itertools import random_permutation, partition, peekable, take, distribute

from .base import Gene, Record
from .individual import GenericIndividual


def filter_genes(genes: t.Iterable[Gene], coupling_threshold: float) -> t.Iterator[Gene]:
    """
    Filters `genes` so that a graph induced by them abides structural constraints.
    """

    def process_group(genes_: t.Iterable[Gene]):
        weak, strong = (partition(lambda g: g.C >= coupling_threshold, genes_))
        strong = peekable(strong)
        if strong.peek(None) is not None:
            return strong
        return take(1, weak)

    groups = groupby(sorted(genes), lambda g: (g.P1, g.P2))
    return chain.from_iterable(process_group(gg) for g, gg in groups)


def _check_mating_group(mating_group: t.List[t.Tuple[GenericIndividual, Record]]):
    # Validate the operation
    if len(mating_group) < 2:
        raise RuntimeError(f'Mating group must contain at least 2 individuals. Got {len(mating_group)}')
    ts = set(indiv.coupling_threshold for indiv, _ in mating_group)
    if len(ts) > 1:
        raise RuntimeError(f'All Individuals in the mating group must have the same coupling threshold. Got {ts}')
    coupling_threshold = ts.pop()
    types = set(indiv.__class__ for indiv, _, in mating_group)
    if len(types) > 1:
        raise RuntimeError(f'All Individuals in the mating group must be of a single type. '
                           f'Got {ts} for mating group {mating_group}')
    ind_type = types.pop()
    max_spaces = set(indiv.max_mut_space for indiv, _ in mating_group)
    if len(max_spaces) > 1:
        raise RuntimeError(f'Multiple values of `max_mut_space` attribute in the mating group.')
    # Perform the recombination
    max_space = max_spaces.pop()
    return coupling_threshold, ind_type, max_space


def recombine_genes_uniformly(mating_group: t.List[t.Tuple[GenericIndividual, Record]],
                              brood_size: int) -> t.List[GenericIndividual]:
    """
    Combines genes of individuals in the `mating_group` in a single pool,
    and uniformly divides these genes into `brood_size` number of individuals.
    :param mating_group: A group of individuals selected to give progeny.
    :param brood_size: A number of offsprings.
    :return: List of offsprings.
    """
    coupling_threshold, ind_type, max_space = _check_mating_group(mating_group)
    pool = random_permutation(chain.from_iterable(indiv.genes() for indiv, _ in mating_group))
    chunks = take(brood_size, distribute(len(mating_group), pool))
    return [ind_type(list(genes), coupling_threshold, max_space) for genes in chunks]


def recombine_into(mating_group: t.List[t.Tuple[GenericIndividual, Record]],
                   brood_size: int) -> t.List[GenericIndividual]:
    coupling_threshold, ind_type, max_space = _check_mating_group(mating_group)
    pool = random_permutation(chain.from_iterable(indiv.genes() for indiv, _ in mating_group))
    chunks = distribute(brood_size, pool)
    return [ind_type(list(genes), coupling_threshold, max_space) for genes in chunks]


def exchange_fraction(mating_group: t.List[t.Tuple[GenericIndividual, Record]],
                      brood_size: int, fraction: float) -> t.List[GenericIndividual]:
    if brood_size > len(mating_group):
        raise ValueError(f'Brood size {brood_size} cannot be larger than the mating group {len(mating_group)}')
    staged_comb, staged_pool = tee(((ind, floor(len(ind) * fraction)) for ind, _ in mating_group), 2)
    pool_samples, samples = tee((take(n, ind.genes()) for ind, n in staged_pool), 2)
    pool = random_permutation(chain.from_iterable(pool_samples))
    recombined = (
        ind.copy().remove_genes(s).add_genes(take(n, pool))
        for (ind, n), s in zip(staged_comb, samples))
    return list(take(brood_size, recombined))
