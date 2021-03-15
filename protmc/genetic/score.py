import typing as t

import numpy as np

from protmc.genetic.base import AbstractIndividual


def gaussian(x: t.Union[np.ndarray, float], mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    :return: Value of the gaussian with mean `mu` and std `sigma` at the point `x`.
    """
    return 1 / (2 * np.pi) ** (1 / 2) / sigma ** 2 * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def gaussian_penalty(x: float, x_peak: float, sigma: float):
    """
    :return: Returns the value of the gaussian with mean `x_peak` and std `sigma` at the point `x`,
    scaled in such a way that its value at `x_peak` equals one.
    Lower values imply higher deviations of `x` from `x_peak` and vice versa.
    """
    return 1 / gaussian(x_peak, x_peak, sigma) * gaussian(x, x_peak, sigma)


def sigma_helper(desired_penalty: float, deviation: float):
    return (-deviation ** 2 / (2 * np.log(desired_penalty))) ** 1/2


def score(indiv: AbstractIndividual,
          min_size: int = 10, max_size: int = 100,
          pen_mut: bool = True, sigma_mut: float = 5, desired_space: float = 5 * np.log(18),
          pen_pos: bool = False, sigma_pos: float = 10, desired_pos: int = 4,
          expose: bool = False) -> float:
    """
    Compute the score (fitness) of the individual.
    The score's base value is the sum of gene scores weighted by the gaussian penalty for deviation from
    either the desired mutation space size or the number of positions.
    :param indiv: An array of indices pointing to genes of the GenePool.
    :param min_size: A min size of an individual; lower sizes imply 0 score.
    :param max_size: A max size of an individual; higher sizes imply 0 score.
    :param pen_mut: Whether to penalize for deviation of mutation space from `desired_space`.
    Note that switching this value off will likely cause oversized individuals.
    :param sigma_mut: Standard deviation (see `gaussian_penalty` docs).
    :param desired_space: A desired size of a mutation space size (estimate).
    :param pen_pos: Whether to penalize for the deviations from the `desired_pos`.
    :param sigma_pos: Standard deviation (see `gaussian_penalty` docs).
    :param desired_pos: Desired number of unique positions within `indiv`
    :param expose: If true print (base_score, duplication_penalty, mut_space_penalty, num_pos_penalty).
    :return: A score capturing how well individual meets our expectations.
    Specifically, with the default weights, the best individual is has no duplicate genes, has
    mutation space size as close to `desired_space` as possible,
    and, of course, a well-scoring composition of genes.
    """
    indiv_size = len(indiv.graph.edges)
    if indiv_size < min_size or indiv_size > max_size:
        return 0

    s = indiv.score

    if pen_mut:
        mutation_space_penalty = gaussian_penalty(
            indiv.mut_space_size, desired_space, sigma_mut)
    else:
        mutation_space_penalty = 1.0

    if pen_pos:
        num_pos_penalty = gaussian_penalty(indiv.n_pos, desired_pos, sigma_pos)
    else:
        num_pos_penalty = 1.0

    if expose:
        print(f'Raw={s},SpacePenalty={mutation_space_penalty},'
              f'PosPenalty={num_pos_penalty},'
              f'Final={s * mutation_space_penalty * num_pos_penalty}')

    return s * mutation_space_penalty * num_pos_penalty


if __name__ == '__main__':
    raise RuntimeError
