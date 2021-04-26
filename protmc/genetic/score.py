import typing as t

import numpy as np

from protmc.genetic.base import AbstractGraphIndividual, AbstractSeqIndividual


def gaussian(x: t.Union[np.ndarray, float], mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    This is a spherical gaussian with a single mean (in case `x` is multidimensional).
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
    """
    Calculate the standard deviation associated with a `desired_penalty` value of `gaussian_penalty`
    when deviation from mean reaches `deviation`.
    """
    return (-deviation ** 2 / (2 * np.log(desired_penalty))) ** 1 / 2


def score(ind: t.Union[AbstractSeqIndividual, AbstractGraphIndividual],
          min_size: int = 10, max_size: int = 100,
          pen_mut: bool = True, sigma_mut: float = 5, desired_space: float = 5 * np.log(18),
          pen_pos: bool = False, sigma_pos: float = 10, desired_pos: int = 4,
          expose: bool = False, penalize_lower: bool = False) -> float:
    """
    Compute the score (fitness) of the individual.
    The score's base value is taken from the `score` property of an individual and
    weighted by the gaussian penalty for deviation from either the desired
    mutation space size or the number of positions.

    In case no penalties are incurred, the function is essentially a `score` getter.
    :param ind: And individual having the `score` property/attribute.
    :param min_size: A min size of an individual; lower sizes imply 0 score.
    :param max_size: A max size of an individual; higher sizes imply 0 score.
    :param pen_mut: Whether to penalize for deviation of mutation space from `desired_space`.
    If `True`, `ind` must have `mut_space_size` property/attribute.
    :param sigma_mut: Standard deviation (see `gaussian_penalty` docs).
    :param desired_space: A desired size of a mutation space size (estimate).
    :param pen_pos: Whether to penalize for the deviations from the `desired_pos`.
     If `True`, `ind` must have `n_pos` property/attribute.
    :param sigma_pos: Standard deviation (see `gaussian_penalty` docs).
    :param desired_pos: Desired number of unique positions within `ind`
    :param expose: If true print (base_score, duplication_penalty, mut_space_penalty, num_pos_penalty).
    :param penalize_lower: If true, penalize deviations symmetrically around the desired number.
    :return: A weighted `score` of an individual.
    """
    ind_size = len(ind)
    if ind_size < min_size or ind_size > max_size:
        return 0

    s = ind.score

    if pen_mut and (penalize_lower or (not penalize_lower and ind.mut_space_size > desired_space)):
        mutation_space_penalty = gaussian_penalty(
            ind.mut_space_size, desired_space, sigma_mut)
    else:
        mutation_space_penalty = 1.0

    if pen_pos and (penalize_lower or (not penalize_lower and ind.n_pos > desired_pos)):
        num_pos_penalty = gaussian_penalty(ind.n_pos, desired_pos, sigma_pos)
    else:
        num_pos_penalty = 1.0

    if expose:
        print(f'Raw={s},SpacePenalty={mutation_space_penalty},'
              f'PosPenalty={num_pos_penalty},'
              f'Final={s * mutation_space_penalty * num_pos_penalty}')

    return s * mutation_space_penalty * num_pos_penalty


if __name__ == '__main__':
    raise RuntimeError
