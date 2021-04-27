import typing as t
from functools import partial
from math import log

import pandas as pd

from ..common.base import AminoAcidDict, AA_pair, AffinityResult, NoReferenceError
from ..common.parsers import parse_population, parse_population_df, parse_bias
from ..common.utils import compute_bias_energy


def energy(
        sequence: str, positions: t.Iterable[str],
        bias_unbound: t.Dict[AA_pair, float], bias_bound: t.Dict[AA_pair, float],
        freq_unbound: t.Dict[str, float], freq_bound: t.Dict[str, float],
        aa_mapping: t.Dict[str, str], temp: float) -> float:
    """
    Compute the energy difference for a given sequence between bound and unbound states.
    :param sequence: A sequence of interest
    :param positions: Positions to sub-set the `sequence`
    :param bias_bound: Parsed bias - a mapping between amino acid pair and a bias value of the "bound" state
    :param bias_unbound: Same as `bias_bound`, but for the "unbound" state
    :param freq_bound: A mapping between sequences and their frequencies for the "bound" state
    :param freq_unbound: Same as `freq_bound` for the "unbound" state
    :param aa_mapping: A mapping between three-letter codes and one-letter codes of amino acids
    :param temp: The temperature of a simulation
    :return:
    """
    bb = compute_bias_energy(sequence, positions, bias_bound, aa_mapping)
    bu = compute_bias_energy(sequence, positions, bias_unbound, aa_mapping)
    fb, fu = freq_bound[sequence], freq_unbound[sequence]
    eb = -temp * log(fb) - bb
    eu = -temp * log(fu) - bu
    return eb - eu


def affinity(
        reference_seq: str,
        pop_unbound: t.Union[str, pd.DataFrame], pop_bound: t.Union[str, pd.DataFrame],
        bias_unbound: str, bias_bound: str,
        temperature: float, threshold: int, positions: t.Iterable[str]) -> pd.DataFrame:
    """
    Computes affinity for all sequences common to both "bound" and "unbound" populations
    relative to a reference sequence.
    :param reference_seq: A reference sequence
    :param pop_unbound: Path to a population of "unbound" states
    :param pop_bound: Path to a population of "bound" states
    :param bias_unbound: Path to the bias values for the "unbound" state
    :param bias_bound: Path to the bias values for the "bound" state
    :param temperature: The temperature of simulation (kT)
    :param threshold: A minimum sequence count
    -> sequences below threshold will be discarded, and the frequencies of the remaining sequences will be recomputed
    :param positions: A list of desired positions (string values)
    :return:
    """
    if not isinstance(positions, t.List):
        positions = list(positions)
    # Count frequencies within the `bound` and `unbound` populations, respectively
    freq_b = (parse_population(pop_bound, threshold)
              if isinstance(pop_bound, str) else parse_population_df(pop_bound, threshold))
    if reference_seq not in freq_b:
        raise NoReferenceError(f'The reference sequence {reference_seq} is not in `bound` population')
    freq_u = (parse_population(pop_unbound, threshold)
              if isinstance(pop_unbound, str) else parse_population_df(pop_unbound, threshold))
    if reference_seq not in freq_u:
        raise NoReferenceError(f'The reference sequence {reference_seq} is not in `unbound` population')
    # Identify sequences common to both populations
    common_ub = set(freq_b) & set(freq_u)

    # Parse bias files
    bias_b = parse_bias(bias_bound) if bias_bound else None
    bias_u = parse_bias(bias_unbound) if bias_unbound else None

    # Energy helper
    energy_ = partial(
        energy, positions=positions,
        freq_bound=freq_b, freq_unbound=freq_u,
        bias_bound=bias_b, bias_unbound=bias_u,
        temp=temperature, aa_mapping=AminoAcidDict().aa_dict)
    # Energy of a reference sequence
    e_ref = energy_(sequence=reference_seq)

    return pd.DataFrame([AffinityResult(s, energy_(sequence=s) - e_ref) for s in common_ub])


if __name__ == '__main__':
    raise RuntimeError
