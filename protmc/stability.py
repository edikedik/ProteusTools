import typing as t
from math import log

import pandas as pd

from protmc.base import AminoAcidDict
from protmc.parsers import parse_bias, parse_ref_energies, parse_population_df
from protmc.utils import compute_bias_energy


def stability(
        population: pd.DataFrame, bias_path: str, eref_path: str, ref_seq: str,
        temp: float, threshold: int, positions: t.Iterable[str]) -> pd.DataFrame:
    """
    Compute the stability of the sequence (-kT*log(frequency) - bias_energy - unfolded_energy)
    :param population: a DataFrame with `seq` and `total_count` columns
    :param bias_path: path to a bias file
    :param eref_path: path to average reference energies
    :param ref_seq: reference sequence
    :param temp: kT temperature
    :param threshold: counting threshold to filter the population
    :param positions: an iterable over positions (string values).
    These are needed to compute bias energy from the bias file.
    :return:
    # TODO: compare results with original implementation
    """
    aa_mappings = AminoAcidDict().aa_dict
    bias = parse_bias(bias_path)
    eref = parse_ref_energies(eref_path)
    freq = parse_population_df(population, count_threshold=threshold)
    if ref_seq not in freq:
        raise ValueError(f'Ref seq {ref_seq} is not in population.')

    def eref_energy(seq: str) -> float:
        seq_mapped = (aa_mappings[c] for c in seq)
        return sum(eref[c] for c in seq_mapped)

    def seq_stability(seq: str):
        seq_freq = freq[seq]
        seq_bias = compute_bias_energy(seq, map(int, positions), bias, aa_mappings)
        seq_eref = eref_energy(seq)
        return -temp * log(seq_freq) - seq_bias - 0

    ref_stability = seq_stability(ref_seq)
    return pd.DataFrame({
        'seq': list(freq),
        'stability': [seq_stability(s) - ref_stability for s in freq]}
    ).sort_values('seq')


if __name__ == '__main__':
    raise RuntimeError
