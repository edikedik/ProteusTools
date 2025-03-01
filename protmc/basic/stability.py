import operator as op
import typing as t
from math import log

import pandas as pd

from protmc.common.base import AminoAcidDict, AA_pair, NoReferenceError
from protmc.common.parsers import parse_bias, parse_population_df
from protmc.common.utils import compute_bias_energy


def stability(
        population: pd.DataFrame, bias: t.Union[t.Mapping[AA_pair, float], str],
        ref_seq: str, temp: float, threshold: int, positions: t.Iterable[str]) -> pd.DataFrame:
    """
    Compute the stability of the sequence (-kT*log(frequency) - bias_energy)
    :param population: a DataFrame with `seq` and `total_count` columns
    :param bias: path to a bias file
    :param ref_seq: reference sequence
    :param temp: kT temperature
    :param threshold: counting threshold to filter the population
    :param positions: an iterable over positions (string values).
    These are needed to compute bias energy from the bias file.
    :return: a DataFrame with two columns - `seq` and `stability`
    """
    if not isinstance(positions, t.List):
        positions = list(positions)
    aa_mappings = AminoAcidDict().aa_dict
    if not isinstance(bias, t.Mapping):
        bias = parse_bias(bias)
    freq = parse_population_df(population, count_threshold=threshold)
    if ref_seq not in freq:
        raise NoReferenceError(f'The reference {ref_seq} is not in population')

    def seq_stability(seq: str):
        """
        Calculates a `seqs`'s stability
        :param seq: sequence
        :return: energy of a sequence based on temperature, sampling and bias
        """
        seq_freq = freq[seq]
        seq_bias = compute_bias_energy(seq, positions, bias, aa_mappings)
        return -temp * log(seq_freq) - seq_bias

    ref_stability = seq_stability(ref_seq)
    return pd.DataFrame({
        'seq': list(freq),
        'stability': [seq_stability(s) - ref_stability for s in freq]}
    ).sort_values('seq')


def stability_position_wise(stability_df: pd.DataFrame, positions: t.Iterable[int]) -> pd.DataFrame:
    def sub(pos_i, pos):
        df = stability_df.copy()
        df['seq'] = df['seq'].apply(op.itemgetter(pos_i))
        df = df.groupby('seq', as_index=False).agg(
            mean_stability=pd.NamedAgg('stability', 'mean'),
            min_stability=pd.NamedAgg('stability', 'min'),
            max_stability=pd.NamedAgg('stability', 'max'))
        df['pos'] = pos
        return df

    return pd.concat([sub(i, p) for i, p in enumerate(positions)])[
        ['pos', 'seq', 'mean_stability', 'min_stability', 'max_stability']
    ].sort_values(['pos', 'seq'])


if __name__ == '__main__':
    raise RuntimeError
