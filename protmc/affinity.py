import operator as op
import sys
import typing as t
from functools import partial
from itertools import product
from math import log

import click
import pandas as pd

from protmc.base import AminoAcidDict, AA_pair, AffinityResult
from protmc.parsers import parse_population, parse_population_df, parse_bias


def compute_bias_energy(
        sequence: str, positions: t.Iterable[int],
        bias: t.Dict[AA_pair, float], aa_mapping: t.Dict[str, str]) -> float:
    """
    Compute a bias for a given subsequence (determined by `positions`)
    :param sequence: A protein sequence
    :param positions: A collection of positions to subset the sequence
    :param bias: A mapping between amino acid pairs and bias values
    :param aa_mapping: A mapping between three-letter and one-letter amino acid codes
    :return: a negative sum of each pair biases inside the subsequence
    """
    pairs = product(zip(positions, sequence), repeat=2)
    pairs = (AA_pair(aa1[0], aa2[0], aa_mapping[aa1[1]], aa_mapping[aa2[1]]) for aa1, aa2 in pairs)
    pairs = filter(lambda pair: pair in bias and pair.pos_j <= pair.pos_i, pairs)
    return -sum(bias[pair] for pair in pairs)


def energy(
        sequence: str, positions: t.Iterable[int],
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
    # print(sequence, temp,
    #       f"bias bu: {round(bu, 2)} bias_bb: {round(bb, 2)} pu_seq: {round(fu, 2)} pb_seq: {round(fb, 2)} "
    #       f"bind_seq_b: {round(eb, 2)} bind_seq_u: {round(eu, 2)} bind_seq: {round(eb - eu, 2)}")
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
    # Count frequencies within the `bound` and `unbound` populations, respectively
    freq_b = (parse_population(pop_bound, threshold)
              if isinstance(pop_bound, str) else parse_population_df(pop_bound, threshold))
    freq_u = (parse_population(pop_unbound, threshold)
              if isinstance(pop_unbound, str) else parse_population_df(pop_unbound, threshold))
    # print("pop_bound\n", sorted(freq_b.items()), "\n", "pop_unbound\n", sorted(freq_u.items()))
    # Identify sequences common to both populations
    common_ub = set(freq_b) & set(freq_u)

    # Parse bias files
    bias_b = parse_bias(bias_bound) if bias_bound else None
    bias_u = parse_bias(bias_unbound) if bias_unbound else None
    # print('bias_bound\n', sorted(bias_b.items()), '\n', 'bias_unbound\n', sorted(bias_u.items()))

    # Energy helper
    energy_ = partial(
        energy, positions=positions,
        freq_bound=freq_b, freq_unbound=freq_u,
        bias_bound=bias_b, bias_unbound=bias_u,
        temp=temperature, aa_mapping=AminoAcidDict().aa_dict)
    # Energy of a reference sequence
    e_ref = energy_(sequence=reference_seq)

    return pd.DataFrame([AffinityResult(s, energy_(sequence=s) - e_ref) for s in common_ub])


@click.command()
@click.argument('pop_unbound', type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument('pop_bound', type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.option('-bu', '--bias_unbound', type=click.Path(dir_okay=False, file_okay=True, exists=True), required=True,
              help='Path to the bias values for the "unbound" state.')
@click.option('-bb', '--bias_bound', type=click.Path(dir_okay=False, file_okay=True, exists=True), required=True,
              help='Path to the bias values for the "bound" state.')
@click.option('-kT', '--temperature', required=True, help='Temperature of the simulation (kT)')
@click.option('-t', '--threshold', default=100, show_default=True, help='Minimum sequence count.')
@click.option('-p', '--positions', required=True, help='Comma-separated list of positions.')
@click.option('-o', '--output', type=click.File('w'), default=sys.stdout,
              help='Path to an output file. If not provided, the program will print to stdout.')
def _affinity(pop_unbound, pop_bound, bias_unbound, bias_bound, temperature, threshold, positions, output):
    """
    Command calculates relative affinities of the sequences using biased populations (proteus.dat files)
    in bound and unbound states.
    # TODO: move this interface to the root ./proteus.py (does not work for now)
    It accepts paths to POP_UNBOUND and POP_BOUND as positional arguments, and the rest is customized via options.

    """
    # Validate options
    positions = positions.split(',')
    if not positions:
        raise click.BadOptionUsage('positions', f'Failed to parse positions.')

    # Get the results
    results = affinity(pop_unbound, pop_bound, bias_unbound, bias_bound, temperature, threshold, positions)

    # Echo the results
    click.echo(f">HEADER|kT:{temperature}|threshold:{threshold}", file=output)
    for seq, aff in sorted(results, key=op.itemgetter(1)):
        click.echo("{:s} {:<10.3f}".format(seq, aff), file=output)


if __name__ == '__main__':
    _affinity()
