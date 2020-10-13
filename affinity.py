import operator as op
import sys
import typing as t
from functools import partial
from itertools import chain, groupby, product
from math import log

import click

from protmc.utils import AminoAcidDict

Population_element = t.NamedTuple(
    'Population_element', [('seq', str), ('count', int)])
AA_pair = t.NamedTuple(
    'AA_pair', [('pos_i', str), ('pos_j', str), ('aa_i', str), ('aa_j', str)])
Pair_bias = t.NamedTuple(
    'Pair_bias', [('aa_pair', AA_pair), ('bias', float)])


def count_sequences(population: str, count_threshold: int) -> t.Dict[str, float]:
    """
    Filters sequences in the `population` having >= `count_threshold` counts.
    Among the filtered sequences, calculates a total number of counts `total`.
    Returns a dict mapping sequences to a fraction (`count / total`) of counts among the filtered sequences.
    :param population: Path to a proteus.dat file (ensure validity externally).
    :param count_threshold: Counting threshold.
    """

    def parse_line(line: str) -> Population_element:
        seq, count = line.split('.')[3].split()[1:3]
        return Population_element(seq, int(count))

    with open(population) as f:
        pop_elements = list(filter(
            lambda el: el.count >= count_threshold,
            map(parse_line, f)))

    total = sum(el.count for el in pop_elements)
    return {el.seq: el.count / total for el in pop_elements}


def parse_bias(bias_in: str) -> t.Dict[AA_pair, float]:
    """
    Parses the bias file into a dict mapping `(pos_i, pos_j, aa_i, aa_j)`
    to a bias accumulated during the adaptive landscape flattening simulation.
    :param bias_in: path to a bias.dat file (validated externally)
    :return:
    """

    def parse_line(line: str) -> t.Tuple[Pair_bias, Pair_bias]:
        line_split = line.rstrip().split()
        pos_i, aa_i, pos_j, aa_j, bias = line_split
        return (
            Pair_bias(AA_pair(pos_i, pos_j, aa_i, aa_j), float(bias)),
            Pair_bias(AA_pair(pos_j, pos_i, aa_j, aa_i), float(bias)))

    with open(bias_in) as f:
        lines = chain.from_iterable(map(parse_line, filter(lambda x: not x.startswith('#'), f)))
        groups = groupby(lines, lambda el: el.aa_pair)
        return {g: sum(x.bias for x in gg) for g, gg in groups}


def compute_bias_energy(
        sequence: str, positions: t.Iterable[int],
        bias: t.Dict[AA_pair, float], aa_mapping: t.Dict[str, str]) -> float:
    """
    :param sequence:
    :param positions:
    :param bias:
    :param aa_mapping:
    :return:
    """
    pairs = product(zip(positions, sequence), repeat=2)
    pairs = (AA_pair(aa1[0], aa2[0], aa_mapping[aa1[1]], aa_mapping[aa2[1]]) for aa1, aa2 in pairs)
    pairs = filter(lambda pair: pair in bias, pairs)
    return -sum(bias[pair] for pair in pairs)


def compute_affinity(
        sequence: str, positions: t.Iterable[int],
        bias_bound: t.Dict[AA_pair, float], bias_unbound: t.Dict[AA_pair, float],
        freq_bound: t.Dict[str, float], freq_unbound: t.Dict[str, float],
        aa_mapping: t.Dict[str, str], temp: float):
    """

    :param sequence:
    :param positions:
    :param bias_bound:
    :param bias_unbound:
    :param freq_bound:
    :param freq_unbound:
    :param aa_mapping:
    :param temp:
    :return:
    """
    bb = compute_bias_energy(sequence, positions, bias_bound, aa_mapping)
    bu = compute_bias_energy(sequence, positions, bias_unbound, aa_mapping)
    fb, fu = freq_bound[sequence], freq_unbound[sequence]
    eb = -temp * log(fb) - bb
    eu = -temp * log(fu) - bu
    return eb - eu


def affinity(pop_unbound, pop_bound, bias_unbound, bias_bound, temperature, threshold, positions):
    # TODO: should compute energy relative to a reference sequence
    # Count frequencies within the `bound` and `unbound` populations, respectively
    freq_b = count_sequences(pop_bound, threshold)
    freq_u = count_sequences(pop_unbound, threshold)
    # Identify sequences common to both populations
    common_ub = set(freq_b) & set(freq_u)

    # Parse bias files
    bias_b = parse_bias(bias_bound) if bias_bound else None
    bias_u = parse_bias(bias_unbound) if bias_unbound else None

    compute_affinity_ = partial(
        compute_affinity, positions=positions,
        freq_bound=freq_b, freq_unbound=freq_u,
        bias_bound=bias_b, bias_unbound=bias_u,
        temp=temperature, aa_mapping=AminoAcidDict().aa_dict)

    return [(s, compute_affinity_(sequence=s)) for s in common_ub]


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
