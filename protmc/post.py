import operator as op
import typing as t
from collections import namedtuple, defaultdict
from functools import partial
from itertools import islice, takewhile, starmap, count, groupby

import pandas as pd
from tqdm import tqdm

from protmc.utils import count_sequences

Summary = namedtuple('summary', [
    'num_unique', 'num_unique_merged', 'coverage',
    'seq_prob_mean', 'seq_prob_std', 'seq_prob_rss'])
Parsed_entry = namedtuple('parsed_entry', ['seq', 'counts', 'energy'])


def analyze_seq(
        seq: str, rich: str, steps: int,
        active: t.Optional[t.List[int]] = None, verbose: bool = False) -> pd.DataFrame:
    """
    Parse the results from .seq and .rich output files and compose a DataFrame comprising
    unique sequences, their counts and probabilities, and mean-min-max energy values
    :param seq: path to a seq file
    :param rich: path to a rich file
    :param steps: a number of steps in the run
    :param active: a list of active positions; if not provided, all positions are considered "active"
    :param verbose: print progress bar
    :return: a DataFrame object with n_rows = number of unique sequences sampled across the run
    """
    Entry = namedtuple('entry', ['seq_entry', 'rich_entry'])

    def take(n, iterable):
        return list(islice(iterable, n))

    def parse_entry(entry: Entry) -> Parsed_entry:
        counts, energy = entry.seq_entry.split()[1:3]
        counts, energy = int(counts), float(energy)
        seq_rich = "".join(entry.rich_entry.split()[1:])
        return Parsed_entry(seq_rich, counts, energy)

    def subset_positions(entry: Parsed_entry, pos: t.List[int]) -> Parsed_entry:
        return Parsed_entry(
            "".join([entry.seq[i] for i in pos]), entry.counts, entry.energy)

    with open(rich) as r, open(seq) as s:
        take(1, r), take(1, s)
        positions = [int(x) for x in r.readline().rstrip().split()[1:]]
        rs = map(op.itemgetter(0), takewhile(bool, (take(3, r) for _ in count())))
        if verbose:
            rs = tqdm(rs, desc='Parsing entries')
        entries = map(parse_entry, starmap(Entry, zip(s, rs)))
        if active:
            active = [positions.index(a) for a in active]
            entries = map(partial(subset_positions, pos=active), entries)
        df = _agg_results(pd.DataFrame(entries), steps)

    return df


def compose_summary(results: pd.DataFrame, mut_space_size: int, num_active: t.Optional[int] = None):
    """
    Further aggregate the results dataframe outputted by `analyze_seq` or `analyze_seq_no_rich`
        to provide the summary of the run (see the Summary object for the list of fields)
    :param results: a results DataFrame
    :param mut_space_size: a number of types in the mutation space
    :param num_active: a number of active positions
    :return: Summary namedtuple object
    """
    counts = count_sequences(results['seq'])
    num_active = num_active or len(results['seq'][0])
    return Summary(
        num_unique=len(results['seq'].unique()),
        num_unique_merged=counts,
        coverage=counts / (mut_space_size ** num_active),
        seq_prob_mean=results['seq_prob'].mean(),
        seq_prob_std=results['seq_prob'].std(),
        seq_prob_rss=((results['seq_prob'] - 1 / len(results)) ** 2).sum())


def analyze_seq_no_rich(
        seq_path: str, matrix_bb: str, steps: int,
        active: t.Optional[t.List[int]] = None,
        verbose: bool = False) -> pd.DataFrame:
    """
    Same as `analyze_seq`, but instead of .rich file one must provide a path to a matrix.bb file to extract mappings.
    :param seq_path: path to a .seq file
    :param matrix_bb: path to a matrix file (diagonals)
    :param steps: a number of steps (Trajectory_Length)
    :param active: a list of active positions
    :param verbose: print progress bar of parsing the .seq file
    :return: a DataFrame with various statistics on the sampled sequences
    """
    pdb_map, rot_map = matrix_mappings(matrix_bb)

    def parse(seq_line: str):
        l_spl = seq_line.split()
        counts, energy = l_spl[1:3]
        positions = list(enumerate(l_spl[3:]))
        if active:
            positions = [positions[pdb_map[str(p)]] for p in active]
        seq = "".join(rot_map[i][p] for i, p in positions)
        return Parsed_entry(seq, int(counts), float(energy))

    with open(seq_path) as f:
        f.readline()
        if verbose:
            f = tqdm(f)
        df = _agg_results(pd.DataFrame(map(parse, f)), steps)

    return df


def matrix_mappings(matrix_bb: str):
    """
    Extract `pdb_pos`->`idx_pos` and `idx_pos`->(`idx_rot`->`aa1`) from the matrix file containing diagonals.
    `pdb_pos` is a position within initial PDB file.
    `idx_pos` is an index of this position within the matrix.
    `idx_rot` is an index of a rotamer for some `idx_pos`.
    Finally, `aa1` is a amino acid one-letter code.
    :param matrix_bb: path to a matrix file with diagonal elements
    :return: a tuple of (`pdb_pos`->`idx_pos` and `idx_pos`->(`idx_rot`->`aa1`) mappings
    """
    def parse_group(group_idx: int, pdb_pos: str, lines: t.Iterator[str]):
        lines = list(lines)
        pdb2idx = (pdb_pos, group_idx)
        idx2rot = (group_idx, dict((str(i), l.split()[2]) for i, l in enumerate(lines)))
        return pdb2idx, idx2rot

    pdb_map, rot_map = defaultdict(), defaultdict()
    with open(matrix_bb) as f:
        groups = groupby(f, lambda x: x.split()[0])
        for (idx, (g, gg)) in enumerate(groups):
            pdb_m, rot_m = parse_group(idx, g, gg)
            pdb_map[pdb_m[0]] = pdb_m[1]
            rot_map[rot_m[0]] = rot_m[1]
    return pdb_map, rot_map


def _agg_results(results: pd.DataFrame, steps):
    # TODO: mean energy is not currently correct -- should be counts * energy / nsteps
    df = results.groupby('seq', as_index=False).agg(
        total_count=pd.NamedAgg(column='counts', aggfunc='sum'),
        # avg_energy=pd.NamedAgg(column='energy', aggfunc='mean'),
        min_energy=pd.NamedAgg(column='energy', aggfunc='min'),
        max_energy=pd.NamedAgg(column='energy', aggfunc='max'))
    df['seq_prob'] = df['total_count'] / steps
    return df.reset_index(drop=True)[
        ['seq', 'total_count', 'seq_prob', 'min_energy', 'max_energy']]


if __name__ == '__main__':
    raise RuntimeError
