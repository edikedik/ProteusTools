import subprocess as sp
import typing as t
from io import StringIO
from itertools import dropwhile, takewhile, groupby, chain

import pandas as pd

from protmc.base import Pair_bias, AA_pair


def parse_population(population: pd.DataFrame, count_threshold: int) -> t.Mapping[str, float]:
    """
    Same as count_sequences in case the population is a DataFrame SUMMARY produced by the Pipeline object
    :param population: SUMMARY.tsv dataframe
    :param count_threshold: counting threshold
    :return:
    """
    df = population.copy()
    df = df[df['total_count'] > count_threshold]
    df['frequency'] = df['total_count'] / df['total_count'].sum()
    return pd.Series(df['frequency'].values, index=df['seq']).to_dict()


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
            Pair_bias(AA_pair(pos_j, pos_i, aa_j, aa_i), float(bias) if pos_i != pos_j else 0.0))

    with open(bias_in) as f:
        lines = chain.from_iterable(map(parse_line, filter(lambda x: not (x.startswith('#') or x == '\n'), f)))
        groups = groupby(sorted(lines, key=lambda el: el.aa_pair), lambda el: el.aa_pair)
        return {g: sum(x.bias for x in gg) for g, gg in groups}


def bias_to_df(bias_file: t.Union[str, StringIO]) -> pd.DataFrame:
    df = pd.read_csv(
        bias_file,
        sep=r'\s+', skiprows=1,
        names=['pos1', 'aa1', 'pos2', 'aa2', 'bias'],
        dtype={'pos1': str, 'pos2': str},
        comment='#').dropna()
    df['var'] = ['-'.join([x.pos1, x.aa1, x.pos2, x.aa2]) for x in df.itertuples()]
    return df


def dump_bias_df(df: pd.DataFrame, path: str, step: t.Optional[int] = None):
    with open(path, 'w') as f:
        if step is not None:
            print(f'# STEP {step}', file=f)
        for _, v, b in df[['var', 'bias']].itertuples():
            print(*v.split('-'), b, file=f)
    return


def read_seq_states(seq_file: str, steps: int) -> t.Optional[str]:
    """
    :param seq_file: path to a .seq file
    :param steps: the steps to extract
    :return:
    """
    with open(seq_file) as f:
        s = [l for i, l in enumerate(f) if i in steps]
    return s or None


def get_seq_state(seq_file: str, step: int) -> t.Optional[str]:
    with open(seq_file) as f:
        f.readline()
        s = [line for line in f if int(line.split()[0]) == step]
    return s[0] if s else None


def get_bias_state(bias_file: str, step: int) -> t.Optional[str]:
    with open(bias_file) as f:
        lines = dropwhile(lambda l: not (l.startswith('#') and int(l.rstrip().split()[2]) == step), f)
        comment = next(lines)
        body = "".join(takewhile(lambda l: not l.startswith('#'), lines))
        bias = comment + body
    return None if not body else bias


def tail(filename: str, n: int) -> str:
    res = sp.run(f'tail -{n} {filename}', capture_output=True, text=True, shell=True)
    if res.stderr:
        raise ValueError(f'Failed with an error {res.stderr}')
    return res.stdout
