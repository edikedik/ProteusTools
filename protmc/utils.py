import typing as t
from itertools import dropwhile, takewhile
from pathlib import Path
import subprocess as sp

import pandas as pd
import seaborn as sns

AA_DICT = """ALA A ACT
CYS C ACT
THR T ACT
GLU E ED
GLH e ED
ASP D ED
ASH d ED
PHE F FW
TRP W FW
ILE I IVL
VAL V IVL
LEU L IVL
LYS K K
LYN k K
MET M M
ASN N NQ
GLN Q NQ
SER S S
ARG R R
TYR Y Y
TYD y Y
HID h H
HIE j H
HIP H H
PRO P PG
GLY G PG"""


class AminoAcidDict:
    def __init__(self, inp: str = AA_DICT):
        self._aa_dict = self._parse_dict(inp)

    @staticmethod
    def _parse_dict(inp):
        inp_split = [x.split() for x in inp.split('\n')]
        return {
            **{line[0]: line[1] for line in inp_split},
            **{line[1]: line[0] for line in inp_split}}

    @property
    def aa_dict(self) -> t.Dict[str, str]:
        return self._aa_dict

    @property
    def proto_mapping(self) -> t.Dict[str, str]:
        return {'e': 'E', 'd': 'D', 'k': 'K', 'y': 'Y', 'j': 'H', 'h': 'H'}


def validate_path(path: str) -> str:
    if not Path(path).exists():
        raise ValueError(f'Invalid path {path}')
    return path


def count_sequences(seqs: t.Iterable[str]) -> int:
    """
    Merges sequence protonation states and counts the number of unique sequences.
    The function is needed to properly calculate coverage of the sequence space.
    :param seqs: a collection of protein sequences
    :return: a number of unique sequences
    """
    proto_mapping = AminoAcidDict().proto_mapping
    mapped_seqs = (
        "".join(proto_mapping[c] if c in proto_mapping else c for c in s) for s in seqs)
    return len(set(mapped_seqs))


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


def bias_to_df(bias_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        bias_file,
        sep=r'\s+', skiprows=1,
        names=['pos1', 'aa1', 'pos2', 'aa2', 'bias'],
        dtype={'pos1': str, 'pos2': str},
        comment='#').dropna()
    df['Var'] = ['-'.join([x.pos1, x.aa1, x.pos2, x.aa2]) for x in df.itertuples()]
    return df


def plot_bias_timeline(bias_file: str):
    pass

    if __name__ == '__main__':
        raise RuntimeError()
