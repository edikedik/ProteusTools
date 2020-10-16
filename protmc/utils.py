import typing as t
from itertools import dropwhile, takewhile
from pathlib import Path
import subprocess as sp

import pandas as pd

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
            **{l[0]: l[1] for l in inp_split},
            **{l[1]: l[0] for l in inp_split}}

    @property
    def aa_dict(self) -> t.Dict[str, str]:
        return self._aa_dict

    @property
    def proto_mapping(self) -> t.Dict[str, str]:
        return {'e': 'E', 'd': 'D', 'k': 'K', 'y': 'Y', 'j': 'H', 'h': 'H'}


def aggregate_counts(paths: t.List, temperatures: t.List[float]) -> pd.DataFrame:
    if len(paths) != len(temperatures):
        raise ValueError(f'Unequal lengths of `paths` ({len(paths)}) and `temperatures` ({len(temperatures)}).')
    return pd.concat(
        [parse_proteus_dat(p, temp) for p, temp in zip(map(validate_path, paths), temperatures)]
    )


def validate_path(path: str) -> str:
    if not Path(path).exists():
        raise ValueError(f'Invalid path {path}')
    return path


def parse_proteus_dat(path: str, temp) -> pd.DataFrame:
    def parse_line(line: str) -> t.Tuple[str, int]:
        seq, count_ = line.split('.')[3].split()[1:3]
        return seq, int(count_)

    with open(path) as f:
        parsed_lines = [parse_line(l) for l in f]

    return pd.DataFrame({
        'seq': [x[0] for x in parsed_lines],
        'num_count': [x[1] for x in parsed_lines],
        'temp': [temp] * len(parsed_lines),
        'file': [path] * len(parsed_lines)
    })


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
        lines = dropwhile(lambda l: l.startswith('#') and int(l.rstrip().split()[2]) == step, f)
        bias = "".join(takewhile(lambda l: not l.startswith('#'), lines))
    return bias or None


def tail(filename: str, n: int):
    res = sp.run(f'tail -{n} {filename}', capture_output=True, text=True, shell=True)
    if res.stderr:
        raise ValueError(f'Failed with an error {res.stderr}')
    return res.stdout


if __name__ == '__main__':
    raise RuntimeError()
