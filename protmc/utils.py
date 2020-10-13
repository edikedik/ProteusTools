import typing as t
from pathlib import Path

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
        seq, count = line.split('.')[3].split()[1:3]
        return seq, int(count)

    with open(path) as f:
        parsed_lines = [parse_line(l) for l in f]

    return pd.DataFrame({
        'seq': [x[0] for x in parsed_lines],
        'num_count': [x[1] for x in parsed_lines],
        'temp': [temp] * len(parsed_lines),
        'file': [path] * len(parsed_lines)
    })


if __name__ == '__main__':
    raise RuntimeError()
