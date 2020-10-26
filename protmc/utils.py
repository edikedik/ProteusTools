import subprocess as sp
import typing as t
from functools import reduce
from itertools import dropwhile, takewhile, combinations, groupby, chain, filterfalse
from pathlib import Path

import biotite.structure as bst
import biotite.structure.io as io
import numpy as np
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

T = t.TypeVar('T')


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


def interacting_pairs(structure_path: str, distance_threshold: float, positions: t.Optional[t.Iterable[int]] = None):
    st = io.load_structure(structure_path)
    ca = st[(st.atom_name == 'CA') & bst.filter_amino_acids(st)]
    if positions is not None:
        ca = ca[np.isin(ca.res_id, list(positions))]
    pairs = np.array(list(combinations(ca.res_id, 2)))
    pairs_idx = np.array(list(combinations(np.arange(len(ca)), 2)))
    dist = bst.index_distance(ca, pairs_idx)
    return pairs[dist < distance_threshold]


def get_reference(position_list: str, positions: t.Optional[t.Container[str]] = None) -> str:
    """
    Extract the reference sequence from the matrix file.
    :param position_list: position_list file with four columns
    (pdb_seq_id, active/inactive, aa3, n_rotamers)
    :param positions: a set of positions to subset the reference (optional)
    :return:
    """
    aa_mapping = AminoAcidDict().aa_dict
    with open(position_list) as f:
        split = filter(lambda l: len(l) == 4, map(lambda l: l.split(), f))
        if positions:
            split = filter(lambda l: l[0] in positions, split)
        return "".join(aa_mapping[l[2]] for l in split)


def space_constraints(
        reference: str, subset: t.Iterable[int], active: t.Iterable[int],
        existing_constraints: t.Optional[t.List[str]] = None) -> t.List[str]:
    mapping = AminoAcidDict().aa_dict
    if not isinstance(subset, t.List):
        subset = list(subset)
    if not isinstance(active, t.List):
        active = list(active)
    if len(active) != len(reference):
        raise ValueError('Length of a reference must match the number of active positions')

    inactive = list(set(active) - set(subset))
    if not inactive:
        raise ValueError('Nothing to constrain')
    inactive_idx = [active.index(pos) for pos in inactive]
    ref_subset = [reference[i] for i in inactive_idx]
    constraints = sorted(
        [f'{pos} {mapping[aa]}' for pos, aa in zip(inactive, ref_subset)],
        key=lambda x: int(x.split()[0]))
    if existing_constraints:
        constraints = merge_constraints(existing=existing_constraints, additional=constraints)
    return constraints


def merge_constraints(existing: t.List[str], additional: t.List[str]):
    total = sorted(existing + additional, key=lambda x: int(x.split()[0]))
    groups = groupby(total, lambda x: int(x.split()[0]))
    return [f'{g} ' + " ".join(set(chain.from_iterable(x.split()[1:] for x in gg))) for g, gg in groups]


def extract_constraints(configs: t.Iterable):
    constraints_ = filterfalse(
        lambda x: x is None,
        (c.get_field_value('Space_Constraints') for c in configs))
    constraints_ = ([c] if isinstance(c, str) else c for c in constraints_)
    return reduce(lambda x, y: merge_constraints(x, y), constraints_)


def adapt_space(pos: t.Union[T, t.Iterable[T]]) -> t.Union[t.List[str], str]:
    if isinstance(pos, int):
        return f'{pos}-{pos}'
    return [f'{p1}-{p2}' for p1, p2 in set(combinations(map(str, pos), 2))]


if __name__ == '__main__':
    raise RuntimeError()
