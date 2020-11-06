import operator as op
import typing as t
from functools import reduce
from io import StringIO
from itertools import combinations, groupby, chain, filterfalse, starmap
from pathlib import Path

import biotite.structure as bst
import biotite.structure.io as io
import numpy as np
import pandas as pd

from protmc.base import AminoAcidDict
from protmc.parsers import bias_to_df, get_bias_state

T = t.TypeVar('T')


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


def sum_bias(bias1, step1, bias2, step2):
    b1, b2 = map(bias_to_df, map(StringIO, starmap(get_bias_state, [(bias1, step1), (bias2, step2)])))
    return pd.concat([b1, b2]).groupby('var', as_index=False).agg({'bias': 'sum'})


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


def merge_constraints(existing: t.List[str], additional: t.List[str]) -> t.List[str]:
    total = sorted(existing + additional, key=lambda x: int(x.split()[0]))
    groups = groupby(total, lambda x: int(x.split()[0]))
    return [f'{g} ' + " ".join(set(chain.from_iterable(x.split()[1:] for x in gg))) for g, gg in groups]


def extract_constraints(configs: t.Iterable) -> t.Optional[t.List[str]]:
    constraints_ = filterfalse(
        lambda x: x is None,
        (c.get_field_value('Space_Constraints') for c in configs))
    constraints_ = [[c] if isinstance(c, str) else c for c in constraints_]
    if not constraints_:
        return None
    return reduce(lambda x, y: merge_constraints(x, y), constraints_)


def adapt_space(pos: t.Union[T, t.Iterable[T]]) -> t.Union[t.List[str], str]:
    if isinstance(pos, int):
        return f'{pos}-{pos}'
    return [f'{p1}-{p2}' for p1, p2 in set(combinations(map(str, pos), 2))]


def infer_mut_space(mut_space_n_types: int, num_active: int, constraints: t.Optional[t.List[str]]):
    num_mut = [mut_space_n_types] * num_active
    if constraints:
        if len(constraints) > num_active:
            raise ValueError('More constrained positions than active ones')
        for i, c in enumerate(constraints):
            num_mut[i] = len(c.split()[1:])
    return reduce(op.mul, num_mut)


if __name__ == '__main__':
    raise RuntimeError()
