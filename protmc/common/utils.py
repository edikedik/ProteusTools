import logging
import operator as op
import typing as t
from collections import defaultdict
from functools import reduce
from io import StringIO
from itertools import combinations, groupby, chain, filterfalse, starmap
from itertools import product
from pathlib import Path
from warnings import warn

import biotite.structure as bst
import biotite.structure.io as io
import numpy as np
import pandas as pd

from protmc.common.base import AminoAcidDict, AA_pair
from protmc.common.parsers import bias_to_df, get_bias_state

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


def compute_bias_energy(
        sequence: str, positions: t.Iterable[str],
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


def sum_bias(bias1, step1, bias2, step2):
    b1, b2 = map(bias_to_df, map(StringIO, starmap(get_bias_state, [(bias1, step1), (bias2, step2)])))
    return pd.concat([b1, b2]).groupby('var', as_index=False).agg({'bias': 'sum'})


def compute_seq_prob(energies: t.List[float], temp: float) -> np.ndarray:
    ns = np.array(energies)
    ex = np.exp((ns / temp) if temp else ns)
    return ex / sum(ex)


def interacting_pairs(
        structure_path: str,
        distance_threshold: float,
        atom_name: str = 'CA',
        positions: t.Optional[t.Iterable[int]] = None):
    """
    Finds residues in structure within distance threshold.
    :param structure_path: path to a structure file
    :param distance_threshold: min distance between elements (non-inclusive)
    :param atom_name: filter atoms to this names (CA, CB, and so on)
    :param positions: filter positions to the ones in this list
    :return: numpy array with shape (N, 2) where N is a number of interacting pairs
    """
    st = io.load_structure(structure_path)
    ca = st[(st.atom_name == atom_name) & bst.filter_amino_acids(st)]
    if positions is not None:
        ca = ca[np.isin(ca.res_id, list(positions))]
    pairs = np.array(list(combinations(np.unique(ca.res_id), 2)))
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
        return "".join(aa_mapping[line[2]] for line in split)


def get_reference_from_structure(structure_path: str, positions: t.Optional[t.Container[int]] = None) -> str:
    aa_mapping = AminoAcidDict().aa_dict
    residues = zip(*bst.get_residues(io.load_structure(structure_path)))
    if positions is not None:
        residues = (r for r in residues if r[0] in positions)
    return "".join([aa_mapping[r[1]] for r in residues])


def space_constraints(
        reference: str, subset: t.Iterable[int], active: t.Iterable[int],
        mutation_space: t.Union[t.List[str], str, None] = None,
        existing_constraints: t.Optional[t.List[str]] = None) -> t.List[str]:
    """
    Constraints all the positions of the {active} - {subset} = {inactive} set members to their native types.
    The function produces a list of values ready to be used in Space_Constraints protMC parameter.
    :param reference: Reference sequence of the {active} positions set.
    :param subset: A subset of positions to exclude from active.
    :param active: A set of active positions, must be equal in length to `reference`.
    :param mutation_space: A list of available types or a path to a mutation_space file.
    If present, constraints will be filtered according to the mutation space.
    :param existing_constraints: A list of existing constraints.
    If there are existing constraints, they will be merged into the output.
    :return: A list of sequence space constraints.
    """
    mapping = AminoAcidDict().aa_dict
    if not isinstance(subset, t.List):
        subset = list(subset)
    if not isinstance(active, t.List):
        active = list(active)
    if len(active) != len(reference):
        raise ValueError('Length of a reference must match the number of active positions')

    inactive = list(set(active) - set(subset))
    if not inactive:
        logging.warning(f'subset {subset} must be equal to {active}, '
                        f'and there is nothing to constrain as a result.')
        return []
    inactive_idx = [active.index(pos) for pos in inactive]
    ref_subset = [reference[i] for i in inactive_idx]
    constraints = sorted(
        [f'{pos} {mapping[aa]}' for pos, aa in zip(inactive, ref_subset)],
        key=lambda x: int(x.split()[0]))
    if mutation_space is not None:
        if isinstance(mutation_space, str) and Path(mutation_space).exists():
            with open(mutation_space) as f:
                mutation_space = {x.rstrip() for x in f if x != '\n'}
        constraints = map(
            lambda l: " ".join(chain([l[0]], filter(lambda aa: aa in mutation_space, l[1:]))),
            (x.split() for x in constraints))
        constraints = list(filter(lambda x: len(x.split()) > 1, constraints))
    if existing_constraints:
        constraints = intersect_constraints(existing_constraints + constraints)
    return constraints


def union_constraints(constraints: t.List[str]) -> t.List[str]:
    """
    Groups constraints by position and takes a union of constraints per position
    :param constraints: A list of strings "pos AA1 AA2 ..."
    """
    key = lambda x: int(x.split()[0])
    groups = groupby(sorted(constraints, key=key), key)
    return [f'{g} ' + " ".join(sorted(set(chain.from_iterable(x.split()[1:] for x in gg)))) for g, gg in groups]


def intersect_constraints(constraints: t.List[str]) -> t.List[str]:
    """
    Groups constraints by position and intersects all constraints
    :param constraints: A list of strings "pos AA1 AA2 ..."
    """
    key = lambda x: int(x.split()[0])
    groups = groupby(sorted(constraints, key=key), key)
    merged = ((g, reduce(lambda x, y: set(x) & set(y), (x.split()[1:] for x in gg))) for g, gg in groups)
    return [f'{g} {" ".join(sorted(gg))}' for g, gg in merged]


def extract_constraints(configs: t.Iterable) -> t.Optional[t.List[str]]:
    constraints_ = filterfalse(
        lambda x: x is None,
        (c.get_field_value('Space_Constraints') for c in configs))
    constraints_ = [[c] if isinstance(c, str) else c for c in constraints_]
    if not constraints_:
        return None
    return reduce(lambda x, y: intersect_constraints(x + y), constraints_)


def adapt_space(pos: t.Union[T, t.Iterable[T]]) -> t.Union[t.List[str], str]:
    if isinstance(pos, int):
        return f'{pos}-{pos}'
    return [f'{p1}-{p2}' for p1, p2 in set(combinations(map(str, pos), 2))]


def infer_mut_space(
        mut_space_n_types: int, num_active: int,
        constraints: t.Optional[t.List[str]],
        merge_proto: bool = True) -> int:
    """
    # TODO: rename to mut_space_size
    Calculate mutation space size.
    :param mut_space_n_types: Number of types available in the initial mutation space.
    :param num_active: Number of active positions.
    :param constraints: Existing `Space_Constraints`
    :param merge_proto: Merge protonation states of `Space_Constraints`.
    It must be done if `mut_space_n_types` excludes protonation states,
    and must not be done otherwise.
    :return:
    """
    num_mut = [mut_space_n_types] * num_active
    proto_mapping = AminoAcidDict().proto_mapping
    if constraints:
        if len(constraints) > num_active:
            raise ValueError('More constrained positions than active ones')
        for i, c in enumerate(constraints):
            aas = c.split()[1:]
            if merge_proto:
                num_mut[i] = len(set(proto_mapping[x] if x in proto_mapping else x for x in aas))
            else:
                num_mut[i] = len(aas)
    return reduce(op.mul, num_mut)


def scale(data: t.Union[t.List[float], np.ndarray], a: float, b: float) -> t.List[float]:
    """
    Linearly scale data between a and b
    :param data: Either a list of numbers or a numpy array
    :param a: lower bound
    :param b: upper bound
    :return: scaled data
    """
    min_, max_ = min(data), max(data)

    def _scale(x):
        return (b - a) * (x - min_) / (max_ - min_) + a

    return _scale(data) if isinstance(data, np.ndarray) else [_scale(x) for x in data]


def decompose_into_singletons(df: pd.DataFrame):
    singleton_scores = defaultdict(lambda: np.nan, {
        (x.pos.split('-')[0], x.seq_subset): x.affinity
        for x in df.itertuples() if len(x.seq_subset) == 1})

    def score_singleton(row, pos):
        return singleton_scores[(row.pos.split('-')[pos], row.seq_subset[pos])]

    is_singleton = df.seq_subset.apply(lambda s: len(s) == 1)
    singletons = df[is_singleton].copy()
    not_singletons = df[~is_singleton].copy()
    singletons['Sai'], singletons['Saj'] = None, None
    not_singletons['Sai'] = [score_singleton(x, 0) for x in not_singletons.itertuples()]
    not_singletons['Saj'] = [score_singleton(x, 1) for x in not_singletons.itertuples()]
    return pd.concat([singletons, not_singletons])


if __name__ == '__main__':
    raise RuntimeError()
