import logging
import operator as op
import typing as t
from functools import reduce
from itertools import product, filterfalse, starmap, chain, combinations
from warnings import warn

import numpy as np
import pandas as pd

from .base import EdgeGene, ParsingParams, ParsingResult, SeqGene
from ..protmc.common.base import AminoAcidDict
from ..protmc.common.utils import scale


def _filter_bounds(df: pd.DataFrame, var_name: str, bound: t.Optional[float] = None, lower: bool = True):
    if bound is None:
        return np.ones(len(df)).astype(bool)
    idx = df[var_name] > bound if lower else df[var_name] < bound
    sign = '>' if lower else '<'
    logging.info(f'There are {idx.sum()} observations with {var_name} {sign} {bound}')
    return idx


def filter_bounds(df: pd.DataFrame, params: ParsingParams) -> pd.DataFrame:
    """
    Filter the "affinity" DataFrame based on stability and affinity bounds provided in `params`.
    """
    cols = params.Results_columns
    df = df.copy()
    lower = [(cols.affinity, params.Affinity_bounds.lower),
             (cols.stability_apo, params.Stability_apo_bounds.lower),
             (cols.stability_holo, params.Stability_holo_bounds.lower)]
    upper = [(cols.affinity, params.Affinity_bounds.upper),
             (cols.stability_apo, params.Stability_apo_bounds.upper),
             (cols.stability_holo, params.Stability_holo_bounds.upper)]
    idx = reduce(op.and_, chain(
        starmap(lambda col, bound: _filter_bounds(df, col, bound, True), lower),
        starmap(lambda col, bound: _filter_bounds(df, col, bound, False), upper)))
    df = df[idx]
    logging.info(f'Filtered to {idx.sum()} observations according to single-variable bounds')

    # Filter by joint stability bounds
    l, h = params.Stability_joint_bounds
    if l is not None:
        idx = (df[cols.stability_apo] > l) & (df[cols.stability_holo] > l)
        df = df[idx]
        logging.info(f'Filtered to {idx.sum()} records with stability_apo & stability_holo > {l}')
    if h is not None:
        idx = (df[cols.stability_apo] < h) & (df[cols.stability_holo] < h)
        df = df[idx]
        logging.info(f'Filtered to {idx.sum()} records with stability_apo & stability_holo < {h}')
    return df


def map_proto_states(df: pd.DataFrame, params: ParsingParams) -> pd.DataFrame:
    """
    Replace sequences in `df` by mapping protonated amino acids to their unprotonated versions.
    """
    df = df.copy()
    proto_map = AminoAcidDict().proto_mapping
    cols = params.Results_columns

    def _map(seq: str):
        return "".join([proto_map[c] if c in proto_map else c for c in seq])

    df[cols.seq_subset] = df[cols.seq_subset].apply(_map)
    df = df.groupby(
        [cols.pos, cols.seq_subset], as_index=False
    ).agg(
        {cols.stability_apo: 'mean', cols.stability_holo: 'mean', cols.affinity: 'mean'}
    )
    return df


def prepare_df(params: ParsingParams) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses a DataFrame, typically an output of AffinitySearch, to be used in the GA.
    The workflow (therefore, the end result) is depends entirely on the provided params.
    See the `ParsingParams` documentation for more details.
    :param params: `ParsingParams` dataclass instance.
    :return: Parsed df ready to be sliced into a `GenePool`. The second element is the DataFrame with singletons.
    """
    cols = params.Results_columns
    if isinstance(params.Results, pd.DataFrame):
        df = params.Results[list(cols)].dropna().copy()
    elif isinstance(params.Results, str):
        df = pd.read_csv(params.Results, sep='\t')[list(cols)].dropna()
    else:
        raise TypeError('Unsupported type of the `Results` attribute')
    logging.info(f'Read initial DataFrame with {len(df)} (non-NaN) records')

    # Which positions were already present? We must know before any filtering
    pos_covered = set(df[cols.pos])
    if params.Exclude_pairs is not None:
        pos_covered |= {f'{p1}-{p2}' for p1, p2 in params.Exclude_pairs}

    if params.Exclude_types:
        ps = {str(x[0]) for x in params.Exclude_types}
        ts = {x[1] for x in params.Exclude_types}
        p1_, p2_ = map(
            lambda i: list(zip(df[cols.pos].apply(lambda x: x.split('-')[i]),
                               df[cols.seq_subset].apply(lambda x: x[i]))),
            [0, 1])
        idx1, idx2 = map(
            lambda p: np.array([x in ps and y in ts for x, y in p]),
            [p1_, p2_])
        df = df[~(idx1 | idx2)]

    df = map_proto_states(df, params)
    logging.info(f'Mapped proto states. Records: {len(df)}')

    # Filter pairs
    def is_singleton(p: str):
        return len(set(p.split('-'))) == 1

    singletons_idx = df[cols.pos].apply(is_singleton)
    pairs = df[~singletons_idx].copy()
    singletons = df[singletons_idx].copy()
    pairs['is_original'] = True
    logging.info(f'Identified {len(singletons)} singletons records and {len(pairs)} pairs records')

    if params.Use_singletons:
        if not len(singletons):
            warn('No singletons; check the input table with the results')
            df = pairs
        else:
            derived_pairs = pd.DataFrame(  # Wrap into df
                filterfalse(  # Leave only new pairs
                    lambda r: r[0] in pos_covered,
                    starmap(  # Wrap into columns
                        lambda p1, p2: (f'{p1[0]}-{p2[0]}', f'{p1[1][0]}{p2[1][0]}',
                                        p1[2] + p2[2], p1[3] + p2[3], p1[4] + p2[4]),
                        filter(  # First position is, by convention, lower than the second
                            lambda x: int(x[0][0]) < int(x[1][0]),
                            product(zip(  # Make combinations of singletons' with themselves
                                singletons[cols.pos].apply(lambda x: x.split('-')[0]),
                                singletons[cols.seq_subset],
                                singletons[cols.stability_apo],
                                singletons[cols.stability_holo],
                                singletons[cols.affinity]), repeat=2)))),
                columns=[cols.pos, cols.seq_subset, cols.stability_apo, cols.stability_holo, cols.affinity])
            derived_pairs['is_original'] = False
            logging.info(f'Derived {len(derived_pairs)} pairs from singletons.')
            df = pd.concat([derived_pairs, pairs]).sort_values(list(cols))
            logging.info(f'Merged {len(pairs)} existing and {len(derived_pairs)} derived pairs. Records: {len(df)}')
    else:
        df = pairs

    if params.Use_couplings and len(singletons):
        score_mapping = {(pos.split('-')[0], seq[0]): aff for _, pos, seq, aff in singletons[
            [cols.pos, cols.seq_subset, cols.affinity]].itertuples()}
        df['coupling'] = [
            round(abs(aff - score_mapping[(pos.split('-')[0], seq[0])] - score_mapping[(pos.split('-')[1], seq[1])]), 4)
            if (pos.split('-')[0], seq[0]) in score_mapping and (pos.split('-')[1], seq[1]) in score_mapping
            else np.nan
            for _, pos, seq, aff in df[[cols.pos, cols.seq_subset, cols.affinity]].itertuples()]
        failed_idx = df['coupling'].isna()
        num_failed = int(failed_idx.sum())
        if num_failed:
            failed_pairs = ",".join(f'{x.pos}_{x.seq_subset}' for x in df[failed_idx].itertuples())
            logging.warning(f'There are {num_failed} pairs with no singleton(s) score(s): {failed_pairs}')
        if params.Default_coupling is not None:
            df.loc[failed_idx, 'coupling'] = params.Default_coupling
            logging.info(f'Set default value {params.Default_coupling} on '
                         f'{num_failed} pairs with no singleton(s) score(s)')
        else:
            df = df[~failed_idx]
            logging.info(f'Excluded {num_failed} pairs with no singleton(s) score(s)')
    else:
        df['coupling'] = np.nan

    # Only now we exclude positions; this solves the issue of using singletons
    # in the context of possibly failed affinity calculations. Indeed, if the
    # calculation has failed, using pairs derived from singletons for such
    # positions would be wrong.
    if params.Exclude_pairs:
        df = df[df[cols.pos].apply(
            lambda p: tuple(map(int, p.split('-'))) not in params.Exclude_pairs)]
        logging.info(f'Excluded pairs {params.Exclude_pairs}. Records: {len(df)}')

    df = filter_bounds(df, params)

    # Cap affinity at certain values
    l, h = params.Affinity_cap
    if l is not None:
        idx = df[cols.affinity] < l
        df.loc[idx, cols.affinity] = l
        logging.info(f'Capped {idx.sum()} affinity records at lower bound {l}')
    if h is not None:
        idx = df[cols.affinity] > h
        logging.info(f'Capped {idx.sum()} affinity records at upper bound {h}')
        df.loc[idx, cols.affinity] = h

    # Convert scores
    scores = np.array(df[cols.affinity])
    if params.Reverse_score:
        scores = -scores
        logging.info(f'Reversed affinity scores sign')
    l, h = params.Scale_range
    if l is not None and h is not None:
        df[cols.affinity] = np.round(scale(scores, l, h), 4)
        logging.info(f'Scaled affinity scores between {l} and {h}')
    return df, singletons


def prepare_graph_pool(df: pd.DataFrame, params: ParsingParams) -> t.Tuple[EdgeGene, ...]:
    """
    Prepare the gene pool -- a tuple of `EdgeGene`s.
    """
    cols = params.Results_columns
    return tuple(
        EdgeGene(int(pos.split('-')[0]), int(pos.split('-')[1]), seq[0], seq[1], score, coupling)
        for _, pos, seq, score, coupling in df[
            [cols.pos, cols.seq_subset, cols.affinity, 'coupling']].itertuples()
    )


def _estimate(seq: t.Tuple[t.Tuple[str, str], ...],
              mapping: t.Mapping[t.Tuple[t.Tuple[str, str], ...], float],
              params: ParsingParams,
              size: int) -> float:
    """
    :param seq: A sequence in the form of (('AA', 'Pos'), ...)
    :param mapping: Mapping of the sequences to energies.
    :param params: A dataclass holding parsing parameters.
    :param size: Max size of larger sequences to start the recursion.
    :return: Sum (!) of sub-sequences' energies.
    """
    if len(seq) == 1:
        try:
            return mapping[seq]
        except KeyError:
            warn(f'Seq {seq} could not be estimated')
            return 0

    combs = combinations(seq, size)
    s = 0
    for c in combs:
        c = tuple(c)
        try:
            s += mapping[c]
        except KeyError:
            s += _estimate(c, mapping, params, size - 1)
    return s


def _aff_mapping(df: pd.DataFrame, params: ParsingParams) -> t.Dict[t.Tuple[t.Tuple[str, str], ...], float]:
    """
    Create the mapping from sequences with lengths <= `params.Seq_size_threshold`
    in the form of (('AA', 'Pos'), ...)to their affinities.
    """
    cols = params.Results_columns
    df = df[df[cols.seq_subset].apply(
        lambda s: len(s) <= params.Seq_size_threshold)][
        [cols.seq_subset, cols.pos, cols.affinity]]
    return {tuple(zip(seq, pos.split('-'))): s for _, seq, pos, s in df.itertuples()}


def estimate_seq_aff(df, params):
    """
    Recursively estimate larger sequence's energy from energies of smaller sequences (up to singletons).
    Warning! The current estimation strategy has been verified only on singletons.
    :param df: A `DataFrame` complying the same standards as required by `params`
    (i.e., having columns specified) in `Results_columns` attribute.
    :param params: A dataclass holding parsing parameters.
    :return: A `DataFrame` with a new column "affinity_estimate".
    """
    df = df.copy()
    cols = params.Results_columns
    mapping = _aff_mapping(df, params)
    df['affinity_estimate'] = [
        _estimate(tuple(zip(s, p.split('-'))), mapping, params, params.Seq_size_threshold)
        for _, s, p in df[[cols.seq_subset, cols.pos]].itertuples()
    ]
    return df


def prepare_singletons(df, params):
    """
    Change "seq" and "pos" columns of singletons from the form "AA", "P-P" to the form "A", "P".
    :param df: A `DataFrame` with columns specified in `params.Results_columns`.
    :param params: A dataclass holding parsing parameters.
    :return: The `DataFrame`, with changed "seq" and "pos" columns for singletons (if any).
    """
    df = df.copy()
    cols = params.Results_columns
    idx = df[cols.pos].apply(lambda x: len(set(x.split('-'))) == 1)
    df.loc[idx, cols.seq_subset] = df.loc[idx, cols.seq_subset].apply(lambda x: x[0])
    df.loc[idx, cols.pos] = df.loc[idx, cols.pos].apply(lambda x: x.split('-')[0])
    return df


def prepare_seq_df(params):
    """
    Prepares a "seq" `DataFrame` with rows ready to be wrapped into `SeqGene`s.
    The workflow (therefore, the end result) is depends entirely on the provided params.
    See the `ParsingParams` documentation for more details.
    :param params: A dataclass holding parsing parameters.
    """
    df = params.Results
    cols = params.Results_columns
    assert isinstance(df, pd.DataFrame)
    df = df.copy().dropna().drop_duplicates()
    logging.info(f'Initial df size: {len(df)}')
    df = map_proto_states(df, params)
    logging.info(f'Mapped proto states. Records: {len(df)}')
    df = prepare_singletons(df, params)
    logging.info(f'Prepared singletons')
    df = estimate_seq_aff(df, params)
    logging.info(f'Estimated affinity from {params.Seq_size_threshold}-sized seqs')
    idx = df[cols.seq_subset].apply(
        lambda x: len(x) > params.Seq_size_threshold)
    idx &= abs(df[cols.affinity] - df['affinity_estimate']) < params.Affinity_diff_threshold
    df = df[~idx]
    logging.info(f'Filtered out {idx.sum()} records due to estimation '
                 f'being accurate to the point of {params.Affinity_diff_threshold}. '
                 f'Records: {len(df)}')
    df = filter_bounds(df, params)
    logging.info(f'Filtered by affinity and stability bounds. Records: {len(df)}')
    n = params.Top_n_seqs
    if n is not None and n > 0:
        df = df.sort_values('affinity', ascending=True).groupby('pos').head(n)
    logging.info(f'Selected {n} best records per position set. Records: {len(df)}')
    return df


def prepare_seq_pool(df, params):
    """Wraps an output of the `prepare_seq_df` into a pool of `SeqGene`s"""
    cols = params.Results_columns
    return tuple(
        SeqGene(seq, tuple(map(int, p.split('-'))), -s)
        for _, seq, p, s in df[[cols.seq_subset, cols.pos, cols.affinity]].itertuples())


def prepare_data(params: ParsingParams) -> ParsingResult:
    """
    A primary interface function. If `params.Seq_df` is `True`,
    will prepare a `DataFrame` provided via `params.Results` and create a pool of `SeqGene`s from it.
    Otherwise, will prepare a pool of `EdgeGenes` for graph-based optimization.
    The filtering workflow can be inferred from `params`,
    `prepare_df` (for the pool of `EdgeGene`s) and
    `prepare_seq_df` (for the pool of `SeqGene`s, and their `logging` output.
    :param params: A dataclass holding parsing parameters.
    :return: A `ParsingParams` namedtuple with three elements:
    (1) a prepared `DataFrame`, (2) a `DataFrame` with singletons
    (without any filtering  applied; `None` in case `params.Seq_df is `True`), and
    (3) a pool of genes for the GA.
    """
    if params.Seq_df:
        df = prepare_seq_df(params)
        return ParsingResult(df, None, prepare_seq_pool(df, params))
    else:
        df, singletons = prepare_df(params)
        return ParsingResult(df, singletons, prepare_graph_pool(df, params))


if __name__ == '__main__':
    raise RuntimeError
