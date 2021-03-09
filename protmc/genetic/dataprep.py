import logging
import typing as t
from functools import reduce
from itertools import product, filterfalse, starmap, chain
from warnings import warn
import operator as op

import numpy as np
import pandas as pd

from protmc.common.base import AminoAcidDict
from protmc.common.utils import scale
from .base import Gene, GenePool, ParsingParams, ParsingResult


def _filter_bounds(df: pd.DataFrame, var_name: str, bound: t.Optional[float] = None, lower: bool = True):
    if bound is None:
        return np.ones(len(df)).astype(bool)
    idx = df[var_name] > bound if lower else df[var_name] < bound
    sign = '>' if lower else '<'
    logging.info(f'There are {idx.sum()} observations with {var_name} {sign} {bound}')
    return idx


def prepare_df(params: ParsingParams) -> pd.DataFrame:
    """
    Parses a DataFrame, typically an output of AffinitySearch, to be used in genetic algorithm.

    :param params: `ParsingParams` dataclass instance.
    :return: Parsed df ready to be sliced into a `GenePool`.
    """
    cols = params.Results_columns
    if isinstance(params.Results, pd.DataFrame):
        df = params.Results[list(cols)].dropna().copy()
    elif isinstance(params.Results, str):
        df = pd.read_csv(params.Results, sep='\t')[list(cols)].dropna()
    else:
        raise TypeError('Unsupported type of the `Results` attribute')
    logging.info(f'Read initial DataFrame with {len(df)} (non-NaN) records')

    # Map protonation states to single types
    def map_proto_states(seq: str) -> str:
        proto_map = AminoAcidDict().proto_mapping
        return "".join([proto_map[c] if c in proto_map else c for c in seq])

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

    df[cols.seq_subset] = df[cols.seq_subset].apply(map_proto_states)
    df = df.groupby(
        [cols.pos, cols.seq_subset], as_index=False
    ).agg(
        {cols.stability_apo: 'mean', cols.stability_holo: 'mean', cols.affinity: 'mean'}
    )
    logging.info(f'Mapped protonated states. Records: {len(df)}')

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
            # Which pairs were already present?
            pairs_covered = {(pos, aa) for _, pos, aa in pairs[[cols.pos, cols.seq_subset]].itertuples()}
            derived_pairs = pd.DataFrame(  # Wrap into df
                filterfalse(  # Leave only new pairs
                    lambda r: (r[0], r[1]) in pairs_covered,
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

    if len(singletons):
        score_mapping = {(pos.split('-')[0], seq[0]): aff for _, pos, seq, aff in singletons[
            [cols.pos, cols.seq_subset, cols.affinity]].itertuples()}
        df['coupling'] = [
            round(abs(aff - score_mapping[(pos.split('-')[0], seq[0])] - score_mapping[(pos.split('-')[1], seq[1])]), 4)
            if (pos.split('-')[0], seq[0]) in score_mapping and (pos.split('-')[1], seq[1]) in score_mapping
            else np.nan
            for _, pos, seq, aff in df[[cols.pos, cols.seq_subset, cols.affinity]].itertuples()]
        failed_idx = df['coupling'].isna()
        if failed_idx.any():
            failed_pairs = ",".join(f'{x.pos}_{x.seq_subset}' for x in df[failed_idx].itertuples())
            logging.warning(f'There are {len(df[failed_idx])} pairs with no singleton(s) score(s): {failed_pairs}')
        df = df[~failed_idx]
    else:
        df['coupling'] = np.nan

    # Only now we exclude positions; this solves the issue of using singletons
    # in the context of possibly failed affinity calculations. Indeed, if the
    # calculation is failed, using pairs derived from singletons for such
    # positions would be wrong.
    if params.Exclude_pairs:
        df = df[df[cols.pos].apply(
            lambda p: tuple(map(int, p.split('-'))) not in params.Exclude_pairs)]
        logging.info(f'Excluded pairs {params.Exclude_pairs}. Records: {len(df)}')

    # Filter by single-variable bounds
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
    df[cols.affinity] = np.round(scale(scores, params.Scale_range.lower, params.Scale_range.upper), 4)
    logging.info(f'Converted and scaled affinity scores. Final records: {len(df)}')
    return df


def prepare_pool(df: pd.DataFrame, params: ParsingParams) -> GenePool:
    """
    Prepare the gene pool
    """
    cols = params.Results_columns
    return tuple(
        Gene(int(pos.split('-')[0]), int(pos.split('-')[1]), seq[0], seq[1], score, coupling)
        for _, pos, seq, score, coupling in df[
            [cols.pos, cols.seq_subset, cols.affinity, 'coupling']].itertuples()
    )


def prepare_data(params: ParsingParams) -> ParsingResult:
    # TODO: docs for the interface function
    df = prepare_df(params)
    return ParsingResult(df, prepare_pool(df, params))


if __name__ == '__main__':
    raise RuntimeError
