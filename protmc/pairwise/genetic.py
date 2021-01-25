import typing as t
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from itertools import chain, product, starmap, filterfalse
from warnings import warn

import genetic.operators as ops
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import ray
from genetic.base import Mutator, Operators, Callback
from genetic.evolvers import GenericEvolver
from numba import njit
from tqdm import tqdm

from protmc.common import AminoAcidDict
from protmc.common.utils import scale

Gene = t.TypeVar('Gene')
Individual = t.TypeVar('Individual')

Positions = t.NamedTuple('Positions', [('fst', np.ndarray),
                                       ('snd', np.ndarray)])
Types = t.NamedTuple('Types', [('fst', np.ndarray),
                               ('snd', np.ndarray),
                               ('map', t.Dict[str, int]),
                               ('map_rev', t.Dict[int, str])])
Bounds = t.NamedTuple('Bounds', [('lower', t.Optional[float]),
                                 ('upper', t.Optional[float])])
Columns = t.NamedTuple('Columns', [('pos', str),
                                   ('seq_subset', str),
                                   ('affinity', str),
                                   ('stability', str)])
Record = t.NamedTuple('Record', [('age', int), ('score', float)])
Stats = t.NamedTuple('Stats', [('NumGenes', int),
                               ('NumPositions', int),
                               ('MutSpaceSize', int),
                               ('Score', float),
                               ('NumCC', int),
                               ('CCNumNodes', int),
                               ('CCNumEdges', int),
                               ('CCSumDegrees', int),
                               ('CCMeanDegree', float),
                               ('CCMeanEdgeScore', float),
                               ('CCRawScore', float)])
IsOriginalCol = 'IsOriginalPair'


@dataclass
class GenePool:
    """
    A collection of genes where populations can be sampled from.
    Has two attributes: (1) `map` is a dictionary mapping genes to unique indices, and (2) `rev` - the reversed `map`.
    """
    map: t.Dict[Gene, int]
    rev: t.Dict[int, Gene]

    def spawn_population(
            self, population_size: int, individual_size: int,
            replace: bool = True) -> t.List[np.ndarray]:
        """
        Create random population sampled from GenePool,with individuals being indices
        (= values) of the `map` attribute.
        :param population_size: The size of the population.
        :param individual_size: The size of the individual.
        :param replace: Sample pool with replacement.
        :return: A list of random individuals.
        """
        idx = np.array(list(self.rev))
        return [np.random.choice(idx, individual_size, replace=replace) for _ in range(population_size)]

    def remap_idx(self, idx: np.ndarray) -> t.List[Gene]:
        return [self.rev[i] for i in idx]


@dataclass
class GeneticParams:
    """
    Dataclass holding parameters varying parameters of the GA.

    Population_size: a number of individuals in a single population.
    Score_kwargs: Keywords arguments for the scoring function.
    Individual_base_size: Initial number of genes in a single individual.
    Brood_size: A number of offsprings produced in a single crossover.
    Number_of_mates: A number of individuals used during crossover.
    Mutable_fraction: A fraction of mutable genes of an individual.
    Mutation_prob: A probability to use regular mutation during the mutation event.
    Deletion_size: A size of a deletion.
    Deletion_prob: A probability to use deletion during the mutation event.
    Acquisition_size: A size of addition sampled from a gene pool.
    Acquisition_prob: A probability to use acquisition during the mutation event.
    Tournaments_selection: A number of tournaments for `ktournament` op during the selection event.
    Tournaments_policy: A number of tournaments for `ktournament` op during the selection policy event.
    """
    Population_size: int
    Score_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    Individual_base_size: int = 50
    Brood_size: int = 1
    Number_of_mates: int = 2
    Mutable_fraction: float = 0.1
    Mutation_prob: float = 0.6
    Deletion_size: int = 1
    Deletion_prob: float = 0.2
    Acquisition_size: int = 1
    Acquisition_prob: float = 0.2
    Tournaments_selection: int = 20
    Tournaments_policy: int = 20
    Use_cache: bool = False


@dataclass
class ParsingParams:
    """
    Dataclass holding changeable params for preparing results for the genetic algorithm.
    It is passed to `prepare_data` function.

    `Results`: Either a path to a "results" table or a DataFrame. The latter holds five columns.
    Their default names are specified in `Results_columns` attribute.
    `Results_columns`: A `Columns` namedtuple, holding column names for
    'position', 'sequence', 'sequence subset', 'affinity', and 'stability'.
    Mind the order! We expect the `RESULTS` DataFrame to have the same columns in that exact order.
    `Affinity_bounds`: A `Bounds` namedtuple holding lower and upper bounds for affinity (both can be `None`).
    `Affinity_cap`: ...
    `Stability_bounds`: A `Bounds` namedtuple holding lower and upper bounds for stability (both can be `None`).
    `Scale_range`: A `Bounds` namedtuple holding lower and upper bounds to scale affinity values.
    After the scaling, we refer to these as "scores".
    `Reverse_score`: Whether to multiply scores by -1.
    """
    Results: t.Union[str, pd.DataFrame]
    Results_columns: Columns = Columns('pos', 'seq_subset', 'affinity', 'stability')
    Affinity_bounds: Bounds = Bounds(None, None)
    Affinity_cap: Bounds = Bounds(None, None)
    Stability_bounds: Bounds = Bounds(None, None)
    Scale_range: Bounds = Bounds(0, 1)
    Reverse_score: bool = True
    Use_singletons: bool = False
    Exclude_types: t.List[t.Tuple[t.Union[str, int], str]] = field(default_factory=list)


@dataclass
class ParsingResults:
    """
    Dataclass holding an output of `prepare_data` function.
    Contains following elements:

    `df`: pandas DataFrame object filtered according to `ParsingParams`
    `GenePool`: GenePool dataclass instance containing the pool of genes.
    `Types`: A namedtuple with four elements:
    (1) an array of indices of amino acid types at the first element of each pair in the `GenePool`;
    (2) the same as (1) for the second element;
    (3) a mapping between string-valued types and their unique indices;
    (4) the (3) reversed for convenience.
    `Positions`: A namedtuple with two elements:
    (1) an array of protein chain positions for the first element of each pair in the `GenePool`,
    (2) the same as (2) for the second element.
    """
    df: pd.DataFrame
    GenePool: GenePool
    Types: Types
    Positions: Positions


class RichIndividual:
    def __init__(self, genes_idx: np.ndarray, parsing_results: ParsingResults,
                 score_fn: t.Callable[[np.ndarray], float]):
        self.genes_idx = genes_idx
        self.genes = parsing_results.GenePool.remap_idx(genes_idx)
        self.graph = self._as_graph()
        self.stats = self._create_stats(score_fn, parsing_results.Positions, parsing_results.Types)
        self.mut_space = self._mutation_space(parsing_results)

    def _as_graph(self) -> nx.MultiGraph:
        return nx.MultiGraph(list(((*map(int, u.pos.split('-')), {'s': u.score}) for u in self.genes)))

    def _create_stats(self, score_fn: t.Callable[[np.ndarray], float], ps: Positions, ts: Types) -> Stats:
        ccs = list(nx.connected.connected_components(self.graph))
        cc = max(ccs, key=len)
        g = self.graph.subgraph(cc)
        scores = [e[2]['s'] for e in g.edges(data=True)]
        degrees = [d[1] for d in nx.degree(g)]
        return Stats(
            NumGenes=len(self.genes),
            NumPositions=count_unique_positions(self.genes_idx, (ps.fst, ps.snd)),
            MutSpaceSize=mut_space_size(self.genes_idx, (ts.fst, ts.snd), (ps.fst, ps.snd), estimate=False),
            Score=score_fn(self.genes_idx),
            NumCC=len(ccs),
            CCNumNodes=len(g.nodes),
            CCNumEdges=len(g.edges),
            CCSumDegrees=sum(degrees),
            CCMeanDegree=float(np.mean(degrees)),
            CCMeanEdgeScore=float(np.mean(scores)),
            CCRawScore=sum(scores)
        )

    def _mutation_space(self, parsing_results: ParsingResults):
        ts1, ts2 = parsing_results.Types.fst, parsing_results.Types.snd
        ps1, ps2 = parsing_results.Positions.fst, parsing_results.Positions.snd
        i_ts = np.hstack((ts1[self.genes_idx], ts2[self.genes_idx]))
        i_ps = np.hstack((ps1[self.genes_idx], ps2[self.genes_idx]))
        masks = [i_ps == u for u in np.unique(i_ps)]
        i_ts_masked = [i_ts[m] for m in masks]
        types = [[parsing_results.Types.map_rev[i] for i in np.unique(x)]
                 for x in i_ts_masked]
        return list(zip(np.unique(i_ps), types))


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
        {cols.stability: 'mean', cols.affinity: 'mean'}
    )

    # Filter pairs
    def is_singleton(p: str):
        return len(set(p.split('-'))) == 1

    singletons_idx = df[cols.pos].apply(is_singleton)
    pairs = df[~singletons_idx]
    singletons = df[singletons_idx]
    pairs['IsOriginalPair'] = True
    if params.Use_singletons:
        if not len(singletons):
            warn('No singletons; check the input table with the results')
            df = pairs
        else:
            pairs_covered = {(pos, aa) for _, pos, aa in pairs[[cols.pos, cols.seq_subset]].itertuples()}
            Row = namedtuple('Row', cols)
            derived_pairs = pd.DataFrame(
                list(
                    filterfalse(
                        lambda r: (r[0], r[1]) in pairs_covered,
                        starmap(
                            lambda p1, p2: Row(f'{p1[0]}-{p2[0]}', f'{p1[1]}{p2[1]}', p1[2] + p2[2], p1[3] + p2[3]),
                            filter(
                                lambda x: int(x[0][0]) < int(x[1][0]),
                                product(zip(
                                    singletons[cols.pos].apply(lambda x: x.split('-')[0]),
                                    singletons[cols.seq_subset],
                                    singletons[cols.stability],
                                    singletons[cols.affinity]), repeat=2))))))
            derived_pairs['IsOriginalPair'] = False
            df = pd.concat([derived_pairs, pairs]).sort_values(list(cols))
    else:
        df = pairs

    # Filter by bounds
    idx = np.ones(len(df)).astype(bool)
    if params.Affinity_bounds.lower is not None:
        idx &= df[cols.affinity] > params.Affinity_bounds.lower
    if params.Affinity_bounds.upper is not None:
        idx &= df[cols.affinity] < params.Affinity_bounds.upper
    if params.Stability_bounds.lower is not None:
        idx &= df[cols.stability] > params.Stability_bounds.lower
    if params.Stability_bounds.upper is not None:
        idx &= df[cols.stability] < params.Stability_bounds.upper
    df = df[idx]

    # Cap affinity at certain values
    l, h = params.Affinity_cap
    if l is not None:
        df.loc[df[cols.affinity] < l, cols.affinity] = l
    if h is not None:
        df.loc[df[cols.affinity] > h, cols.affinity] = h

    # Convert scores
    scores = np.array(df[cols.affinity])
    if params.Reverse_score:
        scores = -scores
    df[cols.affinity] = scale(scores, params.Scale_range.lower, params.Scale_range.upper)

    return df


def prepare_pool(df: pd.DataFrame, params: ParsingParams) -> GenePool:
    """
    Prepare pool of genes - a tuple with two elements:
    (1) mapping between genes and their indices, and (2) the same mapping reversed.
    A gene is a NamedTuple with for elements: (1) '-'-separated pair of positions,
    (2) whole active seq, (3) pair of positions seq, (4) score.
    Other functions working with `GenePool` rely on this particular gene structure.
    :param df: An output of `AffinitySearch`.
    :param params: Parsing parameters.
    """
    cols = params.Results_columns
    Gene_ = namedtuple('Gene', [cols.pos, cols.seq_subset, 'score'])
    genes = (Gene_(*x[1:]) for x in df[
        [cols.pos, cols.seq_subset, cols.affinity]].itertuples())
    genes_mapping = {g: i for i, g in enumerate(genes)}
    genes_mapping_rev = {v: k for k, v in genes_mapping.items()}
    return GenePool(genes_mapping, genes_mapping_rev)


def prepare_types(pool: GenePool) -> Types:
    """
    Takes a `GenePool` as input and parses it to obtain four elements wrapped into `Types` NamedTuple:
    (1) Integer-valued amino-acid types of the first position packed into a numpy array,
    (2) Same as previous, for the second position, (3) Mapping between amino-acid one-letter codes
    and their integer associates, and (4) The latter reversed.
    """
    types_pos1, types_pos2 = map(lambda i: [x[1][i] for x in pool.map], [0, 1])
    types_pool = sorted(set(types_pos1 + types_pos2))
    types_map = {x: i for i, x in enumerate(types_pool)}
    types_map_rev = {v: k for k, v in types_map.items()}
    types_pos1, types_pos2 = map(lambda types: np.array([types_map[x] for x in types]),
                                 [types_pos1, types_pos2])
    return Types(types_pos1, types_pos2, types_map, types_map_rev)


def prepare_pos(pool: GenePool) -> Positions:
    """
    Takes a `GenePool` as input and produces two elements wrapped into `Positions` NamedTuple:
    (1) Integer-valued numpy array of protein sequence positions corresponding to a first character
    in a sequence subset of each gene in the `GenePool`, (2) The same as latter, corresponding to a
    second character.
    """
    return Positions(*map(lambda i: np.array([int(x[0].split('-')[i]) for x in pool.map]), [0, 1]))


def prepare_data(
        params: ParsingParams,
        df_parser: t.Callable[[ParsingParams], pd.DataFrame] = prepare_df,
        pool_parser: t.Callable[[pd.DataFrame, ParsingParams], GenePool] = prepare_pool,
        types_parser: t.Callable[[GenePool], Types] = prepare_types,
        pos_parser: t.Callable[[GenePool], Positions] = prepare_pos) -> ParsingResults:
    """
    TODO: this is an interface function; describe data preparation process here
    :param params:
    :param df_parser:
    :param pool_parser:
    :param types_parser:
    :param pos_parser:
    :return:
    """
    df = df_parser(params)
    pool = pool_parser(df, params)
    return ParsingResults(df, pool, types_parser(pool), pos_parser(pool))


@njit()
def gaussian(x: t.Union[np.ndarray, float], mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    :return: Value of the gaussian with mean `mu` and std `sigma` at the point `x`.
    """
    return 1 / (2 * np.pi) ** (1 / 2) / sigma ** 2 * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


@njit()
def gaussian_penalty(x: float, x_peak: float, sigma: float):
    """
    :return: Returns the value of the gaussian with mean `x_peak` and std `sigma` at the point `x`,
    scaled in such a way that its value at `x_peak` equals one.
    Lower values imply higher deviations of `x` from `x_peak` and vice versa.
    """
    return 1 / gaussian(x_peak, x_peak, sigma) * gaussian(x, x_peak, sigma)


@njit()
def mut_space_size(
        indiv: np.ndarray, ts: t.Tuple[np.ndarray, np.ndarray],
        ps: t.Tuple[np.ndarray, np.ndarray], estimate=True) -> t.Union[int, np.ndarray]:
    """
    Calculate mutation space size.
    :param indiv: An array of indices pointing to genes of the GenePool.
    :param ts: A pair of arrays of amino acid types, which can be indexed by `indiv`.
    :param ps: A pair of arrays of positions, which can be indexed by `indiv`.
    :param estimate: If True returns sum of the logarithms of the position-wise
    mutation space sizes.
    Useful to prevent integer overflow.
    :return: (Estimate of) the mutation space size.
    """
    ts1, ts2 = ts
    ps1, ps2 = ps
    i_ts = np.hstack((ts1[indiv], ts2[indiv]))
    i_ps = np.hstack((ps1[indiv], ps2[indiv]))
    masks = [i_ps == u for u in np.unique(i_ps)]
    i_ts_masked = [i_ts[m] for m in masks]
    n_types = np.array([len(np.unique(x)) for x in i_ts_masked])
    if estimate:
        return np.sum(np.log(n_types))
    return np.prod(n_types)


@njit()
def count_unique_positions(indiv: np.ndarray, ps: t.Tuple[np.ndarray, np.ndarray]) -> int:
    """
    :param indiv: An array of indices pointing to genes of the GenePool.
    :param ps: A pair of arrays of positions, which can be indexed by `indiv`.
    :return: A number of unique protein positions in the `individual`.
    """
    return len(np.unique(np.concatenate((ps[0][indiv], ps[1][indiv]))))


@njit()
def sum_scores(indiv, scores):
    return scores[np.unique(indiv)].sum()


def score(indiv: np.ndarray, scores: np.ndarray,
          mut_space_estimator: t.Callable[[np.ndarray], t.Union[int, float]],
          pos_counter: t.Callable[[np.ndarray], int],
          min_size: int = 10, max_size: int = 100,
          pen_mut: bool = True, sigma_mut: float = 5, desired_space: float = 3.7 * np.log(18),
          pen_dup: bool = True, sigma_dup: float = 10,
          pen_pos: bool = False, sigma_pos: float = 10, desired_pos: int = 4,
          expose: bool = False) -> float:
    """
    Compute the score (fitness) of the individual.
    The score's base value is the sum of gene scores. It's conditionally weighted three times,
    where each weight penalizes the deviation from the expected value.
    :param indiv: An array of indices pointing to genes of the GenePool.
    :param scores: An array of scores which can be indexed by `indiv`.
    :param mut_space_estimator: A callable taking `indiv` and returning the estimate
    size of the mutational space.
    :param pos_counter: A callable taking `indiv` and returning a number of unique positions.
    :param min_size: A min size of an individual; lower sizes imply 0 score.
    :param max_size: A max size of an individual; higher sizes imply 0 score.
    :param pen_mut: Whether to penalize for deviation of mutation space from `desired_space`.
    Note that switching this value off will likely cause oversized individuals.
    :param sigma_mut: Standard deviation (see `gaussian_penalty` docs).
    :param desired_space: A desired size of a mutation space size (estimate).
    :param pen_dup: Whether to penalize gene duplication. It uses `gaussian_penalty` with `x_peak`
    argument equal to the `len(indiv)`.
    Note that switching this off will likely cause individuals consisting of single repeated gene.
    :param sigma_dup: Standard deviation (see `gaussian_penalty` docs).
    :param pen_pos: Whether to penalize for the deviations from the `desired_pos`.
    :param sigma_pos: Standard deviation (see `gaussian_penalty` docs).
    :param desired_pos: Desired number of unique positions within `indiv`
    :param expose: If true print (base_score, duplication_penalty, mut_space_penalty, num_pos_penalty).
    :return: A score capturing how well individual meets our expectations.
    Specifically, with the default weights, the best individual is has no duplicate genes, has
    mutation space size as close to `desired_space` as possible,
    and, of course, a well-scoring composition of genes.
    """
    if len(indiv) < min_size or len(indiv) > max_size:
        return 0

    s = sum_scores(indiv, scores)

    if pen_dup:
        duplication_penalty = gaussian_penalty(
            len(np.unique(indiv)), len(indiv), sigma_dup)
    else:
        duplication_penalty = 1.0

    if pen_mut:
        mutation_space_penalty = gaussian_penalty(
            mut_space_estimator(indiv), desired_space, sigma_mut)
    else:
        mutation_space_penalty = 1.0

    if pen_pos:
        num_pos = pos_counter(indiv)
        num_pos_penalty = gaussian_penalty(num_pos, desired_pos, sigma_pos)
    else:
        num_pos_penalty = 1.0

    if expose:
        print(s, duplication_penalty, mutation_space_penalty, num_pos_penalty)

    return s * duplication_penalty * mutation_space_penalty * num_pos_penalty


def random_combination(mating_group: t.List[t.Tuple[Individual, Record]],
                       brood_size: int) -> t.List[Individual]:
    """
    Mating function. For `len(mating_group) = 2` equivalent to single-point crossover.
    Combines the genes of the `mating_group`s individuals into a single pool,
    then samples `brood_size` number of new individuals from this pool, without replacement.
    If individuals in the `mating_group` have different sizes, offsprings will have
    the exact same sizes in random order.
    :param mating_group: A group of tuples (indiv, rec) to be mated.
    :param brood_size: Number of offsprings.
    """
    individuals = tuple([x[0] for x in mating_group])
    if brood_size > len(individuals):
        raise NotImplementedError(
            f'Currently supports only brood sizes <= than a number of individuals; '
            f'got {brood_size} <= {len(individuals)}')
    sizes = np.array([len(indiv) for indiv in individuals])
    return combine(individuals, brood_size, sizes)


@njit()
def combine(individuals, brood_size, sizes):
    """
    A helper function for the `random_combination` for the faster operation of the latter.
    """
    np.random.shuffle(sizes)
    genes = np.concatenate(individuals)
    return [np.random.choice(genes, size, replace=False)
            for _, size in zip(range(brood_size), sizes)]


@njit()
def mutate_fraction(
        indiv: np.ndarray, pool: np.ndarray,
        mutable_fraction: float) -> np.ndarray:
    """
    Mutate a fraction of `indiv`s genes by sampling from `pool` without replacement.
    Note that some of the genes may stay the same. The efficiency of the operation depends
    on the size difference between `indiv` and `pool`.
    :param indiv: An array of indices pointing to genes of the `pool`.
    :param pool: A pool of indices pointing to genes in the `GenePool`.
    :param mutable_fraction: A fraction of `indiv` allowed to mutate.
    :return: A mutated individual.
    """
    indiv = indiv.copy()
    mask = np.random.binomial(1, mutable_fraction, len(indiv))
    sampled = np.random.choice(pool, mask.sum())
    indiv[np.where(mask == 1)[0]] = sampled
    return indiv


@njit()
def deletion(indiv: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Delete a portion of `indiv`'s genes.
    :param indiv: An array of indices pointing to genes of the `GenePool`.
    :param size: A size of the deletion.
    :return: A mutated individual.
    """
    if size == 0:
        return indiv
    return np.delete(indiv, np.random.randint(0, len(indiv), size))


@njit()
def acquisition(indiv: np.ndarray, pool: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Acquire portion of random genes sampled from `pool`.
    :param indiv: An array of indices pointing to genes of the `GenePool`.
    :param pool: A gene pool.
    :param size: Acquisition size.
    :return: A mutated individual.
    """
    return np.append(indiv, np.random.choice(pool, size=size))


class ChoiceMutator(Mutator):
    def __init__(self, mutators: t.List[t.Callable[[Individual], Individual]], prob: t.List[float]):
        """
        :param mutators: A list of mutating functions.
        :param prob: A list of probabilities.
        Thus, each mutator[i] has a prob[i] usage probability.
        """
        if len(mutators) != len(prob):
            raise ValueError
        self._mutators = mutators
        self._prob = prob

    def _choose(self):
        return np.random.choice(self._mutators, p=self._prob)

    def __call__(self, individuals: t.List[Individual], **kwargs) -> t.List[Individual]:
        """
        For each of the `individuals`, chooses a mutator function from `mutators` attribute
        and applies it to an individual.
        :param individuals: A list of individuals.
        :return: A list of mutated individuals.
        """
        mutators = [self._choose() for _ in range(len(individuals))]
        return [f(indiv) for f, indiv in zip(mutators, individuals)]


class NoPopulations(Exception):
    pass


class ProgressSaver(Callback):
    def __init__(self, freq: int):
        self.freq = freq
        self.generation = 0
        self.acc = []

    def __call__(self, individuals: t.List[Individual], records: t.List[Record], operators: Operators) \
            -> t.Tuple[t.List[Individual], t.List[Record], Operators]:
        self.generation += 1
        if self.generation % self.freq == 0:
            scores = np.array([r.score for r in records])
            self.acc.append((
                self.generation,
                np.mean(scores),
                np.max(scores)))
        return individuals, records, operators


class Genetic:
    """
    Class uses `ParsingResults` returned by `prepare_data` and `GeneticParams` instance to setup
    (Generic*) GA operators and evolver of generations (see `genetic` docs for details).

    Usage following init:
    (1) `spawn_populations` to sample N populations from the gene pool and save them internally.
    (2) `evolve` to evolve X generations of the current populations.
    """

    def __init__(self, parsing_results: ParsingResults, genetic_params: GeneticParams,
                 callbacks: t.Optional[t.List[Callback]] = None):
        self._parsing_results = parsing_results
        self._genetic_params = genetic_params
        self.callbacks = [] or callbacks

        # Setup helpers for operators
        self.score_func = partial(
            score,
            mut_space_estimator=partial(
                mut_space_size,
                ts=(parsing_results.Types.fst, parsing_results.Types.snd),
                ps=(parsing_results.Positions.fst, parsing_results.Positions.snd)),
            pos_counter=partial(
                count_unique_positions,
                ps=(parsing_results.Positions.fst, parsing_results.Positions.snd)),
            scores=np.array([x[-1] for x in parsing_results.GenePool.map]),
            **genetic_params.Score_kwargs)
        self.selector_fn = partial(
            ops.ktournament, genetic_params.Tournaments_selection, lambda x: x.score,
            genetic_params.Number_of_mates, replace=True)
        self.policy_fn = partial(ops.ktournament, genetic_params.Tournaments_policy, lambda x: x.score)
        self.crossover_fn = partial(random_combination, brood_size=genetic_params.Brood_size)
        self._pool = np.array(list(self.parsing_results.GenePool.map.values())).astype(np.int32)

        # Setup operators
        self._estimator = ops.GenericEstimator(
            func=self.score_func, hash_func=lambda x: hash(x.data.tobytes()), cache=genetic_params.Use_cache)
        self._recorder = ops.GenericRecorder(
            start=lambda indiv, score_: Record(0, score_),
            update=lambda indiv, rec: Record(rec.age + 1, rec.score))
        self._selector = ops.GenericSelector(
            selector=self.selector_fn,
            nmates=genetic_params.Number_of_mates)
        self._crossover = ops.GenericCrossover(
            crossover=self.crossover_fn,
            nmates=genetic_params.Number_of_mates,
            broodsize=genetic_params.Brood_size)
        self._mutator = ChoiceMutator(
            mutators=[
                partial(mutate_fraction, pool=self._pool, mutable_fraction=genetic_params.Mutable_fraction),
                partial(deletion, size=genetic_params.Deletion_size),
                partial(acquisition, pool=self._pool, size=genetic_params.Acquisition_size)
            ],
            prob=[genetic_params.Mutation_prob, genetic_params.Deletion_prob, genetic_params.Acquisition_prob])
        self._policy = ops.GenericPolicy(selector=self.policy_fn)
        self._evolver = GenericEvolver()
        self.ops = Operators(self._recorder, self._estimator, self._selector,
                             self._crossover, self._mutator, self._policy)

        # Setup placeholders
        self.populations: t.Optional[t.List[t.List[np.ndarray]]] = None
        self.records: t.Optional[t.List[t.List[Record]]] = None

    def __repr__(self):
        return f"GA with {len(self.populations) if self.populations else 0} populations"

    @property
    def parsing_results(self):
        return self._parsing_results

    @property
    def genetic_params(self):
        return self._genetic_params

    def _spawn_one(self) -> t.List[np.ndarray]:
        return self.parsing_results.GenePool.spawn_population(
            population_size=self.genetic_params.Population_size,
            individual_size=self.genetic_params.Individual_base_size)

    def spawn(self, n: int, overwrite: bool = True) -> t.List[t.List[np.ndarray]]:
        pops = [self._spawn_one() for _ in range(n)]
        if overwrite:
            self.populations = pops
            self.records = None
        return pops

    def flatten(self):
        if self.populations is None:
            raise NoPopulations
        return (list(chain.from_iterable(self.populations)),
                list(chain.from_iterable(self.records)) if self.records else [None] * len(self.populations))

    def select_n_best(self, n: int, overwrite: bool = True):
        """
        Selects `n` best individuals per population based on the score function value.
        :param n: Number of individuals to take per population.
        :param overwrite: Overwrite current population with the selection.
        :return: A top-n selection.
        """

        def sel_population(pop, recs):
            if recs is None:
                recs = [None] * len(pop)
            sel = sorted(zip(pop, recs), key=lambda x: self.score_func(x[0]), reverse=True)
            return [sel[i][0] for i in range(n)], [sel[i][1] for i in range(n)]

        if self.populations is None:
            raise NoPopulations

        records = [None] * len(self.populations) if self.records is None else self.records

        selections = [sel_population(p, r) for p, r in zip(self.populations, records)]
        populations, records = [x[0] for x in selections], [x[1] for x in selections]
        if self.records is None:
            records = None
        if overwrite:
            self.populations, self.records = populations, records
        return populations, records

    def dump(self, path: str = './genetic_results.joblib', compress: int = 3) -> str:
        if self.populations is None:
            raise NoPopulations
        recs = self.records or [None] * len(self.populations)
        return joblib.dump((self.populations, recs), path, compress)[0]

    def evolve(self, num_gen: int, parallel: bool = True, overwrite: bool = True, verbose: bool = False) \
            -> t.Tuple[t.List[t.List[np.ndarray]], t.List[t.List[Record]]]:
        """
        Evolve populations currently held in `populations` attribute.
        :param num_gen: Number of generations to evolve each population.
        :param parallel: Whether to evolve populations in parallel. This is obviously a preferred way.
        To use, first initialize `ray` with `ray.init(num_cpus=X)`.
        :param overwrite: Overwrite `populations` and `records` attributes with the results of the run.
        :param verbose: Progress bar.
        :return: A tuple with (1) a list of evolved populations
        (where each population is a list of individuals, i.e. numpy arrays),
        and (2) a list with lists of individuals' records.
        """
        if self.populations is None:
            raise NoPopulations
        data_to_evolve = zip(self.populations, self.records or [None] * len(self.populations))
        if verbose:
            data_to_evolve = tqdm(data_to_evolve, desc='Evolving population: ', total=len(self.populations))
        if len(self.populations) > 1 and parallel:
            if self.callbacks:
                warn('Parallel execution want preserve internal state of callbacks')
            evolver = partial(self._evolver.evolve, num_gen, self.ops, self.genetic_params.Population_size)
            results = ray.get([_evolve_remote.remote(evolver, x[0], x[1]) for x in data_to_evolve])
        else:
            results = list(map(
                lambda x: self._evolver.evolve(
                    num_gen, operators=self.ops,
                    gensize=self.genetic_params.Population_size,
                    individuals=x[0], records=x[1], callbacks=self.callbacks),
                data_to_evolve
            ))
        pops, recs = [x[0] for x in results], [x[1] for x in results]
        if overwrite:
            self.populations, self.records = pops, recs
        return pops, recs


@ray.remote
def _evolve_remote(evolver, individuals, records):
    return evolver(individuals, records)


if __name__ == '__main__':
    raise RuntimeError
