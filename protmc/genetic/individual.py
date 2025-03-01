import operator as op
import typing as t
from functools import reduce
from itertools import groupby, combinations
from statistics import mean

import networkx as nx
import pandas as pd
from more_itertools import peekable

from .base import EdgeGene, MultiGraphEdge, AbstractGraphIndividual, AbstractSeqIndividual, SeqGene
from .utils import add_gene, genes2_graph, mut_space_size


class GraphIndividual(AbstractGraphIndividual):
    """
    A type of Individual for the `GA` with `MultiGraph` being a core data structure.
    From the latter, three values are derived via explicit call of `upd*` methods:
    score, the mutation space size, and the number of positions.
    """

    def __init__(self, genes: t.Optional[t.Collection[EdgeGene]], coupling_threshold: float = 0,
                 max_mut_space: bool = True, max_num_positions: bool = False,
                 graph: t.Optional[nx.Graph] = None, upd_immediately: bool = True):
        """
        To init, one of the `genes`, `graph` must be provided.
        :param genes: A collection of `EdgeGene`s to create a `MultiGraph` object from.
        Each of the `EdgeGene`s follows the convention that the first position (`P1`)
        is smaller than the second one (`P2`). As `MultiGraph` is undirected,
        it doesn't discriminate between `P1` and `P2` nodes. Use `genes` to retrieve the list
        of `EdgeGenes` or `gene` attribute of the graph's edges.
        :param coupling_threshold: A value to separate "weak" genes (<) from "strong" genes (>=).
        :param max_mut_space: Penalize mutation space size based on the largest CC.
        :param max_num_positions: Penalize the number of positions based on the largest CC.
        :param graph: A `networkx.Graph` object where `EdgeGene`s are stored in `gene` edge attributes.
        If provided, `genes` argument will be ignored.
        :param upd_immediately: Update `score`, `mut_space_size` and `num_pos` during the initialization.
        """
        if not (genes or graph):
            raise ValueError('Nothing to init from')
        self._graph = graph or genes2_graph(genes, coupling_threshold)
        self.coupling_threshold = coupling_threshold
        self.max_mut_space = max_mut_space
        self.max_num_positions = max_num_positions
        self._mut_space_size = 0
        self._score = 0.0
        self._n_pos = 0
        if upd_immediately:
            self.upd()

    def __len__(self) -> int:
        return self._graph.number_of_edges()

    def __repr__(self) -> str:
        return f'GraphIndividual(num_genes={len(self.genes())}, num_pos={self.n_pos}, ' \
               f'score={self.score}, space_size={self.mut_space_size})'

    def copy(self) -> 'GraphIndividual':
        # Mutable graph object is the only one requiring explicit copying
        new = GraphIndividual(
            None, self.coupling_threshold, self.max_mut_space, self.max_num_positions,
            self._graph.copy(), False)
        new._set_mut_space_size(self._mut_space_size)
        new._set_score(self._score)
        new._set_n_pos(self._n_pos)
        return new

    @property
    def n_pos(self) -> int:
        """A number of positions -- nodes of the graph object."""
        return self._n_pos

    def _set_n_pos(self, value: int):
        """Exists for lighter copying."""
        self._n_pos = value

    @property
    def mut_space_size(self) -> float:
        """Mutation space size imposed by the graph's edges."""
        return self._mut_space_size

    def _set_mut_space_size(self, value: float):
        """Exists for lighter copying."""
        self._mut_space_size = value

    @property
    def score(self) -> float:
        """The sum of `S` attributes of the graph's `EdgeGene`s."""
        return self._score

    def _set_score(self, value: float):
        """Exists for lighter copying."""
        self._score = value

    @property
    def graph(self) -> nx.MultiGraph:
        return self._graph

    @property
    def ccs(self) -> t.Iterable[nx.MultiGraph]:
        """Connected components as subgraphs induced by the strong links."""
        induced_graph = self._graph.edge_subgraph(self.strong_links())
        return (induced_graph.subgraph(x) for x in nx.connected_components(induced_graph))

    def genes(self) -> t.List[EdgeGene]:
        """A list of `gene` attributes stored within graph's `EdgeGene`s."""
        return [data['gene'] for _, _, data in self._graph.edges.data()]

    def strong_links(self) -> t.Iterator[MultiGraphEdge]:
        """An iterable over edges with coupling >= `coupling_threshold`."""
        return (e for e in self._graph.edges if self._graph.edges[e]['gene'].C >= self.coupling_threshold)

    def weak_links(self) -> t.Iterator[MultiGraphEdge]:
        """An iterable over edges with coupling < `coupling_threshold`."""
        return (e for e in self._graph.edges if self._graph.edges[e]['gene'].C < self.coupling_threshold)

    def validate(self):
        """
        Checks whether the `graph` object complies with "structural constraints".
        Specifically, there may be only one "weak" or any number of "strong" edges between any pair of nodes.
        The mix of "weak" and "strong" edges is disallowed.
        """

        def is_group_valid(group: t.Iterable[MultiGraphEdge]):
            group = list(group)
            is_weak = all(self.graph.edges[e]['gene'].C < self.coupling_threshold for e in group)
            is_strong = all(self.graph.edges[e]['gene'].C >= self.coupling_threshold for e in group)
            if not (is_weak or is_strong):
                return False
            if is_weak and len(group) != 1:
                return False
            return True

        return all(map(is_group_valid, (gg for _, gg in groupby(self.graph.edges, lambda e: e[:2]))))

    def upd_score(self) -> float:
        """Overwrite the `score` with an updated value."""
        self._score = round(sum(data['gene'].S for _, _, data in self._graph.edges.data()), 4)
        return self._score

    def upd_mut_space_size(self) -> float:
        """Overwrite the `mut_space_size` with an updated value."""
        links = peekable(self.strong_links())
        if not links.peek(0):
            self._mut_space_size = 0
            return 0
        agg = max if self.max_mut_space else sum
        self._mut_space_size = agg(map(mut_space_size, self.ccs))
        return self._mut_space_size

    def upd_n_pos(self) -> int:
        """Overwrite the `n_pos` with an updated value."""
        agg = max if self.max_num_positions else sum
        self._n_pos = agg(map(len, nx.connected_components(self.graph)))
        return self._n_pos

    def upd(self) -> 'GraphIndividual':
        """Call all `upd*` methods."""
        self.upd_score(), self.upd_mut_space_size(), self.upd_n_pos()
        return self

    def remove_genes(self, genes: t.Iterable[EdgeGene], update: bool = True) -> 'GraphIndividual':
        """Remove all of the provided genes. If graph hasn't changed, or `update` is False,
        the removal doesn't result in `upd` method call."""
        graph_changed = False
        for g in genes:
            e = (g.P1, g.P2, g.A1 + g.A2)
            if e in self._graph.edges:
                self._graph.remove_edge(*e)
                graph_changed = True
        if graph_changed and update:
            return self.upd()
        return self

    def add_genes(self, genes: t.Iterable[EdgeGene], update: bool = True) -> 'GraphIndividual':
        """Add provided genes to the graph. If graph hasn't changed, or `update` is False,
        the addition is not followed by the `upd` method call."""
        genes = peekable(genes)
        if not genes.peek(0):
            return self
        graph_changed = reduce(op.and_, map(self._add_gene, genes))
        if graph_changed and update:
            return self.upd()
        return self

    def _add_gene(self, gene: EdgeGene) -> bool:
        return add_gene(gene, self.graph, self.coupling_threshold)


class AverageIndividual(GraphIndividual):
    """
    A sub-type of the `GraphIndividual` where the score between any pair of nodes is averaged.
    This pushes the GA into using fewer edges and more positions per Individual/CC.
    """

    def __repr__(self) -> str:
        return f'AverageIndividual(num_genes={len(self.genes())}, num_pos={self.n_pos}, ' \
               f'score={self.score}, space_size={self.mut_space_size})'

    def copy(self) -> 'AverageIndividual':
        # Mutable graph object is the only one requiring explicit copying
        new = AverageIndividual(
            None, self.coupling_threshold, self.max_mut_space, self.max_num_positions,
            self._graph.copy(), False)
        new._set_mut_space_size(self._mut_space_size)
        new._set_score(self._score)
        new._set_n_pos(self._n_pos)
        return new

    def upd_score(self) -> 'AverageIndividual':
        """Overwrite the `score` with an updated value."""
        self._score = round(
            sum(mean(data['gene'].S for _, _, data in gg) for _, gg in groupby(
                sorted(self._graph.edges.data(), key=lambda e: (e[0], e[1])),
                lambda e: (e[0], e[1]))),
            4)
        return self._score


class SeqIndividual(AbstractSeqIndividual):
    """
    A type of Individual for the GA where the core data structure is a set of `SeqGene`s.
    If the latter do not overlap, they compose a unique sequence.
    """

    def __init__(self, genes: t.Iterable[SeqGene], upd_on_init: bool = True):
        """
        :param genes: An iterable with `SeqGene`s.
        :param upd_on_init: Call `upd*` methods during initialization.
        """
        self._genes = set(genes)
        self._score = 0
        self._n_pos = 0
        if upd_on_init:
            self.upd()

    def __len__(self):
        return len(self.genes())

    def __contains__(self, gene: SeqGene):
        return gene in self._genes

    def __eq__(self, other: 'SeqIndividual'):
        return self._genes == other._genes

    def __hash__(self):
        return sum(hash(g) for g in self._genes)

    def copy(self):
        return SeqIndividual(self._genes.copy(), upd_on_init=True)

    def genes(self) -> t.Set[SeqGene]:
        """Current set of genes."""
        return self._genes

    @property
    def n_pos(self) -> int:
        """A union of genes' positions."""
        return self._n_pos

    @property
    def score(self) -> float:
        """Sum of genes' scores. Zero if `pos_overlap` is `True`."""
        return self._score

    def pos_overlap(self):
        """`True` if the sub-sequences' (genes) positions overlap and `False` otherwise."""
        pairs = combinations((set(g.Pos) for g in self.genes()), 2)
        return any(x & y for x, y in pairs)

    def upd(self):
        """Explicitly update the `score` and `n_pos` based on current genes."""
        if self.pos_overlap():
            self._score = 0
        else:
            self._score = sum(g.S for g in self.genes())
        self._n_pos = len(reduce(op.or_, (set(g.Pos) for g in self.genes())))

    def add_genes(self, genes: t.Iterable[SeqGene], update: bool = True) -> 'SeqIndividual':
        """A union of current and provided genes. Returns new `SeqIndividual` instance.
        If `update` is `False`, the change in genes is not saved."""
        genes_upd = self._genes | set(genes)
        if update:
            self._genes = genes_upd
        return SeqIndividual(genes_upd)

    def remove_genes(self, genes: t.Iterable[SeqGene], update: bool = True) -> 'SeqIndividual':
        """Delete provided genes from the current set. Returns new `SeqIndividual` instance.
        If `update` is `False`, the change in genes is not saved."""
        genes_upd = self._genes - set(genes)
        if update:
            self._genes = genes_upd
        return SeqIndividual(genes_upd)


_GraphIndividual = t.TypeVar('_GraphIndividual', bound=t.Union[GraphIndividual, SeqIndividual])
Individual = t.TypeVar('Individual', bound=t.Union[GraphIndividual, SeqIndividual])


def sep_graph_ind(ind: GraphIndividual) -> t.Iterator[GraphIndividual]:
    """Spawn sub-individuals based on ccs of `ind`."""
    return (GraphIndividual(
        genes=None, coupling_threshold=ind.coupling_threshold,
        max_mut_space=ind.max_mut_space,
        max_num_positions=ind.max_num_positions,
        graph=g, upd_immediately=True) for g in ind.ccs)


def population_to_df(population: t.Iterable[Individual]) -> pd.DataFrame:
    """Create a `DataFrame` from genes of individuals in `population`."""

    def agg_ind(i, ind):
        df = pd.DataFrame(ind.genes())
        df['Ind'] = i
        return df

    return pd.concat([agg_ind(i, ind) for i, ind in enumerate(population)])


def graph_population_from_df(
        df: t.Union[pd.DataFrame, str],
        individual_type: _GraphIndividual = GraphIndividual,
        ind_column: str = 'Ind',
        **kwargs) -> t.List[_GraphIndividual]:
    """
    :param df: A `DataFrame`, where each row can be wrapped into an `EdgeGene`.
    :param individual_type: a type of individual accepting a list of `EdgeGene`s.
    :param ind_column: Column name separating individuals.
    :param kwargs: Passed to each individual during initialization.
    """
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')
    if not isinstance(individual_type, GraphIndividual) and not isinstance(individual_type, AverageIndividual):
        raise ValueError(f'Individual type {individual_type} is not supported.')

    for col in ['P1', 'P2', 'A1', 'A2', 'C', 'S']:
        if col not in df.columns:
            raise ValueError(f'Expected column {col} in the df')

    gene_groups = df.groupby(ind_column).apply(lambda group: [EdgeGene(*g[1:]) for g in group.itertuples()])
    return [individual_type(genes, **kwargs) for genes in gene_groups]


def seq_population_from_df(
        df: t.Union[pd.DataFrame, str],
        **kwargs) -> t.List[SeqIndividual]:
    """
    :param df: A `DataFrame`, where each row can be wrapped into an `SeqGene`.
    :param kwargs:  Passed to each individual during initialization.
    """
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')
    for col in ['Seq', 'Pos', 'S']:
        if col not in df.columns:
            raise ValueError(f'Expected column {col} in the df')

    gene_groups = df.groupby('Ind').apply(
        lambda group: [SeqGene(g.Seq, g.Pos, g.S) for g in group.itertuples()])
    return [SeqIndividual(genes, **kwargs) for genes in gene_groups]


if __name__ == '__main__':
    raise RuntimeError
