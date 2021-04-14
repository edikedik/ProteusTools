import operator as op
import typing as t
from functools import reduce
from itertools import groupby
from statistics import mean

import networkx as nx
import pandas as pd
from more_itertools import peekable

from .base import EdgeGene, MultiGraphEdge, AbstractGraphIndividual, AbstractSeqIndividual, Gene, SeqGene
from .utils import add_gene, genes2_graph, mut_space_size


# TODO: don't forget to state the assumption that gene.P1 < gene.P2


class GraphIndividual(AbstractGraphIndividual):
    # multiple strong links, single weak link
    def __init__(self, genes: t.Optional[t.Collection[EdgeGene]], coupling_threshold: float = 0,
                 max_mut_space: bool = True, max_num_positions: bool = False,
                 graph: t.Optional[nx.Graph] = None, upd_immediately: bool = True):
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
        return self._n_pos

    def _set_n_pos(self, value: int):
        self._n_pos = value

    @property
    def mut_space_size(self) -> float:
        return self._mut_space_size

    def _set_mut_space_size(self, value: float):
        """Exists for lighter copying"""
        self._mut_space_size = value

    @property
    def score(self) -> float:
        return self._score

    def _set_score(self, value: float):
        """Exists for lighter copying"""
        self._score = value

    @property
    def graph(self) -> nx.MultiGraph:
        return self._graph

    @property
    def ccs(self) -> t.Iterable[nx.MultiGraph]:
        """Connected components of the subgraph induced by the strong links"""
        induced_graph = self._graph.edge_subgraph(self.strong_links())
        return (induced_graph.subgraph(x) for x in nx.connected_components(induced_graph))

    def genes(self) -> t.List[EdgeGene]:
        return [data['gene'] for _, _, data in self._graph.edges.data()]

    def strong_links(self) -> t.Iterator[MultiGraphEdge]:
        return (e for e in self._graph.edges if self._graph.edges[e]['gene'].C >= self.coupling_threshold)

    def weak_links(self) -> t.Iterator[MultiGraphEdge]:
        return (e for e in self._graph.edges if self._graph.edges[e]['gene'].C < self.coupling_threshold)

    def validate(self):
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
        self._score = round(sum(data['gene'].S for _, _, data in self._graph.edges.data()), 4)
        return self._score

    def upd_mut_space_size(self) -> float:
        links = peekable(self.strong_links())
        if not links.peek(0):
            self._mut_space_size = 0
            return 0
        agg = max if self.max_mut_space else sum
        self._mut_space_size = agg(map(mut_space_size, self.ccs))
        return self._mut_space_size

    def upd_n_pos(self) -> int:
        agg = max if self.max_num_positions else sum
        self._n_pos = agg(map(len, nx.connected_components(self.graph)))
        return self._n_pos

    def upd(self) -> 'GraphIndividual':
        self.upd_score(), self.upd_mut_space_size(), self.upd_n_pos()
        return self

    def remove_genes(self, genes: t.Iterable[EdgeGene], update: bool = True) -> 'GraphIndividual':
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
        genes = peekable(genes)
        if not genes.peek(0):
            return self
        graph_changed = reduce(op.and_, map(self.add_gene, genes))
        if graph_changed and update:
            return self.upd()
        return self

    def add_gene(self, gene: EdgeGene) -> bool:
        return add_gene(gene, self.graph, self.coupling_threshold)


class AverageIndividual(GraphIndividual):

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
        self._score = round(
            sum(mean(data['gene'].S for _, _, data in gg) for _, gg in groupby(
                sorted(self._graph.edges.data(), key=lambda e: (e[0], e[1])),
                lambda e: (e[0], e[1]))),
            4)
        return self._score


class SeqIndividual(AbstractSeqIndividual):
    def __init__(self, genes: t.Collection[SeqGene], upd_on_init: bool = True):
        self._genes = set(genes)
        self._score = 0
        self._n_pos = 0
        if upd_on_init:
            self.upd()

    def __len__(self):
        return len(self.genes)

    def copy(self):
        return SeqIndividual(self._genes.copy(), upd_on_init=True)

    @property
    def genes(self) -> t.Set[SeqGene]:
        return self._genes

    @property
    def n_pos(self) -> int:
        return self._n_pos

    @property
    def score(self) -> float:
        return self._score

    def upd(self):
        self._score = sum(g.S for g in self.genes)
        self._n_pos = len(reduce(op.or_, (set(g.Pos) for g in self.genes)))

    def add_genes(self, genes: t.Collection[SeqGene], update: bool = False) -> 'SeqIndividual':
        genes_upd = self._genes | set(genes)
        if update:
            self._genes = genes_upd
        return SeqIndividual(genes_upd)

    def remove_genes(self, genes: t.Collection[SeqGene], update: bool = False) -> 'SeqIndividual':
        genes_upd = self._genes - set(genes)
        if update:
            self._genes = genes_upd
        return SeqIndividual(genes_upd)


Individual = t.TypeVar('Individual', bound=GraphIndividual)


def sep_graph_ind(ind: GraphIndividual) -> t.Iterator[GraphIndividual]:
    """Spawn sub-individuals based on ccs"""
    return (GraphIndividual(
        genes=None, coupling_threshold=ind.coupling_threshold,
        max_mut_space=ind.max_mut_space,
        max_num_positions=ind.max_num_positions,
        graph=g, upd_immediately=True) for g in ind.ccs)


def population_to_df(population: t.Iterable[Individual]) -> pd.DataFrame:
    def agg_ind(i, ind):
        df = pd.DataFrame(ind.genes())
        df['Ind'] = i
        return df

    return pd.concat([agg_ind(i, ind) for i, ind in enumerate(population)])


def population_from_df(
        df: t.Union[pd.DataFrame, str],
        individual_type=GraphIndividual,
        **kwargs) -> t.List[GraphIndividual]:
    if isinstance(df, str):
        df = pd.read_csv(df, sep='\t')
    for col in ['P1', 'P2', 'A1', 'A2', 'C', 'S']:
        if col not in df.columns:
            raise ValueError(f'Expected column {col} in the df')

    gene_groups = df.groupby('Ind').apply(lambda group: [EdgeGene(*g[1:]) for g in group.itertuples()])
    return [individual_type(genes, **kwargs) for genes in gene_groups]


if __name__ == '__main__':
    raise RuntimeError
