import operator as op
import typing as t
from functools import reduce
from itertools import chain, groupby, tee, starmap
from math import log
from statistics import mean

import networkx as nx
from more_itertools import peekable

from .base import Gene, MultiGraphEdge, AbstractIndividual, GenePool


# TODO: don't forget to state the assumption that gene.P1 < gene.P2


def mut_space_size(graph: nx.MultiGraph, estimate=True) -> int:
    xs = sorted(chain.from_iterable(((fr, key[0]), (to, key[1])) for fr, to, key in graph.edges))
    gs = (len(set(g)) for _, g in groupby(xs, key=op.itemgetter(0)))
    if estimate:
        return sum(map(log, gs))
    return reduce(op.mul, gs)


def add_gene(gene: Gene, graph: nx.MultiGraph, coupling_threshold: float) -> bool:
    if gene.C >= coupling_threshold:
        return add_strong_gene(gene, graph, coupling_threshold)
    return add_weak_gene(gene, graph)


def get_node_space(graph, node):
    adj_keys = chain.from_iterable(
        ((k, n) for k in v) for n, v in graph[node].items())
    return (k[0] if n > node else k[1] for k, n in adj_keys)


def add_strong_gene(gene: Gene, graph: nx.MultiGraph, coupling_threshold: float) -> bool:
    key = gene.A1 + gene.A2
    edge = (gene.P1, gene.P2, key)
    if not graph.has_edge(gene.P1, gene.P2):
        graph.add_edge(*edge, gene=gene)
        return True
    keys = graph[gene.P1][gene.P2]
    if key in keys:
        return False
    if len(keys) > 1:
        graph.add_edge(*edge, gene=gene)
        return True
    if graph[gene.P1][gene.P2][next(iter(keys))]['gene'].C < coupling_threshold:
        graph.remove_edge(gene.P1, gene.P2, next(iter(keys)))
    graph.add_edge(*edge, gene=gene)
    return True


def add_weak_gene(gene: Gene, graph: nx.MultiGraph) -> bool:
    key = gene.A1 + gene.A2
    edge = (gene.P1, gene.P2, key)
    if any(starmap(
            lambda p, a: p in graph and not any(x == a for x in get_node_space(graph, p)),
            [(gene.P1, gene.A1), (gene.P2, gene.A2)])):
        return False
    if graph.has_edge(gene.P1, gene.P2):
        keys = graph[gene.P1][gene.P2]
        if len(keys) > 1:
            return False
        graph.remove_edge(gene.P1, gene.P2, next(iter(keys)))
    graph.add_edge(*edge, gene=gene)
    return True


def genes2_graph(genes: t.Collection[Gene], coupling_threshold: float) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for gene in genes:
        add_gene(gene, graph, coupling_threshold)
    return graph


class GenericIndividual(AbstractIndividual):
    # multiple strong links, single weak link
    def __init__(self, genes: t.Optional[GenePool], coupling_threshold: float = 0, max_mut_space: bool = True,
                 graph: t.Optional[nx.Graph] = None, upd_immediately: bool = True):
        if not (genes or graph):
            raise ValueError('Nothing to init from')
        self._graph = graph or genes2_graph(genes, coupling_threshold)
        self.coupling_threshold = coupling_threshold
        self.max_mut_space = max_mut_space
        self._mut_space_size = 0
        self._score = 0.0
        if upd_immediately:
            self.upd()

    def __len__(self) -> int:
        return self._graph.number_of_edges()

    def __repr__(self) -> str:
        return f'GenericIndividual(num_genes={len(self.genes())}, num_pos={self.n_pos}, ' \
               f'score={self.score}, space_size={self.mut_space_size})'

    def copy(self) -> 'GenericIndividual':
        new = GenericIndividual(None, self.coupling_threshold, self.max_mut_space, self._graph.copy(), False)
        new.set_mut_space_size(self._mut_space_size)
        new.set_score(self._score)
        return new

    @property
    def n_pos(self) -> int:
        return len(self._graph)

    @property
    def mut_space_size(self) -> float:
        return self._mut_space_size

    def set_mut_space_size(self, value: float):
        """Exists for lighter copying"""
        self._mut_space_size = value

    @property
    def score(self) -> float:
        return self._score

    def set_score(self, value: float):
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

    def genes(self) -> t.List[Gene]:
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

    def upd(self) -> 'GenericIndividual':
        self.upd_score(), self.upd_mut_space_size()
        return self

    def remove_genes(self, genes: t.Iterable[Gene], update: bool = True) -> 'GenericIndividual':
        graph_changed = False
        for g in genes:
            e = (g.P1, g.P2, g.A1 + g.A2)
            if e in self._graph.edges:
                self._graph.remove_edge(*e)
                graph_changed = True
        if graph_changed and update:
            return self.upd()
        return self

    def add_genes(self, genes: t.Iterable[Gene], update: bool = True) -> 'GenericIndividual':
        genes = peekable(genes)
        if not genes.peek(0):
            return self
        graph_changed = reduce(op.and_, map(self.add_gene, genes))
        if graph_changed and update:
            return self.upd()
        return self

    def add_gene(self, gene: Gene) -> bool:
        return add_gene(gene, self.graph, self.coupling_threshold)


class AverageIndividual(GenericIndividual):
    def upd_score(self) -> 'AverageIndividual':
        self._score = round(
            sum(mean(data['gene'].S for _, _, data in gg) for _, gg in groupby(
                sorted(self._graph.edges.data(), key=lambda e: (e[0], e[1])),
                lambda e: (e[0], e[1]))),
            4)
        return self._score


class AverageFlexibleIndividual(AverageIndividual):
    pass


if __name__ == '__main__':
    raise RuntimeError
