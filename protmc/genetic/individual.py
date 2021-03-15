import operator as op
import typing as t
from functools import reduce
from itertools import chain, groupby, tee
from math import log
from statistics import mean

import networkx as nx
from more_itertools import peekable

from .base import Gene, MultiGraphEdge, AbstractIndividual, GenePool


# TODO: don't forget to state the assumption that gene.P1 < gene.P2


def genes2graph(genes: t.Collection[Gene]) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for g in genes:
        graph.add_edge(g.P1, g.P2, key=g.A1 + g.A2, gene=g)
    return graph


def genes2light_graph(genes: t.Collection[Gene], coupling_threshold: float) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    strong_genes = (g for g in genes if g.C >= coupling_threshold)
    weak_genes = (g for g in genes if g.C < coupling_threshold)
    for g in strong_genes:
        graph.add_edge(g.P1, g.P2, g.A1 + g.A2, gene=g)
    for g in weak_genes:
        if g.P1 in graph and g.P2 in graph[g.P1]:
            if len(graph[g.P1][g.P2]) == 0:
                graph.add_edge(g.P1, g.P2, g.A1 + g.A2, gene=g)
        else:
            graph.add_edge(g.P1, g.P2, g.A1 + g.A2, gene=g)
    return graph


def mut_space_size(graph: nx.MultiGraph, estimate=True) -> int:
    xs = sorted(chain.from_iterable(((fr, key[0]), (to, key[1])) for fr, to, key in graph.edges))
    gs = (len(set(g)) for _, g in groupby(xs, key=op.itemgetter(0)))
    if estimate:
        return sum(map(log, gs))
    return reduce(op.mul, gs)


class GenericIndividual(AbstractIndividual):
    # multiple strong links, single weak link
    def __init__(self, genes: GenePool, coupling_threshold: float = 0, max_mut_space: bool = True):
        self._graph = genes2light_graph(genes, coupling_threshold)
        self.coupling_threshold = coupling_threshold
        self.max_mut_space = max_mut_space
        self._n_pos = 0
        self._mut_space_size = 0
        self._score = 0.0
        self.upd()

    def __len__(self) -> int:
        return self._graph.number_of_edges()

    def __repr__(self) -> str:
        return f'GenericIndividual(num_genes={len(self.genes())}, num_pos={self.n_pos}, ' \
               f'score={self.score}, space_size={self.mut_space_size})'

    @property
    def n_pos(self) -> int:
        return len(self._graph)

    @property
    def mut_space_size(self) -> float:
        return self._mut_space_size

    @property
    def score(self) -> float:
        return self._score

    @property
    def graph(self) -> nx.MultiGraph:
        return self._graph

    @property
    def ccs(self) -> t.Iterable[nx.MultiGraph]:

        return (self._graph.subgraph(x) for x in nx.connected_components(
            self._graph.edge_subgraph(self.strong_links())))

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

    def remove_genes(self, genes: GenePool, update: bool = True) -> 'GenericIndividual':
        graph_changed = False
        for g in genes:
            e = (g.P1, g.P2, g.A1 + g.A2)
            if e in self._graph.edges:
                self._graph.remove_edge(*e)
                graph_changed = True
        if graph_changed and update:
            return self.upd()
        return self

    def add_genes(self, genes: GenePool, update: bool = True) -> 'GenericIndividual':
        graph_changed = reduce(op.and_, map(self.add_gene, genes))
        if graph_changed and update:
            return self.upd()
        return self

    def add_gene(self, g: Gene) -> bool:
        e = (g.P1, g.P2, g.A1 + g.A2)
        is_strong = g.C >= self.coupling_threshold
        if e not in self._graph.edges:
            if g.P1 in self._graph and g.P2 in self._graph[g.P1]:
                if len(self._graph[g.P1][g.P2]) == 0:
                    # No edges between two nodes -> add edge
                    self._graph.add_edge(*e, gene=g)
                    return True
                elif len(self._graph[g.P1][g.P2]) == 1:
                    key = list(self._graph[g.P1][g.P2])[0]
                    is_also_strong = self._graph[g.P1][g.P2][key]['gene'].C >= self.coupling_threshold
                    if is_strong and is_also_strong:
                        # If both are strong, add edge
                        self._graph.add_edge(*e, gene=g)
                        return True
                    else:
                        # Whatever the edge is -- replace it
                        self._graph.remove_edge(g.P1, g.P2, key)
                        self._graph.add_edge(*e, gene=g)
                        return True
                    # Otherwise, nothing to do
                elif is_strong:
                    # Assume that multiple edges imply all strong connections
                    # Hence, we add one more strong connection
                    self._graph.add_edge(*e, gene=g)
                    return True
            else:
                # At least one of the nodes is not present -> add edge
                self._graph.add_edge(*e, gene=g)
                return True
        return False


class AverageIndividual(GenericIndividual):
    def upd_score(self) -> 'AverageIndividual':
        self._score = round(
            sum(mean(data['gene'].S for _, _, data in gg) for _, gg in groupby(
                sorted(self._graph.edges.data(), key=lambda e: (e[0], e[1])),
                lambda e: (e[0], e[1]))),
            4)
        return self._score


class AverageFlexibleIndividual(AverageIndividual):

    def add_gene(self, g: Gene) -> bool:
        e = (g.P1, g.P2, g.A1 + g.A2)
        is_strong = g.C >= self.coupling_threshold
        if e not in self._graph.edges:
            if g.P1 in self._graph and g.P2 in self._graph[g.P1]:
                if len(self._graph[g.P1][g.P2]) == 0:
                    # No edges between two nodes -> add edge
                    self._graph.add_edge(*e, gene=g)
                    return True
                elif len(self._graph[g.P1][g.P2]) == 1:
                    key = list(self._graph[g.P1][g.P2])[0]
                    is_also_strong = self._graph[g.P1][g.P2][key]['gene'].C >= self.coupling_threshold
                    if is_strong and is_also_strong:
                        # if both are strong, add edge
                        self._graph.add_edge(*e, gene=g)
                        return True
                    else:
                        # If the edge is weak -- replace it
                        self._graph.remove_edge(g.P1, g.P2, key)
                        self._graph.add_edge(*e, gene=g)
                        return True
                    # Otherwise, nothing to do
                else:
                    # Assume that multiple edges imply all strong connections
                    scores = tee(v['gene'].S for v in self._graph[g.P1][g.P2].values())
                    average_score = mean(scores[0])
                    max_score = max(scores[1])
                    if g.S > average_score and g.S > max_score:
                        self._graph.add_edge(*e, gene=g)
                        return True
            else:
                # At least one of the nodes is not present -> add edge
                self._graph.add_edge(*e, gene=g)
                return True
        return False


if __name__ == '__main__':
    raise RuntimeError
