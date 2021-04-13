import operator as op
import typing as t
from functools import reduce
from itertools import chain, groupby, starmap
from math import log

import networkx as nx

from .base import EdgeGene


def get_attr(graph: nx.Graph, attr: str):
    return [data[attr] for _, _, data in graph.edges.data()]


def mut_space(graph: nx.Graph) -> t.Iterable[t.Tuple[int, t.List[str]]]:
    genes = get_attr(graph, 'gene')
    xs = sorted(chain.from_iterable(((g.P1, g.A1), (g.P2, g.A2)) for g in genes))
    return ((g, sorted(set(x[1] for x in gg))) for g, gg in groupby(xs, key=op.itemgetter(0)))


def mut_space_size(graph: nx.MultiGraph, estimate=True) -> int:
    space = mut_space(graph)
    sizes = (len(gg) for g, gg in space)
    if estimate:
        return sum(map(log, sizes))
    return reduce(op.mul, sizes)


def add_gene(gene: EdgeGene, graph: nx.MultiGraph, coupling_threshold: float) -> bool:
    if gene.C >= coupling_threshold:
        return add_strong_gene(gene, graph, coupling_threshold)
    return add_weak_gene(gene, graph)


def get_node_space(graph, node):
    adj_keys = chain.from_iterable(
        ((k, n) for k in v) for n, v in graph[node].items())
    return (k[0] if n > node else k[1] for k, n in adj_keys)


def add_strong_gene(gene: EdgeGene, graph: nx.MultiGraph, coupling_threshold: float) -> bool:
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


def add_weak_gene(gene: EdgeGene, graph: nx.MultiGraph) -> bool:
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


def genes2_graph(genes: t.Collection[EdgeGene], coupling_threshold: float) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for gene in genes:
        add_gene(gene, graph, coupling_threshold)
    return graph


if __name__ == '__main__':
    raise RuntimeError
