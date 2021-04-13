import operator as op
import typing as t
from functools import reduce
from itertools import chain, groupby, starmap, product
from math import log

import networkx as nx
import numpy as np
from ray.util.multiprocessing import Pool
from tqdm import tqdm

from .base import EdgeGene


def get_attr(graph: nx.Graph, attr: str):
    return [data[attr] for _, _, data in graph.edges.data()]


def mut_space(graph: nx.Graph) -> t.Iterator[t.Tuple[int, t.List[str]]]:
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


def dist(i1: int, g1: nx.Graph, i2: int, g2: nx.Graph) -> t.Tuple[int, int, float]:
    """
    Compute distance between two graphs.
    Dist(g1, g2) = sum of number of different types at each position.
    """
    space1, space2 = map(dict, map(mut_space, [g1, g2]))
    d = 0
    for k in set(list(space1) + list(space2)):
        if k in space1 and k in space2:
            d += len(set(space1[k]).symmetric_difference(set(space2[k])))
            continue
        if k in space1:
            d += len(set(space1[k]))
        if k in space2:
            d += len(set(space2[k]))
    return i1, i2, d


def dist_mat(graphs: t.Sequence[nx.Graph], n_jobs: int = 20):
    """Compute distance matrix using `Dist(g1, g2) = sum of number of different types at each position`."""
    size = len(graphs)
    base = np.zeros(shape=(size, size))
    staged_data = tqdm(product(enumerate(graphs), repeat=2), total=size ** 2, desc='Distance matrix')
    with Pool(n_jobs) as workers:
        distances = workers.map(lambda x: dist(x[0][0], x[0][1], x[1][0], x[1][1]), staged_data)
    for i, j, d in distances:
        base[i][j] = d
        base[j][i] = d
    return base


if __name__ == '__main__':
    raise RuntimeError
