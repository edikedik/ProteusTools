import typing as t
import operator as op
from functools import reduce
from itertools import chain, groupby, combinations

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from logomaker import Logo, sequence_to_matrix

from protmc.genetic.base import SeqGene
from .individual import GraphIndividual, mut_space_size


def collapse_graph(individual: GraphIndividual, average_score: bool = False,
                   average_coupling: bool = True) -> nx.Graph:
    def collapse_edges(pos_pair, edges):
        data = [individual.graph.edges[e_] for e_ in edges]
        score = sum(x['gene'].S for x in data)
        coupling = sum(x['gene'].C for x in data)
        types = ','.join(e[2] for e in edges)
        is_weak = len(edges) == 1 and coupling < individual.coupling_threshold
        style = 'dashed' if is_weak else 'solid'
        if average_score:
            score /= len(edges)
        if average_coupling:
            coupling /= len(edges)
        return *pos_pair, score, coupling, len(edges), types, style

    edges_ = (collapse_edges(g, list(gg)) for g, gg in groupby(
        sorted(individual.graph.edges), lambda x: x[:2]))
    graph = nx.Graph()
    for e in edges_:
        graph.add_edge(e[0], e[1], score=e[2], coupling=e[3], num_edges=e[4], types=e[5], style=e[6])
    return graph


def plot_nx(individual: GraphIndividual, average_score=False, average_coupling=True, ax=None):
    graph = collapse_graph(individual, average_score, average_coupling)
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='circo')
    styles = [d['style'] for _, _, d in graph.edges.data()]
    scores = [d['score'] for _, _, d in graph.edges.data()]

    nx.draw_networkx_nodes(
        graph, pos=pos, node_size=600,
        node_color='#000000',
        edgecolors='#50898a',
        ax=ax)
    edges = nx.draw_networkx_edges(
        graph, pos=pos,
        edge_color=scores,
        style=styles,
        edge_cmap=plt.cm.winter_r,
        width=2.0,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph, pos=pos, font_size=10,
        font_weight='bold',
        font_color='#ffffff',
        ax=ax)
    nx.draw_networkx_edge_labels(
        graph, pos=pos,
        edge_labels={e: graph.edges[e]['num_edges'] for e in graph.edges},
    )
    plt.box(on=None)
    plt.colorbar(edges)
    return ax


def get_mut_space(individual: GraphIndividual):
    genes = individual.genes()
    pairs = sorted(chain(((g.P1, g.A1) for g in genes), ((g.P2, g.A2) for g in genes)))
    return [(g, {x[1] for x in gg}) for g, gg in groupby(pairs, lambda x: x[0])]


def plot_space(
        individual: GraphIndividual,
        positions: t.Iterable[int],
        size=(8, 2), ax=None):
    mut_space = get_mut_space(individual)
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    positions = sorted(positions)
    alphabet_enum = {v: k for k, v in enumerate(alphabet)}
    positions_enum = {v: k for k, v in enumerate(positions)}
    base = np.zeros(shape=(len(positions), len(alphabet)))
    for pos, aas in mut_space:
        for aa in aas:
            base[positions_enum[pos], alphabet_enum[aa]] = 1.0 / len(aas)
    df = pd.DataFrame(base, columns=alphabet)
    logo = Logo(
        df, color_scheme='weblogo_protein', font_name='arial',
        show_spines=False, figsize=size,
        ax=ax)
    if ax is not None:
        ax.set_xticks(list(range(len(positions))), )
        ax.set_xticklabels(positions, rotation=90)
        ax.set_yticks([])
    else:
        plt.xticks(ticks=list(range(len(positions))), labels=positions, rotation=90)
        plt.yticks([])
    return logo


def plot_indiv(i: int, individual: GraphIndividual, ref_positions: t.Iterable[int],
               average_score=False, average_coupling=True, figsize=(6, 8)):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={'height_ratios': [1, 2]})
    sizes = sorted([round(mut_space_size(cc), 2) for cc in individual.ccs])
    if len(sizes) > 1:
        sizes = f'{"+".join(map(str, sizes))}={round(sum(sizes), 2)}'
    else:
        sizes = sizes[0]

    plt.suptitle(
        f'Individual {i}, RawScore: {round(individual.score, 2)}, MutSpaceSizes: {sizes}',
        fontsize=14)

    plot_space(individual, ref_positions, ax=ax1)
    plot_nx(individual, average_score, average_coupling, ax=ax2)
    return fig, ax1, ax2


def map_to_fullseq(genes: t.Collection[SeqGene], refseq: str, mapping: t.Mapping[int, int]):
    refseq = list(refseq)
    flat_seq = chain.from_iterable(((p, s) for p, s in zip(g.Pos, g.Seq)) for g in genes)
    for p, c in flat_seq:
        refseq[mapping[p]] = c
    return "".join(refseq)


def normalize(a):
    return (a - np.min(a)) / (np.max(a) - np.min(a))


def genes2colors(genes: t.Iterable[SeqGene], cm=plt.cm.OrRd):
    scores = [g.S for g in genes]
    return list(cm(normalize(
        np.linspace(min(scores), max(scores), len(scores)))))


def plot_seq(genes: t.Collection[SeqGene], refpos: t.Sequence[int],
             refseq: str, draw_legend: bool = True,
             colormap: t.Optional[t.Mapping[SeqGene, np.ndarray]] = None,
             **kwargs):
    pairs = combinations((set(g.Pos) for g in genes), 2)
    if any(x & y for x, y in pairs):
        raise ValueError('Positions must not overlap')
    mapping = {p: i for i, p in enumerate(refpos)}
    df = sequence_to_matrix(
        map_to_fullseq(genes, refseq, mapping))

    # init logo
    logo = Logo(df, **kwargs)

    # style ticks
    logo.style_xticks(rotation=90)
    logo.ax.xaxis.set_ticklabels(refpos)
    logo.ax.tick_params(left=False, labelleft=False, bottom=False)

    # style target glyphs
    if colormap is None:
        genes_ = sorted(genes, key=lambda x: x.S)
        colors = genes2colors(genes_)
        target = list(zip(genes_, colors))
    else:
        target = [(g, c) for g, c in colormap.items() if g in genes]

    for g, c in target:
        for p, aa in zip(g.Pos, g.Seq):
            logo.style_single_glyph(mapping[p], aa, color=c, edgecolor='black', edgewidth=0.5)

    target_pos = reduce(op.or_, (set(g.Pos) for g in genes))
    for p, aa in zip(refpos, refseq):
        if p not in target_pos:
            logo.style_single_glyph(mapping[p], aa, color='grey', alpha=0.3)

    if draw_legend:
        # style legend
        handles = [mpl.patches.Patch(color=c, label=round(g.S, 2)) for g, c in target]
        logo.ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    return logo


def plot_seqs(seqs: t.Sequence[t.Collection[SeqGene]],
              refpos: t.Sequence[int], refseq: str,
              scores: t.Optional[t.Sequence[float]] = None,
              legend_kwargs: t.Optional[t.Mapping[str, t.Any]] = None,
              **kwargs):
    all_genes = sorted(
        set(chain.from_iterable(seqs)),
        key=lambda g: g.S)
    colors = genes2colors(all_genes)
    colormap = dict(zip(all_genes, colors))

    # base subplots
    fig, axs = plt.subplots(len(seqs), figsize=(7, 0.5 * len(seqs)))
    for seq, ax in zip(seqs, axs):
        logo = plot_seq(seq, refpos, refseq, False, colormap, ax=ax, **kwargs)

    # scores
    if scores:
        assert len(scores) == len(seqs)
        for ax, score in zip(axs, scores):
            ax.set_ylabel(f'S={round(score, 2)}', rotation=0, fontsize=12, ha='right', va='center')

    # legend
    labels = [
        "".join([f'${s}_{{{p}}}$' for s, p in zip(g.Seq, g.Pos)]) + f'-{round(g.S, 2)}'
        for g in all_genes]
    handles = [mpl.patches.Patch(color=c, label=l) for (g, c), l in zip(colormap.items(), labels)]
    if legend_kwargs is None:
        legend_kwargs = {}
    fig.legend(handles=handles, **legend_kwargs)
    fig.tight_layout()
    return logo, ax, fig


if __name__ == '__main__':
    raise RuntimeError
