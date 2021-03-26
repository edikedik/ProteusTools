import operator as op
import typing as t
from collections import defaultdict
from functools import reduce
from itertools import chain, groupby
from warnings import warn

import networkx as nx
from more_itertools import chunked

from protmc.operators import ADAPT, MC
from .base import Gene
from .individual import GenericIndividual
from ..common import AminoAcidDict
from ..common.utils import replace_constraints
from ..pipelines.affinity_search import AffinitySetup, Param_set

CC = t.NamedTuple('CCSetup', [('Positions', t.Tuple[int, ...]), ('Genes', t.Tuple[Gene, ...]),
                              ('MutSpace', t.Tuple[str, ...]), ('MutSpaceSize', int)])
Ind = t.NamedTuple('IndSetup', [('CCs', t.List[CC]), ('WeakLinks', t.List[Gene])])
Worker = t.Union[ADAPT, MC]


def prepare_cc(cc: nx.MultiGraph) -> CC:
    mapping = AminoAcidDict().aa_dict
    genes = tuple(d['gene'] for _, _, d in cc.edges.data())
    pairs_flat = sorted(chain(((g.P1, g.A1) for g in genes), ((g.P2, g.A2) for g in genes)))
    mut_space = [(g, {mapping[x[1]] for x in gg}) for g, gg in groupby(pairs_flat, lambda x: x[0])]
    mut_space_size = reduce(op.mul, map(len, map(op.itemgetter(1), mut_space)))
    positions = tuple(map(op.itemgetter(0), mut_space))
    mut_space = tuple(f'{pos} {" ".join(sorted(types))}' for pos, types in mut_space)
    return CC(positions, genes, mut_space, mut_space_size)


def prepare_individual(ind: GenericIndividual) -> Ind:
    ccs = list(map(prepare_cc, ind.ccs))
    weak_links = [ind.graph.edges[e]['gene'] for e in ind.weak_links()]
    return Ind(ccs, weak_links)


def cc_to_worker_pair(cc: CC, setup: AffinitySetup, param_pair: t.Tuple[Param_set, Param_set],
                      base_constraints: t.List[str]) -> t.Tuple[Worker, Worker]:
    constraints = replace_constraints(base_constraints, cc.MutSpace)
    space_setter = ('MC_PARAMS', 'Space_Constraints', constraints)
    unique_suffix = hash(tuple(cc.MutSpace))
    comb_dir = f'{"-".join(map(str, cc.Positions))}_{cc.MutSpaceSize}_{unique_suffix}'

    def setup_worker(params: Param_set):
        comb, system, mode = params
        worker = setup.setup_worker(comb, system, mode, comb_dir)
        worker.modify_config(field_setters=[space_setter], dump=True)
        return worker

    return setup_worker(param_pair[0]), setup_worker(param_pair[1])


def setup_population(population: t.Iterable[GenericIndividual], setup: AffinitySetup, min_mut_space_size: int = 3):
    if setup.combinations is not None:
        warn('Existing `combinations` attribute will be overwritten')

    mapping = AminoAcidDict().aa_dict
    base_constraints = [f'{p} {mapping[a]}' for p, a in zip(setup.active_pos, setup.reference_seq)]

    prepared = map(prepare_individual, population)
    cc2ind = defaultdict(list)
    for ind in prepared:
        for cc in ind.CCs:
            cc2ind[cc].append(ind)

    setup.combinations = [cc.Positions for cc in cc2ind]
    param_pairs = setup.prepare_params()

    cc2workers, workers = {}, {}
    for cc, chunk in zip(cc2ind, chunked(param_pairs, 2)):
        if cc.MutSpaceSize >= min_mut_space_size:
            ((apo_adapt, apo_mc), (holo_adapt, holo_mc)) = map(
                lambda p: cc_to_worker_pair(cc, setup, p, base_constraints), chunk)
            workers[(apo_adapt.id, apo_mc.id)] = apo_adapt, apo_mc
            workers[(holo_adapt.id, holo_mc.id)] = holo_adapt, holo_mc
            cc2workers[cc] = ((apo_adapt, apo_mc), (holo_adapt, holo_mc))

    return cc2ind, cc2workers, workers


if __name__ == '__main__':
    raise RuntimeError
