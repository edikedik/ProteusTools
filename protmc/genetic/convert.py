import operator as op
import typing as t
from collections import defaultdict
from functools import reduce
from itertools import chain, groupby
from warnings import warn

import networkx as nx
from more_itertools import chunked

from protmc.operators import ADAPT, MC
from .base import CC
from .individual import GraphIndividual
from .utils import get_attr
from ..common import AminoAcidDict
from ..common.utils import union_constraints
from ..pipelines.affinity_search import AffinitySetup, Param_set

Worker = t.Union[ADAPT, MC]


def prepare_cc(cc: nx.MultiGraph, base_constraints: t.List[str]) -> CC:
    """
    Given a fully connected `MultiGraph`, extracts information necessary to setup a pair of workers.
    :param cc: Connected component.
    :param base_constraints: Mutation space constraints imposed by the reference sequence.
    :return: A `CC` named tuple with (1) list of positions, (2) genes,
    (3) mutation space, and (4) mutation space size.
    """
    if not nx.is_connected(cc):
        raise ValueError('Provided graph is not connected')
    mapping = AminoAcidDict().aa_dict
    genes = tuple(get_attr(cc, 'gene'))
    pos_types = sorted(chain(((g.P1, g.A1) for g in genes), ((g.P2, g.A2) for g in genes)))
    pos_group = groupby(pos_types, lambda x: x[0])
    mut_space = [f'{pos} {" ".join(set(mapping[x[1]] for x in types))}' for pos, types in pos_group]
    positions = tuple(int(x.split()[0]) for x in mut_space)
    mut_space = tuple(union_constraints(chain(base_constraints, mut_space)))
    mut_space_size = reduce(op.mul, (len(x.split()[1:]) for x in mut_space))
    return CC(positions, genes, mut_space, mut_space_size)


def cc_to_worker_pair(
        cc: CC, setup: AffinitySetup, param_pair: t.Tuple[Param_set, Param_set]) -> t.Tuple[Worker, Worker]:
    """
    Convert a prepared CC to a pair of workers.
    :param cc: `CC` namedtuple
    :param setup: `AffinitySearch` setup.
    :param param_pair: All the unique parameters for a worker (i.e.,
    list of positions, system, config, and so on).
    :return: A pair of workers (`ADAPT`, `MC`).
    """
    space_setter = ('MC_PARAMS', 'Space_Constraints', list(cc.MutSpace))
    unique_suffix = hash(tuple(cc.MutSpace))
    comb_dir = f'{"-".join(map(str, cc.Positions))}_{cc.MutSpaceSize}_{unique_suffix}'

    def setup_worker(params: Param_set):
        comb, system, mode = params
        worker = setup.setup_worker(comb, system, mode, comb_dir)
        worker.modify_config(field_setters=[space_setter], dump=True)
        return worker

    return setup_worker(param_pair[0]), setup_worker(param_pair[1])


def setup_population(
        population: t.Iterable[GraphIndividual], setup: AffinitySetup,
        mut_space_size_bounds: t.Tuple[t.Optional[float], t.Optional[float]] = (None, None)):
    """
    Prepare a collection of `GraphIndividual`s for the `protMC` sampling.
    Separates individuals into connected components.
    Associates each connected component with a pair of (`ADAPT`, `MC`) workers.
    :param population: An iterable with `GraphIndividual`s.
    :param setup: `AffinitySearch` setup: `combinations` attribute will be overwritten.
    :param mut_space_size_bounds: CCs with mutation space size within these bounds will be retained.
    :return: A tuple of: (1) mapping CC-> `GraphIndividual`, (2) mapping CC->pair of workers,
    and (3) mapping (`ADAPT ID`, `MC ID) -> (`ADAPT` worker, `MC` worker).
    The latter is ready to be used in `AffinitySearch`.
    """
    if setup.combinations is not None:
        warn('Existing `combinations` attribute will be overwritten')

    mapping = AminoAcidDict().aa_dict
    base_constraints = [f'{p} {mapping[a]}' for p, a in zip(setup.active_pos, setup.reference_seq)]

    prepared = map(lambda ind_: map(lambda cc_: prepare_cc(cc_, base_constraints), ind_.ccs), population)
    cc2ind = defaultdict(list)
    for ind in prepared:
        for cc in ind:
            cc2ind[cc].append(ind)

    setup.combinations = [cc.Positions for cc in cc2ind]
    param_pairs = setup.prepare_params()

    cc2workers, workers = {}, {}
    min_size, max_size = mut_space_size_bounds
    for cc, chunk in zip(cc2ind, chunked(param_pairs, 2)):
        if min_size is not None and cc.MutSpaceSize < min_size:
            continue
        if max_size is not None and cc.MutSpaceSize > max_size:
            continue
        ((apo_adapt, apo_mc), (holo_adapt, holo_mc)) = map(
            lambda p: cc_to_worker_pair(cc, setup, p), chunk)
        workers[(apo_adapt.id, apo_mc.id)] = apo_adapt, apo_mc
        workers[(holo_adapt.id, holo_mc.id)] = holo_adapt, holo_mc
        cc2workers[cc] = ((apo_adapt, apo_mc), (holo_adapt, holo_mc))

    return cc2ind, cc2workers, workers


if __name__ == '__main__':
    raise RuntimeError
