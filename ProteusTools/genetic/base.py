import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

EdgeGene = t.NamedTuple(
    'EdgeGene', [('P1', int), ('P2', int),  # position pair
                 ('A1', str), ('A2', str),  # aa pair
                 ('S', float), ('C', float)]  # score and coupling
)
SeqGene = t.NamedTuple(
    'SeqGene', [('Seq', str),  # sequence
                ('Pos', t.Tuple[int, ...]),  # associated positions
                ('S', float)])  # score
Gene = t.TypeVar('Gene', bound=t.Union[EdgeGene, SeqGene])
EarlyStopping = t.NamedTuple(
    'EarlyStopping', [
        ('Rounds', int),  # the number of rounds before stopping
        ('ScoreImprovement', float),
        ('Selector', str)])  # "mean" or "max"
Bounds = t.NamedTuple(
    'Bounds', [('lower', t.Optional[float]),
               ('upper', t.Optional[float])])
Columns = t.NamedTuple(
    'Columns', [('pos', str),  # "-" separated str with positions of the "seq_subset"
                ('seq_subset', str),  # the subset of the active positions really used
                ('affinity', str),  # affinity estimate
                ('stability_apo', str),  # stability estimate for the apo system
                ('stability_holo', str)])  # stability estimate for the holo system
ParsingResult = t.NamedTuple(
    'ParsingResult', [('df', pd.DataFrame),  # filtered DataFrame with genes
                      ('singletons', t.Optional[pd.DataFrame]),  # optional DataFrame with singletons
                      ('pool', t.Tuple[Gene, ...])])  # list of unique genes serving as a pool for GA
Record = t.NamedTuple(
    'Record', [('age', int),  # age of an Individual
               ('score', float)])  # score of an individual
CC = t.NamedTuple('CCSetup', [  # the setup of a connected component conversion
    ('Positions', t.Tuple[int, ...]),
    ('Genes', t.Tuple[EdgeGene, ...]),
    ('MutSpace', t.Tuple[str, ...]),
    ('MutSpaceSize', int)])
MultiGraphEdge = t.Tuple[int, int, str]


@dataclass
class GeneticParams:
    """
    Dataclass holding parameters varying parameters of the GA.
    `Population_size`: A number of individuals in a single population.
    `Coupling_threshold`: Value separating "weak" and "strong" edges.
    `Score_kwargs`: Keywords arguments for the scoring function.
    `Gene_pool`: A collection of genes for the GA.
    `Individual_base_size`: Initial number of genes in a single individual.
    `Brood_size`: A number of offsprings produced in a single crossover.
    `Number_of_mates`: A number of individuals used during crossover.
    `Crossover_mode`: A name of the crossover function.
    `Deletion_size`: A number of genes to delete during mutation.
    `Acquisition_size`: A number of genes to sample from pool during mutation.
    `Mutation_size`: A number of genes to replace from the gene pool.
    `Probabilities`: Probability to select (`mutation`, `deletion`, `acquisition`) actions during the mutation event.
    `Tournaments_selection`: A number of tournaments for `ktournament` op during the selection event.
    `Tournaments_policy`: A number of tournaments for `ktournament` op during the selection policy event.
    `Early_stopping`: Parameters for early stopping: number of founds, score improvement,
    and selector ("max" or "mean" score).
    `Max_mut_space`: Compute the mutation size of an Individual based on the largest CC.
    `Max_num_pos`: Compute the number of positions of an Individual based on the largest CC.
    """
    Population_size: int
    Coupling_threshold: float
    Score_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    Gene_pool: t.List[Gene] = field(default_factory=list)
    Individual_base_size: int = 50
    Brood_size: int = 1
    Number_of_mates: int = 2
    # Use_BucketMutator: bool = False
    Crossover_mode: str = 'recombine_genes_uniformly'
    Deletion_size: int = 1
    Acquisition_size: int = 1
    Mutation_size: int = 1
    Probabilities: t.Tuple[float, float, float] = (0.0, 0.5, 0.5)
    Tournaments_selection: int = 20
    Tournaments_policy: int = 20
    Early_stopping: EarlyStopping = EarlyStopping(50, 0.5, 'max')
    Max_mut_space: bool = True
    Max_num_pos: bool = False


@dataclass
class ParsingParams:
    """
    Dataclass holding changeable params for preparing results for the genetic algorithm.
    It is passed to `prepare_data` function.

    `Results`: Either a path to a "RESULTS.tsv" table or a a parsed DataFrame,
    serving as a starting point for the filtering.
    The `DataFrame` holds five columns, whose default names are specified in `Results_columns` attribute.
    `Results_columns`: `Columns` -- a namedtuple, holding column names for
    'position', 'sequence', 'sequence subset', 'affinity', and 'stability', in this order.
    `Affinity_bounds`: A `Bounds` namedtuple holding lower and upper bounds for affinity.
    `Affinity_cap`: Cap "affinity" using the provided `Bounds`.
    `Stability_apo_bounds`: lower and upper `Bounds` on the stability of the "apo" system.
    `Stability_holo_bounds`: lower and upper `Bounds` on the stability of the "holo" system.
    `Stability_joint_bounds`: lower and upper bounds which both system must pass simultaneously.
    `Scale_range`: A `Bounds` namedtuple holding lower and upper bounds to scale affinity values.
    After the scaling, we refer to these as "scores".
    `Reverse_score`: Whether to multiply scores by -1.
    `Use_singletons`: If `True`, singletons will be used to estimate pairs' affinities.
    `Use_couplings`: If `True`, couplings will be calculated from singletons.
    `Exclude_types`: List of tuples (position, aa) to filter out.
    `Exclude_pairs`: list of tuples (pos1, pos2) to filter out.
    `Default_coupling`: On occasion of failure to calculate coupling from singletons,
    this value will be used instead.
    `Top_n_seqs`: Unique param for "seq" DataFrame -- a number of best-scoring sequences to take per position group.
    `Seq_size_threshold`: Unique param for "seq" DataFrame -- seqs shorter than the provided value
    will be used to estimate the affinity of larger seqs.
    `Seq_df`: Prepare "seq" with genes for `SeqIndividual`.
    `Affinity_diff_threshold: Unique param for "seq" DataFrame -- filter out sequences with
    "affinity" estimates from smaller seqs accurate to the point of the provided value.
    """
    Results: t.Union[str, pd.DataFrame]
    Results_columns: Columns = Columns('pos', 'seq_subset', 'affinity', 'stability_apo', 'stability_holo')
    Affinity_bounds: Bounds = Bounds(None, None)
    Affinity_cap: Bounds = Bounds(None, None)
    Stability_apo_bounds: Bounds = Bounds(None, None)
    Stability_holo_bounds: Bounds = Bounds(None, None)
    Stability_joint_bounds: Bounds = Bounds(None, None)
    Scale_range: Bounds = Bounds(0, 1)
    Reverse_score: bool = True
    Use_singletons: bool = False
    Use_couplings: bool = False
    Exclude_types: t.List[t.Tuple[t.Union[str, int], str]] = field(default_factory=list)
    Exclude_pairs: t.List[t.Tuple[int, int]] = field(default_factory=list)
    Default_coupling: t.Optional[float] = None
    Top_n_seqs: t.Optional[int] = None
    Seq_size_threshold: int = 2
    Seq_df: bool = False
    Affinity_diff_threshold: float = 0.3


class AbstractIndividual(metaclass=ABCMeta):
    """Interface for the GA Individual"""

    def __len__(self):
        pass

    @property
    @abstractmethod
    def score(self) -> float:
        pass

    @abstractmethod
    def add_genes(self, genes: t.Collection[Gene]) -> 'AbstractIndividual':
        pass

    @abstractmethod
    def remove_genes(self, genes: t.Collection[Gene]) -> 'AbstractIndividual':
        pass


class AbstractGraphIndividual(AbstractIndividual):
    """Interface for the GA GraphIndividual"""

    @property
    @abstractmethod
    def graph(self) -> nx.MultiGraph:
        pass

    @property
    @abstractmethod
    def mut_space_size(self) -> int:
        pass

    @property
    @abstractmethod
    def n_pos(self) -> int:
        pass


class AbstractSeqIndividual(AbstractIndividual):
    """Interface for the GA SeqIndividual"""

    @property
    @abstractmethod
    def n_pos(self) -> int:
        pass


if __name__ == '__main__':
    raise RuntimeError
