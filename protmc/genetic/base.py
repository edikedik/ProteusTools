import typing as t
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

EdgeGene = t.NamedTuple('EdgeGene', [('P1', int), ('P2', int), ('A1', str), ('A2', str), ('S', float), ('C', float)])
EarlyStopping = t.NamedTuple('EarlyStopping', [('Rounds', int), ('ScoreImprovement', float), ('Selector', str)])
Bounds = t.NamedTuple('Bounds', [('lower', t.Optional[float]), ('upper', t.Optional[float])])
Columns = t.NamedTuple(
    'Columns', [('pos', str), ('seq_subset', str), ('affinity', str),
                ('stability_apo', str), ('stability_holo', str)])
ParsingResult = t.NamedTuple(
    'ParsingResult', [('df', pd.DataFrame), ('singletons', pd.DataFrame), ('pool', t.Collection[EdgeGene])])
Record = t.NamedTuple('Record', [('age', int), ('score', float)])
MultiGraphEdge = t.Tuple[int, int, str]


@dataclass
class GeneticParams:
    """
    # TODO: revise docs
    Dataclass holding parameters varying parameters of the GA.

    Population_size: a number of individuals in a single population.
    Score_kwargs: Keywords arguments for the scoring function.
    Individual_base_size: Initial number of genes in a single individual.
    Brood_size: A number of offsprings produced in a single crossover.
    Number_of_mates: A number of individuals used during crossover.
    Mutable_fraction: A fraction of mutable genes of an individual.
    Mutation_prob: A probability to use regular mutation during the mutation event.
    Deletion_size: A size of a deletion.
    Deletion_prob: A probability to use deletion during the mutation event.
    Acquisition_size: A size of addition sampled from a gene pool.
    Acquisition_prob: A probability to use acquisition during the mutation event.
    Tournaments_selection: A number of tournaments for `ktournament` op during the selection event.
    Tournaments_policy: A number of tournaments for `ktournament` op during the selection policy event.
    """
    Population_size: int
    Coupling_threshold: float
    Score_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    Gene_pool: t.List[EdgeGene] = field(default_factory=list)
    Individual_base_size: int = 50
    Brood_size: int = 1
    Number_of_mates: int = 2
    Use_BucketMutator: bool = False
    Crossover_mode: str = 'recombine_genes_uniformly'
    Deletion_size: int = 1
    Acquisition_size: int = 1
    Mutation_size: int = 1
    Probabilities: t.Tuple[float, float, float] = (0.0, 0.5, 0.5)
    Tournaments_selection: int = 20
    Tournaments_policy: int = 20
    Early_Stopping: EarlyStopping = EarlyStopping(50, 0.5, 'max')
    Max_mut_space: bool = True
    Max_num_pos: bool = False


@dataclass
class ParsingParams:
    """
    # TODO: revise docs
    Dataclass holding changeable params for preparing results for the genetic algorithm.
    It is passed to `prepare_data` function.

    `Results`: Either a path to a "results" table or a DataFrame. The latter holds five columns.
    Their default names are specified in `Results_columns` attribute.
    `Results_columns`: A `Columns` namedtuple, holding column names for
    'position', 'sequence', 'sequence subset', 'affinity', and 'stability'.
    Mind the order! We expect the `RESULTS` DataFrame to have the same columns in that exact order.
    `Affinity_bounds`: A `Bounds` namedtuple holding lower and upper bounds for affinity (both can be `None`).
    `Affinity_cap`: ...
    `Stability_bounds`: A `Bounds` namedtuple holding lower and upper bounds for stability (both can be `None`).
    `Scale_range`: A `Bounds` namedtuple holding lower and upper bounds to scale affinity values.
    After the scaling, we refer to these as "scores".
    `Reverse_score`: Whether to multiply scores by -1.
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
    Exclude_types: t.List[t.Tuple[t.Union[str, int], str]] = field(default_factory=list)
    Exclude_pairs: t.List[t.Tuple[int, int]] = field(default_factory=list)


class AbstractGraphIndividual(metaclass=ABCMeta):

    @property
    @abstractmethod
    def graph(self) -> nx.MultiGraph:
        pass

    @property
    @abstractmethod
    def score(self) -> float:
        pass

    @property
    @abstractmethod
    def mut_space_size(self) -> int:
        pass

    @property
    @abstractmethod
    def n_pos(self) -> int:
        pass

    @abstractmethod
    def add_genes(self, genes: t.Collection[EdgeGene]) -> 'AbstractGraphIndividual':
        pass

    @abstractmethod
    def remove_genes(self, genes: t.Collection[EdgeGene]) -> 'AbstractGraphIndividual':
        pass


if __name__ == '__main__':
    raise RuntimeError
