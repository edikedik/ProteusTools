import logging
import operator as op
import typing as t
from copy import deepcopy
from itertools import filterfalse, groupby
from pathlib import Path
from subprocess import Popen
from time import time

import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

from protmc import config
from protmc import utils as u
from protmc.affinity import affinity
from protmc.base import NoReferenceError, AminoAcidDict
from protmc.pipeline import Pipeline, PipelineOutput
from protmc.stability import stability

WorkerSetup = t.Tuple[config.ProtMCconfig, config.ProtMCconfig, str]


class AffinitySearch:
    """
    Class for performing affinity search over a combinations of positions.
    It will orchestrate a collection of `AffinityWorker` instances,
    each flattening and sampling the sequence space of a certain combination of positions
    (a subset of all active positions).
    For each of the workers, it will constrain positions outside of the provided subset to the native types.
    For many combinations it is advised to run parallel computations (see `run` method docs).
    """

    def __init__(
            self, positions: t.Iterable[t.Union[t.Collection[int], int]], active: t.List[int],
            apo_base_setup: WorkerSetup, holo_base_setup: WorkerSetup,
            exe_path: str, base_dir: str, ref_seq: t.Optional[str] = None,
            mut_space_n_types: int = 18, mut_space_path: t.Optional[str] = None, count_threshold: int = 10,
            adapt_dir_name: str = 'ADAPT', mc_dir_name: str = 'MC',
            apo_dir_name: str = 'apo', holo_dir_name: str = 'holo',
            temperature: t.Optional[float] = None, id_: t.Optional[str] = None):
        """
        :param positions: Iterable collection of positions or their combinations.
        :param active: A list of active positions (all mutable positions).
        :param ref_seq: Reference sequence (must have the same length as a number of active positions).
        :param apo_base_setup: A tuple with three elements:
        (1) a base config for ADAPT, (2) a base config for MC, and (3) a path to the energy dir.
        :param holo_base_setup: Same as `apo_base_setup`, but for the holo system.
        :param exe_path: A path to the protMC executable.
        :param base_dir: A path to a base directory (dir structure explained below).
        :param mut_space_n_types: A total number of types in the mutation space, excluding protonation states.
        :param count_threshold: A threshold to filter sequences during affinity calculation.
        See documentation for `affinity`.
        :param adapt_dir_name: A name of a directory to store ADAPT simulations.
        :param mc_dir_name: A name of a directory to store MC simulations.
        :param apo_dir_name: A name of a directory to store apo-state simulations.
        :param holo_dir_name: A name of a directory to store holo-state simulations.
        :param temperature: Temperature employed in affinity and stability calculations.
        If not provided, each worker will attempt to infer temperature from configs and raise a ValueError
        in case of inconsistencies.
        :param id_: ID of this object used for logging purposes.
        The directory structure of AffinitySearch is the following:
        ```
        base_dir
            |____ comb_1
                      |____apo
                      |      |____ADAPT
                      |      |      |____RESULTS.tsv
                      |      |      |____ADAPT.dat
                      |      |      |____ADAPT.seq
                      |      |         ...
                      |      |____MC
                      |
                      |____holo
                             |____apo
                             |____holo
            ...
            |---- comb_2
        ```
        `base_dir` holds directories named after the combinations provided via `positions` argument:
        `comb_1`, `comb_2`, and so on.
        Each directory for a combination serves as a `base_dir` for the corresponding `AffinityWorker`.
        Each of the latter, in turn, uses `apo` and `holo` directories as `base_dir` for the `Pipeline` objects.
        Finally, each `Pipeline` will have it's experiment directory named after after either
        `adapt_dir_name` or `mc_dir_name`
        """
        if len(active) != len(ref_seq):
            raise ValueError('Length of the reference sequence has to match the number of active positions')
        self.apo_base_setup, self.holo_base_setup = apo_base_setup, holo_base_setup
        self.exe_path = exe_path
        self.base_dir = base_dir
        self.mut_space_n_types = mut_space_n_types
        self.mut_space_path = mut_space_path
        self.count_threshold = count_threshold
        self.adapt_dir_name, self.mc_dir_name = adapt_dir_name, mc_dir_name
        self.apo_dir_name, self.holo_dir_name = apo_dir_name, holo_dir_name
        self.temperature = temperature
        self.positions = positions
        if not isinstance(positions, t.Tuple):
            self.positions = tuple(self.positions)
        self.active = active
        self.ref_seq = ref_seq or self._infer_ref_seq()
        self.ran_setup, self.ran_workers = False, False
        self.workers: t.Optional[t.Dict[str, AffinityWorker]] = None
        self.summaries: t.Optional[pd.DataFrame] = None
        self.affinities_: t.Optional[pd.DataFrame] = None
        self.stabilities_: t.Optional[pd.DataFrame] = None
        self.results: t.Optional[pd.DataFrame] = None
        self._active_mapping = {pos: i for i, pos in enumerate(sorted(active))}
        self.id = id(self) if id_ is None else id_
        logging.info(f'AffinitySearch {self.id}: initialized')

    def setup_workers(self) -> None:
        """
        Setup `AffinityWorker`'s based on subsets of active positions.
        Will populate the `workers` attribute with prepared `AffinityWorker` instances.
        """
        workers = [self._setup_worker(active_subset=s) for s in self.positions]
        self.workers = {w.id: w for w in workers}
        self.ran_setup = True
        logging.info(f'AffinitySearch {self.id}: ran setup for workers')

    def run_workers(
            self, num_proc: int = 1, cleanup: bool = False,
            cleanup_kwargs: t.Optional[t.Dict] = None, overwrite_summaries: bool = False,
            run_adapt: bool = True, run_mc: bool = True, transfer_bias: bool = True,
            continue_adapt: bool = False, config_changes: t.Optional[t.List[t.Tuple[str, t.Any]]] = None,
            ids: t.Optional[t.Container[str]] = None,
            impose_bias_constraints: bool = False,
            bias_constraints_holo_based: bool = True,
            bias_constraints_mut_space: t.Union[t.Set[str], str] = '',
            bias_constraints_threshold: float = 10,
            bias_constraints_apply_to: t.Tuple[str, ...] = ('apo_adapt', 'apo_mc', 'holo_adapt', 'holo_mc'),
            run_i: int = 0, verbose: bool = True) -> pd.DataFrame:
        """
        Run each of prepared `AffinityWorkers`.
        :param num_proc: Number of processes.
        :param cleanup: If True, each `Pipeline` within each of `AffinityWorker`s will call its `cleanup` method,
        by default removing `seq` and `ener` files (which can be customized via `cleanup_kwargs` argument.
        :param cleanup_kwargs: Pass this dictionary to `cleanup` method of each `Pipeline`.
        :param overwrite_summaries:  Overwrite existing `run_summaries` of a worker, if any.
        This is passed to a `run` method of `AffinityWorker`;
        `summaries` attribute of `AffinitySearch` is overwritten by default.
        :param run_adapt: Run ADAPT mode Pipelines.
        :param run_mc: Run MC mode Pipelines.
        :param transfer_bias: Transfer last ADAPT biases to the MC experiment directories.
        :param continue_adapt: Continue ALF using previously accumulated biases.
        :param config_changes: If `continue_adapt`, apply these changes to `mc_conf` before running.
        :param impose_bias_constraints: See `AffinityWorker.run` docs for details.
        :param bias_constraints_holo_based: See `AffinityWorker.run` docs for details.
        :param bias_constraints_mut_space: See `AffinityWorker.run` docs for details.
        :param bias_constraints_threshold: See `AffinityWorker.run` docs for details.
        :param bias_constraints_apply_to: See `AffinityWorker.run` docs for details.
        :param ids: IDs of selected workers to run.
        :param run_i: Run (consecutive) number to complement summary info.
        :param verbose: Progress bar.
        :return: A DataFrame of summaries comprising run summary
        for each underlining `Pipeline` within each `AffinityWorker`.
        """
        if not self.ran_setup:
            self.setup_workers()

        # Prepare for the run
        mut_space = self.mut_space_path if not bias_constraints_mut_space else bias_constraints_mut_space
        common_args = dict(
            cleanup=cleanup, cleanup_kwargs=cleanup_kwargs, return_self=True,
            run_adapt=run_adapt, run_mc=run_mc, transfer_bias=transfer_bias,
            continue_adapt=continue_adapt, config_changes=config_changes,
            overwrite_summaries=overwrite_summaries,
            impose_bias_constraints=impose_bias_constraints,
            bias_constraints_holo_based=bias_constraints_holo_based,
            bias_constraints_mut_space=mut_space,
            bias_constraints_threshold=bias_constraints_threshold,
            bias_constraints_apply_to=bias_constraints_apply_to)
        workers = list(self.workers.values())
        if ids is not None:
            workers = [w for w in workers if w.id in ids]
            logging.info(f'AffinitySearch {self.id}: will run {len(workers)} out of {len(self.workers)} workers')
        if verbose:
            workers = tqdm(workers, desc='Running workers')

        # Obtain the run results for each of the workers
        if num_proc > 1:
            with Pool(num_proc // 2) as pool:
                results = pool.map(lambda w: w.run(collect_parallel=False, **common_args), workers)
        else:
            results = [w.run(collect_parallel=True, **common_args) for w in workers]

        # Separate results
        summaries = [r[0] for r in results]
        upd_workers = [r[1] for r in results]

        # Update existing workers
        for w in upd_workers:
            self.workers[w.id] = w

        # Process and return summaries
        for s, d in zip(summaries, (w.run_dir for w in upd_workers)):
            s['run_dir'] = d
        summaries = pd.concat(summaries)
        summaries['run_i'] = run_i
        self.summaries = pd.concat([self.summaries, summaries]) if self.summaries is not None else summaries
        logging.info(f'AffinitySearch {self.id}: finished running workers')
        return self.summaries

    def affinities(
            self, dump: bool = True, dump_name: str = 'AFFINITY.tsv',
            num_proc: int = 1, verbose: bool = False) -> t.Optional[pd.DataFrame]:
        """
        Calls `calculate_affinity` method of each `AffinityWorker` and processes the results.
        :param dump: Dump the aggregation of affinity `DataFrame`s into the base directory.
        :param dump_name: Customize the dump name if desired.
        :param num_proc: The number of processes to run.
        Safe to pick `num_proc>=len(self.workers)`.
        :param verbose: Print the progress bar.
        :return: concatenated `affinity` outputs of each worker
        """
        self.affinities_ = self._calculate('affinity', num_proc=num_proc, verbose=verbose)
        if dump:
            self.affinities_.to_csv(f'{self.base_dir}/{dump_name}', sep='\t', index=False)
        logging.info(f'AffinitySearch {self.id}: calculated worker affinities')
        return self.affinities_

    def stabilities(
            self, dump: bool = True, dump_name: str = 'STABILITY.tsv',
            num_proc: int = 1, verbose: bool = False, apo: bool = False) -> t.Optional[pd.DataFrame]:
        """
        Calls `calculate_stability` method of each worker and processes the results.
        :param dump: Dump the aggregation of stabilities into the base directory.
        :param dump_name: Customize the dump name if needed.
        :param num_proc: A max number of processes to use.
        :param verbose: Print the progress bar.
        :param apo: Calculate stability of the apo state (else holo).
        :return: Concatenated outputs of `calculate_stability` of each worker.
        """
        self.stabilities_ = self._calculate('stability', num_proc=num_proc, verbose=verbose, apo=apo)
        if dump:
            self.stabilities_.to_csv(f'{self.base_dir}/{dump_name}', sep='\t', index=False)
        logging.info(f'AffinitySearch {self.id}: calculated worker stabilities')
        return self.stabilities_

    def collect_results(
            self, num_proc: int = 1, verbose: bool = True, apo_stability: bool = True,
            dump: bool = True, dump_name='RESULTS.tsv',
            dump_affinity: bool = False, dump_affinity_name: str = 'AFFINITY.tsv',
            dump_stability: bool = False, dump_stability_name: str = 'STABILITY.tsv'):
        """
        Collects affinities and stabilities in one place - the `results` attribute.
        Calls `affinities` and `stabilities` methods to populate the corresponding class attributes (if empty).
        :param num_proc: The max number of processes allowed.
        :param verbose: Progress bards for affinity and stability calculations.
        :param apo_stability: Stability for the apo state, otherwise for the holo.
        :param dump: Dump collected results into a base directory.
        :param dump_name: Name of the results table.
        :param dump_affinity: passed to `affinities` method.
        :param dump_affinity_name: passed to `affinities` method.
        :param dump_stability: passed to `stabilities` method.
        :param dump_stability_name: passed to `stabilities` method.
        :return:
        """
        if self.affinities_ is None:
            self.affinities_ = self.affinities(
                num_proc=num_proc, verbose=verbose, dump=dump_affinity, dump_name=dump_affinity_name)
        if self.stabilities_ is None:
            self.stabilities_ = self.stabilities(
                num_proc=num_proc, verbose=verbose, dump=dump_stability, dump_name=dump_stability_name,
                apo=apo_stability)
        if self.affinities_ is None:
            logging.error('Could not collect affinities')
            return None
        if self.stabilities_ is None:
            logging.error('Could not collect stabilities')
            return None
        self.results = pd.merge(
            self.affinities_, self.stabilities_, on=['seq', 'seq_subset', 'pos'], how='inner')[
            ['pos', 'seq', 'seq_subset', 'stability', 'affinity']]
        if dump and self.results is not None:
            self.results.to_csv(f'{self.base_dir}/{dump_name}', sep='\t', index=False)
        logging.info(f'AffinitySearch {self.id}: collected results')
        return self.results

    def _calculate(self, what: str, num_proc: int = 1,
                   verbose: bool = False, apo: bool = True) -> t.Optional[pd.DataFrame]:

        def try_calculate(worker: AffinityWorker) -> t.Optional[pd.DataFrame]:
            # Helps to safely attempt a calculation for a worker
            try:
                if what == 'affinity':
                    return worker.calculate_affinity()
                if what == 'stability':
                    return worker.calculate_stability(apo=apo)
                else:
                    raise ValueError(f'What is {what}?')
            except NoReferenceError:
                return None

        def wrap_calculated(worker: AffinityWorker, res: t.Optional[pd.DataFrame]):
            # wraps results of each `AffinityWorker`
            # warns the user if `AffinityWorker` did not yield results
            pos = worker.run_dir.split('/')[-1]
            if res is None:
                logging.warning(f'AffinitySearch {self.id}: no output for pos {pos}')
                return None
            res['pos'] = pos
            return res

        # wrap workers into tqdm object if verbosity is desired
        workers = tqdm(self.workers.values(), desc=f'Calculating {what}') if verbose else self.workers.values()

        # get the affinity DataFrames from `AffinityWorkers`
        if num_proc > 1:
            with Pool(num_proc) as pool:
                results = pool.map(try_calculate, workers)
            for w, r in zip(self.workers.values(), results):
                w.affinity = r  # attributes would remain unchanged relative to the parent process
        else:
            results = list(map(try_calculate, workers))

        # process the affinity DataFrames (if any)
        wrapped = (wrap_calculated(w, r) for w, r in zip(self.workers.values(), results))
        filtered = list(filterfalse(lambda x: x is None, wrapped))

        # none of the workers yielded affinity DataFrame -> return None
        if not filtered:
            logging.warning(f'AffinitySearch {self.id}: no position combination yielded {what} results')
            return None

        # otherwise, concatenate and return the results
        return pd.concat(filtered)

    def _setup_worker(self, active_subset: t.Union[t.List[int], int]) -> 'AffinityWorker':
        """
        Helper function setting up an AffinityWorker based on a subset of active positions
        :param active_subset: A subset of active positions (can be a single value or a list of values)
        :return: an AffinityWorker ready to be executed
        """
        # copy and extract base configs
        apo_adapt_cfg, apo_mc_cfg, apo_matrix = deepcopy(self.apo_base_setup)
        holo_adapt_cfg, holo_mc_cfg, holo_matrix = deepcopy(self.holo_base_setup)

        # compose main varying parameters for the AffinityWorker
        existing_constraints = u.extract_constraints([apo_adapt_cfg, apo_mc_cfg, holo_adapt_cfg, holo_mc_cfg])
        constraints = u.space_constraints(
            reference=self.ref_seq, subset=active_subset, active=self.active,
            mutation_space=self.mut_space_path,
            existing_constraints=existing_constraints or None)
        adapt_space = u.adapt_space(active_subset)
        exp_dir_name = "-".join(map(str, active_subset)) if isinstance(active_subset, t.Iterable) else active_subset

        # put varying parameters into configs
        apo_adapt_cfg.set_field('ADAPT_PARAMS', 'Adapt_Space', adapt_space)
        holo_adapt_cfg.set_field('ADAPT_PARAMS', 'Adapt_Space', adapt_space)
        for cfg in [apo_adapt_cfg, apo_mc_cfg, holo_adapt_cfg, holo_mc_cfg]:
            cfg.set_field('MC_PARAMS', 'Space_Constraints', constraints)

        # setup and return worker
        worker = AffinityWorker(
            apo_setup=(apo_adapt_cfg, apo_mc_cfg, apo_matrix),
            holo_setup=(holo_adapt_cfg, holo_mc_cfg, holo_matrix),
            active_pos=self.active,
            active_subset_mapped=dict(filter(lambda x: x[0] in active_subset, self._active_mapping.items())),
            ref_seq=self.ref_seq,
            mut_space_n_types=self.mut_space_n_types, exe_path=self.exe_path,
            run_dir=f'{self.base_dir}/{exp_dir_name}',
            adapt_dir_name=self.adapt_dir_name, mc_dir_name=self.mc_dir_name,
            apo_dir_name=self.apo_dir_name, holo_dir_name=self.holo_dir_name,
            temperature=self.temperature, id_=exp_dir_name)
        worker.setup()
        return worker

    def _infer_ref_seq(self):
        # find reference sequence from the `position_list` file
        apo_pos = f'{self.apo_base_setup[2]}/position_list.dat'
        holo_pos = f'{self.holo_base_setup[2]}/position_list.dat'
        if Path(apo_pos).exists():
            position_list = apo_pos
        elif Path(holo_pos).exists():
            position_list = holo_pos
        else:
            raise RuntimeError('Could not find position_list.dat in matrix directories.')
        self.ref_seq = u.get_reference(position_list, list(map(str, self.active)))
        if not self.ref_seq:
            raise RuntimeError('Reference sequence is empty')


class AffinityWorker:
    """
    An `AffinityWorker` object encompasses a common sampling strategy used with protMC.
    Specifically, `apo` and `holo` systems (or any equivalent of a two-state system) are flattened and sampled
    to get their stability estimates. Then, affinity is simply a difference in stabilities of these two systems.

    `AffinityWorker` spawns and manages four `Pipeline` objects, corresponding to the simulations `apo adapt`, `apo mc`,
    `holo adapt`, and `holo mc`.
    It first runs both `adapt` simulation in parallel to flatten the landscape.
    This step can be repeated up to convergence (desired level of flattening), with a proper combination of `run` method
    arguments.
    After the flattening step, `AffinityWorker` transfers accumulated biases into experiment directories of the `mc apo`
    and `mc holo` `Pipeline`s, and runs these `Pipelines` in parallel.
    It does (or can do) a lot of things behind the scenes: see `run` documentation for details.
    """
    def __init__(
            self, apo_setup: WorkerSetup, holo_setup: WorkerSetup,
            active_pos: t.Iterable[int], active_subset_mapped: t.Dict[int, int],
            exe_path: str, run_dir: str, ref_seq: str,
            mut_space_n_types: int = 18,
            adapt_dir_name: str = 'ADAPT', mc_dir_name: str = 'MC',
            apo_dir_name: str = 'apo', holo_dir_name: str = 'holo',
            temperature: t.Optional[float] = None, id_: t.Optional[str] = None):
        """
        :param apo_setup: A collection of three items: ADAPT `Config`, MC `Config`,
        and a path to the energy directory of the `apo` system.
        :param holo_setup: Same as `apo_setup`, but for the `holo system.
        :param active_pos: All active (mutable) positions.
        :param active_subset_mapped: A mapping between a subset of active positions actually flattened
        and their position in the ref seq starting from 0
        :param exe_path: A path to the protmc executable.
        :param run_dir: A directory holding the run.
        :param ref_seq: A reference sequence (one-letter codes).
        :param mut_space_n_types: A number of types in the `mutation_space.dat` file.
        :param adapt_dir_name: A name of the dir to hold ADAPT simulation runs.
        :param mc_dir_name: A name of the dir to hold MC simulation runs.
        :param apo_dir_name: A name of the directory to hold `apo` simulation runs.
        :param holo_dir_name: A name of the directory to hold `holo` simulation runs.
        :param temperature: A temperature of a simulation.
        :param id_: Something sensible to distinguish this worker (e.g., `Adapt_Space`).
        """
        self.apo_adapt_cfg, self.apo_mc_cfg, self.apo_matrix_path = apo_setup
        self.holo_adapt_cfg, self.holo_mc_cfg, self.holo_matrix_path = holo_setup
        self.active_pos, self.active_subset_mapped = active_pos, active_subset_mapped
        self.mut_space_n_types = mut_space_n_types
        self.exe_path = exe_path
        self.run_dir = run_dir
        self.adapt_dir_name, self.mc_dir_name = adapt_dir_name, mc_dir_name
        self.apo_dir_name, self.holo_dir_name = apo_dir_name, holo_dir_name
        self.temperature: t.Optional[float] = temperature
        self.apo_adapt_pipe: t.Optional[Pipeline] = None
        self.holo_adapt_pipe: t.Optional[Pipeline] = None
        self.apo_mc_pipe: t.Optional[Pipeline] = None
        self.holo_mc_pipe: t.Optional[Pipeline] = None
        self.apo_adapt_results: t.Optional[PipelineOutput] = None
        self.holo_adapt_results: t.Optional[PipelineOutput] = None
        self.apo_mc_results: t.Optional[PipelineOutput] = None
        self.holo_mc_results: t.Optional[PipelineOutput] = None
        self.run_summaries: t.Optional[pd.DataFrame] = None
        self.ran_setup, self.ran_pipes = False, False
        self.affinity: t.Optional[pd.DataFrame] = None
        self.stability: t.Optional[pd.DataFrame] = None
        self.ref_seq: str = ref_seq
        self._default_bias_input_name = 'ADAPT.inp.dat'
        self.id = id_ or id(self)
        logging.info(f'AffinityWorker {self.id}: initialized')

    def setup(self) -> None:
        def create_pipe(cfg, base_dir, exp_dir, energy_dir, id_):
            # Simplifies the creation of Pipeline by encapsulating common arguments
            return Pipeline(
                base_post_conf=None, exe_path=self.exe_path,
                mut_space_n_types=self.mut_space_n_types, active_pos=list(self.active_pos),
                base_mc_conf=cfg, base_dir=base_dir, exp_dir_name=exp_dir,
                energy_dir=energy_dir, id_=id_)

        def handle_bias_inp(pipe: Pipeline) -> None:
            # Simplifies the assignment of the bias input path
            bias_path = f'{pipe.base_dir}/{pipe.exp_dir_name}/{self._default_bias_input_name}'
            pipe.base_mc_conf.set_field('MC_IO', 'Bias_Input_File', bias_path)

        # base directories for holo and apo Pipelines
        base_apo, base_holo = f'{self.run_dir}/{self.apo_dir_name}', f'{self.run_dir}/{self.holo_dir_name}'

        # create Pipeline objects
        self.apo_adapt_pipe = create_pipe(
            cfg=self.apo_adapt_cfg, base_dir=base_apo, exp_dir=self.adapt_dir_name, energy_dir=self.apo_matrix_path,
            id_=f'AffinityWorker {self.id} apo_adapt')
        self.holo_adapt_pipe = create_pipe(
            cfg=self.holo_adapt_cfg, base_dir=base_holo, exp_dir=self.adapt_dir_name, energy_dir=self.holo_matrix_path,
            id_=f'AffinityWorker {self.id} holo_adapt')
        self.apo_mc_pipe = create_pipe(
            cfg=self.apo_mc_cfg, base_dir=base_apo, exp_dir=self.mc_dir_name, energy_dir=self.apo_matrix_path,
            id_=f'AffinityWorker {self.id} apo_mc')
        self.holo_mc_pipe = create_pipe(
            cfg=self.holo_mc_cfg, base_dir=base_holo, exp_dir=self.mc_dir_name, energy_dir=self.holo_matrix_path,
            id_=f'AffinityWorker {self.id} holo_mc')

        # write bias input paths into configs prior to running setup (thus, dumping configs)
        handle_bias_inp(self.apo_mc_pipe), handle_bias_inp(self.holo_mc_pipe)

        # setup all Pipelines
        self.apo_adapt_pipe.setup()
        self.holo_adapt_pipe.setup()
        self.apo_mc_pipe.setup()
        self.holo_mc_pipe.setup()
        # save the changes made upon configs during the setup
        self._copy_configs()

        # parse the temperature out of configs
        self.temperature = self.temperature or self._get_temperature()

        self.ran_setup = True
        logging.info(f'AffinityWorker {self.id}: completed setting up Pipelines')

    def run(self, cleanup: bool = False,
            cleanup_kwargs: t.Optional[t.Dict] = None,
            collect_parallel: bool = False, overwrite_summaries: bool = False,
            run_adapt: bool = True, run_mc: bool = True, transfer_bias: bool = True,
            continue_adapt: bool = False, config_changes: t.Optional[t.List[t.Tuple[str, t.Any]]] = None,
            return_self: bool = True, filter_results_by_constraints: bool = True,
            impose_bias_constraints: bool = False,
            bias_constraints_holo_based: bool = True,
            bias_constraints_mut_space: t.Union[t.Set[str], str] = '',
            bias_constraints_threshold: float = 10,
            bias_constraints_apply_to: t.Tuple[str, ...] = ('apo_adapt', 'apo_mc', 'holo_adapt', 'holo_mc')) \
            -> t.Tuple[pd.DataFrame, t.Optional['AffinityWorker']]:
        """
        Run AffinityWorker.
        :param cleanup: If True will call the `cleanup` method for each of the four pipelines.
        :param cleanup_kwargs: Pass these kwargs to the `cleanup` method if `cleanup==True`.
        :param collect_parallel: If True will collect results for MC and ADAPT in parallel,
        using Pool of two workers.
        :param overwrite_summaries:  Overwrite existing `run_summary`, if any.
        :param run_adapt: Run ADAPT mode Pipelines.
        :param run_mc: Run MC mode Pipelines.
        :param transfer_bias: Transfer last ADAPT bias to the MC experiment directory.
        :param continue_adapt: Continue ALF using previously accumulated bias.
        :param config_changes: If `continue_adapt`, apply these changes to `mc_conf` before running.
        :param return_self: Whether to return `self` (useful for CPU-parallelization)
        :param filter_results_by_constraints:
        :param impose_bias_constraints: Whether to impose constraints of a mutation space based on previous bias.
        If `True`, types having bias values above the threshold will be excluded from the mutation space.
        If there are existing constraints in either of the `ADAPT` configs, they will be merged
        and concatenated with the bias-derived constraints.
        :param bias_constraints_holo_based: Use "holo adapt" pipe to derive constraints. Otherwise use "apo adapt" pipe.
        Bias is taken from the `last_bias` attribute of the corresponding `Pipeline` object.
        :param bias_constraints_mut_space: Either a path to the `mutation_space` file,
        or a set of amino acid three letter codes, defining the mutation space.
        :param bias_constraints_threshold: Threshold for the bias values.
        Will only use biases with the energy higher than the threshold.
        :param bias_constraints_apply_to: Apply the derived constraints to the following `Pipelines`.
        Full list of pipelines with correct namings is given in the default value.
        :return: a DataFrame with 4 rows (Summary's returned by the Pipeline's).
        """
        start = time()
        if not self.ran_setup:
            logging.info(f'AffinityWorker {self.id}: running setup with default arguments')
            self.setup()

        if not (run_adapt or run_mc):
            raise ValueError(f'AffinityWorker {self.id}: nothing to run')

        run_common_args = dict(
            wait=True, collect=True, collect_parallel=collect_parallel,
            cleanup=cleanup, cleanup_kwargs=cleanup_kwargs)

        if run_adapt or continue_adapt:
            if continue_adapt:
                f'AffinityWorker {self.id}: setting up ADAPT continuation'
                self.apo_adapt_pipe.setup_continuation(
                    new_exp_name=self.apo_adapt_pipe.exp_dir_name,
                    mc_config_changes=config_changes)
                self.holo_adapt_pipe.setup_continuation(
                    new_exp_name=self.holo_adapt_pipe.exp_dir_name,
                    mc_config_changes=config_changes)
            if impose_bias_constraints:
                if not bias_constraints_mut_space:
                    raise ValueError('Provide mutation space to use bias-based constraints')
                logging.info(f'AffinityWorker {self.id}: Constraining {bias_constraints_apply_to} pipelines '
                             f'based on previous bias')
                self.bias_based_constraints(
                    mut_space=bias_constraints_mut_space,
                    bias_threshold=bias_constraints_threshold,
                    holo_based=bias_constraints_holo_based,
                    apply_to=bias_constraints_apply_to)
            logging.info(f'AffinityWorker {self.id}: running ADAPT pipes')
            self._run_pipes(self.apo_adapt_pipe, self.holo_adapt_pipe, 'adapt', **run_common_args)

        if transfer_bias:
            # Transfer ADAPT biases into MC experiment directory
            self.transfer_biases()

        if run_mc:
            logging.info(f'AffinityWorker {self.id}: running MC pipes')
            if config_changes is not None:
                self.apo_mc_pipe.setup(mc_config_changes=config_changes)
                self.holo_mc_pipe.setup(mc_config_changes=config_changes)
            self._run_pipes(self.apo_mc_pipe, self.holo_mc_pipe, 'mc', **run_common_args)

        if filter_results_by_constraints:
            self.filter_res()

        # collect summaries into a single DataFrame
        self.collect_summaries(overwrite=overwrite_summaries)
        self.ran_pipes = True
        end = time()
        logging.info(f'AffinityWorker {self.id}: finished running pipes in {end - start}s')
        return self.run_summaries, self if return_self else None

    def collect_adapt(self, parallel: bool = False) -> None:
        """
        Collect the results of ADAPT simulation.
        This will rewrite attributes, holding adapt pipes and their run results.
        :param parallel: collect Pipelines in parallel
        """
        pipes = [self.apo_adapt_pipe, self.holo_adapt_pipe]
        ((self.apo_adapt_results, self.apo_adapt_pipe),
         (self.holo_adapt_results, self.holo_adapt_pipe)) = self._collect_pipes(*pipes, parallel=parallel)

    def collect_mc(self, parallel: bool = False) -> None:
        """
        Collect the results of MC simulation.
        This will rewrite attributes holding mc pipes and their run results.
        :param parallel: collect Pipelines in parallel
        """
        pipes = [self.apo_mc_pipe, self.holo_mc_pipe]
        ((self.apo_mc_results, self.apo_mc_pipe),
         (self.holo_mc_results, self.holo_mc_pipe)) = self._collect_pipes(*pipes, parallel=parallel)

    def collect_summaries(self, overwrite: bool = False):
        """
        Combine summaries of all Pipeline objects into a single DataFrame and write it into a `run_summaries` attribute.
        Appends to existing `run_summaries` by default.
        """
        summaries = zip(
            [self.apo_adapt_results, self.holo_adapt_results, self.apo_mc_results, self.holo_mc_results],
            ['apo_adapt', 'holo_adapt', 'apo_mc', 'holo_mc']
        )
        summaries = filter(lambda x: x[0] is not None, summaries)
        run_summaries = pd.concat([self._wrap_summary(r.summary, p) for r, p in summaries])
        self.run_summaries = (
            run_summaries if self.run_summaries is None or overwrite
            else pd.concat([self.run_summaries, run_summaries]).drop_duplicates())
        return self.run_summaries

    def calculate_affinity(self, count_threshold: int = 100) -> pd.DataFrame:
        """
        The method basically handles IO to run the externally defined `affinity` function.
        :return: an output of the `affinity` function
        :param count_threshold: Counting threshold; sequences sampled less than this number are excluded
        from the population prior to affinity calculation.
        ran on populations of sequences and biases attained by this worker
        """
        if not self.ran_pipes:
            self.run()

        # Handle biases
        apo_bias, holo_bias = self._get_bias_paths()

        # Find the populations
        pop_apo, pop_holo = self._get_populations()

        # Compute the affinity
        self.affinity = affinity(
            reference_seq=self.ref_seq,
            pop_unbound=pop_apo,
            pop_bound=pop_holo,
            bias_unbound=apo_bias,
            bias_bound=holo_bias,
            temperature=self.temperature,
            threshold=count_threshold,
            positions=list(map(str, self.active_pos)))
        logging.info(f'AffinityWorker {self.id}: finished calculating affinity')
        self._subset_seq(self.affinity)
        return self.affinity

    def calculate_stability(self, count_threshold: int = 100, apo: bool = True) -> pd.DataFrame:
        """
        The method handles IO to run externally defined `stability` function.
        :param apo: calculate stability of the apo state (otherwise, the holo state)
        :param count_threshold: Counting threshold; sequences sampled less than this number are excluded
        from the population prior to stability calculation.
        :return: an output of the `stability` function -- a DataFrame with `seq` and `stability` columns.
        """
        if not self.ran_pipes:
            self.run()
        biases = self._get_bias_paths()
        pops = self._get_populations()
        bias, pop = (biases[0], pops[0]) if apo else (biases[1], pops[1])
        self.stability = stability(
            population=pop,
            bias=bias,
            ref_seq=self.ref_seq,
            temp=self.temperature,
            threshold=count_threshold,
            positions=list(map(str, self.active_pos)))
        logging.info(f'AffinityWorker {self.id}: finished calculating stability')
        self._subset_seq(self.stability)
        return self.stability

    def transfer_biases(self):
        def transfer_bias(adapt_pipe: Pipeline, mc_pipe: Pipeline):
            """
            Transfers last bias state from the ADAPT simulation into MC exp directory.
            """
            if adapt_pipe.last_bias is None:
                logging.warning(f'AffinityWorker {self.id}: last bias for {adapt_pipe.id} is absent')
                adapt_pipe.store_bias()
                if adapt_pipe.last_bias is None:
                    raise ValueError(f'AffinityWorker {self.id}: could not find bias for {adapt_pipe.id}')
            bias_path = f'{mc_pipe.base_dir}/{mc_pipe.exp_dir_name}/{self._default_bias_input_name}'
            adapt_pipe.dump_bias(path=bias_path)
            return None

        transfer_bias(self.apo_adapt_pipe, self.apo_mc_pipe)
        transfer_bias(self.holo_adapt_pipe, self.holo_mc_pipe)
        return self.apo_mc_pipe, self.holo_mc_pipe

    def bias_based_constraints(
            self, mut_space: t.Union[t.Set[str], str],
            bias_threshold: float = 10, holo_based: bool = True,
            apply_to: t.Tuple[str, ...] = ('apo_adapt', 'apo_mc', 'holo_adapt', 'holo_mc')) -> t.List[str]:
        pipes = list(
            map(
                op.itemgetter(1),
                filter(
                    lambda x: x[0] in apply_to,
                    (('apo_adapt', self.apo_adapt_pipe),
                     ('apo_mc', self.apo_mc_pipe),
                     ('holo_adapt', self.holo_adapt_pipe),
                     ('holo_mc', self.holo_mc_pipe))
                )
            )
        )
        if not pipes:
            raise ValueError(f'AffinityWorker {self.id}: no pipes to apply to')
        existing = u.extract_constraints([p.mc_conf for p in pipes])
        pipe = self.holo_adapt_pipe if holo_based else self.apo_adapt_pipe
        bias = pipe.last_bias
        if bias is None:
            raise ValueError(f'There is no bias accumulated for the Pipe {pipe.id}')
        if isinstance(mut_space, str):
            if not Path(mut_space).exists():
                raise ValueError(f'Invalid mut space path {mut_space}')
            with open(mut_space) as f:
                mut_space = {x.rstrip() for x in f if x != '\n'}

        pos = list(bias['var'].apply(lambda x: tuple(x.split('-')[:2])))
        pos += list(bias['var'].apply(lambda x: tuple(x.split('-')[2:])))
        df = pd.DataFrame({'pos': pos, 'bias': list(bias['bias']) * 2})
        df = df.groupby('pos', as_index=False).agg(
            above_threshold=pd.NamedAgg('bias', lambda x: (x > bias_threshold).all()))
        exclude_grouped = groupby(
            df[df.above_threshold]['pos'],
            op.itemgetter(0))
        exclude = ((pos, set(aa for _, aa in group)) for pos, group in exclude_grouped)
        constraints = ((pos, mut_space - e) for pos, e in exclude)
        constraints = [f'{pos} {" ".join(c)}' for pos, c in constraints]
        constraints = u.merge_constraints(existing, constraints)

        for p in pipes:
            p.setup(
                mc_config_changes=[('Space_Constraints', constraints)],
                continuation=True)
        return constraints

    def filter_res(self, seq_col: str = 'seq') -> None:
        aa_mapping = AminoAcidDict().aa_dict
        pipes = (self.apo_adapt_pipe, self.apo_mc_pipe, self.holo_adapt_pipe, self.holo_mc_pipe)
        constraints_split = map(
            lambda x: x.split(),
            u.extract_constraints([pipe.mc_conf for pipe in pipes]))
        constraints_of_subset = [
            (self.active_subset_mapped[int(c[0])], [aa_mapping[x] for x in c[1:]]) for c in constraints_split
            if int(c[0]) in self.active_subset_mapped
        ]

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            return df[df[seq_col].apply(lambda x: all(x[i] in c for i, c in constraints_of_subset))]

        for p in pipes:
            if p.results is not None and p.results.seqs is not None:
                prev_len = len(p.results.seqs)
                seqs = _filter(p.results.seqs)
                summary = p.collect_summary(seqs=seqs)
                p.results = PipelineOutput(seqs, summary)
                logging.info(
                    f'AffinityWorker {self.id}: filtered out {prev_len - len(p.results.seqs)} rows '
                    f'in the results of the {p.id}')
        self._copy_results()
        return

    def _run_pipes(
            self, pipe_apo: Pipeline, pipe_holo: Pipeline, pipes_type: str,
            wait: bool = True, collect: bool = False, collect_parallel: bool = False,
            cleanup: bool = False, cleanup_kwargs: t.Optional[t.Dict] = None) -> t.Tuple[Popen, Popen]:
        """
        Call apo and holo ADAPT parallel subprocesses.
        :param wait: Wait for simulations to finish.
        :param collect: Collect simulation's results.
        :param collect_parallel: Use a Pool of two workers to collect the results.
        :param cleanup: if True will call the `cleanup` method for each of the four pipelines
        :param cleanup_kwargs: pass these kwargs to the `cleanup` method if `cleanup==True`
        :return: (apo, holo) Popen objects and (optionally) adapt pipelines
        """
        if pipes_type not in ['adapt', 'mc']:
            raise ValueError(f'Incorrect argument `pipes_type` {pipes_type}')
        # Spawn ADAPT subprocesses
        proc = (pipe_apo.run_non_blocking(), pipe_holo.run_non_blocking())

        if wait:
            # Wait for ADAPT subprocesses to finish
            for p in proc:
                p.communicate()
        if collect:
            if not wait:
                raise ValueError('Must wait for results before collecting.')
            if pipes_type == 'adapt':
                self.collect_adapt(parallel=collect_parallel)
            else:
                self.collect_mc(parallel=collect_parallel)

        if cleanup:
            if not wait:
                raise ValueError('Must wait for results before cleaning.')
            if not collect:
                logging.error(f'AffinityWorker {self.id}: cleaning up the results without their prior collection '
                              f'means useless run!')
            cleanup_kwargs = {} if cleanup_kwargs is None else cleanup_kwargs
            pipe_apo.cleanup(**cleanup_kwargs)
            pipe_holo.cleanup(**cleanup_kwargs)
        return proc

    def _get_bias_paths(self) -> t.Tuple[t.Optional[str], t.Optional[str]]:
        apo_bias = self.apo_mc_cfg.get_field_value('Bias_Input_File')
        holo_bias = self.holo_mc_cfg.get_field_value('Bias_Input_File') or apo_bias
        return apo_bias, holo_bias

    def _get_populations(self) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        pop_apo, pop_holo = self.apo_mc_pipe.results.seqs, self.holo_mc_pipe.results.seqs
        if pop_apo is None:
            pop_apo_path = f'{self.apo_mc_pipe.exp_dir}/{self.apo_mc_pipe.default_results_dump_name}'
            if not Path(pop_apo_path).exists():
                raise RuntimeError(f'Could not find the population at {pop_apo_path}')
            pop_apo = pd.read_csv(pop_apo_path, sep='\t')
        if pop_holo is None:
            pop_holo_path = f'{self.holo_mc_pipe.exp_dir}/{self.holo_mc_pipe.default_results_dump_name}'
            if not Path(pop_holo_path).exists():
                raise RuntimeError(f'Could not find the population at {pop_holo_path}')
            pop_holo = pd.read_csv(pop_holo_path, sep='\t')
        return pop_apo, pop_holo

    def _get_temperature(self) -> float:
        # Run only after pipes has been setup
        pipes = [self.apo_adapt_pipe, self.apo_mc_pipe, self.holo_adapt_pipe, self.holo_mc_pipe]
        temps = (p.mc_conf.get_field_value('Temperature') for p in pipes)
        temps = (temp[0] if isinstance(temp, t.List) else temp for temp in temps)
        temps = set(map(float, temps))
        if len(temps) != 1:
            raise ValueError(f'Every pipe should have the same `Temperature` for the first walker parameter. '
                             f'Got {temps} instead.')
        return float(temps.pop())

    @staticmethod
    def _collect_res(pipe: Pipeline) -> t.Tuple[PipelineOutput, Pipeline]:
        # Convention is to use walker you really care about as the first one
        n_walkers = [1, pipe.mc_conf.get_field_value('Replica_Number'),
                     pipe.mc_conf.get_field_value('Trajectory_Number')]
        n_walkers = max(map(int, filterfalse(lambda x: x is None, n_walkers)))
        walker = 0 if n_walkers > 1 else None
        return pipe.collect_results(
            dump_results=True, dump_bias=True, parallel=False,
            walker=walker, return_self=True)

    def _collect_pipes(self, pipe_apo: Pipeline, pipe_holo: Pipeline, parallel: bool) \
            -> t.Tuple[t.Tuple[PipelineOutput, Pipeline], t.Tuple[PipelineOutput, Pipeline]]:
        # Collect results for a pair of Pipeline's
        if parallel:
            with Pool(2) as workers:
                (res_apo, pipe_apo_), (res_holo, pipe_holo_) = workers.map(self._collect_res, [pipe_apo, pipe_holo])
        else:
            (res_apo, pipe_apo_), (res_holo, pipe_holo_) = map(self._collect_res, [pipe_apo, pipe_holo])
        # Create a new column with the sequence subset in both of the results
        self._subset_seq(res_apo.seqs), self._subset_seq(res_holo.seqs)
        return (res_apo, pipe_apo_), (res_holo, pipe_holo_)

    def _subset_seq(self, df: pd.DataFrame, seq_col_name: str = 'seq', new_col_name: str = 'seq_subset') -> None:
        """
        Creates a new column in the results `df` based on existing 'seq_col_name` column -- `new_col_name`,
        using `self.active_subset_map` attribute to subset the sequences.
        """
        df[new_col_name] = df[seq_col_name].apply(
            lambda x: "".join(x[i] for i in self.active_subset_mapped.values()))

    @staticmethod
    def _wrap_summary(summary: pd.DataFrame, pipeline: str) -> pd.DataFrame:
        summary = summary.copy()
        summary['pipeline'] = pipeline
        return summary

    def _copy_configs(self):
        self.apo_adapt_cfg = self.apo_adapt_pipe.mc_conf.copy()
        self.holo_adapt_cfg = self.holo_adapt_pipe.mc_conf.copy()
        self.apo_mc_cfg = self.apo_mc_pipe.mc_conf.copy()
        self.holo_mc_cfg = self.holo_mc_pipe.mc_conf.copy()

    def _copy_results(self):
        self.apo_adapt_results = self.apo_adapt_pipe.results
        self.apo_mc_results = self.apo_mc_pipe.results
        self.holo_adapt_results = self.holo_adapt_pipe.results
        self.holo_mc_results = self.holo_mc_pipe.results


def infer_mut_space(
        df: pd.DataFrame, pos_mapping: t.Mapping[int, int],
        stability_lower: t.Optional[float], stability_upper: t.Optional[float],
        affinity_lower: t.Optional[float], affinity_upper: t.Optional[float]):
    """
    Restricts the mutation space of a position given the results of AffinitySearch.

    Filters the `df` given provided boundaries (optional).
    For each position, subsets the `df`, leaving only entries with this position involved,
    and obtains the amino acid types at this positions.
    :param df: expects to get a DataFrame with four columns: (1) pos, (2) seq, (3) stability, (4) affinity
    (i.e., produced by `AffinitySearch.collect_results` method).
    :param pos_mapping: mapping between a position (i.e., the one used in a `pos` column) and its index in the `seq`
    :param stability_lower: lower boundary for stability
    :param stability_upper: upper boundary for stability
    :param affinity_lower: lower boundary for affinity
    :param affinity_upper: upper boundary for affinity
    :return:
    """
    df_expected = ('pos', 'seq', 'stability', 'affinity')
    if any(x not in df.columns for x in df_expected):
        raise ValueError(f'Expected input `df` to have {df_expected} columns')
    df = _filter_results(df, stability_lower, stability_upper, affinity_lower, affinity_upper)

    def mut_space_for_pos(pos):
        pos_i = pos_mapping[int(pos)]
        sub = df[df.pos.apply(lambda x: pos in x)]
        return set(sub.seq.apply(op.itemgetter(pos_i)))

    return {p: mut_space_for_pos(p) for p in map(str, pos_mapping)}


def _filter_results(
        df: pd.DataFrame, stability_lower: t.Optional[float], stability_upper: t.Optional[float],
        affinity_lower: t.Optional[float], affinity_upper: t.Optional[float]) -> pd.DataFrame:
    df = df.copy()
    if stability_lower is not None:
        df = df[df.stability > stability_lower]
    if stability_upper is not None:
        df = df[df.stability < stability_upper]
    if affinity_lower is not None:
        df = df[df.affinity > affinity_lower]
    if affinity_upper is not None:
        df = df[df.affinity < affinity_upper]
    return df


if __name__ == '__main__':
    raise RuntimeError
