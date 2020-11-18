import logging
import typing as t
from copy import deepcopy
from itertools import filterfalse
from pathlib import Path

import pandas as pd
from multiprocess.pool import Pool
from tqdm import tqdm

from protmc import config
from protmc import utils as u
from protmc.base import NoReferenceError
from protmc.affinity import affinity
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
    For many combinations it is advised to run parallel computations (consult with the `run` method docs).
    """

    def __init__(
            self, positions: t.Iterable[t.Union[t.Collection[int], int]], active: t.List[int],
            apo_base_setup: WorkerSetup, holo_base_setup: WorkerSetup,
            exe_path: str, base_dir: str, ref_seq: t.Optional[str] = None,
            mut_space_n_types: int = 18, count_threshold: int = 10,
            adapt_dir_name: str = 'ADAPT', mc_dir_name: str = 'MC',
            apo_dir_name: str = 'apo', holo_dir_name: str = 'holo'):
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
        self.count_threshold = count_threshold
        self.adapt_dir_name, self.mc_dir_name = adapt_dir_name, mc_dir_name
        self.apo_dir_name, self.holo_dir_name = apo_dir_name, holo_dir_name
        self.positions = positions
        if not isinstance(positions, t.Tuple):
            self.positions = tuple(self.positions)
        self.active = active
        self.ref_seq = ref_seq or self._infer_ref_seq()
        self.ran_setup, self.ran_workers = False, False
        self.workers: t.Optional[t.List[AffinityWorker]] = None
        self.summaries: t.Optional[pd.DataFrame] = None
        self.affinities_: t.Optional[pd.DataFrame] = None
        self.stabilities_: t.Optional[pd.DataFrame] = None
        self.results: t.Optional[pd.DataFrame] = None
        self.id = id(self)
        logging.info(f'AffinitySearch {self.id}: initialized')

    def setup_workers(self) -> None:
        """
        Setup `AffinityWorker`'s based on subsets of active positions.
        Will populate the `workers` attribute with prepared `AffinityWorker` instances.
        """
        self.workers = [self._setup_worker(active_subset=s) for s in self.positions]
        self.ran_setup = True
        logging.info(f'AffinitySearch {self.id}: ran setup for workers')

    def run_workers(self, num_proc: int = 1, cleanup: bool = False,
                    cleanup_kwargs: t.Optional[t.Dict] = None) -> pd.DataFrame:
        """
        Run each of prepared `AffinityWorkers`.
        :param num_proc: Number of processes.
        In general, each process will run a separate `AffinityWorker`.
        The latter takes 2-3 CPU units (2 for `Pipeline`s, 1 for results aggregation).
        However, if some config specifies `n` walkers, the corresponding `Pipeline` will take `n` CPUs.
        Hence, in the latter case one should carefully count the maximal `num_proc`.
        WARNING: currently, even if `num_proc=1`, each worker will take 2 CPUs.
        :param cleanup: If True, each `Pipeline` within each of `AffinityWorker`s will call its `cleanup` method,
        by default removing `seq` and `ener` files (which can be customized via `cleanup_kwargs` argument.
        :param cleanup_kwargs: Pass this dictionary to `cleanup` method of each `Pipeline`.
        :return: A DataFrame of summaries comprising run summary
        for each underlining `Pipeline` within each `AffinityWorker`.
        """
        if not self.ran_setup:
            self.setup_workers()

        # Obtain the run results for each of the workers
        common_args = dict(cleanup=cleanup, cleanup_kwargs=cleanup_kwargs, return_self=True)
        if num_proc > 1:
            with Pool(num_proc) as workers:
                results = workers.map(lambda w: w.run(parallel=False, **common_args), self.workers)
        else:
            results = [w.run(parallel=True, **common_args) for w in self.workers]

        # Separate results
        summaries = [r[0] for r in results]
        self.workers = [r[1] for r in results]

        # Process and return summaries
        for res, d in zip(summaries, (w.run_dir for w in self.workers)):
            res['run_dir'] = d
        self.summaries = pd.concat(summaries)
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
            self.affinities_, self.stabilities_, on=['seq', 'pos'], how='inner')[
            ['pos', 'seq', 'stability', 'affinity']]
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
        workers = tqdm(self.workers, desc=f'Calculating {what}') if verbose else self.workers

        # get the affinity DataFrames from `AffinityWorkers`
        if num_proc > 1:
            with Pool(num_proc) as pool:
                results = pool.map(try_calculate, workers)
            for w, r in zip(self.workers, results):  # attributes would be unchanged within different processes
                w.affinity = r
        else:
            results = list(map(try_calculate, workers))

        # process the affinity DataFrames (if any)
        wrapped = (wrap_calculated(w, r) for w, r in zip(self.workers, results))
        filtered = list(filterfalse(lambda x: x is None, wrapped))

        # none of the workers yielded affinity DataFrame -> return None
        if not filtered:
            logging.warning(f'AffinitySearch {self.id}: no position combination yielded {what} results')
            return None

        # otherwise, concatenate, dump and return the results
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
            active_pos=self.active, ref_seq=self.ref_seq,
            mut_space_n_types=self.mut_space_n_types, exe_path=self.exe_path,
            run_dir=f'{self.base_dir}/{exp_dir_name}',
            count_threshold=self.count_threshold,
            adapt_dir_name=self.adapt_dir_name, mc_dir_name=self.mc_dir_name,
            apo_dir_name=self.apo_dir_name, holo_dir_name=self.holo_dir_name)
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
    def __init__(
            self, apo_setup: WorkerSetup, holo_setup: WorkerSetup,
            active_pos: t.Iterable[int], exe_path: str, run_dir: str, ref_seq: str,
            mut_space_n_types: int = 18, count_threshold: int = 10,
            adapt_dir_name: str = 'ADAPT', mc_dir_name: str = 'MC',
            apo_dir_name: str = 'apo', holo_dir_name: str = 'holo'):
        self.apo_adapt_cfg, self.apo_mc_cfg, self.apo_matrix_path = apo_setup
        self.holo_adapt_cfg, self.holo_mc_cfg, self.holo_matrix_path = holo_setup
        self.active_pos = active_pos
        self.mut_space_n_types = mut_space_n_types
        self.exe_path = exe_path
        self.run_dir = run_dir
        self.count_threshold = count_threshold
        self.adapt_dir_name, self.mc_dir_name = adapt_dir_name, mc_dir_name
        self.apo_dir_name, self.holo_dir_name = apo_dir_name, holo_dir_name
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
        self.temperature: t.Optional[float] = None
        self.affinity: t.Optional[pd.DataFrame] = None
        self.stability: t.Optional[pd.DataFrame] = None
        self.ref_seq: str = ref_seq
        self.default_bias_input_name = 'ADAPT.inp.dat'
        self.id = id(self)
        logging.info(f'AffinityWorker {self.id}: initialized')

    def setup(self) -> None:
        def create_pipe(cfg, base_dir, exp_dir, energy_dir):
            # Simplifies the creation of Pipeline by encapsulating common arguments
            return Pipeline(
                base_post_conf=None, exe_path=self.exe_path,
                mut_space_n_types=self.mut_space_n_types, active_pos=list(self.active_pos),
                base_mc_conf=cfg, base_dir=base_dir, exp_dir_name=exp_dir, energy_dir=energy_dir)

        def handle_bias_inp(pipe: Pipeline) -> None:
            # Simplifies the assignment of the bias input path
            bias_path = f'{pipe.base_dir}/{pipe.exp_dir_name}/{self.default_bias_input_name}'
            pipe.base_mc_conf.set_field('MC_IO', 'Bias_Input_File', bias_path)

        # base directories for holo and apo Pipelines
        base_apo, base_holo = f'{self.run_dir}/{self.apo_dir_name}', f'{self.run_dir}/{self.holo_dir_name}'

        # create Pipeline objects
        self.apo_adapt_pipe = create_pipe(
            cfg=self.apo_adapt_cfg, base_dir=base_apo, exp_dir=self.adapt_dir_name, energy_dir=self.apo_matrix_path)
        self.holo_adapt_pipe = create_pipe(
            cfg=self.holo_adapt_cfg, base_dir=base_holo, exp_dir=self.adapt_dir_name, energy_dir=self.holo_matrix_path)
        self.apo_mc_pipe = create_pipe(
            cfg=self.apo_mc_cfg, base_dir=base_apo, exp_dir=self.mc_dir_name, energy_dir=self.apo_matrix_path)
        self.holo_mc_pipe = create_pipe(
            cfg=self.holo_mc_cfg, base_dir=base_holo, exp_dir=self.mc_dir_name, energy_dir=self.holo_matrix_path)

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
        self.temperature = self._get_temperature()

        self.ran_setup = True
        logging.info(f'AffinityWorker {self.id}: completed setting up Pipelines')

    def run(self, parallel: bool = True, cleanup: bool = False,
            cleanup_kwargs: t.Optional[t.Dict] = None,
            return_self: bool = True) -> t.Tuple[pd.DataFrame, t.Optional['AffinityWorker']]:
        """
        Run AffinityWorker.
        :param parallel: if True will collect results for MC and ADAPT in parallel
        :param cleanup: if True will call the `cleanup` method for each of the four pipelines
        :param cleanup_kwargs: pass these kwargs to the `cleanup` method if `cleanup==True`
        :param return_self: whether to return `self` (useful for CPU-parallelization)
        :return: a DataFrame with 4 rows (Summary's returned by the Pipeline's).
        """

        def collect_res(pipe: Pipeline) -> t.Tuple[PipelineOutput, Pipeline]:
            # Convention is to use walker you really care about as a first one
            n_walkers = [1, pipe.mc_conf.get_field_value('Replica_Number'),
                         pipe.mc_conf.get_field_value('Trajectory_Number')]
            n_walkers = max(map(int, filterfalse(lambda x: x is None, n_walkers)))
            walker = 0 if n_walkers > 1 else None
            return pipe.collect_results(
                dump_results=True, dump_bias=True, parallel=False, walker=walker, return_self=True)

        def collect_pipes(pipe_apo, pipe_holo):
            # Collect results for a pair of Pipeline's
            if parallel:
                with Pool(2) as workers:
                    (res_apo, pipe_apo_), (res_holo, pipe_holo_) = workers.map(collect_res, [pipe_apo, pipe_holo])
            else:
                (res_apo, pipe_apo_), (res_holo, pipe_holo_) = map(collect_res, [pipe_apo, pipe_holo])
            return (res_apo, pipe_apo_), (res_holo, pipe_holo_)

        def wrap_summary(summary: pd.DataFrame, pipeline: str) -> pd.DataFrame:
            summary = summary.copy()
            summary['pipeline'] = pipeline
            return summary

        if not self.ran_setup:
            logging.info(f'AffinityWorker {self.id}: running setup with default arguments')
            self.setup()
        adapt_pipes = [self.apo_adapt_pipe, self.holo_adapt_pipe]
        mc_pipes = [self.apo_mc_pipe, self.holo_mc_pipe]

        logging.info(f'AffinityWorker {self.id}: running ADAPT pipes')
        # Spawn ADAPT subprocesses
        adapt_proc = [p.run_non_blocking() for p in adapt_pipes]
        # Wait for ADAPT subprocesses to finish
        for proc in adapt_proc:
            proc.communicate()

        # Transfer ADAPT biases into MC experiment directory
        self._transfer_biases()

        # Immediately start detached subprocesses for running MC
        logging.info(f'AffinityWorker {self.id}: running MC pipes')
        mc_proc = [p.run_non_blocking() for p in mc_pipes]

        # Meanwhile collect the results of ADAPT runs
        logging.info(f'AffinityWorker {self.id}: aggregating ADAPT results')
        ((self.apo_adapt_results, self.apo_adapt_pipe),
         (self.holo_adapt_results, self.holo_adapt_pipe)) = collect_pipes(*adapt_pipes)

        # Wait for MC subprocesses to finish
        for proc in mc_proc:
            proc.communicate()

        # Collect the results of MC runs
        logging.info(f'AffinityWorker {self.id}: aggregating MC results')
        ((self.apo_mc_results, self.apo_mc_pipe),
         (self.holo_mc_results, self.holo_mc_pipe)) = collect_pipes(*mc_pipes)

        logging.info(f'AffinityWorker {self.id}: finished running pipes')

        # optionally cleanup each pipeline
        if cleanup:
            logging.info(f'AffinityWorker {self.id}: cleaning up pipes')
            kw = {} if cleanup_kwargs is None else cleanup_kwargs
            for p in [self.apo_adapt_pipe, self.apo_mc_pipe, self.holo_adapt_pipe, self.holo_mc_pipe]:
                p.cleanup(**kw)

        # collect summaries into a single DataFrame
        self.run_summaries = pd.concat([
            wrap_summary(r.summary, p) for r, p in zip(
                [self.apo_adapt_results, self.apo_mc_results, self.apo_mc_results, self.holo_mc_results],
                ['apo_adapt', 'holo_adapt', 'apo_mc', 'holo_mc'])]
        )
        self.ran_pipes = True
        return self.run_summaries, self if return_self else None

    def calculate_affinity(self) -> pd.DataFrame:
        """
        The method basically handles IO to run the externally defined `affinity` function.
        :return: an output of the `affinity` function
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
            threshold=self.count_threshold,
            positions=list(map(str, self.active_pos)))
        logging.info(f'AffinityWorker {self.id}: finished calculating affinity')
        return self.affinity

    def calculate_stability(self, apo: bool = True) -> pd.DataFrame:
        """
        The method handles IO to run externally defined `stability` function.
        :param apo: calculate stability of the apo state (otherwise, the holo state)
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
            threshold=self.count_threshold,
            positions=list(map(str, self.active_pos)))
        logging.info(f'AffinityWorker {self.id}: finished calculating stability')
        return self.stability

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

    def _get_temperature(self):
        # Run only after pipes has been setup
        pipes = [self.apo_adapt_pipe, self.apo_mc_pipe, self.holo_adapt_pipe, self.holo_mc_pipe]
        temps = (p.mc_conf.get_field_value('Temperature') for p in pipes)
        temps = (temp[0] if isinstance(temp, t.List) else temp for temp in temps)
        temps = set(map(float, temps))
        if len(temps) != 1:
            raise ValueError(f'Every pipe should have the same `Temperature` for the first walker parameter. '
                             f'Got {temps} instead.')
        return float(temps.pop())

    def _transfer_biases(self):
        def transfer_bias(adapt_pipe: Pipeline, mc_pipe: Pipeline):
            """
            Transfers last bias state from the ADAPT simulation into MC experiments directory.
            Modifies the MC pipeline config accordingly
            :param adapt_pipe:
            :param mc_pipe:
            :return:
            """
            adapt_cfg = adapt_pipe.mc_conf
            bias_path = adapt_cfg.get_field_value('Adapt_Output_File')
            output_period = adapt_cfg.get_field_value('Adapt_Output_Period')
            n_steps = adapt_cfg.get_field_value('Trajectory_Length')
            last_step = (n_steps // output_period) * output_period
            bias = u.get_bias_state(bias_path, last_step)
            with open(f'{mc_pipe.exp_dir}/{self.default_bias_input_name}', 'w') as f:
                print(bias.rstrip(), file=f)
            return None

        transfer_bias(self.apo_adapt_pipe, self.apo_mc_pipe)
        transfer_bias(self.holo_adapt_pipe, self.holo_mc_pipe)
        return self.apo_mc_pipe, self.holo_mc_pipe

    def _copy_configs(self):
        self.apo_adapt_cfg = self.apo_adapt_pipe.mc_conf.copy()
        self.holo_adapt_cfg = self.holo_adapt_pipe.mc_conf.copy()
        self.apo_mc_cfg = self.apo_mc_pipe.mc_conf.copy()
        self.holo_mc_cfg = self.holo_mc_pipe.mc_conf.copy()


if __name__ == '__main__':
    raise RuntimeError
