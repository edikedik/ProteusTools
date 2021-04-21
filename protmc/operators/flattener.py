import logging
import typing as t
from shutil import copyfile
from warnings import warn

import ray

from protmc.common.base import AbstractExecutor, AbstractPoolExecutor, Id, AbstractWorker
from protmc.operators.worker import ADAPT, MC, Worker


class ExecutorError(Exception):
    def __init__(self, executor: AbstractExecutor, worker: AbstractWorker, error: Exception):
        self.message = f'Executor {executor.id} failed to execute {worker.id} with an error {error}'


class Flattener(AbstractPoolExecutor):
    """
    A higher-order class orchestrating the execution of a pair of workers: `MC` and `ADAPT`,
    whose purpose is to flatten the sequence landscape.
    """
    def __init__(self, id_: Id = None, reference: t.Optional[str] = None,
                 predicate_adapt: t.Optional[t.Callable[[ADAPT], bool]] = None,
                 predicate_mc: t.Optional[t.Callable[[MC], bool]] = None,
                 executor_adapt: t.Optional[AbstractExecutor] = None,
                 executor_mc: t.Optional[AbstractExecutor] = None,
                 count_threshold: int = 1, step_threshold: int = 300):
        """
        :param id_: Unique identifier. Defaults to `id(self)`.
        :param reference: Reference sequence. In case it is provided,
        its presence serves as additional flattening criterion.
        :param predicate_adapt: A callable accepting `ADAPT` object and returning
        `True` if the object is considered "flattened" (and `False` otherwise).
        :param predicate_mc: A callable accepting `MC` object and returning
        `True` if the object is considered "flattened" (and `False` otherwise).
        :param executor_adapt: `Executor` operator with `ADAPT` execution strategy.
        :param executor_mc:  `Executor` operator with `MC` execution strategy.
        :param count_threshold: Threshold to consider the `reference` counted, if the latter is provided.
        Note that separate eponymous argument exists at he level of workers.
        :param step_threshold: A maximum number of steps to execute flattening.
        """
        super().__init__(id_)
        self.predicate_mc = predicate_mc
        self.predicate_adapt = predicate_adapt
        self.reference = reference
        self.count_threshold = count_threshold
        self.step_threshold = step_threshold
        if executor_mc is None and executor_adapt is None:
            raise ValueError('Must provide at least one of the executors')
        self.executor_adapt = executor_adapt or executor_mc
        self.executor_mc = executor_mc or executor_adapt

    def get_id(self):
        return self._id

    def adapt_passes(self, adapt: ADAPT) -> bool:
        return ((self.predicate_adapt is None or self.predicate_adapt(adapt)) and
                (self.reference is None or self.reference_sampled(adapt)))

    def mc_passes(self, mc: MC) -> bool:
        return ((self.predicate_mc is None or self.predicate_mc(mc)) and
                (self.reference is None or self.reference_sampled(mc)))

    def reference_sampled(self, worker: Worker):
        return (worker.seqs is None or self.reference in set(
            worker.seqs.loc[worker.seqs['total_count'] >= self.count_threshold, 'seq']))

    def transfer_bias(self, adapt: ADAPT, mc: MC) -> None:
        """
        Transfer the current `ADAPT` from its working directory to the `MC` working directory.
        Modifies `MC`'s config so that it points to the transferred bias.
        """
        # Check that there is an accumulated bias
        if adapt.bias is None or adapt.bias.bias is None:
            raise RuntimeError()
        # Dump the last bias' step the into working dirs of both workers
        adapt_bias_path = f'{adapt.params.working_dir}/{adapt.params.input_bias_name}'
        adapt.bias.dump(adapt_bias_path, last_step=True, tsv=False)
        logging.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- '
                     f'saved current bias to {adapt_bias_path}')
        mc_bias_path = f'{mc.params.working_dir}/{mc.params.input_bias_name}'
        # Copying is faster than calling `dump`
        copyfile(adapt_bias_path, mc_bias_path)
        logging.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- '
                     f'transferred bias from {adapt_bias_path} to {mc_bias_path}')
        # Modify configs to use dumped bias
        adapt.modify_config(field_setters=[('MC_IO', 'Bias_Input_File', adapt_bias_path)], dump=True)
        mc.modify_config(field_setters=[('MC_IO', 'Bias_Input_File', mc_bias_path)], dump=True)

    def __call__(self, pair: t.Tuple[ADAPT, MC]) -> t.Tuple[ADAPT, MC, int]:
        """
        Run the flattening loop.
        :param pair: A pair of (1) `ADAPT` worker and (2) `MC` worker.
        :return: A tuple of `ADAPT`, `MC` and the number of steps until the execution stopped.
        :raises ExecutionError: if executor fails on any of the workers.
        """
        adapt, mc = pair
        step = 0
        if self.adapt_passes(adapt) and self.mc_passes(mc):
            logging.error(f'Both workers appear to be flattened at step 0. '
                          f'Check your stopping criteria')
        while not self.adapt_passes(adapt) or not self.mc_passes(mc):
            logging.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- starting step {step}. '
                         f'MC pass: {self.mc_passes(mc)}, ADAPT pass: {self.adapt_passes(adapt)}')
            try:
                adapt = self.executor_adapt(adapt)
            except Exception as e:
                raise ExecutorError(self.executor_adapt, adapt, e)
            self.transfer_bias(adapt, mc)
            try:
                mc = self.executor_mc(mc)
            except Exception as e:
                raise ExecutorError(self.executor_mc, mc, e)
            logging.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- finished at step {step}')
            step += 1
            if step >= self.step_threshold:
                logging.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- terminating '
                             f'due to reaching step threshold {self.step_threshold}')
                break
        logging.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- finished at step {step}')
        return adapt, mc, step


def flatten_remotely(
        workers: t.Sequence[t.Tuple[ADAPT, MC]], flattener: Flattener,
        stop_on_fail: bool = False) -> t.List[t.Tuple[ADAPT, MC]]:
    """
    A helper function for parallel `Flattener`s execution via `ray` remote function.
    :param workers: A pair of workers to flatten.
    :param flattener: An initialized `Flattener` object.
    :param stop_on_fail: If the execution fails, return the workers immediately.
    :return:
    """
    logging.info(f'Setting up Flattener {flattener.id} on {len(workers)} worker pairs')
    handles = [flatten_pair.remote(p, flattener) for p in workers]
    all_ids = {(adapt.id, mc.id) for adapt, mc in workers}
    done_ids = set()
    done = []
    while handles:
        pair_done, handles = ray.wait(handles)
        adapt, mc, step = ray.get(pair_done[0])
        if not step:
            logging.warning(f'Failed on {adapt.id} and {mc.id}')
            if stop_on_fail:
                return done
        else:
            logging.info(f'Flattened {adapt.id} and {mc.id} in {step} steps')
        done_ids |= {(adapt.id, mc.id)}
        remain = all_ids - done_ids
        logging.info(f'Remaining {len(remain)} our of {len(all_ids)}: {remain}')
        done.append((adapt, mc))
    return done


@ray.remote
def flatten_pair(pair: t.Tuple[ADAPT, MC], flattener: Flattener) \
        -> t.Tuple[ADAPT, MC, int]:
    try:
        adapt, mc, step = flattener(pair)
    except ValueError as e:
        adapt, mc, step = pair[0], pair[1], 0
        warn(f'Flattener {flattener.id} failed on pair ({adapt.id}, {mc.id}) with an error {e}')
    return adapt, mc, step


if __name__ == '__main__':
    raise RuntimeError
