import logging
import typing as t
from shutil import copyfile

from protmc.common.base import AbstractExecutor, AbstractPoolExecutor, Id
from protmc.operators.worker import ADAPT, MC, Worker


class Flattener(AbstractPoolExecutor):
    def __init__(self, id_: Id = None, reference: t.Optional[str] = None,
                 predicate_mc: t.Optional[t.Callable[[MC], bool]] = None,
                 predicate_adapt: t.Optional[t.Callable[[ADAPT], bool]] = None,
                 count_threshold: int = 1, log_path: t.Optional[str] = None, loglevel=logging.INFO):
        super().__init__(id_)
        self.predicate_mc = predicate_mc
        self.predicate_adapt = predicate_adapt
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(loglevel)
        if log_path is not None:
            self.logger.addHandler(logging.FileHandler(log_path, 'a+'))
        self.reference = reference
        self.count_threshold = count_threshold

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
        # Check that there is an accumulated bias
        if adapt.bias is None or adapt.bias.bias is None:
            raise RuntimeError()
        # Dump the last bias' step the into working dirs of both workers
        adapt_bias_path = f'{adapt.params.working_dir}/{adapt.params.input_bias_name}'
        adapt.bias.dump(adapt_bias_path, last_step=True, tsv=False)
        self.logger.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- '
                         f'saved current bias to {adapt_bias_path}')
        mc_bias_path = f'{mc.params.working_dir}/{mc.params.input_bias_name}'
        # Copying is faster than calling `dump`
        copyfile(adapt_bias_path, mc_bias_path)
        self.logger.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- '
                         f'transferred bias from {adapt_bias_path} to {mc_bias_path}')
        # Modify configs to use dumped bias
        adapt.modify_config(field_setters=[('MC_IO', 'Bias_Input_File', adapt_bias_path)], dump=True)
        mc.modify_config(field_setters=[('MC_IO', 'Bias_Input_File', mc_bias_path)], dump=True)

    def apply(self, executor: AbstractExecutor, pool: t.Tuple[ADAPT, MC],
              mc_executor: t.Optional[AbstractExecutor] = None) -> t.Tuple[ADAPT, MC]:
        adapt, mc = pool
        step = 0
        if self.adapt_passes(adapt) and self.mc_passes(mc):
            self.logger.error(f'Both workers appear to be flattened at step 0. '
                              f'Check your stopping criteria')
        while not self.adapt_passes(adapt) or not self.mc_passes(mc):
            self.logger.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- starting step {step}')
            try:
                adapt = executor(adapt)
            except Exception as e:
                msg = f'Executor {executor.id} failed to execute {adapt.id} with an error {e}'
                self.logger.error(msg)
                raise ValueError(msg)
            self.transfer_bias(adapt, mc)
            try:
                mc = mc_executor(mc) if mc_executor else executor(mc)
            except Exception as e:
                msg = f'Executor {mc_executor.id} failed to execute {mc.id} with an error {e}'
                self.logger.error(msg)
                raise ValueError(msg)
            step += 1
        self.logger.info(f'Flattener {self.id} of {adapt.id} and {mc.id} -- finished at step {step}')
        return adapt, mc


if __name__ == '__main__':
    raise RuntimeError
