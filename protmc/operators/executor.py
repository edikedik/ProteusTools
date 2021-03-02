import logging
import typing as t

from protmc.common.base import AbstractExecutor, AbstractCallback, Id
from protmc.operators.worker import ADAPT, Worker


class GenericExecutor(AbstractExecutor):
    def __init__(self, id_: Id = None, store_bias: bool = True,
                 callbacks: t.Optional[t.List[AbstractCallback]] = None,
                 collect_seqs: bool = True, collect_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
                 compose_summary: bool = True, summary_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
                 cleanup: bool = True, cleanup_kwargs: t.Optional[t.Dict[str, t.Any]] = None):
        super().__init__(id_)
        self.store_bias = store_bias
        self.callbacks = callbacks
        self.collect_seqs = collect_seqs
        self.collect_kwargs = {} if collect_kwargs is None else collect_kwargs
        self.compose_summary = compose_summary
        self.summary_kwargs = {} if summary_kwargs is None else summary_kwargs
        self.cleanup: bool = cleanup
        self.cleanup_kwargs = {} if cleanup_kwargs is None else cleanup_kwargs

    def __call__(self, worker: Worker) -> Worker:
        worker.setup_io()
        logging.info(f'GenericExecutor {self.id} -- starting executing worker {worker.id}')
        worker.run()
        logging.debug(f'GenericExecutor {self.id} -- finished running worker {worker.id}')
        if self.collect_seqs:
            worker.collect_seqs(**self.collect_kwargs)
            logging.debug(f'GenericExecutor {self.id} -- collected sequences for {worker.id}')
        if self.collect_seqs and self.compose_summary:
            worker.compose_summary(**self.summary_kwargs)
            logging.debug(f'GenericExecutor {self.id} -- composed summary for {worker.id}')
        if self.cleanup:
            worker.cleanup(**self.cleanup_kwargs)
            logging.debug(f'GenericExecutor {self.id} -- cleaned up {worker.id}')
        if self.store_bias and isinstance(worker, ADAPT):
            worker.store_bias()
            logging.debug(f'GenericExecutor {self.id} -- stored bias of {worker.id}')
        if self.callbacks is not None:
            for callback in self.callbacks:
                worker = callback(worker)
            logging.debug(f'GenericExecutor {self.id} -- finished applying callbacks to {worker.id}')
        logging.info(f'GenericExecutor {self.id} -- finished executing worker {worker.id}')
        return worker


if __name__ == '__main__':
    raise RuntimeError
