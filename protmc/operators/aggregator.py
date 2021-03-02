import typing as t

import pandas as pd

from protmc.basic.stability import stability
from protmc.common.base import AbstractAggregator, Id
from protmc.operators.worker import MC


class AffinityAggregator(AbstractAggregator):
    def __init__(self, ref_seq: str, temperature: float, count_threshold: int,
                 positions: t.Iterable[str], id_: Id = None, dump: bool = True,
                 dump_name: str = 'stability.tsv'):
        super().__init__(id_)
        self.ref_seq = ref_seq
        self.temperature = temperature
        self.count_threshold = count_threshold
        self.positions: t.Tuple[str] = tuple(positions)
        self.dump = dump
        self.dump_name = dump_name

    def stability(self, worker: MC):
        bias_path = worker.params.config.get_field_value('Bias_Input_File')
        if bias_path is None:
            raise ValueError(f'AffinityAggregator {self.id} -- `Bias_Input_File` is empty for {worker.id}')
        if worker.seqs is None:
            raise ValueError(f'AffinityAggregator {self.id} -- no sequences for {worker.id}')
        return stability(
            worker.seqs, bias_path, self.ref_seq, self.temperature,
            self.count_threshold, self.positions)

    def aggregate(self, workers: t.Tuple[MC, MC], suffixes: t.Tuple[str] = ('_apo', '_holo')):
        st1, st2 = map(self.stability, workers)
        df = pd.merge(st1, st2, on='seq', suffixes=suffixes, how='outer')
        if self.dump:
            w1_base, w2_base = map(lambda w: '/'.join(
                w.params.working_dir.split('/')[:-2]), workers)
            if w1_base == w2_base:
                base = w1_base
            else:
                base = './'
            df.to_csv(f'{base}/{self.dump_name}', sep='\t', index=False)
        return df


if __name__ == '__main__':
    raise RuntimeError
