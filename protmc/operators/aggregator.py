import typing as t
from glob import glob
from warnings import warn

import pandas as pd
from tqdm import tqdm

from protmc.basic.stability import stability
from protmc.common.base import AbstractAggregator, Id, NoReferenceError
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
            # TODO: this way of getting "MC root" dir is quite dumb
            w1_base, w2_base = map(lambda w: '/'.join(
                w.params.working_dir.split('/')[:-1]), workers)
            if w1_base == w2_base:
                base = w1_base
            else:
                base = './'
            df.to_csv(f'{base}/{self.dump_name}', sep='\t', index=False)
        return df


def aggregate_from_base(
        base_dir: str, ref_seq: str, ref_pos: t.Sequence[int],
        pos_parser: t.Callable[[str], t.List[str]] = lambda x: x.split('-'),
        temperature: float = 0.6, count_threshold: int = 100,
        holo: str = 'holo', apo: str = 'apo', mc: str = 'MC',
        bias_name: str = 'ADAPT.inp.dat', seqs_name: str = 'RESULTS.tsv'):

    ref_pos_str = list(map(str, ref_pos))
    ref_pos_mapping = {p: i for i, p in enumerate(ref_pos_str)}

    def affinity_df(pair_base):
        pop_apo = pd.read_csv(f'{pair_base}/{apo}/{mc}/{seqs_name}', sep='\t')
        pop_holo = pd.read_csv(f'{pair_base}/{holo}/{mc}/{seqs_name}', sep='\t')
        bias_apo = f'{pair_base}/{apo}/{mc}/{bias_name}'
        bias_holo = f'{pair_base}/{holo}/{mc}/{bias_name}'
        stability_apo = stability(pop_apo, bias_apo, ref_seq, temperature, count_threshold, ref_pos_str)
        stability_holo = stability(pop_holo, bias_holo, ref_seq, temperature, count_threshold, ref_pos_str)
        df = pd.merge(stability_apo, stability_holo, on='seq', how='outer', suffixes=['_apo', '_holo'])
        df['affinity'] = df['stability_holo'] - df['stability_apo']
        positions = pos_parser(pair_base)
        df['seq_subset'] = df['seq'].apply(lambda s: ''.join(s[ref_pos_mapping[p]] for p in positions))
        df['pos'] = '-'.join(positions)
        return df

    paths = tqdm(glob(f'{base_dir}/*'), desc='Aggregating workers')
    dfs = []
    for p in paths:
        try:
            dfs.append(affinity_df(p))
        except (NoReferenceError, ValueError, KeyError) as e:
            warn(f'Could not aggregate worker {p} due to {e}')

    return pd.concat(dfs)


if __name__ == '__main__':
    raise RuntimeError
