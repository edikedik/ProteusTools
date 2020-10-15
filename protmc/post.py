import operator as op
import typing as t
from collections import namedtuple
from functools import partial
from itertools import islice, takewhile, starmap, count

import pandas as pd


def analyze(seq: str, rich: str, steps: int, active: t.Optional[t.List[int]] = None):
    Entry = namedtuple('entry', ['seq_entry', 'rich_entry'])
    Parsed_entry = namedtuple('parsed_entry', ['seq', 'counts', 'energy'])

    def take(n, iterable):
        return list(islice(iterable, n))

    def parse_entry(entry: Entry) -> Parsed_entry:
        counts, energy = entry.seq_entry.split()[1:3]
        counts, energy = int(counts), float(energy)
        seq_rich = "".join(entry.rich_entry.split()[1:])
        return Parsed_entry(seq_rich, counts, energy)

    def subset_positions(entry: Parsed_entry, pos: t.List[int]) -> Parsed_entry:
        return Parsed_entry(
            "".join([entry.seq[i] for i in pos]), entry.counts, entry.energy)

    with open(rich) as r, open(seq) as s:
        take(1, r), take(1, s)
        positions = [int(x) for x in r.readline().rstrip().split()[1:]]
        rs = map(op.itemgetter(0), takewhile(bool, (take(3, r) for _ in count())))
        entries = map(parse_entry, starmap(Entry, zip(s, rs)))
        if active:
            active = [positions.index(a) for a in active]
            entries = map(partial(subset_positions, pos=active), entries)
        df = pd.DataFrame(entries)
        df = df.groupby('seq', as_index=False).agg(
            total_count=pd.NamedAgg(column='counts', aggfunc='count'),
            avg_energy=pd.NamedAgg(column='energy', aggfunc='mean'),
            min_energy=pd.NamedAgg(column='energy', aggfunc='min'),
            max_energy=pd.NamedAgg(column='energy', aggfunc='max'))
        df['seq_prob'] = df['total_count'] / steps * 100

    return df.reset_index(drop=True)
