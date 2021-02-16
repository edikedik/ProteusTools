import typing as t

import pandas as pd

from protmc.common.utils import split_before


class Bias:
    def __init__(self):
        self.bias = None

    def read_adapt_output(self, path: str, overwrite: bool = True) -> pd.DataFrame:
        def wrap_chunk(chunk: t.List[str]):
            step = int(chunk[0].split()[-1])
            body = ((int(y[0]), y[1], int(y[2]), y[3], float(y[4])) for y in (x.split() for x in chunk[1:]))
            df = pd.DataFrame(body, columns=['pos1', 'aa1', 'pos2', 'aa2', 'bias'])
            df['var'] = ['-'.join([x.pos1, x.aa1, x.pos2, x.aa2]) for x in df.itertuples()]
            df['step'] = step
            return df[['step', 'var', 'bias']]

        with open(path) as f:
            lines = filter(lambda x: x != '\n', f)
            chunks = split_before(lines, lambda x: x.startswith('#'))
            bias_df = pd.concat([wrap_chunk(c) for c in chunks])
        if overwrite:
            self.bias = bias_df

        return bias_df

    def read_bias_df(self, path: str, overwrite: bool = True) -> pd.DataFrame:
        bias_df = pd.read_csv(path, sep='\t')
        if overwrite:
            self.bias = bias_df
        return bias_df

    def dump(self, path: str, step: int, last_step: bool = True):
        pass

    def center_at_reference(self):
        pass

    def combine(self, other: 'Bias'):
        pass
