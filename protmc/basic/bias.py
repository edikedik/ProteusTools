import typing as t
from io import TextIOWrapper

import pandas as pd

from protmc.common.utils import split_before


class Bias:
    """
    Encapsulates common operations with bias outputted by protMC in ADAPT mode.
    The core data structure available via `bias` attribute is a pandas `DataFrame`,
    comprised of three columns:
    (1) step -- a step of the output,
    (2) var -- a variable in the format "Pos1-AA1-Pos2-AA2",
    (3) bias -- a value of the bias.
    """
    def __init__(self, bias: t.Optional[pd.DataFrame] = None):
        """
        :param bias: Optional `DataFrame` with 'step', 'var' and 'bias' columns.
        """
        if bias is not None:
            if set(bias.columns) != {'step', 'var', 'bias'}:
                raise ValueError(f'Expected columns ["step", "var", "bias"], got {bias.columns}')
        self.bias: t.Optional[pd.DataFrame] = bias

    def __len__(self):
        return 0 if self.bias is None else len(self.bias)

    def copy(self):
        return Bias(self.bias.copy() if self.bias is not None else None)

    def read_adapt_output(self, path: str, overwrite: bool = True) -> 'Bias':
        """
        Parse ADAPT output (typically named "ADAPT.dat", with output step specified via "# STEP ...")
        into a `DataFrame`.
        :param path: Valid path to a ".dat" file.
        :param overwrite: Overwrite current `bias` attribute.
        :return: New `Bias` instance initialized by the parsed `DataFrame`.
        """
        def wrap_chunk(chunk: t.List[str]):
            step = int(chunk[0].split()[-1])
            body = ((int(y[0]), y[1], int(y[2]), y[3], float(y[4])) for y in (x.split() for x in chunk[1:]))
            df = pd.DataFrame(body, columns=['pos1', 'aa1', 'pos2', 'aa2', 'bias'])
            df['var'] = ['-'.join(map(str, x[1:-1])) for x in df.itertuples()]
            df['step'] = step
            return df[['step', 'var', 'bias']]

        with open(path) as f:
            lines = filter(lambda x: x != '\n', f)
            chunks = split_before(lines, lambda x: x.startswith('#'))
            bias_df = pd.concat([wrap_chunk(c) for c in chunks]).sort_values(['step', 'var'])

        if overwrite:
            self.bias = bias_df

        return Bias(bias_df)

    def read_bias_df(self, path: t.Union[str, TextIOWrapper], overwrite: bool = True) -> 'Bias':
        """
        Read "bias" stored in "tsv" format.
        """
        bias_df = pd.read_csv(path, sep='\t')
        if set(bias_df.columns) != {'step', 'var', 'bias'}:
            raise ValueError(f'Expected tsv file with columns `step`, `var`, `bias`. Got {bias_df.columns}')
        if overwrite:
            self.bias = bias_df
        return Bias(bias_df)

    def dump(self, path: str, last_step: bool = True, all_steps: bool = False,
             step: t.Optional[int] = None, tsv: bool = False) -> None:
        """
        Dump current bias.
        :param path: Path to save bias.
        :param last_step: Flag indicating to save only last step.
        :param all_steps: Flag indicating to save all steps.
        :param step: Specific step to save.
        :param tsv: Flag indicating whether to save as table or in ".dat" format.
        """
        if last_step:
            step = max(self.bias['step'])
        if all_steps or step is None:
            df = self.bias
        else:
            df = self.bias[self.bias['step'] == step]
        if tsv:
            df.to_csv(path, sep='\t', index=False)
        else:
            with open(path, 'w') as f:
                for s, g in df.groupby('step'):
                    print(f'# STEP {s}', file=f)
                    for _, v, b in g[['var', 'bias']].itertuples():
                        print(*v.split('-'), round(b, 4), file=f)

    def center_at_ref(self, ref_states: t.Mapping[str, str], overwrite: bool = True) -> 'Bias':
        """
        Center current bias values at reference sequence.
        Specifically, find reference sequence bias and subtract it from other bias values.
        :param ref_states: A mapping from positions to reference amino acid type,
        (e.g., {"1" -> "VAL", "2" -> "ALA", ...})
        :param overwrite: Overwrite current `bias` attribute with the results.
        :return: New `Bias` instance initialized by the results.
        """
        df = self.bias.copy()
        var_split = df['var'].apply(lambda x: x.split('-'))
        df['p1'] = [x[0] for x in var_split]
        df['p2'] = [x[2] for x in var_split]

        def center_state(group):
            p1, p2 = group['p1'].iloc[0], group['p2'].iloc[0]
            a1, a2 = ref_states[p1], ref_states[p2]
            v = f'{p1}-{a1}-{p2}-{a2}'
            offset = float(group.loc[group['var'] == v, 'bias'])
            group['bias'] -= offset
            return group

        df = df.groupby(['step', 'p1', 'p2']).apply(center_state)
        if overwrite:
            self.bias = df
        return Bias(df)

    def squeeze(self, step: t.Optional[int] = None, overwrite: bool = True) -> 'Bias':
        """
        Filter currently stored bias to a specific step.
        :param step: If not provided, the maximum step will be used.
        :param overwrite: Overwrite currently stored `bias` with the results.
        :return:
        """
        if step is None:
            step = max(self.bias['step'])
        df = self.bias[self.bias['step'] == step]
        if overwrite:
            self.bias = df
        return Bias(df)

    def update(self, other: 'Bias', overwrite: bool = True) -> 'Bias':
        """
        Update currently stored bias with the new one.
        Assumes that the provided `other` is the result
        of bias development continuation for the same system.

        The update takes effect only if the smallest "step" in `other`
        is larger than the biggest "step" in the current bias, which
        normally happens when the "ADAPT" run is simply restarted,
        thereby resetting the step counter.
        If this condition is not met, the function will simply concatenate
        `other` with the current bias.

        The update will take the latest step in the current bias.
        The "step" and "bias" values of this state will serve as a
        starting point for "step" and "bias" values of `other`.
        """
        if other.bias is None:
            return self

        bias_upd = other.bias.copy()
        first_step_in_upd = min(bias_upd['step'])
        last_step_in_bias = max(self.bias['step'])

        # Here we assume that the update is a consequence of restarting ADAPT
        # and thereby resetting step counter. Hence, we add last written step
        # of the current bias to the first step of the update
        if first_step_in_upd <= last_step_in_bias:
            bias_upd['step'] += last_step_in_bias
        # Otherwise -- no update; we simply concatenate two biases
        else:
            return self.concat(other, overwrite)

        last_bias_idx = self.bias['step'] == last_step_in_bias
        bias_upd = pd.merge(bias_upd, self.bias[last_bias_idx], on='var', how='inner',
                            suffixes=['_upd', '_old'])
        bias_upd['bias'] = bias_upd['bias_upd'] + bias_upd['bias_old']
        bias_upd = bias_upd.drop(
            columns=['bias_upd', 'bias_old', 'step_old']
        ).rename(
            columns={'step_upd': 'step'})
        bias_upd['step'] = bias_upd['step'].astype(int)
        if bias_upd.isna().any().any():
            raise ValueError('Problem with updating the bias -- some values are empty. '
                             'Inspect both `self` and `other`')
        df = pd.concat([self.bias, bias_upd]).reset_index(drop=True).sort_values(['step', 'var'])
        if overwrite:
            self.bias = df
        return Bias(df)

    def concat(self, other: 'Bias', overwrite: bool = True) -> 'Bias':
        """
        Concatenate currently stored bias with `other`.
        It is assumed that current bias and other are parts of the same run,
        where `other` encapsulates the later bias development stage.
        """
        first_step_in_upd = min(other.bias['step'])
        last_step_in_bias = max(self.bias['step'])
        if first_step_in_upd < last_step_in_bias:
            raise ValueError(
                f'The first step {first_step_in_upd} is smaller '
                f'than the last step {last_step_in_bias} of the bias in memory. '
                f'This implies that the `other` is not a continuation.')
        df = pd.concat([self.bias, other.bias]).reset_index(drop=True)
        if overwrite:
            self.bias = df
        return Bias(df)


if __name__ == '__main__':
    raise RuntimeError
