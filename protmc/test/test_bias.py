from io import StringIO
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import pytest

from protmc.basic import Bias


def random_variable():
    aas_pool = ['ALA', 'SER', 'THR', 'PHE', 'GLN']
    pos_pool = [1, 2, 3, 4, 5]
    a1, a2 = np.random.choice(aas_pool, 2)
    p1, p2 = np.random.choice(pos_pool, 2)
    return f'{p1}-{a1}-{p2}-{a2}'


@pytest.fixture()
def random_bias() -> Bias:
    num_steps = np.random.randint(1, 10)
    step_size = np.random.randint(3, 50)
    variables = [random_variable() for _ in range(step_size)]
    dfs = []
    for step in range(1, num_steps + 1):
        dfs.append(pd.DataFrame({
            'step': [step] * step_size,
            'var': variables,
            'bias': np.random.rand(step_size)
        }))
    return Bias(pd.concat(dfs))


@pytest.fixture()
def bias_dat(random_bias):
    df = random_bias.bias
    with StringIO() as f:
        for s in sorted(set(df['step'])):
            print(f'# STEP {s}', file=f)
            for _, v, b in df[['var', 'bias']].itertuples():
                print(*v.split('-'), round(b, 4), file=f)
        f.seek(0)
        return f.read()


@pytest.fixture()
def bias_tsv(random_bias):
    df = random_bias.bias
    with StringIO() as f:
        df.to_csv(path_or_buf=f, sep='\t', index=False)
        f.seek(0)
        return f.read()


@pytest.fixture()
def df_ini():
    return pd.DataFrame(
        {'step': [1, 1, 2, 2],
         'var': ['X', 'Y', 'X', 'Y'],
         'bias': [0, 0, 1, -1]}
    )


def test_bias_init(bias_dat, bias_tsv):
    # Empty init
    assert Bias().bias is None

    # Non-empty init
    df = pd.DataFrame(StringIO(bias_tsv))
    assert Bias(df).bias is not None

    # Init from dat
    with NamedTemporaryFile('w') as f:
        f.write(bias_dat)
        f.seek(0)
        bias = Bias()
        bias.read_adapt_output(f.name)
        assert bias.bias is not None

    # Init from tsv
    bias = Bias()
    bias.read_bias_df(StringIO(bias_tsv))
    assert bias.bias is not None


def test_random_bias_update(random_bias):
    b = random_bias.bias
    first = min(b['step'])
    last = max(b['step'])
    if first != last:
        b1 = Bias(b)
        # Update with the same bias
        upd = b1.update(random_bias, overwrite=False).bias
        # Nothing was overwritten
        assert (b1.bias == b).all().all()
        # The length is doubled
        assert len(upd) == len(b) * 2
        # The largest step of the update is twice the largest step of initial
        assert max(upd['step']) == max(b['step']) * 2
        # The steps of the update can be inferred via the last step of the original
        assert list(upd['step']) == list(b['step']) + list(b['step'] + last)
        # The bias value of the last step is twice this value of the original (1 + 1 = 2)
        assert list(upd['bias'])[-1] == list(b['bias'])[-1] * 2
    else:
        with pytest.raises(ValueError):
            Bias(random_bias.bias).update(random_bias)

    with pytest.raises(ValueError):
        b_ = Bias(b.iloc[:len(b) // 2])
        Bias(b).update(b_)


def test_manual_bias_update(df_ini):
    b_ini = Bias(df_ini)

    def test_upd_empty():
        upd = b_ini.update(Bias())
        assert upd is b_ini

    def test_common_case():
        df_upd = pd.DataFrame(
            {'step': [1, 1, 2, 2],
             'var': ['X', 'Y', 'X', 'Y'],
             'bias': [1, -1, 2, -2]}
        )
        upd = b_ini.update(Bias(df_upd), overwrite=False)

        assert len(upd) == 8
        assert (b_ini.bias == upd.bias[:4]).all().all()
        assert list(upd.bias['step']) == [1, 1, 2, 2, 3, 3, 4, 4]
        assert list(upd.bias['var']) == ['X', 'Y'] * 4
        assert list(upd.bias['bias']) == [0, 0, 1, -1, 2, -2, 3, -3]

    def test_less_upd_vars():
        df_upd = pd.DataFrame(
            {'step': [1, 2],
             'var': ['X', 'X'],
             'bias': [1, -1]}
        )
        upd = b_ini.update(Bias(df_upd), overwrite=False)

        assert len(upd) == 6
        assert list(upd.bias['step']) == [1, 1, 2, 2, 3, 4]
        assert list(upd.bias['var']) == ['X', 'Y', 'X', 'Y', 'X', 'X']
        assert list(upd.bias['bias']) == [0, 0, 1, -1, 2, 0]

    def test_more_upd_vars():
        # variable is ignored
        df_upd = pd.DataFrame(
            {'step': [1, 1, 1, 2, 2, 2],
             'var': ['X', 'Y', 'Z', 'X', 'Y', 'Z'],
             'bias': [1, -1, -2, 2, -2, 3]}
        )
        upd = b_ini.update(Bias(df_upd), overwrite=False)

        assert len(upd) == 8
        assert list(upd.bias['step']) == [1, 1, 2, 2, 3, 3, 4, 4]
        assert list(upd.bias['var']) == ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
        assert list(upd.bias['bias']) == [0, 0, 1, -1, 2, -2, 3, -3]

    def test_min_step_larger():
        df_upd = pd.DataFrame(
            {'step': [3, 3, 4, 4],
             'var': ['X', 'Y', 'X', 'Y'],
             'bias': [1, -1, 2, -2]}
        )
        upd = b_ini.update(Bias(df_upd), overwrite=False)

        assert len(upd) == 8
        assert list(upd.bias['step']) == [1, 1, 2, 2, 3, 3, 4, 4]
        assert list(upd.bias['bias']) == [0, 0, 1, -1, 1, -1, 2, -2]

    test_upd_empty()
    test_common_case()
    test_less_upd_vars()
    test_more_upd_vars()
    test_min_step_larger()


def test_centering():
    def test_one_pos():
        df_ini = pd.DataFrame(
            {'step': [1, 1, 2, 2],
             'var': ['1-X-1-X', '1-X-1-Y'] * 2,
             'bias': [1, 1, 2, 2]}
        )
        b_ini = Bias(df_ini)
        with pytest.raises(KeyError):
            b_ini.center_at_ref({'2': 'X'}, overwrite=False)
        upd = b_ini.center_at_ref({'1': 'X'}, overwrite=False)
        assert len(upd) == len(b_ini)
        assert list(upd.bias['step']) == [1, 1, 2, 2]
        assert list(upd.bias['bias']) == [0, 0, 0, 0]

    def test_two_pos():
        df_ini = pd.DataFrame(
            {'step': [1] * 3 + [2] * 3,
             'var': ['1-X-1-X', '1-X-2-Y', '1-X-2-Z'] * 2,
             'bias': [10, 2, 3, 20, 4, 6]}
        )
        b_ini = Bias(df_ini)
        upd = b_ini.center_at_ref({'1': 'X', '2': 'Y'}, overwrite=False)
        assert list(upd.bias['bias']) == [0, 0, 1, 0, 0, 2]

    test_one_pos()
    test_two_pos()
