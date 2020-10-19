import re
import typing as t
from collections.abc import MutableMapping
from copy import deepcopy
from itertools import chain

_Value = t.Union[str, float, int]
_Values = t.Union[_Value, t.List[_Value], None]


class ProtMCfield:
    """
    A Field of the protMC config file.
    """

    def __init__(
            self, field_name: str, field_values: _Values,
            comment: t.Optional[str] = None, default_value: _Values = None):
        self.field_name = field_name
        self.is_default = field_values == default_value or field_values is default_value
        self.is_empty = field_values is None
        self.field_values = field_values if isinstance(field_values, t.List) else [field_values]
        self.default_value = default_value
        self.comment = comment

    def __str__(self):
        return self._format()

    def __repr__(self):
        return f'Field: {self.field_name} | Values: {self.field_values}'

    def _format(self):
        head = f'# {self.comment}\n<{self.field_name}>' if self.comment else f'<{self.field_name}>'
        body = "\n".join(map(str, self.field_values))
        tail = f'</{self.field_name}>'
        return f'{head}\n{body}\n{tail}'


class ProtMCfieldGroup(MutableMapping):
    """
    A meaningful group of fields of the protMC config file.
    """

    def __init__(self, group_name: str, group_fields: t.Iterable[ProtMCfield], comment: t.Optional[str] = None):
        self._store = dict(zip((f.field_name for f in group_fields), group_fields))
        self.group_name = group_name
        self.comment = comment

    def __getitem__(self, item):
        return self._store[item] if item in self._store else None

    def __setitem__(self, key: str, value: t.Union[_Values, ProtMCfield]):
        if isinstance(value, ProtMCfield):
            self._store[key] = value
        else:
            default = self._store[key].default_value if key in self._store else None
            comment = self._store[key].comment if key in self._store else None
            self._store[key] = ProtMCfield(
                field_name=key, field_values=value, default_value=default, comment=comment)

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __str__(self):
        return self._format()

    def __repr__(self) -> str:
        return f'Group: {self.group_name} | Num fields: {len(self._store)}'

    def _format(self):
        head = f'# GROUP START: {self.group_name.upper()}\n# {self.comment}'
        body = "\n".join([f'{field}\n' for field in self.values()])
        tail = f'# GROUP END: {self.group_name.upper()}'
        return f'{head}\n\n{body}\n{tail}'


class ProtMCconfig(MutableMapping):
    """
    Container for the proteus config
    """

    def __init__(self, mode: t.Union[str, ProtMCfield, None], groups: t.Optional[t.List[ProtMCfieldGroup]]):
        self.mode = ProtMCfield('Mode', mode, 'The mode of the Proteus run') if isinstance(mode, str) else mode
        self._store = dict(zip((g.group_name for g in groups), groups))

    @property
    def fields(self):
        field_gen = ((f for f in g.values()) for g in self._store.values())
        yield from chain.from_iterable(field_gen)

    def __getitem__(self, item):
        return self._store[item] if item in self._store else None

    def __setitem__(self, key: str, value: ProtMCfieldGroup):
        if not isinstance(value, ProtMCfieldGroup):
            raise ValueError('Attempting to set and item that is not a field group')
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __str__(self) -> str:
        return self._format()

    def __repr__(self) -> str:
        return f'{list(self._store)}'

    def _format(self) -> str:
        head = f'# {"=" * 30} PROTEUS CONFIG FILE {"=" * 30} \n\n{self.mode}\n\n'
        body = "\n".join(f'{g}\n\n' for g in self._store.values())
        return f'{head}\n{body}'

    def copy(self):
        return deepcopy(self)

    def get_field(self, field_name: str):
        for f in self.fields:
            if f.field_name == field_name:
                return f
        return None

    def get_field_value(self, field_name: str):
        f = self.get_field(field_name)
        if f is None:
            return None
        if len(f.field_values) == 1:
            return f.field_values[0]
        return f.field_values

    def change_field(self, field_name: str, field_values: _Values):
        for g_name, group in self.items():
            for f_name, f in group.items():
                if f_name == field_name:
                    self._store[g_name][f_name] = field_values

    def rm_field(self, field_name: str):
        for g_name, group in self.items():
            try:
                del self._store[g_name][field_name]
            except KeyError:
                pass

    def rm_empty_fields(self):
        for f in [f.field_name for f in self.fields if f.is_empty]:
            self.rm_field(f)

    def rm_default_fields(self):
        for f in [f.field_name for f in self.fields if f.is_default]:
            self.rm_field(f)

    def dump(self, path: str, rm_empty: bool = True) -> None:
        """
        Dump a config into a file.
        """
        if not len(self):
            raise ValueError('No groups in the config')
        if rm_empty:
            self.rm_empty_fields()
        with open(path, 'w') as f:
            print(self._format(), file=f)


def parse_field(field_name: str, config: str) -> t.Optional[ProtMCfield]:
    """
    Capture a field from the proteus config
    :param field_name: the name of the field to capture
    :param config: a string with a config chunk
    :return: None if captured nothing else `ProtMCfield` object
    """
    pattern = re.compile(f'(#(.*?)\n)?<{field_name}>\n((.|\n)*)\n<\/{field_name}>')
    try:
        capture = re.findall(pattern, config)[0]
    except IndexError:
        return None
    return ProtMCfield(
        field_name=field_name,
        field_values=capture[2].split('\n'),
        comment=capture[1].strip() if capture[1] else None
    )


def parse_fields(config: str) -> t.Optional[t.List[ProtMCfield]]:
    """
    Capture all fields available in the `config`
    :param config: a string with a config chunk
    :return:
    """

    def wrap_match(match: t.Tuple) -> ProtMCfield:
        return ProtMCfield(
            field_name=match[2],
            field_values=match[3].split('\n'),
            comment=match[1] if match[1] else None
        )

    pattern = re.compile(r'(#(.*?)\n)?<(\w+)>\n((.|\n)*)\n<\/(\3)>')
    capture = re.findall(pattern, config)
    if not capture:
        return None
    return [wrap_match(m) for m in capture]


def parse_groups(config: str) -> t.Optional[t.List[ProtMCfieldGroup]]:
    """
    Capture all groups available in the `config`.
    A valid config group has a specific format depicted below.

    ```
    # GROUP START: {group_name}
    # {group_comment}

    {field1}
    {field2}
    ...

    # GROUP END: {group_name}
    ```

    :param config: a string with a config chunk
    :return: None if no groups were found else a list of `ProtMCfieldGroup`s
    """

    def wrap_match(match: t.Tuple) -> ProtMCfieldGroup:
        return ProtMCfieldGroup(
            group_name=match[0],
            group_fields=parse_fields(match[2]),
            comment=match[1]
        )

    pattern = re.compile(r'# GROUP START: ([\w\s]+)\n# ([\w\s]+)\n([\s\S]*)# GROUP END: (\1)')
    capture = re.findall(pattern, config)
    if not capture:
        return None
    return [wrap_match(m) for m in capture]


def parse_config(config: str) -> ProtMCconfig:
    """
    Parses the whole config
    :param config: a protMC config file (opened externally)
    :return: `ProtMCconfig` object
    :raises: ValueError on a failure to parse a config
    """
    mode = parse_field('Mode', config)
    if not mode:
        raise ValueError('There is no <Mode> field in the provided config')
    body = parse_groups(config)
    if body is None:
        fields = parse_fields(config)
        if not fields:
            raise ValueError('No fields were found in the provided config')
        body = [ProtMCfieldGroup(
            group_name='GENERAL', group_fields=fields,
            comment='General group is created in the absence of other groups')]
    return ProtMCconfig(
        mode=mode,
        groups=body
    )


def rm_defaults(group: ProtMCfieldGroup) -> ProtMCfieldGroup:
    return ProtMCfieldGroup(
        group_name=group.group_name,
        group_fields=[f for f in group.values() if f.is_default])


def load_default_config(mode: str):
    if mode not in ['MONTECARLO', 'ADAPT', 'POST']:
        raise ValueError(f'Unsupported mode {mode}')

    # ================ Adaptive mode fields and groups ================================================================
    adapt_parameters = (
        ProtMCfield('Adapt_Space', None, 'Defines which positions are flattened in Adapt mode'),
        ProtMCfield('Adapt_Mono_Period', 5000, 'The frequency for updating the single-position bias terms'),
        ProtMCfield('Adapt_Pair_Period', 5000, 'The frequency for updating the paired-position bias terms'),
        ProtMCfield('Adapt_Mono_Speed', 50, 'E_0 term to calculate bias increment (single terms)'),
        ProtMCfield('Adapt_Pair_Speed', 50, 'E_0 term to calculate bias increment (paired terms)'),
        ProtMCfield('Adapt_Mono_Height', 0.2, 'h term to calculate bias increment (single terms)'),
        ProtMCfield('Adapt_Pair_Height', 0.2, 'h term to calculate bias increment (paired terms)'),
        ProtMCfield('Adapt_Mono_Offset', 0, 'The step number where the first period begins (single terms)'),
        ProtMCfield('Adapt_Pair_Offset', 0, 'The step number where the first period begins (paired terms)'),
    )
    adapt_io = (
        ProtMCfield('Adapt_Output_File', 'adapt_out.dat', 'File to store bias values at the end of each period'),
        ProtMCfield('Adapt_Output_Period', None, 'A period for outputting bias values to a file of choice'),
    )
    adapt_parameters_g = ProtMCfieldGroup(
        group_name='ADAPT_PARAMS',
        group_fields=adapt_parameters,
        comment='These parameters are specific to adapt mode and control the process of the bias development'
    )
    adapt_io_g = ProtMCfieldGroup(
        group_name='ADAPT_IO',
        group_fields=adapt_io,
        comment='Parameters controlling IO specific to the ADAPT mode'
    )
    # ================ MC mode fields and groups ======================================================================
    mc_general = (
        ProtMCfield('Group_Definition', None, '', None),
        ProtMCfield('Label', None, '', None),
        ProtMCfield('Optimization_Configuration', None, '', None),
        ProtMCfield('Position_Weights', None, '', None),
        ProtMCfield('Random_Generator', 'mt19937', 'RNG; must be available within GSL', 'mt19937'),
        ProtMCfield('Ref_Ener', None, 'Reference (unfolded state) energies for all or selected positions', None),
        ProtMCfield('Rseed_Definition', None, 'Seed for the RNG', None),
        ProtMCfield('Space_Constraints', None, 'Constraints of the sequence-structure space', None),
        ProtMCfield('Step_Definition_Proba', None, 'The detailed set of move probabilities', None),
        ProtMCfield('Swap_Period', None, 'A swap frequency between replicas', None),
        ProtMCfield('Temperature', 0.65, 'kT in kcal/mol (one value per replica in case of REMC)', 0.65),
        ProtMCfield('Trajectory_Length', 1000000, 'Number of sampling steps', 1000000),
        ProtMCfield('Trajectory_Number', 1, 'A number of MC trajectories to run in parallel', 1),
        ProtMCfield('Replica_Number', 1, 'A number of replicas for REMC', 1),
        ProtMCfield('Weight_Exchange_File', None, 'A set of probabilities for backbone exchange moves', None)
    )
    mc_tweaks = (
        ProtMCfield('Dielectric_Parameter', 1.0, 'A value dividing the electrostatic energy term', 1.0),
        ProtMCfield('GB_BMAX', 10.0, 'A threshold for GB solvation radii', 1.0),
        ProtMCfield('GB_Method', 'False', 'Whether to use FDB GB method', 'False'),
        ProtMCfield('GB_Neighbor_Threshold', 0.0, 'Distance threshold to take into account for solvation radii', 0.0),
        ProtMCfield('Protein_Dielectric', 4.0, 'Dielectric constant values (FDB method)', 4.0),
        ProtMCfield('Reset_Energies', 100, 'Frequency of recomputing energies from scratch', 100),
        ProtMCfield('Solv_Neighbor_Threshold', 0.0, 'An energy threshold for neighbors (FDB method)', 0.0),
        ProtMCfield('Surf_Energy_Factor', 1.0, 'A factor multiplying the surface energy term', 1.0),
    )
    mc_io = (
        ProtMCfield('Bias_Input_File', None, 'Path to a file with existing bias potential', None),
        ProtMCfield('Energy_Directory', '../matrix', 'A path to a directory holding energy matrix files', None),
        ProtMCfield('Print_Threshold', None, '', None),
        ProtMCfield('Print_BSolv', None, '', None),
        ProtMCfield('Seq_Input', None, 'Existing state to continue the simulation from', None),
        ProtMCfield('Seq_Input_File', None, 'Path to existing state (.seq file) to continue the simulation from', None),
        ProtMCfield('Seq_Output_File', 'output.seq', 'The file to output a `trajectory` of sampled sequences',
                    'output.seq'),
        ProtMCfield('Ener_Output_File', 'output.ener', 'A path to a file to dump energy values', 'output.ener'),
    )
    mc_general_g = ProtMCfieldGroup(
        group_name='MC_PARAMS',
        group_fields=mc_general,
        comment='General parameters controlling the course of the MC simulation'
    )
    mc_tweaks_g = ProtMCfieldGroup(
        group_name='MC_TWEAKS',
        group_fields=mc_tweaks,
        comment='Terms controlling energy function behavior; careful attitude is implied'
    )
    mc_io_g = ProtMCfieldGroup(
        group_name='MC_IO',
        group_fields=mc_io,
        comment='Parameters controlling IO of the MC simulation'
    )
    # =========== POST mode fields and groups =========================================================================
    post_fields = (
        ProtMCfield('Energy_Directory', '../matrix', 'A path to a directory holding energy matrix files', None),
        ProtMCfield('Seq_Input_File', None, 'Path to existing state (.seq file) to continue the simulation from'),
        ProtMCfield('Fasta_File', 'output.rich', 'Path to a .rich file with fasta-like format'),
    )
    post_fields_g = ProtMCfieldGroup(
        group_name='POST_PARAMS',
        group_fields=post_fields,
        comment='Specific POST mode parameters'
    )
    # =========== Configs  ============================================================================================
    configs = {
        'ADAPT': ProtMCconfig(
            mode='ADAPT',
            groups=[adapt_parameters_g, adapt_io_g, mc_general_g, mc_io_g, mc_tweaks_g]
        ),
        'MONTECARLO': ProtMCconfig(
            mode='MONTECARLO',
            groups=[mc_general_g, mc_io_g, mc_tweaks_g]
        ),
        'POST': ProtMCconfig(
            mode='POSTPROCESS',
            groups=[post_fields_g]
        )
    }

    return configs[mode]


if __name__ == '__main__':
    raise RuntimeError
