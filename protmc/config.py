import re
import typing as t
from copy import deepcopy
from pathlib import Path

_Value = t.Union[str, float, int]
_Values = t.Union[_Value, t.List[_Value], None]


# TODO: create a flag controlling whether the field should be formatted with empty or default values

class ProtMCfield:
    """
    A Field of the protMC config file.
    """

    def __init__(
            self, field_name: str, field_values: _Values, comment: t.Optional[str] = None,
            default_value: _Values = None, format_default: bool = False):
        self.field_name = field_name
        self.is_default = field_values is None or field_values == default_value
        self.field_values = field_values if isinstance(field_values, t.List) else [field_values]
        self.default_value = default_value
        self.format_default = format_default
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


class ProtMCfieldGroup:
    """
    A meaningful group of fields of the protMC config file.
    """

    def __init__(self, group_name: str, group_fields: t.Iterable[ProtMCfield], comment: t.Optional[str] = None):
        self.group_name = group_name
        self.group_fields = group_fields if isinstance(group_fields, t.List) else list(group_fields)
        self.comment = comment

    def __str__(self):
        return self._format()

    def __repr__(self):
        return f'Group: {self.group_name} | Num fields: {len(self.group_fields)}'

    def _format(self):
        head = f'# GROUP START: {self.group_name.upper()}\n# {self.comment}'
        body = "\n".join([f'{field}\n' for field in self.group_fields])
        tail = f'# GROUP END: {self.group_name.upper()}'
        return f'{head}\n\n{body}\n{tail}'


class ProtMCconfig:
    """
    Container for the proteus config
    """

    def __init__(self, mode: t.Union[str, ProtMCfield, None], groups: t.Optional[t.List[ProtMCfieldGroup]]):
        self.mode = ProtMCfield('Mode', mode, 'The mode of the Proteus run') if isinstance(mode, str) else mode
        self.groups = groups

    def read(self, path: str) -> None:
        """
        Read a proteus config file and substitute its fields into this `ProtMCconfig` instance.
        :param path: valid path to a config file.
        """
        if not Path(path).exists():
            raise ValueError(f'Invalid path {path}')
        with open(path) as f:
            new = parse_config(f.read())
        self.mode = new.mode
        self.groups = new.groups

    def dump(self, path: str, dump_default: bool = False) -> None:
        """
        Dump a config into a file.
        """
        if not self.groups:
            raise ValueError('No groups in the config')
        groups = deepcopy(self.groups)
        if not dump_default:
            for g in groups:
                g.group_fields = [f for f in g.group_fields if not f.is_default]
        config = ProtMCconfig(mode=self.mode, groups=groups)
        with open(path, 'w') as f:
            print(config._format(), file=f)

    def __str__(self) -> str:
        return self._format()

    def __repr__(self) -> str:
        return f'ProtMC config with {0 if not self.groups else len(self.groups)} group(s)'

    def _format(self) -> str:
        head = f'# {"=" * 30} PROTEUS CONFIG FILE {"=" * 30} \n\n{self.mode}\n\n'
        body = "\n".join(f'{g}\n\n' for g in self.groups)
        return f'{head}\n{body}'


def parse_field(field_name: str, config: str) -> t.Optional[ProtMCfield]:
    """
    Capture a field from the proteus config
    :param field_name: the name of the field to capture
    :param config: a string with a config chunk
    :return: None if captured nothing else `ProtMCfield` object
    """
    pattern = re.compile(f'(#([\w\s]+)\n)?<{field_name}>\n((.|\n)*)\n<\/{field_name}>')
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

    pattern = re.compile(r'(#([\w\s]+)\n)?<(\w+)>\n((.|\n)*)\n<\/(\3)>')
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
    if not body:
        fields = parse_fields(config)
        if not fields:
            raise ValueError('No fields were found in the provided config')
        body = ProtMCfieldGroup(
            group_name='GENERAL', group_fields=fields,
            comment='General group is created in the absence of other groups')
    return ProtMCconfig(
        mode=mode,
        groups=body
    )


def load_default_config(mode: str):
    if mode not in ['MC', 'ADAPT', 'POST']:
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
        group_name='ADAPT mode parameters',
        group_fields=adapt_parameters,
        comment='These parameters are specific to adapt mode and control the process of the bias development'
    )
    adapt_io_g = ProtMCfieldGroup(
        group_name='ADAPT mode IO parameters',
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
        ProtMCfield('Energy_Output_File', 'output.ener', 'A path to a file to dump energy values', 'output.ener'),
    )
    mc_general_g = ProtMCfieldGroup(
        group_name='MC General params',
        group_fields=mc_general,
        comment='General parameters controlling the course of the MC simulation'
    )
    mc_tweaks_g = ProtMCfieldGroup(
        group_name='MC energy function tweaks',
        group_fields=mc_tweaks,
        comment='Terms controlling energy function behavior; careful attitude is implied'
    )
    mc_io_g = ProtMCfieldGroup(
        group_name='MC IO params',
        group_fields=mc_io,
        comment='Parameters controlling IO of the MC simulation'
    )
    # =========== POST mode fields and groups =========================================================================
    post_fields = (
        ProtMCfield('Seq_Input_File', None, 'Path to existing state (.seq file) to continue the simulation from'),
        ProtMCfield('Fasta_File', 'output.rich', 'Path to a .rich file with fasta-like format'),
    )
    post_fields_g = ProtMCfieldGroup(
        group_name='POST config parameters',
        group_fields=post_fields,
        comment='Specific POST mode parameters'
    )
    # =========== Configs  ============================================================================================
    configs = {
        'ADAPT': ProtMCconfig(
            mode='ADAPT',
            groups=[adapt_parameters_g, adapt_io_g, mc_general_g, mc_io_g, mc_tweaks_g]
        ),
        'MC': ProtMCconfig(
            mode='MC',
            groups=[mc_general_g, mc_io_g, mc_tweaks_g]
        ),
        'POST': ProtMCconfig(
            mode='POST',
            groups=[post_fields_g]
        )
    }

    return configs[mode]


if __name__ == '__main__':
    raise RuntimeError
