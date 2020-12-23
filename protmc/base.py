import typing as t

import pandas as pd

Summary = t.NamedTuple('Summary', [
    ('num_unique', int), ('num_unique_merged', int), ('coverage', float),
    ('seq_prob_mean', float), ('seq_prob_std', float), ('seq_prob_rss', float)])
Population_element = t.NamedTuple('Population_element', [('seq', str), ('count', int)])
AA_pair = t.NamedTuple('AA_pair', [('pos_i', str), ('pos_j', str), ('aa_i', str), ('aa_j', str)])
PairBias = t.NamedTuple('PairBias', [('aa_pair', AA_pair), ('bias', float)])
AffinityResult = t.NamedTuple('AffinityResults', [('seq', str), ('affinity', float)])
AffinityResults = t.NamedTuple('AffinityResults', [('run_dir', str), ('affinity', float)])
PipelineOutput = t.NamedTuple('PipelineOutput', [('seqs', pd.DataFrame), ('summary', pd.DataFrame)])
ParsedEntry = t.NamedTuple('ParsedEntry', [('seq', str), ('counts', int), ('energy', float)])

_AA_DICT = """ALA A ACT
CYS C ACT
THR T ACT
GLU E ED
GLH e ED
ASP D ED
ASH d ED
PHE F FW
TRP W FW
ILE I IVL
VAL V IVL
LEU L IVL
LYS K K
LYN k K
MET M M
ASN N NQ
GLN Q NQ
SER S S
ARG R R
TYR Y Y
TYD y Y
HID h H
HIE j H
HIP H H
PRO P PG
GLY G PG"""


class AminoAcidDict:
    def __init__(self, inp: str = _AA_DICT):
        self._aa_dict = self._parse_dict(inp)

    @staticmethod
    def _parse_dict(inp):
        inp_split = [x.split() for x in inp.split('\n')]
        return {
            **{line[0]: line[1] for line in inp_split},
            **{line[1]: line[0] for line in inp_split}}

    @property
    def aa_dict(self) -> t.Dict[str, str]:
        return self._aa_dict

    @property
    def proto_mapping(self) -> t.Dict[str, str]:
        return {'e': 'E', 'd': 'D', 'k': 'K', 'y': 'Y', 'j': 'H', 'h': 'H',
                'GLH': 'GLU', 'ASH': 'ASP', 'LYN': 'LYS', 'TYD': 'TYR', 'HID': 'HIP', 'HIE': 'HIP'}


class NoReferenceError(Exception):
    pass


if __name__ == '__main__':
    raise RuntimeError
