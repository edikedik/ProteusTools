import subprocess as sp
import typing as t
from pathlib import Path

from protmc.config import ProtMCconfig


class Runner:
    def __init__(self, run_dir: str, exe_path: str, config: ProtMCconfig, log_path: t.Optional[str] = None):
        self.run_dir = Path(run_dir)
        if self.run_dir.exists() and not self.run_dir.is_dir():
            raise ValueError(f'Invalid path {run_dir}')
        self.run_dir.mkdir(exist_ok=True, parents=True)
        self._exe_path = Path(exe_path)
        if not (self._exe_path.exists() and self._exe_path.is_file()):
            raise ValueError(f'Invalid protMC executable path {exe_path}')
        self._config = config
        self._mode = config.mode.field_values[0]
        self._config_path = f'{run_dir}/{self._mode}.conf'
        self._config.dump(self._config_path)
        self._log_path = log_path or f'{run_dir}/{self._mode}.log'
        self._run_command = f'{self._exe_path} < {self._config_path} > {self._log_path}'

    def run(self):
        try:
            sp.run(self._run_command, shell=True, check=True)
        except sp.CalledProcessError as e:
            output = sp.run(self._run_command, shell=True, check=False, capture_output=True, text=True)
            raise RuntimeError(f'Failed to run protMC with an error {e} and output {output}')
