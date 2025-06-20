import os
import json
import subprocess
import sys


class Challenge:
    def __init__(self, name: str):
        self.name = name
        self._config = None
        self._base_algorithm = None

    @property
    def config_path(self) -> str:
        return os.path.join('challenges', 'configs', f'{self.name}.json')

    @property
    def config(self) -> dict:
        if self._config is None:
            with open(self.config_path, 'r') as f:
                self._config = json.load(f)
        return self._config

    @property
    def base_algorithm_path(self) -> str:
        return os.path.join('challenges', 'base_algos', f'{self.name}_{self.config["base_algorithm"]}.py')

    @property
    def base_algorithm(self) -> str:
        if self._base_algorithm is None:
            with open(self.base_algorithm_path, 'r') as f:
                self._base_algorithm = f.read()
        return self._base_algorithm
    
    @property
    def challenge_path(self) -> str:
        return os.path.join('challenges', f'{self.name}.py')

    @property
    def parameters(self) -> dict:
        return self.config["parameters"]
        
    @property
    def seed(self) -> int:
        return self.config["seed"]
        
    @property
    def num_instances(self) -> int:
        return self.config["num_instances"]
    
    def evaluate_algorithm(
        self, 
        algorithm_path: str,
        evaluation_path: str,
        visualisation_path: str,
        num_visualisations: int = 0
    ):
        cmd = [
            sys.executable,
            self.challenge_path,
            algorithm_path,
            json.dumps(self.parameters),
            str(self.seed),
            str(self.num_instances),
            "--evaluation", evaluation_path,
            "--output", visualisation_path,
            "--visualisations", str(num_visualisations)
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        if result.returncode != 0:
            raise RuntimeError(f"Algorithm evaluation failed: {result.stderr}")