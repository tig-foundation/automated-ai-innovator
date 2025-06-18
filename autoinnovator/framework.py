from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Union
from copy import deepcopy
from .llm import LLM
from .challenge import Challenge
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import time
import os
import json
import traceback

def read_if_exists(path: str) -> Optional[Union[dict, str]]:
    if os.path.exists(path):
        with open(path, 'r') as f:
            if path.endswith('.json'):
                return json.load(f)
            else:
                return f.read()
    return None

@dataclass
class Candidate:
    id: int
    generation: int
    output_dir: str
    success: bool

    @property
    def error_path(self) -> str:
        return os.path.join(self.output_dir, 'error.txt')

    @property
    def error(self) -> Optional[str]:
        return read_if_exists(self.error_path)

    @property
    def prompt_path(self) -> str:
        return os.path.join(self.output_dir, 'prompt.json')

    @property
    def prompt(self) -> Optional[dict]:
        return read_if_exists(self.prompt_path)
    
    @property
    def response_path(self) -> str:
        return os.path.join(self.output_dir, 'response.json')

    @property
    def response(self) -> Optional[dict]:
        return read_if_exists(self.response_path)
    
    @property
    def evaluation_path(self) -> str:
        return os.path.join(self.output_dir, 'evaluation.json')

    @property
    def evaluation(self) -> Optional[Dict[str, Any]]:
        return read_if_exists(self.evaluation_path)
    
    @property
    def algorithm_path(self) -> str:
        return os.path.join(self.output_dir, 'algorithm.py')

    @property
    def algorithm(self) -> Optional[str]:
        return read_if_exists(self.algorithm_path)


@dataclass
class Context:
    experiment_dir: str
    candidates_per_generation: int
    num_generations: int
    curr_generation: int
    candidates: List[List[Candidate]]


class AutoInnovatorBase(ABC):
    def __init__(self, llm: LLM, challenge: Challenge):
        self.llm = llm
        self.challenge = challenge
    
    @abstractmethod
    def create_prompt_kwargs(self, candidate: Candidate, ctx: Context) -> dict:
        """
        Create the prompt arguments for LLM.send_prompt.
        This must include the prompt. Optionally can include system_prompt, temperature and any additional parameters.

        See LLM.send_prompt for details on the parameters.
        """
        raise NotImplementedError
    
    @abstractmethod
    def extract_algorithm_code(self, response: dict) -> str:
        """
        Extract the algorithm code from the LLM response.
        This should return a string containing the Python code for the algorithm.

        If the response does not contain valid code, it should raise an exception.
        """
        raise NotImplementedError

    def run_candidate(
        self, 
        candidate_id: int,
        ctx: Context,
        num_visualisations: int = 4
    ) -> Candidate:
        candidate = Candidate(
            id=candidate_id,
            generation=ctx.curr_generation,
            output_dir=os.path.join(ctx.experiment_dir, f'generation_{ctx.curr_generation:03}', f'candidate_{candidate_id:03}'),
            success=False
        )
        os.makedirs(candidate.output_dir, exist_ok=True)
        try:
            start = time()
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: create_prompt_kwargs (starting)")
            prompt_kwargs = self.create_prompt_kwargs(candidate, ctx)
            with open(candidate.prompt_path, 'w') as f:
                json.dump(prompt_kwargs, f)
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: create_prompt_kwargs (done, took {time() - start:.2f} seconds)")

            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: llm.send_prompt (starting)")
            response = self.llm.send_prompt(**prompt_kwargs)
            with open(candidate.response_path, 'w') as f:
                json.dump(response, f)
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: llm.send_prompt (done, took {time() - start:.2f} seconds)")

            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: extract_algorithm_code (starting)")
            algorithm_code = self.extract_algorithm_code(response)
            with open(candidate.algorithm_path, 'w') as f:
                f.write(algorithm_code)
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: extract_algorithm_code (done, took {time() - start:.2f} seconds)")

            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: evaluate_algorithm (starting)")
            self.challenge.evaluate_algorithm(
                candidate.algorithm_path, 
                candidate.evaluation_path,
                candidate.output_dir,
                num_visualisations
            )
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: evaluate_algorithm (done, took {time() - start:.2f} seconds)")
            candidate.success = True
        except Exception as e:
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: error {str(e)}")
            with open(candidate.error_path, 'w') as f:
                f.write(f"{traceback.format_exc()}\n{str(e)}")

        return candidate

    def run(
        self, 
        num_generations: int,
        candidates_per_generation: int,
        experiment_dir: str,
        num_visualisations_per_candidate: int = 4
    ):
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"Starting AutoInnovator run with {num_generations} generations and {candidates_per_generation} candidates per generation.")
        print(f"Experiment directory: {experiment_dir}")

        # Handle Generation 0: base_algorithm only
        print("Running generation 0 with base algorithm...")
        candidate = Candidate(
            id=0,
            generation=0,
            output_dir=os.path.join(experiment_dir, 'generation_000', 'candidate_000'),
            success=True
        )
        os.makedirs(candidate.output_dir, exist_ok=True)

        with open(candidate.algorithm_path, 'w') as f:
            f.write(self.challenge.base_algorithm)
        self.challenge.evaluate_algorithm(
            candidate.algorithm_path, 
            candidate.evaluation_path,
            candidate.output_dir,
            num_visualisations_per_candidate
        )

        # Run subsequent generations
        candidates = [[candidate]]  # Start with generation 0
        for gen in range(1, num_generations + 1):
            print(f"Running generation {gen}")
            ctx = Context(
                experiment_dir=experiment_dir,
                candidates_per_generation=candidates_per_generation,
                num_generations=num_generations,
                curr_generation=gen,
                candidates=candidates
            )

            with ThreadPoolExecutor(max_workers=candidates_per_generation) as executor:
                futures = [
                    executor.submit(self.run_candidate, candidate, ctx, num_visualisations_per_candidate) 
                    for candidate in range(candidates_per_generation)
                ]
                generation_results = [f.result() for f in futures]
                candidates.append(generation_results)