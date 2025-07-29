from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from copy import deepcopy
from .llm import LLM
from .challenge import Challenge
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import time
import os
import json
import traceback
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

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
    
    @property
    def visualisation_path(self) -> str:
        return os.path.join(self.output_dir, 'visualisation_{i:03}.png')


@dataclass
class Context:
    experiment_dir: str
    llm: LLM
    challenge: Challenge
    candidates_per_generation: int
    num_generations: int
    curr_generation: int
    candidates: List[List[Candidate]]

    @property
    def results_plot_path(self) -> str:
        return os.path.join(self.experiment_dir, 'results_plot.png')


class AutoInnovatorBase(ABC):    
    @abstractmethod
    def create_prompt_kwargs(self, candidate: Candidate, ctx: Context) -> dict:
        """
        Create the prompt arguments for LLM.send_prompt.
        This must include the prompt. Optionally can include system_prompt, temperature and any additional parameters.

        See LLM.send_prompt for details on the parameters.
        """
        raise NotImplementedError
    
    @abstractmethod
    def extract_algorithm_code(self, candidate: Candidate, ctx: Context) -> str:
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
            if ctx.curr_generation > 0:
                start = time()
                print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: create_prompt_kwargs (starting)")
                prompt_kwargs = self.create_prompt_kwargs(candidate, ctx)
                with open(candidate.prompt_path, 'w') as f:
                    json.dump(prompt_kwargs, f)
                print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: create_prompt_kwargs (done, took {time() - start:.2f} seconds)")

                start = time()
                print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: llm.send_prompt (starting)")
                response = ctx.llm.send_prompt(**prompt_kwargs)
                with open(candidate.response_path, 'w') as f:
                    json.dump(response, f)
                print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: llm.send_prompt (done, took {time() - start:.2f} seconds)")

                start = time()
                print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: extract_algorithm_code (starting)")
                algorithm_code = self.extract_algorithm_code(candidate, ctx)
                with open(candidate.algorithm_path, 'w') as f:
                    f.write(algorithm_code)
                print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: extract_algorithm_code (done, took {time() - start:.2f} seconds)")
            
            else:
                # For generation 0, we use the base algorithm directly
                algorithm_code = ctx.challenge.base_algorithm
                with open(candidate.algorithm_path, 'w') as f:
                    f.write(algorithm_code)

            start = time()
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: evaluate_algorithm (starting)")
            ctx.challenge.evaluate_algorithm(
                candidate.algorithm_path, 
                candidate.evaluation_path,
                candidate.visualisation_path,
                num_visualisations
            )
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: evaluate_algorithm (done, took {time() - start:.2f} seconds)")
            candidate.success = True
        except Exception as e:
            print(f"Gen {ctx.curr_generation}, Candidate {candidate_id}: error {str(e)}")
            with open(candidate.error_path, 'w') as f:
                f.write(f"{traceback.format_exc()}\n{str(e)}")

        return candidate

    def plot_results(self, ctx: Context):
        y = {
            k: [] 
            for k, v in ctx.candidates[0][0].evaluation.items()
            if isinstance(v, (int, float))
        }
        x = []
        for gen in range(ctx.curr_generation + 1):
            for candidate in ctx.candidates[gen]:
                if not candidate.success:
                    continue
                x.append(gen)
                evaluation = candidate.evaluation
                for k in y:
                    y[k].append(evaluation[k])

        x = np.array(x)
        y = {k: np.array(v) for k, v in y.items()}

        num_plots = len(y)
        plt.figure(figsize=(10, 5 * num_plots))
        for i, k in enumerate(sorted(y)):
            plt.subplot(num_plots, 1, i + 1)

            if ctx.curr_generation > 0:
                slope, intercept, r_value, p_value, std_err = linregress(x, y[k])
                y_pred = slope * x + intercept
                plt.plot(x, y_pred, linestyle='--', color='red')

            plt.scatter(x, y[k], color='blue', marker='.')
            plt.title(k)
            plt.xticks(np.arange(0, ctx.num_generations + 1))
            plt.grid(True)
            
        plt.xlabel('Generation')
        plt.tight_layout()
        plt.savefig(ctx.results_plot_path)
        plt.close()
        print(f"Results plot saved to {ctx.results_plot_path}")

    def run(
        self,
        llm: LLM,
        challenge: Challenge,
        num_generations: int,
        candidates_per_generation: int,
        experiment_dir: str,
        num_visualisations_per_candidate: int = 4,
        on_generation_done: Optional[callable] = None,
    ) -> Context:
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"Starting AutoInnovator run with {num_generations} generations and {candidates_per_generation} candidates per generation.")
        print(f"Experiment directory: {experiment_dir}")

        # Run subsequent generations
        ctx = Context(
            experiment_dir=experiment_dir,
            llm=llm,
            challenge=challenge,
            candidates_per_generation=candidates_per_generation,
            num_generations=num_generations,
            curr_generation=0,
            candidates=[]
        )
        for gen in range(num_generations + 1):
            print(f"Running generation {gen}")
            ctx.curr_generation = gen
            if gen > 0:
                with ThreadPoolExecutor(max_workers=candidates_per_generation) as executor:
                    futures = [
                        executor.submit(self.run_candidate, candidate_id, ctx, num_visualisations_per_candidate) 
                        for candidate_id in range(candidates_per_generation)
                    ]
                    generation_results = [f.result() for f in futures]
                    ctx.candidates.append(generation_results)
            else:
                candidate = self.run_candidate(0, ctx, num_visualisations_per_candidate)
                ctx.candidates.append([candidate])

            print(f"Generation {gen} completed. Plotting results...")
            self.plot_results(ctx)

            if on_generation_done:
                try:
                    on_generation_done(ctx)
                except Exception as e:
                    print(f"Error in on_generation_done callback: {str(e)}")
                    print(traceback.format_exc())
        
        return ctx