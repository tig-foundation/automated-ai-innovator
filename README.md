# TIG automated innovator prototype

LLM-powered automated innovator for TIG ([document](https://www.overleaf.com/project/682e1044aa6cfd5a37fb5f5b)).


<details>
<summary><b>Instructions to set up local Python environment 💻 </b></summary>

To run the notebooks, we need to install software dependencies in Python 3 (3.7 or higher).
    
1. First, open the terminal and create a Python 3 virtual environment in the repository directory

```
mkdir venv/
python3 -m venv venv/
```

Note it is a good habit to specify the exact Python version `python3.xx -m venv venv/`
    
2. Now activate it 
    
```
. venv/bin/activate
```
    
3. Finally, install the required dependencies 
    
```
python3 -m pip install -r requirements.txt
```

4. Set up the `.env` file in the root directory for local paths and variables

```
NANOGPT_APIKEY="123Example"
VENV_PATH="/your/venv/path/..."
```

   
</details>


<details>
<summary><b>Instructions to set up IPython kernel for Jupyter notebooks</b></summary>


1. Add the new environment to Jupyter kernels 

```
python3 -m ipykernel install --user --name=TIG_LLMinnovator
```
  
2. Now one should be able to run the notebooks with all dependencies available using the `TIG_LLMinnovator` IPython kernel.

</details>



## Usage

Each challenge class folder in the root repository directory `challengeXYZ/` specifies all code and prompts to start the automated innovator loop. In particular, every challenge class folder should contain the `challengeXYZ/template.py` file, which specifies the prompts: 
- `algorithm_prompt`
    Prompt to specify the challenge we are trying to solve and what the algorithm needs to take in as arguments and return as outputs
- `first_algorithm`
    A working first code implementation of a candidate algorithm
- `feedback_prompt`
    A prompt describing the structure of the evaluation feedback metrics the LLM will receive at each iteration

and the following functions:
- `EvaluateAlgorithm(instances_seed: int, challenge_params: dict, algo_script_file: str)`
    The function that generates a challenge instance from a given `challenge_params` dictionary and loads a candidate algorithm and runs it to evaluate the performance/validity, returning the results in a dictionary `evaluation_results`
- `ConstructFeedback(evaluation_results: dict)`
    The function that constructs the evaluation feedback prompt with the evaluation results from the candidate algorithm run, consistent with the structure described in `feedback_prompt`

Within the challenge directory, one should define instance classes `challengeXYZ/instanceXYZ/` and specify a `challengeXYZ/instanceXYZ/config.py` file to set the challenge and LLM parameters for all instances in this group. This file has to specify:
- `LLM_temperature`
    A float for the temperature of text generation
- `LLM_name`
    A string for the NanoGPT model to query, e.g. "deepseek-chat", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview-05-20", "chatgpt-4o-latest"
- `challenge_params`
    A dictionary that specifies parameters for challenge instance generation in `EvaluateAlgorithm()`

A full example challenge is given in `KDE/` on the kernel density estimation (KDE) challenge. 


### `run_linear_evolution_batch.py`

This script manages the execution of a batch of simple linear evolution chain automated innovator loops, each specified by one of the seeds in the `--seeds` flag. It creates a subfolder for each run based on the random seed, number of innovation loop iterations and execution time `seedXYZitersNtimeTTTT/`, where all algorithm proposals are stored in separate Python scripts named `algo1.py`, `algo2.py`, ... `algoN.py` in chronological order. In addition, it stores all intermediate reasoning and evaluation results in corresponding JSON files `reasoning.json` and `evaluation_feedback.json`, respectively. One can control the maximum number of simultaneous launches at a time through `--max_parallel_runs` as is done in this example: 
```
python3 run_linear_evolution_batch.py --env_name .env --seeds 1 2 3 4 --experiment_foldername KDE --config_foldername test --max_prompt_iters 8 --max_parallel_runs 4 --startup_interval_sec 1.0
```

### `run_linear_evolution_instance.py`

This script executes an automated LLM innovation loop as a single linear evolution chain, where we specify the following arguments as in the example below: 
```
python3 run_linear_evolution_instance.py --env_name .env --seed 123 --experiment_foldername KDE --config_foldername test --max_prompt_iters 8
```