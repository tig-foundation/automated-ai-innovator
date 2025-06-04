The following scripts are used for running linear evolution experiments:

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