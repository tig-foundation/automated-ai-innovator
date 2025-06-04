### Defining challenge classes

Each challenge class folder in this directory `challengeXYZ/` specifies all code and prompts to start the automated innovator loop. In particular, every challenge class folder should contain the `challengeXYZ/template.py` file, which specifies the prompts: 
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

Full example challenges are given in this folder.



### Challenge instance evolution configs

Within the challenge directory, one should define problem instance groups associated with a `challengeXYZ/config_NAME.py` file to set the challenge and LLM parameters for all instances in this group with name set by the user. This file has to specify:
- `LLM_temperature`
    A float for the temperature of text generation
- `LLM_name`
    A string for the NanoGPT model to query, e.g. "deepseek-chat", "claude-3-7-sonnet-20250219", "gemini-2.5-flash-preview-05-20", "chatgpt-4o-latest"
- `challenge_params`
    A dictionary that specifies parameters for challenge instance generation in `EvaluateAlgorithm()`

Example configs are provided per challenge class.