"""
Script to run a simple linear evolution chain on a challenge problem
"""

import argparse
from dotenv import load_dotenv, find_dotenv
import json
import os
import textwrap
import traceback
import time

import autoinnovator


response_format_prompt = textwrap.dedent("""
    Format your output using XML tags as "<python>Code here</python><reasoning>Reasoning here</reasoning>", where you output code between "<python>" tags such that I can simply cut out the parts between the python tag and write it directly into some Python script file that I can use to import your function as a symbol into the evaluation script, and similarly your reasoning/thoughts/additional metadata into "<reasoning>" tags.
    In particular, you should include notes of ideas you tried that worked and did not work well in your reasoning response, as I will keep feeding this back your latest code modifications and you can remember why you made a particular change based on past experience to avoid making the same mistake or modification twice.
""")


def build_system_prompt(algorithm_prompt, challenge_params, feedback_prompt, response_format_prompt):
    return textwrap.dedent(f"""
        Code up an algorithm in Python 3 with the following goals and specifications: {algorithm_prompt}
        I will run your algorithm on a set of problem instances and report back to you the algorithm code I ran with your provided reasoning of writing this algorithm as well as evaluation results feedback.
        Problem instances are generated with the following parameters: {challenge_params}
        All prompts I send you thus contains three parts marked by "ALGORITHM:", "REASONING:" and "EVALUATION:".
        The evaluation feedback contains: {feedback_prompt}
        If there is a bug, you will instead receive the corresponding error from the Python interpreter.
        We keep iterating to improve your candidate algorithm target score. Keep your responses short: first part is only your code annotated with comments to explain where necessary; second part is your summary of changes and your reasoning/thoughts on why you chose those.
        For your response, adhere to the format: {response_format_prompt}
        To kickstart the improvement cycle, I start with providing the seed algorithm implementation under "ALGORITHM:" and report back to you the corresponding evaluation feedback under "EVALUATION:" in the prompt format as discussed previously.
    """)


        
def setup_parser(description):
    """
    Setup parsers for command line arguments
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description=description,
    )

    # run flags
    parser.add_argument("--env_name", action="store", type=str, required=True)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument("--seed", default=123, type=int)

    # file paths
    parser.add_argument("--path_to_challenges", default='../challenges/', type=str)
    parser.add_argument("--experiment_foldername", action="store", type=str, required=True)
    parser.add_argument("--config_name", action="store", type=str, required=True)

    # loop flags
    parser.add_argument("--max_prompt_iters", default=256, type=int)

    return parser




def main():

    # command line arguments
    parser = setup_parser(
        "Run automated LLM innovator MVP linear evolution chain."
    )
    args = parser.parse_args()

    # venv
    load_dotenv(find_dotenv(args.env_name))
    NANOGPT_APIKEY = os.getenv("NANOGPT_APIKEY")

    # setup
    LLM_prompt_time_ms = int(time.time() * 1000)  # ms, include as we don't have seed control of LLM API
    challenge_dir = args.path_to_challenges + args.experiment_foldername + '/'
    checkpoint_dir = args.experiment_foldername + '/' + args.config_name + '/'

    instance_foldername = f'seed{args.seed}iters{args.max_prompt_iters}time{LLM_prompt_time_ms}'
    exp_fulldir = checkpoint_dir + instance_foldername + '/'
    os.makedirs(exp_fulldir, exist_ok=True)

    run_expname = args.experiment_foldername + '_' + args.config_name
    logger = autoinnovator.logger.setup_logger(
        f'run_linear_evolution_instance_{run_expname}', 
        exp_fulldir + 'run.log', 
    )
    
    # challenge config
    config_file = challenge_dir + f'config_{args.config_name}.py'
    LLM_temperature, LLM_name, challenge_params = autoinnovator.utils.import_model_specific_symbols(
        config_file, ['LLM_temperature', 'LLM_name', 'challenge_params'])
    
    # system prompt generation and import Python function symbols
    template_file = challenge_dir + 'template.py'
    algorithm_prompt, feedback_prompt, first_algorithm = autoinnovator.utils.import_model_specific_symbols(
        template_file, ['algorithm_prompt', 'feedback_prompt', 'first_algorithm'])
    system_prompt = build_system_prompt(
        algorithm_prompt, challenge_params, feedback_prompt, response_format_prompt)

    EvaluateAlgorithm, ConstructFeedback = autoinnovator.utils.import_model_specific_symbols(
        template_file, ['EvaluateAlgorithm', 'ConstructFeedback'])
    
    # create the LLM API model
    logger.info("Building LLM API model...")
    api = autoinnovator.llm.OpenAIAPI(provider="nanogpt")
    api.set_api_key(NANOGPT_APIKEY)

    model = autoinnovator.llm.BaseLLM(LLM_name, LLM_temperature)
    model.set_api(api)
    model.set_system_prompt(system_prompt)

    with open(exp_fulldir + "system_prompt.txt", "w") as f:
        f.write(system_prompt)
    
    # innovation loop
    logger.info("Starting LLM-innovation evolution loop...")
    start_time = time.time()

    pycode = first_algorithm
    reasontext = "This is the initial seed program.\n"
    retry_err = ""
    algo_id = 1

    evalfeedback_prompt_dict, reasontext_dict = {}, {}
    while algo_id < args.max_prompt_iters + 1:

        if retry_err == "":  # LLM respnose structure was valid, export and evaluate program
            with open(exp_fulldir + f'algo{algo_id}.py', 'w') as f:
                f.write(pycode)
            
            try:  # see if the LLM Python program executes
                eval_results = EvaluateAlgorithm(args.seed, challenge_params, exp_fulldir + f'algo{algo_id}.py')
                try:
                    evalfeedback_prompt = ConstructFeedback(eval_results)
                except Exception as err:
                    error_str = traceback.format_exc()
                    evalfeedback_prompt = 'ConstructFeedback error: ' + str(err) + "\nTraceback: " + error_str

            except Exception as err:  # return Python interpreter error as feedback
                error_str = traceback.format_exc()
                evalfeedback_prompt = 'EvaluateAlgorithms error: ' + str(err) + "\nTraceback: " + error_str

            if args.verbose:
                logger.info("Evaluation feedback: " + evalfeedback_prompt)
            evalfeedback_prompt_dict[algo_id] = evalfeedback_prompt
            with open(exp_fulldir + "evaluation_feedback.json", "w") as f:
                json.dump(evalfeedback_prompt_dict, f)

            if algo_id == args.max_prompt_iters:
                break  # finish with final feedback

        logger.info(f"Sending algorithm {algo_id} prompt...")
        full_prompt = retry_err + "ALGORITHM:\n" + pycode + "REASONING:\n" + reasontext + \
            "EVALUATION:\n" + evalfeedback_prompt
        if args.verbose:
            logger.info("Full prompt sent:\n" + full_prompt)

        receive_status, receive_message = model.send_prompt(full_prompt)
        if receive_status:
            try:
                pycode = receive_message.split('<python>')[1].split('</python>')[0]
                reasontext = receive_message.split('<reasoning>')[1].split('</reasoning>')[0]

                reasontext_dict[algo_id] = reasontext
                with open(exp_fulldir + "reasoning.json", "w") as f:
                    json.dump(reasontext_dict, f)

                retry_err = ""
                algo_id += 1

            except Exception as err:  # retry at same algorithm index
                logger.info("Proposed program was invalid, retrying generation...")
                retry_err = 'Response formatting error: ' + str(err) + f'\nReminder about response formatting: {response_format_prompt}\nRetry previous prompt:\n'
            
        else:
            error_str = traceback.format_exc()
            logger.error(receive_message + "\nTraceback: " + error_str)
            raise ValueError(receive_message + "\nTraceback: " + error_str)


    loop_elapsed_time = time.time() - start_time
    logger.info(f"Completed loop in {loop_elapsed_time:.2f} seconds...")


if __name__ == "__main__":
    main()