"""
Script to run multiple linear evolution chains in parallel in batches
"""

import argparse
import os
import subprocess
import textwrap
import time

from dotenv import load_dotenv, find_dotenv



def GenerateBashCmds(
    venv_path: str, 
    env_name: str, 
    seeds: list[int], 
    experiment_foldername: str, 
    config_name: str, 
    max_prompt_iters: int, 
    verbose: bool, 
):
    """
    Generate linear evolution instance launches within a Python virtual environment
    """
    bash_cmds = []
    for seed in seeds:
        verbose_str = "--verbose" if verbose else ""
        bash_cmd = textwrap.dedent(f"""
            . {venv_path}
            python3 run_linear_evolution_instance.py --env_name {env_name} --seed {seed} --experiment_foldername {experiment_foldername} --config_name {config_name} --max_prompt_iters {max_prompt_iters} {verbose_str}
        """)

        bash_cmds.append(bash_cmd)

    return bash_cmds



def RunCommands(
    processes: list[str],
    max_parallel_cmds: int,
    startup_interval_sec: float, 
):
    """
    Run commands in parallel using multiprocessing
    
    :param list[str] processes: list of commands to run using subprocess.Popen
    :param int max_parallel_cmds: list of maximum number of commands to run in parallel per device
    :param float startup_interval_sec: interval between starting each command to avoid too much synchronous load
    """
    active_processes = []
    for i in range(len(processes)):
        print(f'Launching process {i + 1}/{len(processes)}...')
        
        if len(active_processes) < max_parallel_cmds:
            active_processes.append(subprocess.Popen(['bash', '-c', processes[i]]))
            time.sleep(startup_interval_sec)

        while len(active_processes) == max_parallel_cmds:
            for p in active_processes:
                if p.poll() is not None:  # remove process that finished
                    active_processes.remove(p)
                    # p.returncode
            
            time.sleep(0.1)  # wait 100 milliseconds before polling again

    # wait for all processes to finish
    for p in active_processes:
        p.wait()



def main():
    # parser
    parser = argparse.ArgumentParser(
        usage="%(prog)s [options]",
        description="Launch multiple automated LLM innovator linear evolution instances.",
    )

    parser.add_argument(
        "-v", "--version", action="version", version=f"{parser.prog} version 1.0.0"
    )

    # run flags
    parser.add_argument("--env_name", action="store", type=str, required=True)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument("--seeds", default=[1, 2, 3], type=int, nargs='+')
    parser.add_argument("--max_parallel_runs", default=4, type=int)
    parser.add_argument("--startup_interval_sec", default=0.0, type=float)

    # file paths
    parser.add_argument("--experiment_foldername", action="store", type=str, required=True)
    parser.add_argument("--config_name", action="store", type=str, required=True)

    # loop flags
    parser.add_argument("--max_prompt_iters", default=256, type=int)

    args = parser.parse_args()
    
    # load environment variables
    load_dotenv(find_dotenv(args.env_name))
    venv_path = os.getenv("VENV_PATH")

    
    try:
        processes = GenerateBashCmds(
            venv_path, 
            args.env_name, 
            args.seeds, 
            args.experiment_foldername, 
            args.config_name, 
            args.max_prompt_iters, 
            args.verbose, 
        )

        RunCommands(
            processes,
            args.max_parallel_runs,
            args.startup_interval_sec,
        )
  
    except Exception as e:
        print(f"Error in executing scripts: {e}")
        raise e
        
    print('Batch run finished.')



if __name__ == "__main__":
    main()