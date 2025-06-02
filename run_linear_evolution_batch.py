"""
Script to run multiple challenge instances in a linear evolution fashion in parallel in batches
"""

import argparse
import os
import subprocess
import time

from dotenv import load_dotenv, find_dotenv

from src.utils import import_model_specific_symbols, setup_logger



def RunCommands(
    logger, 
    processes: list[list[str]],
    devices: list[str],
    max_parallel_cmds_per_iter: list[int],
    startup_interval_sec: float, 
):
    """
    Run commands in parallel using multiprocessing, potentially spreading over multiple devices
    
    :param list[list[str]] processes: list of commands to run using subprocess.Popen
    :param list[str] devices: list of devices to run commands on
    :param list[int] max_parallel_cmds_per_iter: list of maximum number of commands to run in parallel per device
    :param float startup_interval_sec: interval between starting each command to avoid too much synchronous load
    """
    active_processes = {name: [] for name in devices}
    for i in range(len(processes)):
        logger.info(f'Launching bash process {i + 1}/{len(processes)}...')
        
        for use_device, max_per_device in zip(devices, max_parallel_cmds_per_iter):
            if len(active_processes[use_device]) < max_per_device:
                active_processes[use_device].append(subprocess.Popen(processes[i] + [use_device]))
                time.sleep(startup_interval_sec)
                break

        while all([
            len(active_processes[use_device]) == max_per_device \
            for use_device, max_per_device in zip(devices, max_parallel_cmds_per_iter)
        ]):
            for use_device in devices:
                for p in active_processes[use_device]:
                    if p.poll() is not None:  # remove process that finished
                        active_processes[use_device].remove(p)
                        # p.returncode
                        
            time.sleep(0.1)  # wait 100 milliseconds before polling again

    for use_device in devices:  # wait for all processes to finish
        for p in active_processes[use_device]:
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
    parser.add_argument("--scripting_dir", type=str, required=True)
    parser.add_argument("--bashes_folder", default="bashes/", type=str)
    parser.add_argument("--env_name", type=str, required=True)

    # operation flags
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument("--devices", default=["cpu"], nargs='+', type=str)
    parser.add_argument("--max_parallel_searches_per_iter", default=[1], nargs='+', type=int)
    parser.add_argument("--startup_interval_sec", default=0.0, type=float)

    # config
    parser.add_argument("--experiment_foldername", type=str, required=True)

    args = parser.parse_args()
    
    # load environment variables
    load_dotenv(find_dotenv(args.env_name))

    checkpoint_path = os.getenv("CHECKPOINTPATH")
    venv_path = os.getenv("VENVPATH")

    # setup names and folders
    base_dir = args.scripting_dir + f'{args.experiment_foldername}/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(base_dir + args.bashes_folder):
        os.makedirs(base_dir + args.bashes_folder)

    logger_name = 'run_instances_batch_' + args.experiment_foldername
    logger = setup_logger(logger_name, base_dir + args.bashes_folder + 'run.log')

    # symbols
    logger.info(f'Loading Python symbols...')
    template_file = args.scripting_dir + args.experiment_foldername + '/template.py'
    
    try:
        BuildBashScripts, = import_model_specific_symbols(
            template_file, ['BuildBashScripts']
        )
            
    except Exception as e:
        logger.error(f"Error loading symbols from {template_file}: {e}")
        raise e


    ###
    # build bash scripts #
    ###
    logger.info(f'Building bash scripts...')
    
    try:
        scriptnames = BuildBashScripts(
            args.scripting_dir, 
            checkpoint_path, 
            venv_path, 
            args.experiment_foldername, 
            args.bashes_folder, 
            args.output_optimiser_states,
            args.double_arrays, 
            args.verbose, 
        )
        
    except Exception as e:
        logger.error(f"Error in building scripts: {e}")
        raise e

    ###
    # execute bash scripts #
    ###
    
    try:
        processes = [["bash", f'{scriptname}.sh'] for scriptname in scriptnames]

        RunCommands(
            logger,
            processes,
            args.devices,
            args.max_parallel_searches_per_iter,
            args.startup_interval_sec,
        )
  
    except Exception as e:
        logger.error(f"Error in executing scripts: {e}")
        raise e
        
    logger.info('Challenge instance ensemble finished.')



if __name__ == "__main__":
    main()