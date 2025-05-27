"""
Script to run innovator submissions in the benchmarker training environment
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

import src
from src.utils import import_model_specific_symbols, setup_logger



def build_system_prompt(algorithm_prompt, feedback_prompt, first_algorithm):
    return f"""
Code up an algorithm in Python 3 with the following goals and specifications: {algorithm_prompt}
I will run your algorithm on a set of problem instances and report back to you: {feedback_prompt}
If there is a bug, you will receive the corresponding error from the Python interpreter.
We keep iterating to improve your candidate algorithm target score. Keep your responses short: first part is only your code annotated with comments to explain where necessary; second part is your summary of changes and your reasoning/thoughts on why you chose those.
Format your output using XML tags, where you output code between "<python>" tags such that I can simply cut out the parts between the python tag and write it directly into some Python script file that I can use to import your function as a symbol into the evaluation script, and your reasoning/thoughts/additional metadata into "<reasoning>" tags.
We start with the first algorithm implementation: {first_algorithm}
The first prompt I send you are the evaluation feedback results from running this algorithm.
"""


        
def setup_parser(description):
    """
    Setup parsers for command line arguments
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description=description,
    )

    # run flags
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--double_arrays", dest="double_arrays", action="store_true")
    parser.set_defaults(double_arrays=False)
    parser.add_argument("--deterministic_run", dest="deterministic_run", action="store_true")
    parser.set_defaults(deterministic_run=False)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--jitter", default=1e-6, type=float)

    # file paths
    parser.add_argument("--innovator_filename", action="store", type=str, required=True)
    parser.add_argument("--experiment_name", action="store", type=str, required=True)
    parser.add_argument("--checkpoint_dir", action="store", type=str, required=True)

    # synthetic data flags
    parser.add_argument("--kernel_type", action="store", type=str, required=True)
    parser.add_argument("--num_dims", type=int, required=True)
    parser.add_argument("--num_data_pts", default=1024, type=int)
    parser.add_argument("--num_test_pts", default=512, type=int)
    parser.add_argument("--noise_std", default=0.2, type=float)
    parser.add_argument("--kernel_lengthscale", default=0.3, type=float)
    parser.add_argument("--RFF_num_feats", default=0, type=int)

    # data loader flags
    parser.add_argument("--train_fraction", default=0.8, type=float)
    parser.add_argument("--dataloader_workers", default=1, type=int)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)

    # model flags
    parser.add_argument("--hidden_layers", nargs='+', type=int, required=True)
    parser.add_argument("--activation_function", type=str, required=True)
    parser.add_argument("--final_layers_frozen", default=2, type=int)

    # training flags
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--stopping_patience", default=30, type=int)
    parser.add_argument("--stopping_threshold", default=0, type=float)
    parser.add_argument("--output_optimization_trajectory", dest="output_optimization_trajectory", action="store_true")
    parser.set_defaults(output_optimization_trajectory=False)

    # environment variables
    parser.add_argument("--environ_vars", default=[], nargs='+', type=str)

    return parser




def main():

    parser = setup_parser(
        "Run innovator optimiser submission on a challenge instance in the benchmarker training environment."
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    logger = setup_logger('run_challenge_instance_', args.checkpoint_dir + args.experiment_name + '.log')

    # environment
    for var in args.environ_vars:
        var_comps = var.split('=')
        key, value = var_comps[0], var[len(var_comps[0]) + 1:]
        os.environ[key] = value

    # set the device
    dev = src.device.get_device(args.device)
    logger.info(f"Using device: {dev}")
    
    # import the optimizer and scheduler constructors
    OptimizerInitState, OptimiserQueryAtParams, OptimizerStep = import_model_specific_symbols(
        args.innovator_filename + '.py', ['OptimizerInitState', 'OptimiserQueryAtParams', 'OptimizerStep'])

    # create the model
    logger.info("Building neural network...")
    seed = args.seed + 1234  # add an arbitrary seed offset

    neural_network = model_constructor(
        seed, input_dims, out_dims, args.hidden_layers, args.activation_function, args.final_layers_frozen)
    neural_network.to(dev)


    # time the training function
    logger.info("Training neural network...")
    seed = args.seed

    try:
        start_time = time.time()
        train_metrics, best_model, optimization_trajectory = training_loop(
            logger, 
            neural_network, 
            seed, 
            dev, 
            train_datasampler, 
            train_dataloader, 
            val_dataloader, 
            OptimizerInitState, 
            OptimiserQueryAtParams, 
            OptimizerStep, 
            max_epochs=args.max_epochs, 
            stopping_patience=args.stopping_patience, 
            stopping_threshold=args.stopping_threshold, 
            output_optimization_trajectory=args.output_optimization_trajectory, 
            output_trajectory_device=torch.device('cpu'),
            deterministic_run=args.deterministic_run,
            verbose=args.verbose, 
        )

        logger.info(f'Training time: {time.time() - start_time:.2f} seconds')
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise e

    # evaluate the model on the test set
    logger.info("Evaluating neural network...")
    
    neural_network.load_state_dict(best_model)
    neural_network.to(dev)

    test_loss = evaluate_on_test_data(neural_network, test_dataloader, dev)

    # evaluate test data loss on ground truth function
    gt_test_loss = loss_criterion(
        torch.from_numpy(gt_f_i_test), torch.from_numpy(y_i_test)).item()
    
    # evaluate test data loss of baseline GP model
    # GP_test_loss = loss_criterion(
    #     torch.from_numpy(GP_f_i_test), torch.from_numpy(y_i_test)).item()
    
    eval_metrics = {
        'fitted_model_test_loss': test_loss,
        'ground_truth_test_loss': gt_test_loss,
        #'GP_test_loss': GP_test_loss,
    }

    
    # save the results
    logger.info("Saving training results...")
    
    output_metrics = {
        'eval_metrics': eval_metrics,
        'train_metrics': train_metrics,
        'config': vars(args),
    }

    save_path = args.checkpoint_dir + args.experiment_name + '_metrics.json'
    with open(save_path, 'w') as f:
        json.dump(output_metrics, f)

    if args.output_optimization_trajectory:
        save_path = args.checkpoint_dir + args.experiment_name + '_trajectories.pckl'
        with open(save_path, 'wb') as f:
            pickle.dump(optimization_trajectory, f)

    logger.info("Saving neural network checkpoint...")
    save_path = args.checkpoint_dir + args.experiment_name + '_model.pth'
    torch.save(best_model, save_path)



if __name__ == "__main__":
    main()