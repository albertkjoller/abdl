import os
import json
from pathlib import Path

import textwrap
import argparse
from typing import List, Tuple, Optional
from types import MethodType

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier

from src.data.toy_example import generate_moons, generate_multiclass
from src.methods.target_input_distribution import TargetInputDistribution, MultivariateGaussian
from src.methods.acquisition_functions import AcquisitionFunction, Random, VariationRatios, MinimumMargin, Entropy, BALD, EPIG

from src.models.utils import GP_sample
from src.methods.toy_example import run_active_learning_loop_toy

from src.models.llla import SimpleLLLA

def parse_arguments():
    # Create parser
    parser = argparse.ArgumentParser(
        prog='PROG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Active Bayesian Deep Learning
            -----------------------------
                This script can run a variety of active learning acquistion functions
                to be used for running experiments. It is the main engine for 
                producing results associated to this project work.
        '''),
        epilog="Feel free to reach out if you have trouble reproducing the results! - albertkjoller"
    )

    ### DATASET PARAMETERS ###
    parser.add_argument('--dataset', type=str, required=True,
                        help='Choose which dataset to run with - either "2D_moons", "2D_multiclass." or ...')
    parser.add_argument('--size_initial', type=int, default=5,
                        help='Number of samples per class in initial training set.')
    parser.add_argument('--size_test', type=int, default=100,
                        help='Number of samples in test set.')
    parser.add_argument('--size_pool', type=int, default=100,
                        help='Number of samples in the pool. Relevant for 2D tasks.')
    
    ### MODEL PARAMETERS ###
    parser.add_argument('--model', type=str, default='SimpleLLLA',
                        help='Name of the model to be trained.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='The learning rate.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs.')
    parser.add_argument('--val_every_step', type=int, default=1,
                        help='How frequently to run validation if deep learning setting.')
    parser.add_argument('--model_verbose', type=int, default=0,
                        help='Whether to print model training progress.')

    ### ACTIVE LEARNING PARAMETERS ###  
    parser.add_argument('--acq_functions', nargs='+', action='extend', type=str, required=True,
                        help='List acquisition functions to explore, e.g. Random, BALD or EPIG.',)
    parser.add_argument('--num_queries', type=int, default=50,
                        help='Number of iterations of querying data points from the pool.')
    parser.add_argument('--samples_per_query', type=int, default=1,
                        help='How many samples to query per iteration.')
    parser.add_argument('--n_posterior_samples', type=int, default=5000,
                        help='Number of MC samples to approximate posterior for BALD and EPIG estimates.')
    parser.add_argument('--target_dist', type=str,
                        help='The target input distribution. Required when running with EPIG acquisition.')
    parser.add_argument('--mu', nargs='+', action='extend', type=float,
                        help='Mean of the target distribution.')
    parser.add_argument('--sigma', type=float,
                        help='Spread of the target distribution.')
    parser.add_argument('--n_target_dist_samples', type=int, default=100,
                        help='Number of MC samples to draw from the target input distribution for EPIG estimates.')
    
    ### EXPERIMENT PARAMETERS ###
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='The name of the experiment that is running.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Where to store the results (and plots).')
    parser.add_argument('--seed_range', nargs=2, action='extend', type=int, required=True,
                        help='Range of random seeds - provides for uncertainty bounds.')
    parser.add_argument('--save_fig', action='store_true',
                        help='Determines whether to save figures of the run.')
    parser.add_argument('--animate', action='store_true',
                        help='Determines whether to animate the saved figures.')

    args = parser.parse_args()
    if args.save_fig:
        args.samples_per_query = None
    
    return args

def get_dataset(args, seed):
    if args.dataset == '2D_moons':
        num_classes = 2
        Xtrain, ytrain, Xtest, ytest, Xpool, ypool = generate_moons(N_initial_per_class=args.size_initial, N_test=args.size_test, N_pool=args.size_pool, noise=0.2, seed=seed)
    elif args.dataset == '2D_multiclass':
        num_classes = 4
        Xtrain, ytrain, Xtest, ytest, Xpool, ypool = generate_multiclass(N_initial_per_class=args.size_initial, N_test=args.size_test, N_pool=args.size_pool, num_classes=num_classes, noise=0.35, seed=seed)
    else:
        raise NotImplementedError("The chosen dataset does not exist...")
    
    if args.model == 'SimpleLLLA':
        return (
            torch.FloatTensor(Xtrain).to(args.device), torch.LongTensor(ytrain).to(args.device), 
            torch.FloatTensor(Xpool).to(args.device), torch.LongTensor(ypool).to(args.device), 
            torch.FloatTensor(Xtest).to(args.device), torch.LongTensor(ytest).to(args.device), 
            num_classes
        )
    else:
        return Xtrain, ytrain, Xtest, ytest, Xpool, ypool, num_classes

def get_model(args, num_classes, seed):
    if args.model == 'GPClassifier':
        # Define model
        model           = GaussianProcessClassifier(1.0 * RBF(1.0))
        model.sample    = MethodType( GP_sample, model )
        model.fit_      = MethodType(  lambda self, Xtrain, ytrain, Xval, yval: self.fit(Xtrain, ytrain) , model )
        
    elif args.model == 'SimpleLLLA':
        model   = SimpleLLLA(args=args, num_classes=num_classes, seed=seed)
        model.to(args.device)
    else:
        raise NotImplementedError("The chosen model type does not exist...")
    return model

def get_acquisition_fun(acq_fun: str, seed: int, args) -> Tuple[AcquisitionFunction, Optional[TargetInputDistribution]]:
    if acq_fun == 'Random':
        return Random(query_n_points=args.samples_per_query), None
    elif acq_fun == 'VariationRatios':
        return VariationRatios(query_n_points=args.samples_per_query), None
    elif acq_fun == 'MinimumMargin':
        return MinimumMargin(query_n_points=args.samples_per_query), None
    elif acq_fun == 'Entropy':
        return Entropy(query_n_points=args.samples_per_query), None
    elif acq_fun == 'BALD':
        return BALD(query_n_points=args.samples_per_query, n_posterior_samples=args.n_posterior_samples), None
    elif acq_fun == 'EPIG':
        target_input_dist = get_target_input_distribution(args)
        return EPIG(
            query_n_points=args.samples_per_query, 
            target_input_distribution=target_input_dist,
            n_posterior_samples=args.n_posterior_samples, 
            n_target_input_samples=args.n_target_dist_samples, 
            seed=seed,
        ), target_input_dist
    # elif acq_fun == 'GeneralEPIG':
    #     target_input_dist = get_target_input_distribution(args)
    #     return GeneralEPIG(
    #         query_n_points=args.samples_per_query, 
    #         target_input_distribution=target_input_dist,
    #         n_posterior_samples=args.n_posterior_samples, 
    #         n_target_input_samples=args.n_target_dist_samples, 
    #         seed=seed,
    #     ), target_input_dist

    else:
        raise NotImplementedError("The chosen acquition function does not exist...")

def get_target_input_distribution(args) -> TargetInputDistribution:
    if args.target_dist == '2DGaussian':
        return MultivariateGaussian(mu=args.mu, Sigma=np.array([[1, 0], [0, 1]]) / args.sigma**2)
    # elif args.target_dist == '2DGaussian_v2':
    #     return MultivariateGaussian(mu=[0.75, 0.0], Sigma=np.array([[1, 0], [0, 1]]) / 8)
    else:
        raise NotImplementedError("The chosen target input distribution does not exist...")
    
def get_active_learning_loop(args):
    if args.dataset in ['2D_moons', '2D_multiclass']:
        return run_active_learning_loop_toy
    else:
        raise NotImplementedError("The chosen active learning loop does not exist...")

def setup_save_information(args):
    # Get version number for specific experiment name
    if not os.path.isdir(Path(args.save_dir) / args.experiment_name):
        exp_version = 0
    else:
        exp_version = max([int(version_.split('version')[1]) for version_ in next(os.walk(Path(args.save_dir) / args.experiment_name))[1]]) + 1
    
    # Set save path
    save_path = Path(args.save_dir) / f'{args.experiment_name}/version{exp_version}'
    # Create folder if experiment and version does not already exist
    os.makedirs(save_path, exist_ok=False)

    # Save setting of args in experiment folder
    with open(save_path / 'args.json', 'wt') as f:
        json.dump(vars(args), f)

    return save_path
    
if __name__ == '__main__':

    # Parse arguments and run setup
    args        = parse_arguments()
    save_path   = setup_save_information(args)
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = torch.device('cpu') #  if torch.cuda.is_available() else torch.device('cpu')

    # Run experiment as defined in inputs
    for seed in range(args.seed_range[0], args.seed_range[1] + 1):
        for acq_function in args.acq_functions:
            print(f"{'-'*10} RUNNING {acq_function} ACQUISITION FUNCTION (SEED = {seed}) {'-'*10}")

            # Set seed
            np.random.seed(seed)
            # Define acquisition function
            acq_fun, target_input_dist = get_acquisition_fun(acq_fun=acq_function, seed=seed, args=args)

            # Load data
            Xtrain, ytrain, Xtest, ytest, Xpool, ypool, num_classes = get_dataset(args, seed=seed)
            
            # Define model
            args.save_dir_model = save_path / f'{acq_fun.name}/seed{seed}'
            model               = get_model(args, num_classes, seed)

            # Run active learning loop
            _, _, _, _ = get_active_learning_loop(args)(
                model,
                acq_fun=acq_fun,
                target_input_distribution=target_input_dist,
                Xtrain=Xtrain, ytrain=ytrain,
                Xtest=Xtest, ytest=ytest, 
                Xpool=Xpool, ypool=ypool,
                n_iterations=args.num_queries,
                save_fig=args.save_fig, 
                animate=args.animate, fps=1,
                num_classes=num_classes,
                P=55,
                seed=seed,
                save_dir=save_path,
            )

# BINARY POOLSIZE:
# python run.py --experiment_name 2D_binary_poolsize --dataset 2D_moons --size_initial 2 --size_test 200 --size_pool 100 --model GPClassifier --model_verbose 0 --acq_functions EPIG --num_queries 50 --samples_per_query 1 --n_posterior_samples 5000 --save_dir /work3/s194253/projects/abdl/reports/ --seed_range 0 100
# python run.py --experiment_name 2D_binary_poolsize --dataset 2D_moons --size_initial 2 --size_test 200 --size_pool 1000 --model GPClassifier --model_verbose 0 --acq_functions EPIG --num_queries 50 --samples_per_query 1 --n_posterior_samples 5000 --save_dir /work3/s194253/projects/abdl/reports/ --seed_range 0 100
# python run.py --experiment_name 2D_binary_poolsize --dataset 2D_moons --size_initial 2 --size_test 200 --size_pool 10000 --model GPClassifier --model_verbose 0 --acq_functions EPIG --num_queries 50 --samples_per_query 1 --n_posterior_samples 5000 --save_dir /work3/s194253/projects/abdl/reports/ --seed_range 0 100

# MULTICLASS POOLSIZE:
# python run.py --experiment_name 2D_multiclass_poolsize --dataset 2D_multiclass --size_initial 2 --size_test 200 --size_pool 100 --model GPClassifier --model_verbose 0 --acq_functions EPIG --num_queries 50 --samples_per_query 1 --n_posterior_samples 5000 --save_dir /work3/s194253/projects/abdl/reports/ --seed_range 0 100
# python run.py --experiment_name 2D_multiclass_poolsize --dataset 2D_multiclass --size_initial 2 --size_test 200 --size_pool 1000 --model GPClassifier --model_verbose 0 --acq_functions EPIG --num_queries 50 --samples_per_query 1 --n_posterior_samples 5000 --save_dir /work3/s194253/projects/abdl/reports/ --seed_range 0 100
# python run.py --experiment_name 2D_multiclass_poolsize --dataset 2D_multiclass --size_initial 2 --size_test 200 --size_pool 10000 --model GPClassifier --model_verbose 0 --acq_functions EPIG --num_queries 50 --samples_per_query 1 --n_posterior_samples 5000 --save_dir /work3/s194253/projects/abdl/reports/ --seed_range 0 100


# EPIG:
# python run.py --experiment_name epig_test --dataset 2D_moons --size_initial 5 --size_test 200 --size_pool 100 --model GPClassifier --model_verbose 0 --acq_functions EPIG --num_queries 50 --samples_per_query 1 --n_posterior_samples 5000 --save_dir /work3/s194253/projects/abdl/reports/ --seed_range 0 1 --save_fig --target_dist 2DGaussian --mu 2.1 0.45 --sigma 2 --n_target_dist_samples 1000
# python run.py --experiment_name epig_test --dataset 2D_moons --size_initial 5 --size_test 200 --size_pool 100 --model GPClassifier --model_verbose 0 --acq_functions EPIG --num_queries 50 --samples_per_query 1 --n_posterior_samples 5000 --save_dir /work3/s194253/projects/abdl/reports/ --seed_range 0 1 --save_fig --target_dist 2DGaussian --mu -2.0 0.45 --sigma 2 --n_target_dist_samples 1000
         