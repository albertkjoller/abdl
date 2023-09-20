import os
from pathlib import Path

import textwrap
import argparse
from typing import List

from src.visualization.toy_example import plot_moons, plot_multiclass, show_density_grid, show_acquisition_grid, plot_performance_curves

def parse_arguments():
    # Create parser
    parser = argparse.ArgumentParser(
        prog='PROG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Active Bayesian Deep Learning
            -----------------------------
                This script can evaluate a previously run experiment. 
                It is the main engine for visualizing results associated 
                to this project work.
        '''),
        epilog="Feel free to reach out if you have trouble reproducing the results! - albertkjoller"
    )

    ### DATASET PARAMETERS ###
    parser.add_argument('--dataset', type=str, required=True,
                        help='Choose which dataset to run with - either "2D-moons", "2D-multiclass." or ...')
    parser.add_argument('--size_initial', type=int, default=5,
                        help='Number of samples in initial training set.')
    parser.add_argument('--size_test', type=int, default=100,
                        help='Number of samples in test set.')
    parser.add_argument('--size_pool', type=int, default=100,
                        help='Number of samples in the pool. Relevant for 2D tasks.')
    parser.add_argument('--noise_level', type=float, default=0.075,
                        help='Noise level for the 2D toy datasets.')
    
    ### MODEL PARAMETERS ###
    parser.add_argument('--model', type=str, default='GPClassifier',
                        help='Name of the model to be trained.')

    ### ACTIVE LEARNING PARAMETERS ###
    parser.add_argument('--acq_functions', nargs='+', action='extend', type=List[str], required=True,
                        help='List acquisition functions to explore, e.g. Random, BALD or EPIG.',)
    parser.add_argument('--num_queries', type=int, default=50,
                        help='Number of iterations of querying data points from the pool.')
    parser.add_argument('--samples_per_query', type=int, default=1,
                        help='How many samples to query per iteration.')
    parser.add_argument('--n_posterior_samples', type=int, default=5000,
                        help='Number of MC samples to approximate posterior for BALD and EPIG estimates.')
    parser.add_argument('--n_target_dist_samples', type=int, default=100,
                        help='Number of MC samples to draw from the target input distribution for EPIG estimates.')
    
    ### EXPERIMENT PARAMETERS ###
    parser.add_argument('--seeds', nargs='+', action='extend', type=int, required=True,
                        help='Random seeds - provides for uncertainty bounds.')
    parser.add_argument('--save_fig', action='store_true',
                        help='Determines whether to use save figures of the run.')
    parser.add_argument('--animate', action='store_true',
                        help='Determines whether to animate the saved figures.')

    return parser.parse_args()

if __name__ == '__main__':

    # Parse arguments
    args = parse_arguments()

# python evaluation.py