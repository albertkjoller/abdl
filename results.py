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
    
    ### MODEL PARAMETERS ###
    # parser.add_argument('--model', type=str, default='GPClassifier',
    #                     help='Name of the model to be trained.')

    ### ACTIVE LEARNING PARAMETERS ###
    parser.add_argument('--acq_functions', nargs='+', action='extend', type=List[str], required=True,
                        help='List acquisition functions to explore, e.g. Random, BALD or EPIG.',)

    ### EXPERIMENT PARAMETERS ###
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Random seeds - provides for uncertainty bounds.')

    parser.add_argument('--seeds', nargs='+', action='extend', type=int, required=True,
                        help='Random seeds - provides for uncertainty bounds.')
    parser.add_argument('--save_fig', action='store_true',
                        help='Determines whether to use save figures of the run.')

    return parser.parse_args()

if __name__ == '__main__':

    # Parse arguments
    args = parse_arguments()

# python evaluation.py