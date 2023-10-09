from typing import Optional, List, Tuple, Union

import os
import pickle
from collections import defaultdict

import torch
import numpy as np

import imageio
from PIL import Image
import matplotlib.pyplot as plt

from src.visualization.toy_example import plot_example
from src.methods.acquisition_functions import AcquisitionFunction
from src.methods.target_input_distribution import TargetInputDistribution

convert_to_numpy    = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x    
def update_model_score(model, performance, Xtrain, ytrain, Xtest, ytest):  
    performance['train'].append((model.predict_proba(Xtrain).argmax(axis=1) == convert_to_numpy(ytrain)).mean())
    performance['test'].append((model.predict_proba(Xtest).argmax(axis=1) == convert_to_numpy(ytest)).mean())
    performance['N_points'].append(Xtrain.__len__())
    return performance

def update_datasets_and_performance(Xtrain, ytrain, Xpool, ypool, X_next_query, y_next_query, query_idxs):
    if isinstance(Xtrain, torch.Tensor):
        device  = Xtrain.device
        Xtrain  = torch.vstack([Xtrain, X_next_query])
        ytrain  = torch.hstack([ytrain, y_next_query])
        Xpool   = torch.FloatTensor(np.delete(Xpool.cpu().numpy(), query_idxs[0], 0)).to(device)
        ypool   = torch.LongTensor(np.delete(ypool.cpu().numpy(), query_idxs[0], 0)).to(device)
    else:
        Xtrain  = np.vstack([Xtrain, X_next_query])
        ytrain  = np.hstack([ytrain, y_next_query])
        Xpool   = np.delete(Xpool, query_idxs[0], 0)
        ypool   = np.delete(ypool, query_idxs[0], 0)
    
    return Xtrain, ytrain, Xpool, ypool

def run_active_learning_loop_toy(
    model,
    acq_fun: AcquisitionFunction,
    Xtrain: Union[torch.Tensor, np.ndarray], Xtest: Union[torch.Tensor, np.ndarray], Xpool: Union[torch.Tensor, np.ndarray], 
    ytrain: Union[torch.Tensor, np.ndarray], ytest: Union[torch.Tensor, np.ndarray], ypool: Union[torch.Tensor, np.ndarray],
    n_iterations: int = 1, 
    save_dir: str = '../reports/2D_toy', 
    save_fig: bool = False, verbose: int = 1, animate: bool = False, fps: int = 10.,
    zoom: Tuple[List[float], List[float]] = ([-5, 5], [-5, 5]),
    num_classes: int = 2,
    P: int = 100,
    seed: int = 0,
    target_input_distribution: Optional[TargetInputDistribution] = None,
):

    # Create folder for saving results
    os.makedirs(f'{save_dir}/{acq_fun.name}/seed{seed}', exist_ok=True)
    if save_fig:
        os.makedirs(f'{save_dir}/{acq_fun.name}/seed{seed}/images', exist_ok=True)
    img_list = []

    # Run active learning loop
    performance = defaultdict(list)
    for i in range(n_iterations):

        # Train model
        model.fit_(Xtrain, ytrain, Xtest, ytest)
        # Update performance dictionary  
        performance = update_model_score(model, performance, Xtrain, ytrain, Xtest, ytest)

        # Get acquisition function score and the items to query
        _, query_idxs               = acq_fun(Xpool, model=model)
        X_next_query, y_next_query  = Xpool[query_idxs[0]], ypool[query_idxs[0]]

        if save_fig:
        
            # Visualize
            axs = plot_example(model, Xtrain, Xtest, Xpool, ytrain, ytest, acq_fun, X_next_query, num_classes=num_classes, P=P, auto_zoom=True, zoom=None)
            
            # Plot target input distribution on top
            if target_input_distribution is not None:
                target_input_distribution.plot_2D(ax=axs[2 - int(num_classes == 4)], zoom=zoom)
                
            img_list.append(f'{save_dir}/{acq_fun.name}/seed{seed}/images/iteration{i}.png')
            plt.savefig(f'{save_dir}/{acq_fun.name}/seed{seed}/images/iteration{i}.png')
            plt.close()

        ### UPDATE TRAINING SET AND POOL ACCORDINGLY ###
        Xtrain, ytrain, Xpool, ypool = update_datasets_and_performance(Xtrain, ytrain, Xpool, ypool, X_next_query, y_next_query, query_idxs)
            
        if verbose == 1:
            print(
                f"Train acc.: {performance['train'][i]:.2f} |",
                f"Test acc.: {performance['test'][i]:.2f} |",
                f"|ytrain|: {performance['N_points'][i]} points   |",
                f"Remaining pool: {Xpool.__len__()} points   |",
                f"Query: [{X_next_query[0]:.2f}, {X_next_query[1]:.2f}]",
                sep='   ',
            )

    # Fit final model
    model.fit_(Xtrain, ytrain, Xtest, ytest)
    # Get performance
    performance = update_model_score(model, performance, Xtrain, ytrain, Xtest, ytest)

    if verbose == 1:
        print(
            f"Train acc.: {performance['train'][i+1]:.2f}   |",
            f"Test acc.: {performance['test'][i+1]:.2f}   |",
            f"|ytrain|: {performance['N_points'][i+1]} points  |",
            sep='   ',
        )
    
    # Store results
    with open(f'{save_dir}/{acq_fun.name}/seed{seed}/performance.pkl', 'wb') as f:
        pickle.dump(performance, f)

    if save_fig and animate:
        # Load and sort images
        images = [Image.open(path) for path in img_list]

        # Set output path for GIF
        output_gif_path = f'{save_dir}/{acq_fun.name}/querying_process.gif'
        # Save GIF
        imageio.mimsave(output_gif_path, images, duration=(1000 * 1/fps))

    return model, performance, Xtrain, ytrain