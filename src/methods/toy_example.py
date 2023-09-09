import os
import numpy as np
import matplotlib.pyplot as plt

import imageio
from PIL import Image

import pickle
from collections import defaultdict

from src.methods.acquisition_functions import AcquisitionFunction
from src.visualization.toy_example import show_acquisition_grid, show_density_grid

def run_active_learning_loop(
    model,
    acq_fun: AcquisitionFunction,
    Xtrain: np.ndarray, Xtest: np.ndarray, Xpool: np.ndarray, 
    ytrain: np.ndarray, ytest: np.ndarray, ypool: np.ndarray,
    n_iterations: int = 1, 
    save_dir: str = '../reports/2D_toy', 
    save_fig=False, verbose=1, animate=False, fps=10.,
    zoom=([-2, 3], [-2, 2.5]),
    num_classes=2,
    seed = 0,
):

    # Create folder for saving results
    os.makedirs(f'{save_dir}/{"binary" if num_classes == 2 else "multiclass"}/{acq_fun.name}', exist_ok=True)
    os.makedirs(f'{save_dir}/{"binary" if num_classes == 2 else "multiclass"}/{acq_fun.name}/images', exist_ok=True)
    img_list = []

    # Run active learning loop
    performance = defaultdict(list)
    for i in range(n_iterations):

        # Train model
        model.fit(Xtrain, ytrain)
        
        # Update performance dictionary
        performance['train'].append(model.score(Xtrain, ytrain))
        performance['test'].append(model.score(Xtest, ytest))
        performance['N_points'].append(Xtrain.__len__())

        # Get acquisition function score and the items to query
        _, query_idxs               = acq_fun(Xpool, model=model)
        X_next_query, y_next_query  = Xpool[query_idxs[0]], ypool[query_idxs[0]]

        if save_fig:
            ### PLOT DECISION BOUNDARY AND ACQUISITION FUNCTION ###
            fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

            # Plot model uncertainty across grid
            axs[0] = show_density_grid(model, Xtrain, Xtest, ytrain, ytest, zoom=zoom, ax=axs[0], fig=fig, num_classes=num_classes)
            if num_classes == 2:
                axs[0].legend()

            # Plot acquisition function across a grid
            axs[1] = show_acquisition_grid(model, acq_fun, Xtrain, ytrain, Xpool, zoom=zoom, P=35 if acq_fun.name == 'BALD' else 100, ax=axs[1], fig=fig, num_classes=num_classes)
            axs[1].scatter(X_next_query[0], X_next_query[1], color='orange', marker=(5, 1), s=100, label='New query')
            axs[1].legend(loc='upper right' if num_classes == 2 else 'center left')

            img_list.append(f'{save_dir}/{"binary" if num_classes == 2 else "multiclass"}/{acq_fun.name}/images/iteration{i}.png')
            plt.savefig(f'{save_dir}/{"binary" if num_classes == 2 else "multiclass"}/{acq_fun.name}/images/iteration{i}.png')
            plt.close()

        ### UPDATE TRAINING SET AND POOL ACCORDINGLY ###
        Xtrain  = np.vstack([Xtrain, X_next_query])
        ytrain  = np.hstack([ytrain, y_next_query])
        Xpool   = np.delete(Xpool, query_idxs[0], 0)
        ypool   = np.delete(ypool, query_idxs[0], 0)

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
    model.fit(Xtrain, ytrain)
    # Get performance
    performance['train'].append(model.score(Xtrain, ytrain))
    performance['test'].append(model.score(Xtest, ytest))
    performance['N_points'].append(Xtrain.__len__())

    if verbose == 1:
        print(
            f"Train acc.: {performance['train'][i+1]:.2f}   |",
            f"Test acc.: {performance['test'][i+1]:.2f}   |",
            f"|ytrain|: {performance['N_points'][i+1]} points  |",
            sep='   ',
        )
    
    # Store results
    with open(f'{save_dir}/{"binary" if num_classes == 2 else "multiclass"}/{acq_fun.name}/performance_seed{seed}.pkl', 'wb') as f:
        pickle.dump(performance, f)

    if save_fig and animate:
        # Load and sort images
        images = [Image.open(path) for path in img_list]

        # Set output path for GIF
        output_gif_path = f'{save_dir}/{"binary" if num_classes == 2 else "multiclass"}/{acq_fun.name}/querying_process.gif'
        # Save GIF
        imageio.mimsave(output_gif_path, images, duration=(1000 * 1/fps))

    return model, performance, Xtrain, ytrain