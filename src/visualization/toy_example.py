from typing import List, Tuple

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_sine(Xtrain, ytrain, xaxis, gp_samples, N):
    for i in range(2):
        plt.scatter(Xtrain[ytrain==i], ytrain[ytrain==i], color=f"C{i}", label=f'Class {i}')

    np.random.seed(42)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    xaxis   = np.arange(0, 2*np.pi, 6 / N)
    labels  = (sigmoid(np.sin(2*np.pi*xaxis / 4)) > 0.5).astype(int) # threshold a sigmoid
    plt.plot(xaxis, labels, color='gray', ls='--', label='True function')

    # for i in range(5):
    plt.plot(xaxis, gp_samples[1].mean(axis=0), label='Mean of GP samples')

def plot_moons(Xtrain, ytrain, Xtest, ytest, Xpool, ypool):
    fig = plt.figure(figsize=(16, 5))

    ax = fig.add_subplot(131)
    for i in range(2):
        ax.scatter(Xtrain[ytrain == i, 0], Xtrain[ytrain == i, 1], color=f'C{i}', label=f'Class {i}')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim([-1.5, 2.5]); ax.set_ylim([-1, 1.5])
    ax.set_title('Initial training data, $\mathcal{D}_{train}$')
    ax.legend()

    ax = fig.add_subplot(132)
    for i in range(2):
        ax.scatter(Xtest[ytest == i, 0], Xtest[ytest == i, 1], color=f'C{i}', label=f'Class {i}')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim([-1.5, 2.5]); ax.set_ylim([-1, 1.5])
    ax.set_title('Test data, $\mathcal{D}_{test}$')
    ax.legend()

    ax = fig.add_subplot(133)
    ax.scatter(Xpool[:, 0], Xpool[:, 1], color=f'gray')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim([-1.5, 2.5]); ax.set_ylim([-1, 1.5])
    ax.set_title('Unlabelled pool, $\mathcal{D}_{pool}$')
    plt.show()

def plot_multiclass(Xtrain, ytrain, Xtest, ytest, Xpool, ypool, num_classes=4):
    zoom = ([-1.5, 1.5], [-1, 4])

    fig = plt.figure(figsize=(16, 5))

    ax = fig.add_subplot(131)
    for i in range(num_classes):
        ax.scatter(Xtrain[ytrain == i, 0], Xtrain[ytrain == i, 1], color=f'C{i}', label=f'Class {i}')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim(zoom[0]); ax.set_ylim(zoom[1])
    ax.set_title('Initial training data, $\mathcal{D}_{train}$')
    ax.legend()

    ax = fig.add_subplot(132)
    for i in range(num_classes):
        ax.scatter(Xtest[ytest == i, 0], Xtest[ytest == i, 1], color=f'C{i}', label=f'Class {i}')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim(zoom[0]); ax.set_ylim(zoom[1])
    ax.set_title('Test data, $\mathcal{D}_{test}$')
    ax.legend()

    ax = fig.add_subplot(133)
    ax.scatter(Xpool[:, 0], Xpool[:, 1], color=f'gray')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim(zoom[0]); ax.set_ylim(zoom[1])
    ax.set_title('Unlabelled pool, $\mathcal{D}_{pool}$')
    plt.show()

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

def get_density_grid(model, x1_low, x1_high, x2_low, x2_high, P=200):
    # Generate a sample 2D density grid
    x1              = np.linspace(x1_low, x1_high, P)
    x2              = np.linspace(x2_low, x2_high, P)
    X1, X2          = np.meshgrid(x1, x2)
    XX              = np.column_stack((X1.ravel(), X2.ravel()))
    XX = torch.FloatTensor(XX).to(model.args.device) if model.__class__.__name__ == 'SimpleLLLA' else XX

    # Get uncertainty output from model
    density_grid = model.predict_proba(XX)
    return x1, x2, density_grid, XX

convert_to_numpy    = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x    

def show_density_grid(model, Xtrain, Xtest, ytrain, ytest, P=200, figsize=(6,5), auto_zoom=True, zoom=None, ax=None, fig=None, num_classes=2):
    if auto_zoom:
        zoom = ([-1, 1], [-0.5, 3.5]) if num_classes == 4 else ([-5, 5], [-5, 5])
    else:
        assert zoom is not None

    # Get density grid
    x1, x2, density_grid, XX = get_density_grid(model, x1_low=zoom[0][0], x1_high=zoom[0][1], x2_low=zoom[1][0], x2_high=zoom[1][1], P=P)

    # Plot density grid
    if num_classes == 2:
        im = ax.pcolormesh(x1, x2, density_grid[:, 0].reshape(P, P), cmap=plt.cm.RdBu_r, norm=colors.Normalize(vmin=0., vmax=1.), shading='auto')

        # Plot training points
        ax.scatter(convert_to_numpy(Xtrain[:, 0]), convert_to_numpy(Xtrain[:, 1]), color='k', s=50, label='Training data')
        # Plot all data, colored by respective category
        for i in range(num_classes):
            ax.scatter(convert_to_numpy(Xtrain[ytrain == i, 0]), convert_to_numpy(Xtrain[ytrain == i, 1]), color=f'C{i if i!=3 else i+1}', label=f'Class {i}', s=20)
        
        # Add label
        ax.set_xlabel('$x_1$');      
        ax.set_ylabel('$x_2$')
        ax.set_xlim(zoom[0]);    
        ax.set_ylim(zoom[1])

        ax.set_title(f'Posterior predictive')

        # Add colorbar
        add_colorbar(im, fig, ax)
    else:
        gs = plt.GridSpec(2, 2, left=0.2, right=0.5, bottom=0.1, top=0.9)
        subplots_axs = [plt.subplot(gs[i]) for i in range(4)]
        for i, subplot_ax in enumerate(subplots_axs):
            im_ = subplot_ax.pcolormesh(x1, x2, density_grid[:, i].reshape(P, P), cmap=plt.cm.RdBu_r, norm=colors.Normalize(vmin=0.0, vmax=1.0), shading='auto')

            # Plot training points
            subplot_ax.scatter(convert_to_numpy(Xtrain[:, 0]), convert_to_numpy(Xtrain[:, 1]), color='k', s=50, label='Training data')
            # Plot all data, colored by respective category
            for class_idx in range(num_classes):
                subplot_ax.scatter(convert_to_numpy(Xtrain[ytrain == class_idx, 0]), convert_to_numpy(Xtrain[ytrain == class_idx, 1]), color=f'C{class_idx if class_idx!=3 else class_idx+1}', label=f'Class {class_idx}', s=20)

            subplot_ax.set_title(f'$P(y={i} | $' + '$\mathcal{D}_{train})$')
            if i // 2 == 0:
                subplot_ax.set_xticks([], [])      
            if i % 2 == 1:
                subplot_ax.set_yticks([], [])
                add_colorbar(im_, fig, subplot_ax)
            
    return ax, (x1, x2, density_grid, XX)

def show_acquisition_grid(model, acq_fun, Xtrain, ytrain, Xpool, density_grid_outputs, P=200, ax=None, fig=None, num_classes=2, auto_zoom=True, zoom = None, normalize: bool = False):
    # Get density grid outputs
    x1, x2, _, XX = density_grid_outputs

    if auto_zoom:
        zoom = ([-1, 1], [-0.5, 3.5]) if num_classes == 4 else ([-5, 5], [-5, 5])
    else:
        assert zoom is not None

    # Get acquisition function values
    acq_scores, _           = acq_fun(XX, return_sorted=False, model=model)
    acq_score_grid          = acq_scores.reshape(P, P)
    if normalize:
        acq_score_grid      = (acq_score_grid - acq_score_grid.min()) / (acq_score_grid.max() - acq_score_grid.min())

    # Plot density grid
    im = ax.pcolormesh(x1, x2, acq_score_grid, cmap=plt.cm.Greens, norm=colors.Normalize(), shading='auto')

    # Plot unlabelled pool
    ax.scatter(convert_to_numpy(Xpool[:, 0]), convert_to_numpy(Xpool[:, 1]), color=f'gray', label=f'Unlabelled pool', s=20, alpha=0.5)
    # Plot training points
    ax.scatter(convert_to_numpy(Xtrain[:, 0]), convert_to_numpy(Xtrain[:, 1]), color='k', s=50, label='Training data')
    # Color training points by their respective category
    for i in range(num_classes):
        ax.scatter(convert_to_numpy(Xtrain[ytrain == i, 0]), convert_to_numpy(Xtrain[ytrain == i, 1]), color=f'C{i if i!=3 else i+1}', label=f'Class {i}', s=20)

    # Add label
    ax.set_xlabel('$x_1$');      
    ax.set_xlim(zoom[0]);    
    ax.set_ylim(zoom[1])
    ax.set_title(f'{"Normalized " if normalize == True else ""} Acquisition function - {acq_fun.name}')
    # Add colorbar
    add_colorbar(im, fig, ax)
    return ax

def plot_example(model, Xtrain, Xtest, Xpool, ytrain, ytest, acq_fun, next_query, num_classes, P=150, zoom=None, auto_zoom=True):
    
    ### PLOT DECISION BOUNDARY AND ACQUISITION FUNCTION ###
    fig, axs = plt.subplots(1, 3 - int(num_classes == 4), figsize=(18 - int(num_classes == 4) * 6, 5), sharey=True, sharex=True)

    # Plot model uncertainty across grid
    axs[0], density_grid_output = show_density_grid(model, Xtrain, Xtest, ytrain, ytest, num_classes=num_classes, P=P, auto_zoom=auto_zoom, zoom=zoom, ax=axs[0], fig=fig)
    if num_classes == 2:
        axs[0].legend()

    if num_classes == 2:
        decisions_bls = np.argmax(density_grid_output[2], axis=1).reshape((len(density_grid_output[0]), len(density_grid_output[1])))
        axs[1].pcolormesh(density_grid_output[0], density_grid_output[1], 1-decisions_bls, alpha=0.8, cmap=plt.cm.RdBu_r, shading='auto')        
        axs[1].set_title('Decision boundary')
        axs[1].set_xlabel('$x_1$')
        axs[1].grid(None)

    # Plot acquisition function across a grid
    axs[2 - int(num_classes == 4)] = show_acquisition_grid(model, acq_fun, Xtrain, ytrain, Xpool, density_grid_output, num_classes=num_classes, P=P, auto_zoom=auto_zoom, zoom=zoom, ax=axs[2 - int(num_classes == 4)], fig=fig)
    axs[2 - int(num_classes == 4)].scatter(convert_to_numpy(next_query[0]), convert_to_numpy(next_query[1]), color='orange', marker=(5, 1), s=100, label='New query')
    axs[2 - int(num_classes == 4)].legend()
    return axs

def plot_performance_curves(results, experiments: Tuple[str, str], acq_functions: List[str]):

    fig, axs        = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
    for i, experiment in enumerate(experiments):
        # Plot binary case
        pd.DataFrame.from_dict(results[experiment]).plot(x='N_points', y=acq_functions, ax=axs[i])
        for acq_fun in acq_functions:
            axs[i].fill_between(
                results[experiment]['N_points'], 
                results[experiment][acq_fun] - results[experiment][f'{acq_fun}_std'], 
                results[experiment][acq_fun] + results[experiment][f'{acq_fun}_std'],
                alpha=0.3,
            )
        axs[i].set_ylabel('Accuracy')
        axs[i].set_xlabel('$N$ training points')
        axs[i].set_title(experiment.upper())
        axs[i].set_xticks(results[experiment]['N_points'][::5], results[experiment]['N_points'][::5])
        axs[i].legend(loc='lower right')

        # # Plot multiclass case
        # pd.DataFrame.from_dict(results['multiclass']).plot(x='N_points', y=acq_functions, ax=axs[1])
        # for acq_fun in acq_functions:
        #     axs[1].fill_between(
        #         results['multiclass']['N_points'], 
        #         results['multiclass'][acq_fun] - results['multiclass'][f'{acq_fun}_std'], 
        #         results['multiclass'][acq_fun] + results['multiclass'][f'{acq_fun}_std'],
        #         alpha=0.3,
        #     )
        # axs[1].set_xlabel('$N$ training points')
        # axs[1].set_title('Multiclass')
        # axs[1].set_xticks(results['multiclass']['N_points'][::5], results['multiclass']['N_points'][::5])
        # axs[1].legend(loc='lower right')
    return fig