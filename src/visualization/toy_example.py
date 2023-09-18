from typing import List

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
    plt.legend()

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
    fig = plt.figure(figsize=(16, 5))

    ax = fig.add_subplot(131)
    for i in range(num_classes):
        ax.scatter(Xtrain[ytrain == i, 0], Xtrain[ytrain == i, 1], color=f'C{i}', label=f'Class {i}')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim([-0.5, 0.5]); ax.set_ylim([-0.5, 0.5])
    ax.set_title('Initial training data, $\mathcal{D}_{train}$')
    ax.legend()

    ax = fig.add_subplot(132)
    for i in range(num_classes):
        ax.scatter(Xtest[ytest == i, 0], Xtest[ytest == i, 1], color=f'C{i}', label=f'Class {i}')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim([-0.5, 0.5]); ax.set_ylim([-0.5, 0.5])
    ax.set_title('Test data, $\mathcal{D}_{test}$')
    ax.legend()

    ax = fig.add_subplot(133)
    ax.scatter(Xpool[:, 0], Xpool[:, 1], color=f'gray')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xlim([-0.5, 0.5]); ax.set_ylim([-0.5, 0.5])
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

    # Get uncertainty output from model
    density_grid    = model.predict_proba(XX)# [:, 0].reshape(P, P) # binary case
    return x1, x2, density_grid, XX

def show_density_grid(model, Xtrain, Xtest, ytrain, ytest, zoom=([-2, 2], [-2, 2]), P=200, figsize=(6,5), ax=None, fig=None, num_classes=2):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
        ax      = ax[0][0]

    # Get density grid
    x1, x2, density_grid, _ = get_density_grid(model, x1_low=zoom[0][0], x1_high=zoom[0][1], x2_low=zoom[1][0], x2_high=zoom[1][1], P=P)

    # Plot density grid
    if num_classes == 2:
        im = ax.pcolormesh(x1, x2, density_grid[:, 0].reshape(P, P), cmap=plt.cm.RdBu_r, norm=colors.CenteredNorm(0.5), shading='auto')

        # Plot training points
        ax.scatter(Xtrain[:, 0], Xtrain[:, 1], color='k', s=50, label='Training data')

        # Plot all data, colored by respective category
        # X_, y_ = np.append(Xtrain, Xtest, axis=0), np.append(ytrain, ytest)
        for i in range(num_classes):
            ax.scatter(Xtrain[ytrain == i, 0], Xtrain[ytrain == i, 1], color=f'C{i if i!=3 else i+1}', label=f'Class {i}', s=20)

        # Add label
        ax.set_xlabel('$x_1$');      
        ax.set_ylabel('$x_2$')
        ax.set_xlim(zoom[0]);    
        ax.set_ylim(zoom[1])
        ax.set_title(f'Decision boundary')

        # Add colorbar
        add_colorbar(im, fig, ax)
    else:
        gs = plt.GridSpec(2, 2, left=0.2, right=0.5, bottom=0.1, top=0.9)
        subplots_axs = [plt.subplot(gs[i]) for i in range(4)]
        for i, subplot_ax in enumerate(subplots_axs):
            im_ = subplot_ax.pcolormesh(x1, x2, density_grid[:, i].reshape(P, P), cmap=plt.cm.RdBu_r, norm=colors.Normalize(vmin=0.0, vmax=1.0), shading='auto')

            # Plot training points
            subplot_ax.scatter(Xtrain[:, 0], Xtrain[:, 1], color='k', s=50, label='Training data')

            # Plot all data, colored by respective category
            for class_idx in range(num_classes):
                subplot_ax.scatter(Xtrain[ytrain == class_idx, 0], Xtrain[ytrain == class_idx, 1], color=f'C{class_idx if class_idx!=3 else class_idx+1}', label=f'Class {class_idx}', s=20)

            subplot_ax.set_title(f'$P(y={i} | $' + '$\mathcal{D}_{train})$')
            if i // 2 == 0:
                subplot_ax.set_xticks([], [])      
            if i % 2 == 1:
                subplot_ax.set_yticks([], [])
                add_colorbar(im_, fig, subplot_ax)
            
    return ax

def show_acquisition_grid(model, acq_fun, Xtrain, ytrain, Xpool, zoom=([-2, 2], [-2, 2]), P=200, ax=None, fig=None, num_classes=2):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, squeeze=False)
        ax      = ax[0][0]

    # Get density grid
    x1, x2, _, XX = get_density_grid(model, x1_low=zoom[0][0], x1_high=zoom[0][1], x2_low=zoom[1][0], x2_high=zoom[1][1], P=P)
    
    # Get acquisition function values
    acq_scores, _           = acq_fun(XX, return_sorted=False, model=model)
    acq_score_grid          = acq_scores.reshape(P, P)

    # Plot density grid
    im = ax.pcolormesh(x1, x2, acq_score_grid, cmap=plt.cm.Greens, norm=colors.Normalize(), shading='auto')

    # Plot unlabelled pool
    ax.scatter(Xpool[:, 0], Xpool[:, 1], color=f'gray', label=f'Unlabelled pool', s=20, alpha=0.5)
    # Plot training points
    ax.scatter(Xtrain[:, 0], Xtrain[:, 1], color='k', s=50, label='Training data')
    # Color training points by their respective category
    for i in range(num_classes):
        ax.scatter(Xtrain[ytrain == i, 0], Xtrain[ytrain == i, 1], color=f'C{i if i!=3 else i+1}', label=f'Class {i}', s=20)

    # Add label
    ax.set_xlabel('$x_1$');      
    ax.set_ylabel('$x_2$')
    ax.set_xlim(zoom[0]);    
    ax.set_ylim(zoom[1])
    ax.set_title(f'Acquisition function - {acq_fun.name}')

    # Add colorbar
    add_colorbar(im, fig, ax)
    return ax

def plot_performance_curves(results, acq_functions: List[str]):

    fig, axs        = plt.subplots(1, 2, sharey=True, figsize=(12, 4))

    # Plot binary case
    pd.DataFrame.from_dict(results['binary']).plot(x='N_points', y=acq_functions, ax=axs[0])
    for acq_fun in acq_functions:
        axs[0].fill_between(
            results['binary']['N_points'], 
            results['binary'][acq_fun] - results['binary'][f'{acq_fun}_std'], 
            results['binary'][acq_fun] + results['binary'][f'{acq_fun}_std'],
            alpha=0.3,
        )
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('$N$ training points')
    axs[0].set_title('Binary')
    axs[0].set_xticks(results['binary']['N_points'][1::5], results['binary']['N_points'][1::5])
    axs[0].legend(loc='lower right')

    # Plot multiclass case
    pd.DataFrame.from_dict(results['multiclass']).plot(x='N_points', y=acq_functions, ax=axs[1])
    for acq_fun in acq_functions:
        axs[1].fill_between(
            results['multiclass']['N_points'], 
            results['multiclass'][acq_fun] - results['multiclass'][f'{acq_fun}_std'], 
            results['multiclass'][acq_fun] + results['multiclass'][f'{acq_fun}_std'],
            alpha=0.3,
        )
    axs[1].set_xlabel('$N$ training points')
    axs[1].set_title('Multiclass')
    axs[1].set_xticks(results['multiclass']['N_points'][::5], results['multiclass']['N_points'][::5])
    axs[1].legend(loc='lower right')
    return fig