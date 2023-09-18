from typing import Union, List
import numpy as np
import matplotlib.pyplot as plt

class TargetInputDistribution:

    def __init__(self, name: str):
        self.name           = name

    def __call__(self,):
        return None
    
    def sample(self, n_samples: int = 100):
        return None
    
class MultivariateGaussian(TargetInputDistribution):

    def __init__(self, mu: Union[list, np.ndarray], Sigma: Union[List[list], np.ndarray]):
        # Set parameters of the distribution
        self.mu     = mu        
        self.Sigma  = Sigma

        super().__init__(name='MultivariateGaussian')

    def sample(self, n_samples: int, seed: int = 0):
        # Set seed for consistency 
        np.random.seed(seed)

        # Sample from a multivariate normal distribution
        return np.random.multivariate_normal(mean=self.mu, cov=self.Sigma, size=n_samples)

    def plot_2D(self, ax, zoom, z_factor: float = 1.):
        # Create a grid of x and y values
        x = np.linspace(zoom[0][0], zoom[0][1], 100)
        y = np.linspace(zoom[1][0], zoom[0][1], 100)
        X, Y = np.meshgrid(x, y)

        # Calculate the 2D Gaussian values for each point in the grid
        Z = (1 / (2 * np.pi * self.Sigma[0][0] * self.Sigma[1][1]) *
            np.exp(-((X - self.mu[0])**2 / (2 * self.Sigma[0][0]**2) + (Y - self.mu[1])**2 / (2 * self.Sigma[1][1]**2))))

        # Plot the contour lines of the Gaussian
        contour = ax.contour(X, Y, Z * z_factor, cmap='gray') 
