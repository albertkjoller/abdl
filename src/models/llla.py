import numpy as np
from math import sqrt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from torch.distributions.multivariate_normal import MultivariateNormal

from backpack import extend, backpack, extensions

from src.models.train_model import train_model

class SimpleLLLA(nn.Module):
    def __init__(self, args, num_classes: int = 2, seed: int = 42):
        super(SimpleLLLA, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.feature_extr   = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.Linear(32, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )      
        self.classifier = nn.Linear(32, num_classes, bias=False)
        
        # Model parameters
        self.num_classes    = num_classes
        self.seed           = seed    
        self.args           = args  

        # LLLA parameters
        self.LLLA_fitted            = False
        self.variance_factor        = 1.
    
    def forward(self, x: torch.FloatTensor):
        x = self.feature_extr(x)
        return self.classifier(x)

    def fit_(self, Xtrain, ytrain, Xtest, ytest):
        # Reset modle weights
        self.__init__(self.args, self.num_classes, self.seed)
        # Add to device
        self.to(self.args.device)
        # Define optimizer
        optimizer   = optim.Adam(self.parameters(), lr=self.args.lr)
        # Run model training
        train_model(self, optimizer, Xtrain, ytrain, Xtest, ytest, self.args.epochs, self.args.val_every_step, self.args.save_dir_model, self.seed, self.args.model_verbose)
        # "Train" LLLA 
        self.fit_LLLA(Xtrain, ytrain)
        # Set predict function to posterior
        self.predict_proba = self.predict_posterior_proba

    def predict_MAP_proba(self, X: torch.FloatTensor):
        outputs = self.forward(X)
        return softmax(outputs, dim=1).cpu().numpy()
    
    def predict_posterior_proba(self, X: torch.FloatTensor, seed: int = 0):
        posterior_samples   = self.sample(X=X, n_samples=self.args.n_posterior_samples, seed=seed)
        probs = softmax(posterior_samples, dim=0).mean(axis=1).T  
        return probs.cpu().numpy()
    
    def fit_LLLA(self, Xtrain: torch.FloatTensor, ytrain: torch.LongTensor):
        ### PARTS OF CODE DONE BY THESE PERSONS: https://github.com/wiseodd/last_layer_laplace/blob/master/bnn_laplace_multiclass.ipynb  
        self.LLLA_fitted = True

        self.W_last_layer   = list(self.parameters())[-1]
        shape_W             = self.W_last_layer.shape
        device              = self.W_last_layer.device

        # Use BackPACK to get the Kronecker-factored last-layer covariance
        extend(self.classifier)
        loss_func = extend(nn.CrossEntropyLoss(reduction='sum'))

        loss = loss_func(self(Xtrain), ytrain)
        with backpack(extensions.KFAC()):
            loss.backward()

        # The Kronecker-factored Hessian of the negative log-posterior
        A, B = self.W_last_layer.kfac

        # The weight decay used for training is the Gaussian prior's precision
        prec0 = 5e-4

        # The posterior covariance's Kronecker factors
        self.U = torch.inverse(A + sqrt(prec0)*torch.eye(shape_W[0]).to(device))
        self.V = torch.inverse(B + sqrt(prec0)*torch.eye(shape_W[1]).to(device))

    def sample(self, X: torch.FloatTensor, n_samples: int = 1000, seed: int = 0):
        ### PARTS OF CODE DONE BY THESE PERSONS: https://github.com/wiseodd/last_layer_laplace/blob/master/bnn_laplace_multiclass.ipynb  
        assert self.LLLA_fitted, "You must explicitly fit the LLLA layer before trying to sample!" 
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # feed input through first part of network
        with torch.no_grad():
            z = self.feature_extr(X)
            # MAP prediction
            Wmap = z @ self.W_last_layer.T

            # v is the induced covariance. 
            # See Appendix B.1 of https://arxiv.org/abs/2002.10118 for the detail of the derivation.
            v = torch.diag(z @ self.V @ z.T).reshape(-1, 1, 1) * self.U 
            
            # TODO: check reshaping error in samples
            output_dist = MultivariateNormal(Wmap, v * self.variance_factor)
            posterior_samples     = torch.zeros((self.num_classes, n_samples, len(X)))
            for n in range(n_samples):
                posterior_samples[:, n, :] = output_dist.rsample().T 
                
        return posterior_samples