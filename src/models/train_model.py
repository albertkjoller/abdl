import numpy as np
from tqdm import tqdm

import torch
from torch.nn.functional import log_softmax, nll_loss

# TODO: include batches
def train_model(model, optimizer, Xtrain, ytrain, Xval, yval, epochs, val_step, save_dir, seed, verbose=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

    current_best_loss   = np.inf
    pbar                = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in pbar:
        model.train()

        # Set grad zero
        optimizer.zero_grad()

        # Get predictions
        outputs         = model(Xtrain)
        log_probs       = log_softmax(outputs, dim=1)

        # Compute loss
        loss    = nll_loss(log_probs, ytrain)
        loss.backward()
        # Optimize
        optimizer.step()

        # Compute accuracy for batch
        _, pred_class   = torch.topk(outputs, k=1)
        train_acc       = torch.mean((pred_class.flatten() == ytrain).float()).cpu()
        train_loss      = loss.item()
        
        if epoch % val_step == 0:
            model.eval()
            with torch.no_grad():
                # Get predictions
                outputs         = model(Xval)
                log_probs       = log_softmax(outputs, dim=1)
                
                # Compute loss
                loss    = nll_loss(log_probs, yval)

                # Compute accuracy for batch
                _, pred_class   = torch.topk(outputs, k=1)
                val_acc         = torch.mean((pred_class.flatten() == yval).float()).cpu()
                val_loss        = loss.item()

            if val_loss < current_best_loss:
                current_best_loss   = val_loss
                best_model          = model
                best_epoch          = epoch 
                
                torch.save(model.state_dict(), f'{save_dir}/{model.__class__.__name__}_best.pth')
    
        if verbose:
            pbar.set_description(f"EPOCH {epoch+1}/{epochs}: Train acc. = {train_acc.item():.4f} \t | Validation acc. = {val_acc:.4f} \t | Train loss = {train_loss:.4f} \t | Validation loss = {val_loss:.4f}")
    
    if verbose:
        print(f"BEST EPOCH = {best_epoch}")
    return best_model